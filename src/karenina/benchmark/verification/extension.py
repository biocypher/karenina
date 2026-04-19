"""Extend a prior verification run along three axes: judges, answerers, replicates.

Mirrors the error-analysis facade pattern: result-set-in, result-set-out. A
``ReplayStore`` built from the prior results serves the answering stage for
triples already covered; new answerers or replicates miss and run live.
``VerificationConfig.skip_triples`` tells the batch runner to drop tasks
already present in ``prior_results`` so those rows pass through verbatim.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from karenina.replay import capture_from_result_set
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationConfig
from karenina.schemas.verification.model_identity import ModelIdentity

if TYPE_CHECKING:
    from karenina.benchmark import Benchmark

logger = logging.getLogger(__name__)


def extend_verification_run(
    benchmark: Benchmark,
    prior_results: VerificationResultSet,
    config: VerificationConfig,
    *,
    run_name: str | None = None,
    question_ids: list[str] | None = None,
    async_enabled: bool | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> VerificationResultSet:
    """Extend a prior verification run along any combination of three axes.

    The axes, all optional and composable in a single call:

    1. **Parsing models (judges)**: new judge(s) in ``config.parsing_models``.
    2. **Answering models**: new answerer(s) in ``config.answering_models``.
    3. **Replicates**: higher ``config.replicate_count`` than the prior fan-out.

    A :class:`~karenina.replay.ReplayStore` is built from ``prior_results`` so
    that prior ``(question, answerer, replicate)`` combinations serve the
    answering stage from replay. New answerers, new replicates, or any
    combination thereof miss the store and run answering live. Parsing
    always runs live. Triples already covered by ``prior_results`` are
    filtered out of the task queue so prior rows pass through the merge
    verbatim. The returned set matches the shape of a joint run with the
    full ``(answerers × judges × replicates)`` matrix.

    Args:
        benchmark: The :class:`Benchmark` whose questions and templates the
            prior run evaluated. Question IDs on ``prior_results`` must exist
            on this benchmark.
        prior_results: Result set from an earlier ``run_verification`` call
            to extend. Must be non-empty.
        config: Verification configuration for the extension. Must describe
            the **final** state (full union, not deltas):

            - ``parsing_models`` = every judge you want in the merged output
              (old judges from ``prior_results`` + any new judges).
            - ``answering_models`` = every answerer you want in the merged
              output. Must be a superset of the answering identities in
              ``prior_results`` (compared via
              :meth:`ModelIdentity.from_model_config`).
            - ``replicate_count`` = the final number of replicates. Must be
              ``>=`` the fan-out observed in ``prior_results``.
            - ``replay_store`` must be ``None`` (the extension owns that slot).
        run_name: Optional override for the merged run name. Defaults to the
            run name carried by ``prior_results`` when all rows agree; raises
            ``ValueError`` on disagreement when no override is supplied.
        question_ids: Optional subset of question IDs to re-judge. Defaults
            to every question in ``prior_results``.
        async_enabled: Forwarded to ``run_verification``.
        progress_callback: Forwarded to ``run_verification``.

    Returns:
        Merged ``VerificationResultSet`` with ``len(prior_results.results)
        + len(new.results)`` rows, all stamped with the effective ``run_name``.

    Raises:
        ValueError: On any input validation failure (empty prior set, missing
            parsing models, answering identity mismatch, replicate count
            mismatch, pre-populated ``replay_store``, or inconsistent run
            names without an override).
    """
    _validate(prior_results, config, run_name)

    effective_run_name = run_name or _infer_run_name(prior_results)
    effective_question_ids = (
        list(question_ids)
        if question_ids is not None
        else sorted({r.metadata.question_id for r in prior_results.results})
    )

    replay_store = capture_from_result_set(
        prior_results,
        include_parsed=False,
        include_agent_traces=True,
        only_successful=False,
        replicate_selector="all",
    )
    logger.info(
        "extend_judgment: captured replay store with %d entries from %d prior results",
        len(replay_store.entries),
        len(prior_results.results),
    )

    skip_triples = frozenset(
        (
            r.metadata.question_id,
            r.metadata.answering.canonical_key,
            r.metadata.parsing.canonical_key,
            r.metadata.replicate,
        )
        for r in prior_results.results
    )

    observed_replicates = _observed_replicate_count(prior_results)
    prior_answering_keys: set[str] = {r.metadata.answering.canonical_key for r in prior_results.results}
    new_answering_keys = sorted(
        {ModelIdentity.from_model_config(m, role="answering").canonical_key for m in config.answering_models}
        - prior_answering_keys
    )
    added_replicates = config.replicate_count - observed_replicates
    logger.info(
        "extend_judgment: new answerers=%s, added replicates=%d, skip_triples=%d",
        new_answering_keys or "[]",
        added_replicates,
        len(skip_triples),
    )

    extended_config = config.model_copy(update={"replay_store": replay_store, "skip_triples": skip_triples})

    new_results = benchmark.run_verification(
        config=extended_config,
        question_ids=effective_question_ids,
        run_name=effective_run_name,
        async_enabled=async_enabled,
        progress_callback=progress_callback,
    )

    merged = _merge(prior_results, new_results, effective_run_name)
    logger.info(
        "extend_judgment: merged %d prior + %d new = %d total results under run_name=%r",
        len(prior_results.results),
        len(new_results.results),
        len(merged.results),
        effective_run_name,
    )
    return merged


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def _validate(
    prior_results: VerificationResultSet,
    config: VerificationConfig,
    run_name_override: str | None,
) -> None:
    if not prior_results.results:
        raise ValueError("prior_results must contain at least one VerificationResult")

    if not config.parsing_models:
        raise ValueError("config.parsing_models must be non-empty (supply the new judges)")

    if not config.answering_models:
        raise ValueError("config.answering_models must be non-empty")

    if config.replay_store is not None:
        raise ValueError(
            "config.replay_store must be None; extend_judgment builds the replay store internally from prior_results"
        )

    _validate_answering_identity(prior_results, config)
    _validate_replicate_count(prior_results, config)

    if run_name_override is None:
        # Probe run_name consistency early; _infer_run_name will raise the real error.
        _infer_run_name(prior_results)


def _validate_answering_identity(
    prior_results: VerificationResultSet,
    config: VerificationConfig,
) -> None:
    prior_answering: set[str] = {r.metadata.answering.canonical_key for r in prior_results.results}
    configured_answering: set[str] = {
        ModelIdentity.from_model_config(m, role="answering").canonical_key for m in config.answering_models
    }
    missing = prior_answering - configured_answering
    if missing:
        raise ValueError(
            "config.answering_models does not cover the answering identities in "
            f"prior_results. Missing keys: {sorted(missing)}. Configured keys: "
            f"{sorted(configured_answering)}."
        )


def _observed_replicate_count(prior_results: VerificationResultSet) -> int:
    counts_per_triple: Counter[tuple[str, str, str]] = Counter()
    replicate_values: dict[tuple[str, str, str], set[int | None]] = {}
    for r in prior_results.results:
        key = (
            r.metadata.question_id,
            r.metadata.answering.canonical_key,
            r.metadata.parsing.canonical_key,
        )
        replicate_values.setdefault(key, set()).add(r.metadata.replicate)
    for key, values in replicate_values.items():
        counts_per_triple[key] = len(values)
    return max(counts_per_triple.values()) if counts_per_triple else 0


def _validate_replicate_count(
    prior_results: VerificationResultSet,
    config: VerificationConfig,
) -> None:
    # config.replicate_count must be >= the replicate fan-out observed in
    # prior_results. Equal => pure judge/answerer extension; greater => also
    # adds replicates. Less is rejected: replicate reduction is out of scope.
    observed = _observed_replicate_count(prior_results)
    if config.replicate_count < observed:
        raise ValueError(
            f"config.replicate_count={config.replicate_count} is lower than the "
            f"replicate fan-out observed in prior_results (={observed}). Replicate "
            "reduction is not supported by extend_judgment; pass "
            f"replicate_count>={observed}."
        )


def _infer_run_name(prior_results: VerificationResultSet) -> str:
    names: set[str] = {r.metadata.run_name for r in prior_results.results if r.metadata.run_name is not None}
    if len(names) == 0:
        raise ValueError("prior_results rows have no run_name; pass run_name= explicitly to extend_judgment")
    if len(names) > 1:
        raise ValueError(
            f"prior_results rows carry inconsistent run_names ({sorted(names)}); "
            "pass run_name= explicitly to extend_judgment"
        )
    return next(iter(names))


# ----------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------


def _merge(
    prior_results: VerificationResultSet,
    new_results: VerificationResultSet,
    run_name: str,
) -> VerificationResultSet:
    merged_rows = list(prior_results.results) + list(new_results.results)
    for row in merged_rows:
        if row.metadata.run_name != run_name:
            row.metadata.run_name = run_name

    scenario_results: list[Any] | None = None
    if prior_results.scenario_results or new_results.scenario_results:
        scenario_results = list(prior_results.scenario_results or []) + list(new_results.scenario_results or [])

    errors: list[tuple[str, BaseException]] | None = None
    if prior_results.errors or new_results.errors:
        errors = list(prior_results.errors or []) + list(new_results.errors or [])

    return VerificationResultSet(
        results=merged_rows,
        scenario_results=scenario_results,
        errors=errors,
    )
