"""Extend a prior verification run along three axes: judges, answerers, replicates.

Mirrors the error-analysis facade pattern: result-set-in, result-set-out. A
``ReplayStore`` built from the prior results serves the answering stage for
triples already covered; new answerers or replicates miss and run live.
``VerificationConfig.skip_triples`` tells the batch runner to drop tasks
already present in ``prior_results`` so those rows pass through verbatim.

This module also hosts :func:`extend_rubric_run`, a sibling facade for
attaching a new rubric to a prior run. Unlike the judge/answerer/replicate
extension, rubric extension enriches prior rows in-place: answering is
replayed, template parsing and verification are skipped
(``evaluation_mode="rubric_only"``), and the resulting rubric scores are
merged trait-by-trait onto the prior rows. Shape and row count are
preserved; same-name trait collisions raise.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from karenina.replay import capture_from_result_set
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity

if TYPE_CHECKING:
    from karenina.benchmark import Benchmark
    from karenina.schemas.entities.rubric import (
        AgenticRubricTrait,
        CallableRubricTrait,
        LLMRubricTrait,
        MetricRubricTrait,
        RegexRubricTrait,
        Rubric,
    )

logger = logging.getLogger(__name__)


def extend_template_run(
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
        "extend_template: captured replay store with %d entries from %d prior results",
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
        "extend_template: new answerers=%s, added replicates=%d, skip_triples=%d",
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
        "extend_template: merged %d prior + %d new = %d total results under run_name=%r",
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
            "config.replay_store must be None; extend_template builds the replay store internally from prior_results"
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
            "reduction is not supported by extend_template; pass "
            f"replicate_count>={observed}."
        )


def _infer_run_name(prior_results: VerificationResultSet) -> str:
    names: set[str] = {r.metadata.run_name for r in prior_results.results if r.metadata.run_name is not None}
    if len(names) == 0:
        raise ValueError("prior_results rows have no run_name; pass run_name= explicitly to extend_template")
    if len(names) > 1:
        raise ValueError(
            f"prior_results rows carry inconsistent run_names ({sorted(names)}); "
            "pass run_name= explicitly to extend_template"
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


# ----------------------------------------------------------------------
# Rubric extension: attach a new rubric to an existing run
# ----------------------------------------------------------------------


_RUBRIC_TRAIT_BUCKETS: tuple[str, ...] = (
    "llm_trait_scores",
    "llm_trait_labels",
    "regex_trait_scores",
    "callable_trait_scores",
    "agentic_trait_scores",
    "agentic_trait_investigation_traces",
)


def extend_rubric_run(
    benchmark: Benchmark,
    prior_results: VerificationResultSet,
    config: VerificationConfig,
    *,
    run_name: str | None = None,
    question_ids: list[str] | None = None,
    async_enabled: bool | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> VerificationResultSet:
    """Attach a new rubric to an existing verification run.

    The caller sets the new rubric on ``benchmark`` (global and/or
    per-question) *before* invoking this helper. A replay store built
    from ``prior_results`` serves the answering stage, the pipeline
    runs in ``evaluation_mode="rubric_only"`` (template verification
    skipped), and the resulting rubric scores are merged
    trait-by-trait onto copies of the prior rows. Row count is
    preserved: the merged set has exactly ``len(prior_results.results)``
    entries.

    Args:
        benchmark: The :class:`Benchmark` carrying the new rubric(s).
        prior_results: Result set from an earlier ``run_verification``
            call. Must be non-empty.
        config: Verification configuration describing the **same shape**
            as ``prior_results`` (``answering_models``,
            ``parsing_models``, ``replicate_count`` must equal the
            observed prior shape). ``replay_store`` must be ``None``;
            the helper owns that slot. ``evaluation_mode`` must be
            either the default ``"template_only"`` or ``"rubric_only"``
            (the helper rewrites it internally).
        run_name: Optional override for the merged run name. Defaults
            to the run name carried by ``prior_results``.
        question_ids: Optional subset of question IDs. Defaults to every
            question present in ``prior_results``.
        async_enabled: Forwarded to ``run_verification``.
        progress_callback: Forwarded to ``run_verification``.

    Returns:
        ``VerificationResultSet`` of enriched prior rows (same count as
        ``prior_results.results``) carrying the new rubric scores.

    Raises:
        ValueError: On any input validation failure (empty prior set,
            shape mismatch, pre-populated ``replay_store``, unsupported
            ``evaluation_mode``, no rubric attached, metric traits
            present, inconsistent run names without override, trait
            name collision, or shape corruption at merge time).
    """
    _validate_rubric_extension(benchmark, prior_results, config, run_name)

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
        "extend_rubric: captured replay store with %d entries from %d prior results",
        len(replay_store.entries),
        len(prior_results.results),
    )

    extended_config = config.model_copy(update={"replay_store": replay_store, "evaluation_mode": "rubric_only"})

    new_results = benchmark.run_verification(
        config=extended_config,
        question_ids=effective_question_ids,
        run_name=effective_run_name,
        async_enabled=async_enabled,
        progress_callback=progress_callback,
    )

    enriched = _enrich_with_rubric(prior_results, new_results, effective_run_name)
    logger.info(
        "extend_rubric: enriched %d prior rows under run_name=%r",
        len(enriched.results),
        effective_run_name,
    )
    return enriched


def _validate_rubric_extension(
    benchmark: Benchmark,
    prior_results: VerificationResultSet,
    config: VerificationConfig,
    run_name_override: str | None,
) -> None:
    if not prior_results.results:
        raise ValueError("prior_results must contain at least one VerificationResult")

    if not config.answering_models:
        raise ValueError("config.answering_models must be non-empty")

    if not config.parsing_models:
        raise ValueError("config.parsing_models must be non-empty")

    if config.replay_store is not None:
        raise ValueError(
            "config.replay_store must be None; extend_rubric builds the replay store internally from prior_results"
        )

    if config.evaluation_mode not in ("template_only", "rubric_only"):
        raise ValueError(
            "config.evaluation_mode must be 'template_only' or 'rubric_only' for extend_rubric "
            f"(got {config.evaluation_mode!r}); the helper rewrites it to 'rubric_only' internally"
        )

    _validate_answering_identity(prior_results, config)
    _validate_parsing_identity(prior_results, config)
    _validate_replicate_count_equals(prior_results, config)

    rubric = _collect_active_rubric(benchmark, prior_results)
    if rubric is None:
        raise ValueError(
            "benchmark has no rubric attached (global or per-question) for any question in prior_results; "
            "set a rubric before calling extend_rubric"
        )
    if rubric.metric_traits:
        raise ValueError(
            "Metric traits are not supported by extend_rubric in v1; "
            "remove them from the rubric or use template_and_rubric on a fresh run"
        )

    if run_name_override is None:
        _infer_run_name(prior_results)


def _validate_parsing_identity(
    prior_results: VerificationResultSet,
    config: VerificationConfig,
) -> None:
    prior_parsing: set[str] = {r.metadata.parsing.canonical_key for r in prior_results.results}
    configured_parsing: set[str] = {
        ModelIdentity.from_model_config(m, role="parsing").canonical_key for m in config.parsing_models
    }
    if prior_parsing != configured_parsing:
        raise ValueError(
            "config.parsing_models does not match the parsing identities in prior_results. "
            f"Configured: {sorted(configured_parsing)}. Observed in prior: {sorted(prior_parsing)}."
        )


def _validate_replicate_count_equals(
    prior_results: VerificationResultSet,
    config: VerificationConfig,
) -> None:
    observed = _observed_replicate_count(prior_results)
    if observed == 0:
        return
    if config.replicate_count != observed:
        raise ValueError(
            f"config.replicate_count={config.replicate_count} does not match the replicate "
            f"fan-out observed in prior_results (={observed}). extend_rubric preserves the "
            "prior shape; pass replicate_count equal to the observed count."
        )


def _collect_active_rubric(
    benchmark: Benchmark,
    prior_results: VerificationResultSet,
) -> Rubric | None:
    """Return a synthetic Rubric merging every trait that will run under extend_rubric.

    Combines the global rubric with every per-question rubric bound to a
    question appearing in ``prior_results``. The result is used only for
    validation (presence check + metric-trait rejection); the pipeline
    still resolves rubrics per-question at runtime.
    """
    from karenina.schemas.entities import Rubric

    llm: list[LLMRubricTrait] = []
    regex: list[RegexRubricTrait] = []
    callbl: list[CallableRubricTrait] = []
    metric: list[MetricRubricTrait] = []
    agentic: list[AgenticRubricTrait] = []

    global_rubric = benchmark._rubric_manager.get_global_rubric()
    if global_rubric is not None:
        llm.extend(global_rubric.llm_traits)
        regex.extend(global_rubric.regex_traits)
        callbl.extend(global_rubric.callable_traits)
        metric.extend(global_rubric.metric_traits)
        agentic.extend(global_rubric.agentic_traits)

    qids = {r.metadata.question_id for r in prior_results.results}
    for qid in qids:
        try:
            qrubric = benchmark._rubric_manager.get_merged_rubric_for_question(qid)
        except Exception:  # noqa: BLE001
            qrubric = None
        if qrubric is None:
            continue
        # merged_rubric already includes the global traits; extend from a
        # fresh view to avoid duplicate collection. We only pick up
        # question-specific additions by diffing trait names.
        all_current: list[
            LLMRubricTrait | RegexRubricTrait | CallableRubricTrait | MetricRubricTrait | AgenticRubricTrait
        ] = [
            *llm,
            *regex,
            *callbl,
            *metric,
            *agentic,
        ]
        global_names = {trait.name for trait in all_current}
        for lt in qrubric.llm_traits:
            if lt.name not in global_names:
                llm.append(lt)
        for rt in qrubric.regex_traits:
            if rt.name not in global_names:
                regex.append(rt)
        for ct in qrubric.callable_traits:
            if ct.name not in global_names:
                callbl.append(ct)
        for mt in qrubric.metric_traits:
            if mt.name not in global_names:
                metric.append(mt)
        for at in qrubric.agentic_traits:
            if at.name not in global_names:
                agentic.append(at)

    if not (llm or regex or callbl or metric or agentic):
        return None

    return Rubric(
        llm_traits=llm,
        regex_traits=regex,
        callable_traits=callbl,
        metric_traits=metric,
        agentic_traits=agentic,
    )


def _enrich_with_rubric(
    prior_results: VerificationResultSet,
    new_results: VerificationResultSet,
    run_name: str,
) -> VerificationResultSet:
    new_by_triple: dict[tuple[str, str, str, int | None], VerificationResult] = {}
    for row in new_results.results:
        key = (
            row.metadata.question_id,
            row.metadata.answering.canonical_key,
            row.metadata.parsing.canonical_key,
            row.metadata.replicate,
        )
        if key in new_by_triple:
            raise ValueError(
                f"extend_rubric: rubric-only run produced duplicate rows for triple {key}; shape corruption"
            )
        new_by_triple[key] = row

    enriched_rows: list[VerificationResult] = []
    for prior_row in prior_results.results:
        key = (
            prior_row.metadata.question_id,
            prior_row.metadata.answering.canonical_key,
            prior_row.metadata.parsing.canonical_key,
            prior_row.metadata.replicate,
        )
        if key not in new_by_triple:
            raise ValueError(f"extend_rubric: no rubric-only row produced for prior triple {key}; shape corruption")
        merged_row = prior_row.model_copy(deep=True)
        if merged_row.rubric is None:
            from karenina.schemas.verification.result_components import VerificationResultRubric

            merged_row.rubric = VerificationResultRubric()
        _merge_rubric_onto_row(merged_row, new_by_triple[key])
        if merged_row.metadata.run_name != run_name:
            merged_row.metadata.run_name = run_name
        enriched_rows.append(merged_row)

    return VerificationResultSet(
        results=enriched_rows,
        scenario_results=list(prior_results.scenario_results) if prior_results.scenario_results else None,
        errors=list(prior_results.errors) if prior_results.errors else None,
    )


def _merge_rubric_onto_row(target: VerificationResult, source: VerificationResult) -> None:
    """Union source.rubric.* trait dicts onto target.rubric.* in place.

    Raises ValueError on same-name trait collision within a bucket.
    """
    src_rubric = source.rubric
    dst_rubric = target.rubric
    if src_rubric is None or dst_rubric is None:
        return

    # Mark that rubric evaluation has now been performed on this row.
    if src_rubric.rubric_evaluation_performed:
        dst_rubric.rubric_evaluation_performed = True
    if src_rubric.rubric_evaluation_strategy is not None:
        dst_rubric.rubric_evaluation_strategy = src_rubric.rubric_evaluation_strategy

    for bucket in _RUBRIC_TRAIT_BUCKETS:
        src_dict = getattr(src_rubric, bucket)
        if not src_dict:
            continue
        dst_dict = getattr(dst_rubric, bucket)
        if dst_dict is None:
            setattr(dst_rubric, bucket, dict(src_dict))
            continue
        collisions = set(dst_dict) & set(src_dict)
        if collisions:
            raise ValueError(
                f"extend_rubric: trait name collision in {bucket} for traits "
                f"{sorted(collisions)}; rename or remove the colliding trait"
            )
        merged = dict(dst_dict)
        merged.update(src_dict)
        setattr(dst_rubric, bucket, merged)

    # Dynamic-rubric metadata: union with collision rejection on skipped-trait names.
    if src_rubric.dynamic_rubric_skipped_traits:
        dst_skipped = dst_rubric.dynamic_rubric_skipped_traits or {}
        collisions = set(dst_skipped) & set(src_rubric.dynamic_rubric_skipped_traits)
        if collisions:
            raise ValueError(
                f"extend_rubric: trait name collision in dynamic_rubric_skipped_traits for traits {sorted(collisions)}"
            )
        merged_skipped = dict(dst_skipped)
        merged_skipped.update(src_rubric.dynamic_rubric_skipped_traits)
        dst_rubric.dynamic_rubric_skipped_traits = merged_skipped

    if src_rubric.dynamic_rubric_promoted_traits:
        existing = list(dst_rubric.dynamic_rubric_promoted_traits or [])
        for t in src_rubric.dynamic_rubric_promoted_traits:
            if t not in existing:
                existing.append(t)
        dst_rubric.dynamic_rubric_promoted_traits = existing

    if src_rubric.trait_provenance:
        dst_prov = dst_rubric.trait_provenance or {}
        merged_prov = dict(dst_prov)
        merged_prov.update(src_rubric.trait_provenance)
        dst_rubric.trait_provenance = merged_prov
