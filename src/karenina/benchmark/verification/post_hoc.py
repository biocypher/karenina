"""Evaluate rubrics over already-stored verification results.

This is the post-hoc path used by trace characterizations: stored rows are
judged against a rubric without regenerating answers or modifying the source
results. Evaluation runs through TaskEval in rubric-only mode, so deterministic
and model-backed traits share the standard verification pipeline semantics.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Hashable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Any

from karenina.benchmark.task_eval.task_eval import TaskEval
from karenina.benchmark.verification.async_lifecycle import get_async_portal, set_async_portal
from karenina.benchmark.verification.portal_pool import sequential_portal
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import Rubric
from karenina.schemas.results import ResultRowKey
from karenina.schemas.verification.config import VerificationConfig
from karenina.schemas.verification.result import VerificationResult

logger = logging.getLogger(__name__)

PostHocItem = tuple[ResultRowKey, str, "RowContext"]
PreparedPostHocItem = tuple[PostHocItem, Rubric, str | None]


@dataclass(frozen=True, slots=True)
class RowContext:
    """Optional question and reference answer supplied to a row's judge."""

    question: str = ""
    ground_truth: str | None = None


@dataclass(frozen=True, slots=True)
class PostHocJudgment:
    """Rubric scores for one stored row or parser-collapsed trace."""

    key: ResultRowKey
    sibling_keys: tuple[ResultRowKey, ...] = ()
    scores: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    representative_result_id: str | None = None
    sibling_result_ids: tuple[str, ...] = ()


def build_rubric_only_config(parsing_model: ModelConfig) -> VerificationConfig:
    """Build the minimal configuration for judging stored text."""
    return VerificationConfig(
        parsing_models=[parsing_model],
        parsing_only=True,
        evaluation_mode="rubric_only",
        abstention_enabled=False,
        sufficiency_enabled=False,
    )


def _extract_scores_and_labels(outcome: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Flatten rubric scores and literal labels from a TaskEval outcome."""
    scores: dict[str, Any] = {}
    labels: dict[str, str] = {}
    global_eval = getattr(outcome, "global_eval", None)
    if global_eval is None:
        return scores, labels
    for result_group in global_eval.verification_results.values():
        for result in result_group:
            if result.rubric is None:
                continue
            scores.update(result.rubric.get_all_trait_scores())
            labels.update(result.rubric.get_llm_trait_labels())
    return scores, labels


def _extract_evaluation_error(outcome: Any) -> str | None:
    """Return the first TaskEval failure, if row evaluation failed."""
    global_eval = getattr(outcome, "global_eval", None)
    if global_eval is None:
        return "TaskEval returned no global evaluation"
    for failures in global_eval.failed_questions.values():
        if failures:
            return str(failures[0])
    return None


def _evaluate_one(
    item: PostHocItem,
    rubric: Rubric,
    config: VerificationConfig,
    result_id: str | None = None,
) -> PostHocJudgment:
    """Evaluate one prepared item and capture failures as judgment data."""
    key, text, context = item
    try:
        task = TaskEval(task_id=f"post-hoc-{key.question_id}")
        task.log(text)
        if context.question or context.ground_truth is not None:
            task.add_question(
                {
                    "id": key.question_id,
                    "question": context.question,
                    "raw_answer": context.ground_truth or "",
                }
            )
        task.add_rubric(rubric)
        outcome = task.evaluate(config, run_name=key.run_name)
        evaluation_error = _extract_evaluation_error(outcome)
        if evaluation_error is not None:
            return PostHocJudgment(
                key=key,
                sibling_keys=(key,),
                representative_result_id=result_id,
                sibling_result_ids=(result_id,) if result_id is not None else (),
                error=evaluation_error,
            )
        scores, labels = _extract_scores_and_labels(outcome)
        if not scores or all(score is None for score in scores.values()):
            return PostHocJudgment(
                key=key,
                sibling_keys=(key,),
                representative_result_id=result_id,
                sibling_result_ids=(result_id,) if result_id is not None else (),
                error="TaskEval produced no usable rubric scores",
            )
        return PostHocJudgment(
            key=key,
            sibling_keys=(key,),
            representative_result_id=result_id,
            sibling_result_ids=(result_id,) if result_id is not None else (),
            scores=scores,
            labels=labels,
        )
    except Exception as error:
        logger.warning("Post-hoc rubric evaluation failed for %s", key, exc_info=True)
        return PostHocJudgment(
            key=key,
            sibling_keys=(key,),
            representative_result_id=result_id,
            sibling_result_ids=(result_id,) if result_id is not None else (),
            error=str(error),
        )


def _evaluate_prepared_one(
    prepared: PreparedPostHocItem,
    config: VerificationConfig,
) -> PostHocJudgment:
    """Evaluate one item carrying its row-specific rubric."""
    item, rubric, result_id = prepared
    return _evaluate_one(item, rubric, config, result_id)


def _evaluate_prepared_one_on_portal(
    prepared: PreparedPostHocItem,
    config: VerificationConfig,
    portal: Any,
) -> PostHocJudgment:
    """Evaluate one item with the caller's shared async portal installed."""
    previous = get_async_portal()
    set_async_portal(portal)
    try:
        return _evaluate_prepared_one(prepared, config)
    finally:
        set_async_portal(previous)


def _evaluate_prepared(
    items: Iterable[PreparedPostHocItem],
    config: VerificationConfig,
    *,
    max_workers: int,
) -> Iterator[PostHocJudgment]:
    """Evaluate prepared items serially or with bounded concurrency."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    existing_portal = get_async_portal()
    portal_context = nullcontext(existing_portal) if existing_portal is not None else sequential_portal()
    with portal_context as portal:
        if max_workers == 1:
            for item in items:
                yield _evaluate_prepared_one(item, config)
            return

        item_iterator = iter(items)
        in_flight_cap = max_workers * 2
        executor = ThreadPoolExecutor(max_workers=max_workers)
        in_flight: set[Future[PostHocJudgment]] = set()
        try:
            for item in itertools.islice(item_iterator, in_flight_cap):
                in_flight.add(
                    executor.submit(
                        _evaluate_prepared_one_on_portal,
                        item,
                        config,
                        portal,
                    )
                )
            while in_flight:
                completed, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed:
                    yield future.result()
                for item in itertools.islice(item_iterator, in_flight_cap - len(in_flight)):
                    in_flight.add(
                        executor.submit(
                            _evaluate_prepared_one_on_portal,
                            item,
                            config,
                            portal,
                        )
                    )
        finally:
            # Keep the shared portal alive until already-running judgments
            # finish. Pending work is cancelled when a consumer stops early.
            executor.shutdown(wait=True, cancel_futures=True)


def evaluate_rubric_on_texts(
    items: Iterable[PostHocItem],
    rubric: Rubric,
    parsing_model: ModelConfig,
    *,
    max_workers: int = 4,
) -> Iterator[PostHocJudgment]:
    """Judge prepared key, text, and context triples against a rubric.

    Args:
        items: Prepared stored-text evaluation inputs.
        rubric: Traits to evaluate.
        parsing_model: Model used by traits that require an LLM.
        max_workers: Maximum concurrent row evaluations.

    Yields:
        One judgment per input, in completion order when concurrent.

    Raises:
        ValueError: If ``max_workers`` is less than one.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    config = build_rubric_only_config(parsing_model)
    prepared = ((item, rubric, None) for item in items)
    yield from _evaluate_prepared(prepared, config, max_workers=max_workers)


def _default_text_selector(result: VerificationResult) -> str:
    """Select the stored raw response, or empty text when absent."""
    if result.template is None:
        return ""
    return result.template.raw_llm_response


def evaluate_rubric_on_results(
    results: Iterable[VerificationResult],
    rubric: Rubric,
    parsing_model: ModelConfig,
    *,
    text_selector: Callable[[VerificationResult], str] | None = None,
    row_context: Callable[[VerificationResult], RowContext] | None = None,
    row_filter: Callable[[VerificationResult], bool] | None = None,
    rubric_factory: Callable[[VerificationResult, Rubric], Rubric] | None = None,
    sibling_identity: Callable[[VerificationResult], Hashable] | None = None,
    collapse_parser_siblings: bool = True,
    max_workers: int = 4,
) -> Iterator[PostHocJudgment]:
    """Judge stored result rows, optionally once per generated trace.

    Args:
        results: Stored verification results.
        rubric: Traits to evaluate.
        parsing_model: Model used by traits that require an LLM.
        text_selector: Optional stored-text selector.
        row_context: Optional question and reference-answer selector.
        row_filter: Optional predicate used to exclude rows.
        rubric_factory: Optional factory for a row-specific rubric.
        sibling_identity: Optional parser-independent grouping identity.
        collapse_parser_siblings: Judge rows sharing trace identity once.
        max_workers: Maximum concurrent row evaluations.

    Yields:
        Post-hoc judgments with every represented row key in ``sibling_keys``.
    """
    select_text = text_selector or _default_text_selector
    select_context = row_context or (lambda _result: RowContext())

    prepared: list[PreparedPostHocItem] = []
    siblings: dict[Hashable, list[tuple[ResultRowKey, str]]] = {}
    judgment_groups: dict[str, Hashable] = {}
    for result in results:
        if row_filter is not None and not row_filter(result):
            continue
        key = ResultRowKey.from_result(result)
        result_id = result.metadata.result_id
        selected_rubric = rubric_factory(result, rubric) if rubric_factory else rubric
        if not collapse_parser_siblings:
            prepared.append(((key, select_text(result), select_context(result)), selected_rubric, result_id))
            continue
        trace_identity: Hashable = sibling_identity(result) if sibling_identity is not None else key.trace_identity
        if trace_identity in siblings:
            siblings[trace_identity].append((key, result_id))
            continue
        siblings[trace_identity] = [(key, result_id)]
        judgment_groups[result_id] = trace_identity
        prepared.append(((key, select_text(result), select_context(result)), selected_rubric, result_id))

    config = build_rubric_only_config(parsing_model)
    for judgment in _evaluate_prepared(prepared, config, max_workers=max_workers):
        if collapse_parser_siblings:
            representative_id = judgment.representative_result_id
            if representative_id is None:
                raise RuntimeError("Stored-result judgment has no representative result id")
            grouped = siblings[judgment_groups[representative_id]]
            yield replace(
                judgment,
                sibling_keys=tuple(key for key, _result_id in grouped),
                sibling_result_ids=tuple(result_id for _key, result_id in grouped),
            )
        else:
            yield judgment
