"""Rerun and analyze judgments for traces with a blank final assistant message."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from karenina.benchmark import PostHocJudgment, RowContext, evaluate_rubric_on_results
from karenina.schemas.entities.rubric import LLMRubricTrait, RegexRubricTrait, Rubric
from karenina.schemas.verification import VerificationResult
from paper.common.bootstrap import input_path
from paper.common.qa_results import QAResultRow, iter_results, load_benchmark_text
from paper.config import (
    OTP_BENCHMARK_JSONLD,
    OTP_MCP_RESULTS,
    RESPONSE_EMPTY_TRAILING_JUDGMENTS,
)
from paper.otp_response_characterization.config import (
    EMPTY_TRAILING_CLASSES,
    FAILURE_SHAPE_TRAITS,
    gpt_oss_judge,
)

logger = logging.getLogger(__name__)

TRAIT_NAME = "MissingFinalMessageOutcome"
CLASSIFICATIONS = tuple(EMPTY_TRAILING_CLASSES)

RUBRIC_PROMPT = """\
Classify why this MCP agent trace ended with an empty final assistant message.

Decide whether the trace already contains enough information to answer the
benchmark question correctly, even though the model failed to produce a final
assistant message.

Use exactly one class:
- answer_present_no_final_message: the correct answer, or an answer-equivalent
  tool result, is present in the trace.
- wrong_result_no_final_message: the trace reaches a substantive result, but
  the result conflicts with or misses the reference answer.
- no_answer_gave_up: no usable answer is present. The trace mainly contains
  failed tool calls, schema exploration, irrelevant work, or an incomplete
  search path.

Judge the trace against the reference answer. Related entities or partial
context are not sufficient. Prefer no_answer_gave_up when there is no clear
answer-bearing result.
"""


class MissingFinalJudgment(BaseModel):
    """One literal judgment over a unique generated trace."""

    model_config = ConfigDict(extra="allow")

    representative_result_id: str
    sibling_result_ids: list[str] = Field(min_length=1)
    question_id: str
    question_text: str
    reference_answer: str | None = None
    answering_model: str
    answering_identity: str | None = None
    replicate: int | None = None
    raw_trace_sha256: str
    classification: str
    classification_score: int | None = None
    judge_model: str | None = None
    error: str | None = None


def load_judgments(path: Path) -> list[MissingFinalJudgment]:
    """Load and validate stored missing-final-message judgment records."""
    judgments: list[MissingFinalJudgment] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                judgments.append(MissingFinalJudgment.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"Invalid judgment at {path}:{line_number}: {error}") from error
    return judgments


def _trace_hash(text: str) -> str:
    """Return the stable SHA256 identity of a stored trace."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_source_joins(
    judgments: Iterable[MissingFinalJudgment],
    source_rows: Iterable[VerificationResult],
) -> list[MissingFinalJudgment]:
    """Cross-check judgments and parser siblings against validated QA results.

    Returns:
        Materialized judgments after every source join and class check passes.

    Raises:
        ValueError: If classes, identities, sibling sets, or source joins are
            invalid.
    """
    materialized = list(judgments)
    source_by_id: dict[str, QAResultRow] = {}
    for result in source_rows:
        row = QAResultRow.from_result(result, "mcp")
        if row.result_id in source_by_id:
            raise ValueError(f"Duplicate result id in MCP source: {row.result_id}")
        source_by_id[row.result_id] = row

    claimed_ids: set[str] = set()
    seen_trace_keys: set[tuple[str, str, int | None, str]] = set()
    for judgment in materialized:
        if judgment.error is not None:
            raise ValueError(f"Stored judgment contains an evaluation error: {judgment.error}")
        if judgment.classification not in CLASSIFICATIONS:
            raise ValueError(f"Unknown missing-final-message class: {judgment.classification!r}")
        sibling_ids = judgment.sibling_result_ids
        if len(sibling_ids) != len(set(sibling_ids)):
            raise ValueError(f"Duplicate sibling ids for {judgment.representative_result_id}")
        if judgment.representative_result_id not in sibling_ids:
            raise ValueError("Representative result id is absent from its sibling set")
        overlap = claimed_ids.intersection(sibling_ids)
        if overlap:
            raise ValueError(f"Result ids occur in multiple judgments: {sorted(overlap)}")
        claimed_ids.update(sibling_ids)

        missing = [result_id for result_id in sibling_ids if result_id not in source_by_id]
        if missing:
            raise ValueError(f"Judgment sibling ids are absent from the MCP source: {missing}")
        siblings = [source_by_id[result_id] for result_id in sibling_ids]
        identities = {
            (row.question_id, row.answerer, row.replicate, _trace_hash(row.raw_trace))
            for row in siblings
        }
        if len(identities) != 1:
            raise ValueError("Judgment sibling rows do not represent one generated trace")
        trace_key = next(iter(identities))
        if trace_key in seen_trace_keys:
            raise ValueError("Duplicate generated trace in stored judgments")
        seen_trace_keys.add(trace_key)
        if trace_key[0] != judgment.question_id or trace_key[1] != judgment.answering_model:
            raise ValueError("Stored judgment metadata disagrees with its source trace")
        if trace_key[2] != judgment.replicate or trace_key[3] != judgment.raw_trace_sha256:
            raise ValueError("Stored judgment replicate or trace hash disagrees with its source")
    return materialized


def summarize_judgments(
    judgments: Iterable[MissingFinalJudgment],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize stored classifications overall and by answerer."""
    rows = list(judgments)
    class_counts = Counter(row.classification for row in rows)
    sibling_counts: Counter[str] = Counter()
    by_answerer: Counter[tuple[str, str]] = Counter()
    sibling_by_answerer: Counter[tuple[str, str]] = Counter()
    for row in rows:
        sibling_count = len(row.sibling_result_ids)
        sibling_counts[row.classification] += sibling_count
        by_answerer[(row.answering_model, row.classification)] += 1
        sibling_by_answerer[(row.answering_model, row.classification)] += sibling_count

    unknown = set(class_counts).difference(CLASSIFICATIONS)
    if unknown:
        raise ValueError(f"Unknown missing-final-message classes: {sorted(unknown)}")
    total_unique = len(rows)
    total_siblings = sum(sibling_counts.values())
    summary = pd.DataFrame(
        [
            {
                "classification": classification,
                "unique_traces": class_counts[classification],
                "sibling_rows": sibling_counts[classification],
                "unique_fraction": class_counts[classification] / total_unique if total_unique else 0.0,
                "sibling_row_fraction": (
                    sibling_counts[classification] / total_siblings if total_siblings else 0.0
                ),
            }
            for classification in CLASSIFICATIONS
        ]
    )
    answerers = sorted({row.answering_model for row in rows})
    by_answerer_frame = pd.DataFrame(
        [
            {
                "answering_model": answerer,
                "classification": classification,
                "unique_traces": by_answerer[(answerer, classification)],
                "sibling_rows": sibling_by_answerer[(answerer, classification)],
            }
            for answerer in answerers
            for classification in CLASSIFICATIONS
        ]
    )
    return summary, by_answerer_frame


def _live_rubric() -> Rubric:
    """Build the three-class missing-final-message rubric."""
    return Rubric(
        llm_traits=[
            LLMRubricTrait(
                name=TRAIT_NAME,
                summary="classifies a blank final assistant response",
                description=RUBRIC_PROMPT,
                kind="literal",
                min_score=None,
                max_score=None,
                classes=EMPTY_TRAILING_CLASSES,
                deep_judgment_enabled=False,
                deep_judgment_excerpt_enabled=False,
                deep_judgment_max_excerpts=None,
                deep_judgment_fuzzy_match_threshold=None,
                deep_judgment_excerpt_retry_attempts=None,
                deep_judgment_search_enabled=False,
                include_ground_truth=True,
                higher_is_better=None,
            )
        ]
    )


def _missing_trace_identity(result: VerificationResult) -> tuple[str, str, int | None, str]:
    """Return a generated-trace identity that includes its text hash."""
    raw_trace = result.template.raw_llm_response if result.template else ""
    return (
        result.metadata.question_id,
        result.metadata.answering.canonical_key,
        result.metadata.replicate,
        _trace_hash(raw_trace),
    )


def _empty_trailing_trait() -> RegexRubricTrait:
    """Return the deterministic selector used before literal judgment."""
    return next(
        trait
        for trait in FAILURE_SHAPE_TRAITS
        if trait.name == "EmptyTrailingAI"
    )


def _select_empty_trailing_results(limit: int | None = None) -> list[VerificationResult]:
    """Select parser siblings for unique traces ending in a blank AI message."""
    selector = Rubric(regex_traits=[_empty_trailing_trait()])
    selected_groups: list[PostHocJudgment] = []
    for judgment in evaluate_rubric_on_results(
        iter_results(input_path(OTP_MCP_RESULTS)),
        selector,
        gpt_oss_judge(),
        sibling_identity=_missing_trace_identity,
        max_workers=1,
    ):
        if judgment.scores.get("EmptyTrailingAI") is True:
            selected_groups.append(judgment)
            if limit is not None and len(selected_groups) == limit:
                break

    selected_ids = {
        result_id
        for judgment in selected_groups
        for result_id in judgment.sibling_result_ids
    }
    selected = [
        result
        for result in iter_results(input_path(OTP_MCP_RESULTS))
        if result.metadata.result_id in selected_ids
    ]
    if len(selected) != len(selected_ids):
        raise ValueError("Selected missing-final-message rows do not rejoin the MCP source")
    return selected


def _fresh_judgments(limit: int | None = None) -> list[MissingFinalJudgment]:
    """Rerun the literal rubric over every selected generated trace."""
    selected = _select_empty_trailing_results(limit)
    benchmark = load_benchmark_text(input_path(OTP_BENCHMARK_JSONLD))
    by_result_id = {result.metadata.result_id: result for result in selected}
    logger.info(
        "Calling GPT-OSS for %d unique missing-final-message traces",
        len({_missing_trace_identity(result) for result in selected}),
    )
    evaluated = evaluate_rubric_on_results(
        selected,
        _live_rubric(),
        gpt_oss_judge(),
        row_context=lambda result: RowContext(
            question=benchmark[result.metadata.question_id][0],
            ground_truth=benchmark[result.metadata.question_id][1],
        ),
        sibling_identity=_missing_trace_identity,
        max_workers=1,
    )

    records: list[MissingFinalJudgment] = []
    for judgment in evaluated:
        if judgment.error is not None:
            raise RuntimeError(f"Live missing-final-message judgment failed: {judgment.error}")
        representative_id = judgment.representative_result_id
        if representative_id is None:
            raise RuntimeError("Live judgment has no representative result id")
        result = by_result_id[representative_id]
        classification = judgment.labels.get(TRAIT_NAME)
        if classification not in CLASSIFICATIONS:
            raise RuntimeError(f"Live judge returned an invalid class: {classification!r}")
        question, reference = benchmark[result.metadata.question_id]
        raw_trace = result.template.raw_llm_response if result.template else ""
        records.append(
            MissingFinalJudgment(
                representative_result_id=representative_id,
                sibling_result_ids=list(judgment.sibling_result_ids),
                question_id=result.metadata.question_id,
                question_text=question,
                reference_answer=reference,
                answering_model=result.metadata.answering.model_name,
                answering_identity=result.metadata.answering.canonical_key,
                replicate=result.metadata.replicate,
                raw_trace_sha256=_trace_hash(raw_trace),
                classification=classification,
                classification_score=judgment.scores.get(TRAIT_NAME),
                judge_model="gpt-oss-120b",
            )
        )
    return records


def _write_judgments(path: Path, judgments: Iterable[MissingFinalJudgment]) -> None:
    """Write fresh stochastic judgments as JSONL for audit and reuse."""
    with path.open("w", encoding="utf-8") as handle:
        for judgment in judgments:
            handle.write(judgment.model_dump_json() + "\n")


def run(out_dir: Path, *, reuse_stored_judgments: bool = False) -> None:
    """Rerun the rubric by default and write missing-final-message tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if reuse_stored_judgments:
        logger.info("Reusing archived missing-final-message judgments")
        judgments = load_judgments(
            input_path(RESPONSE_EMPTY_TRAILING_JUDGMENTS)
        )
    else:
        judgments = _fresh_judgments()
        _write_judgments(out_dir / "missing_final_message_judgments.jsonl", judgments)
    judgments = validate_source_joins(
        judgments,
        iter_results(input_path(OTP_MCP_RESULTS)),
    )
    summary, by_answerer = summarize_judgments(judgments)
    summary.to_csv(out_dir / "missing_final_message_summary.tsv", sep="\t", index=False)
    by_answerer.to_csv(
        out_dir / "missing_final_message_summary_by_answerer.tsv",
        sep="\t",
        index=False,
    )


def run_smoke(limit: int) -> None:
    """Judge a small live slice of missing-final-message traces."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    for judgment in _fresh_judgments(limit):
        logger.info(
            "Fresh class for %s: %s",
            judgment.representative_result_id,
            judgment.classification,
        )
