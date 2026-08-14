"""Deterministic no-tool-call analysis."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from karenina.benchmark import evaluate_rubric_on_results
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities.rubric import Rubric
from karenina.schemas.results import ResultRowKey
from karenina.schemas.verification import VerificationResult
from paper.common.bootstrap import input_path
from paper.common.qa_results import REFERENCE_JUDGE, QAResultRow, iter_results
from paper.config import OTP_MCP_RESULTS
from paper.otp_response_characterization.config import (
    ANSWERERS,
    FAILURE_SHAPE_TRAITS,
    NO_TOOL_CALL_TRAIT,
    POST_HOC_WORKERS,
    gpt_oss_judge,
)

logger = logging.getLogger(__name__)

OUTCOMES = ("pass", "fail-content", "abstain", "infra")
ANSWERER_METRICS = {
    "gpt-oss-120b": "gpt_oss",
    "claude-sonnet-4-6": "claude_sonnet",
    "claude-opus-4-6": "claude_opus",
    "claude-haiku-4-5-20251001": "claude_haiku",
    "qwen3.5-a3b": "qwen35",
    "qwen3.6-a3b": "qwen36",
    "qwen3.5-122b-a10b": "qwen35_122b",
}
EMPTY_TRACE_TRAIT = next(
    trait for trait in FAILURE_SHAPE_TRAITS if trait.name == "EmptyTrace"
)


def _track_results(
    results: Iterable[VerificationResult],
    rows_by_key: dict[ResultRowKey, QAResultRow],
) -> Iterator[VerificationResult]:
    """Record reduced row metadata while yielding validated MCP results."""
    for result in results:
        key = ResultRowKey.from_result(result)
        if key in rows_by_key:
            raise ValueError(f"Duplicate QA result key: {key}")
        rows_by_key[key] = QAResultRow.from_result(result, "mcp")
        yield result


def score_no_tool_rows(
    results: Iterable[VerificationResult],
    parsing_model: ModelConfig,
) -> pd.DataFrame:
    """Evaluate no-tool and empty-trace traits through the TaskEval facade."""
    rows_by_key: dict[ResultRowKey, QAResultRow] = {}
    records: list[dict[str, object]] = []
    judgments = evaluate_rubric_on_results(
        _track_results(results, rows_by_key),
        Rubric(regex_traits=[NO_TOOL_CALL_TRAIT, EMPTY_TRACE_TRAIT]),
        parsing_model,
        collapse_parser_siblings=True,
        max_workers=POST_HOC_WORKERS,
    )
    for judgment in judgments:
        if judgment.error is not None:
            raise RuntimeError(f"TaskEval failed for no-tool row {judgment.key}: {judgment.error}")
        no_tool = judgment.scores.get(NO_TOOL_CALL_TRAIT.name)
        empty = judgment.scores.get(EMPTY_TRACE_TRAIT.name)
        if not isinstance(no_tool, bool) or not isinstance(empty, bool):
            raise ValueError(f"TaskEval returned incomplete no-tool scores for {judgment.key}")
        for sibling_key in judgment.sibling_keys:
            row = rows_by_key[sibling_key]
            records.append(
                {
                    "question_id": row.question_id,
                    "answerer": row.answerer,
                    "judge": row.parser,
                    "replicate": row.replicate,
                    "outcome_class": row.outcome,
                    "no_tool_call": no_tool,
                    "empty_trace": empty,
                }
            )
    return pd.DataFrame(records)


def build_no_tool_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize nonempty no-tool rows overall and for the reference judge."""
    unknown = set(scores["outcome_class"].unique()).difference(OUTCOMES)
    if unknown:
        raise ValueError(f"Unknown no-tool outcome classes: {sorted(unknown)}")

    metrics: list[tuple[str, int | float]] = []

    def add_cohort(cohort: pd.DataFrame, prefix: str) -> None:
        empty = cohort["empty_trace"].astype(bool)
        selected = cohort[cohort["no_tool_call"].astype(bool) & ~empty]
        outcome_counts = selected["outcome_class"].value_counts().to_dict()
        if sum(int(outcome_counts.get(outcome, 0)) for outcome in OUTCOMES) != len(selected):
            raise ValueError("No-tool outcome partition does not cover the selected cohort")

        pass_count = int(outcome_counts.get("pass", 0))
        metrics.extend(
            [
                (f"{prefix}empty_trace_count", int(empty.sum())),
                (f"{prefix}no_tool_nonempty_count", len(selected)),
                (f"{prefix}no_tool_pass_count", pass_count),
                (
                    f"{prefix}no_tool_pass_rate",
                    100 * pass_count / len(cohort) if len(cohort) else 0.0,
                ),
                (f"{prefix}no_tool_fail_count", int(outcome_counts.get("fail-content", 0))),
                (f"{prefix}no_tool_abstain_count", int(outcome_counts.get("abstain", 0))),
                (f"{prefix}no_tool_infra_count", int(outcome_counts.get("infra", 0))),
            ]
        )
        passes = selected[selected["outcome_class"] == "pass"]
        for answerer in ANSWERERS:
            metric_prefix = ANSWERER_METRICS[answerer]
            marker = "ref_judge_no_tool" if "ref_judge" in prefix else "no_tool"
            metrics.append(
                (
                    f"{metric_prefix}_{marker}_pass_count",
                    int((passes["answerer"] == answerer).sum()),
                )
            )

    add_cohort(scores, "mcp_")
    add_cohort(scores[scores["judge"] == REFERENCE_JUDGE], "mcp_ref_judge_")
    return pd.DataFrame(metrics, columns=["metric", "value"])


def run(out_dir: Path) -> None:
    """Write the no-tool-call summary into ``out_dir``."""
    logger.info("Evaluating the deterministic no-tool-call trait")
    scores = score_no_tool_rows(
        iter_results(input_path(OTP_MCP_RESULTS)),
        gpt_oss_judge(),
    )
    summary = build_no_tool_summary(scores)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "no_tool_call_summary.tsv", sep="\t", index=False)
