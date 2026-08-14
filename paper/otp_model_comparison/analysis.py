"""Deterministic result tables for the Open Targets model comparison."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from karenina.schemas.verification import VerificationResult
from paper.otp_model_comparison.config import REFERENCE_PARSER

LONG_FORM_COLUMNS = [
    "question_id",
    "answerer",
    "regime",
    "replicate",
    "parser",
    "outcome_class",
    "failure_group",
    "failure_stage",
    "tokens_answerer",
    "trace_length",
    "execution_time",
]


def outcome_class(result: VerificationResult) -> str:
    """Map structured failure metadata into the experiment outcome classes."""
    failure = result.metadata.failure
    if failure is None:
        return "pass"
    groups = {
        "content": "fail-content",
        "abstained": "abstain",
        "autofail": "infra",
        "retry": "infra",
        "system": "infra",
    }
    try:
        return groups[failure.group.value]
    except KeyError as exc:
        raise ValueError(f"Unknown verification failure group: {failure.group.value}") from exc


def build_long_form(results: Iterable[VerificationResult], regime: str) -> pd.DataFrame:
    """Build one validated row per question, answerer, parser, and replicate."""
    rows: list[dict[str, object]] = []
    for result in results:
        template = result.template
        usage = template.usage_metadata if template else None
        answer_usage = usage.get("answer_generation", {}) if usage else {}
        rows.append(
            {
                "question_id": result.metadata.question_id,
                "answerer": result.metadata.answering.model_name,
                "regime": regime,
                "replicate": result.metadata.replicate,
                "parser": result.metadata.parsing.model_name,
                "outcome_class": outcome_class(result),
                "failure_group": result.metadata.failure.group.value if result.metadata.failure else None,
                "failure_stage": result.metadata.failure.stage if result.metadata.failure else None,
                "tokens_answerer": answer_usage.get("total_tokens"),
                "trace_length": len(template.trace_messages or []) if template else 0,
                "execution_time": result.metadata.execution_time,
            }
        )
    return pd.DataFrame(rows, columns=LONG_FORM_COLUMNS)


def write_analysis(parametric: pd.DataFrame, mcp: pd.DataFrame, output_dir: Path) -> None:
    """Write the deterministic model-comparison analysis tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    populated = [frame for frame in (parametric, mcp) if not frame.empty]
    frame = pd.concat(populated, ignore_index=True) if populated else pd.DataFrame(columns=LONG_FORM_COLUMNS)
    frame.to_csv(output_dir / "results_long_form.tsv", sep="\t", index=False)
    (
        frame.groupby(["regime", "answerer", "parser", "outcome_class"], dropna=False)
        .size()
        .reset_index(name="rows")
        .to_csv(output_dir / "outcome_counts.tsv", sep="\t", index=False)
    )
    reference = frame[frame["parser"] == REFERENCE_PARSER].copy()
    reference["is_pass"] = reference["outcome_class"] == "pass"
    (
        reference.groupby(["answerer", "regime", "replicate"], dropna=False)["is_pass"]
        .mean()
        .reset_index(name="pass_rate")
        .to_csv(output_dir / "pass_rate_by_replicate.tsv", sep="\t", index=False)
    )
    tokens = frame.dropna(subset=["tokens_answerer"]).drop_duplicates(
        ["question_id", "answerer", "regime", "replicate"]
    )
    (
        tokens.groupby(["answerer", "regime"])["tokens_answerer"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .to_csv(output_dir / "answer_tokens.tsv", sep="\t", index=False)
    )
    (
        reference.groupby(["question_id", "answerer", "regime"])["is_pass"]
        .agg(pass_count="sum", replicate_count="count")
        .reset_index()
        .to_csv(output_dir / "question_pass_counts.tsv", sep="\t", index=False)
    )
