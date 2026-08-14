"""Trace-length and answer-generation token analysis."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from paper.common.bootstrap import input_path
from paper.common.qa_results import REFERENCE_JUDGE, QAResultRow, iter_rows
from paper.config import OTP_MCP_RESULTS
from paper.otp_response_characterization.config import ANSWERERS

logger = logging.getLogger(__name__)

KEY_COLUMNS = ["question_id", "answerer", "regime", "replicate"]


def build_longform(rows: Iterable[QAResultRow]) -> pd.DataFrame:
    """Build the reduced long-form table required for trace statistics."""
    records = [
        {
            "question_id": row.question_id,
            "answerer": row.answerer,
            "regime": row.regime,
            "replicate": row.replicate,
            "judge": row.parser,
            "outcome_class": row.outcome,
            "trace_length": row.trace_message_count,
            "tokens_answerer": row.tokens_answerer,
        }
        for row in rows
    ]
    return pd.DataFrame(records)


def token_lookup(longform: pd.DataFrame) -> pd.DataFrame:
    """Return one consistent token count per generated answer.

    Raises:
        ValueError: If parser siblings store conflicting positive token counts.
    """
    populated = longform[longform["tokens_answerer"].notna() & longform["tokens_answerer"].gt(0)]
    distinct = populated.groupby(KEY_COLUMNS, dropna=False)["tokens_answerer"].nunique()
    if distinct.gt(1).any():
        bad = distinct[distinct.gt(1)].index[0]
        raise ValueError(f"Parser siblings store conflicting answerer token counts for {bad}")
    return populated.drop_duplicates(KEY_COLUMNS)[KEY_COLUMNS + ["tokens_answerer"]]


def reference_rows(longform: pd.DataFrame) -> pd.DataFrame:
    """Return reference-judge rows enriched with sibling token metadata."""
    reference = longform[longform["judge"] == REFERENCE_JUDGE].copy()
    if reference.duplicated(KEY_COLUMNS).any():
        raise ValueError("Reference-judge rows contain duplicate generated-answer keys")
    return reference.drop(columns="tokens_answerer").merge(
        token_lookup(longform),
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )


def common_correct_questions(
    reference: pd.DataFrame,
    answerers: Iterable[str],
) -> set[str]:
    """Find questions every selected answerer passed in every MCP replicate."""
    mcp = reference[reference["regime"] == "mcp"]
    available_replicates = set(mcp["replicate"].dropna().astype(int).unique())
    if not available_replicates:
        return set()

    per_answerer: list[set[str]] = []
    for answerer in answerers:
        cohort = mcp[mcp["answerer"] == answerer]
        passing: set[str] = set()
        for question_id, group in cohort.groupby("question_id"):
            observed = set(group["replicate"].dropna().astype(int))
            if observed == available_replicates and group["outcome_class"].eq("pass").all():
                passing.add(str(question_id))
        per_answerer.append(passing)
    return set.intersection(*per_answerer) if per_answerer else set()


def trace_length_stats(
    reference: pd.DataFrame,
    answerers: Sequence[str] = tuple(ANSWERERS),
) -> pd.DataFrame:
    """Summarize trace lengths on the current common-correct question set."""
    common = common_correct_questions(reference, answerers)
    selected = reference[
        (reference["regime"] == "mcp") & reference["question_id"].isin(common) & reference["outcome_class"].eq("pass")
    ]
    rows: list[dict[str, int | float | str]] = []
    for answerer in answerers:
        values = selected[selected["answerer"] == answerer]["trace_length"].to_numpy(dtype=float)
        if not values.size:
            rows.append(
                {
                    "answerer": answerer,
                    "n": 0,
                    "median": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "pct_gt_10": np.nan,
                    "count_gt_10": 0,
                    "pct_gt_20": np.nan,
                    "count_gt_20": 0,
                    "max": np.nan,
                }
            )
            continue
        rows.append(
            {
                "answerer": answerer,
                "n": int(values.size),
                "median": float(np.median(values)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
                "pct_gt_10": float((values > 10).mean() * 100),
                "count_gt_10": int((values > 10).sum()),
                "pct_gt_20": float((values > 20).mean() * 100),
                "count_gt_20": int((values > 20).sum()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def trace_length_summary(
    reference: pd.DataFrame,
    answerers: Sequence[str] = tuple(ANSWERERS),
) -> pd.DataFrame:
    """Return mean, median, and interquartile trace lengths by answerer."""
    common = common_correct_questions(reference, answerers)
    selected = reference[
        (reference["regime"] == "mcp") & reference["question_id"].isin(common) & reference["outcome_class"].eq("pass")
    ]
    rows: list[dict[str, float | str]] = []
    for answerer in answerers:
        values = selected[selected["answerer"] == answerer]["trace_length"].to_numpy(dtype=float)
        rows.append(
            {
                "answerer": answerer,
                "mean": float(np.mean(values)) if values.size else np.nan,
                "median": float(np.median(values)) if values.size else np.nan,
                "p25": float(np.percentile(values, 25)) if values.size else np.nan,
                "p75": float(np.percentile(values, 75)) if values.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def right_wrong_tokens(
    reference: pd.DataFrame,
    answerers: Sequence[str] = tuple(ANSWERERS),
) -> pd.DataFrame:
    """Summarize answerer tokens for correct and content-failed MCP answers."""
    rows: list[dict[str, int | float | str]] = []
    for answerer in answerers:
        cohort = reference[(reference["regime"] == "mcp") & (reference["answerer"] == answerer)]
        right = cohort[cohort["outcome_class"] == "pass"]["tokens_answerer"].dropna()
        wrong = cohort[cohort["outcome_class"] == "fail-content"]["tokens_answerer"].dropna()

        def percentile(values: pd.Series, q: int) -> float:
            return float(np.percentile(values.to_numpy(dtype=float), q)) if len(values) else np.nan

        rows.append(
            {
                "answerer": answerer,
                "right_median": percentile(right, 50),
                "right_q1": percentile(right, 25),
                "right_q3": percentile(right, 75),
                "wrong_median": percentile(wrong, 50),
                "wrong_q1": percentile(wrong, 25),
                "wrong_q3": percentile(wrong, 75),
                "n_right": len(right),
                "n_wrong": len(wrong),
            }
        )
    return pd.DataFrame(rows)


def run(out_dir: Path) -> None:
    """Write trace-length and token tables into ``out_dir``."""
    logger.info("Building the reduced MCP trace table")
    longform = build_longform(
        iter_rows(input_path(OTP_MCP_RESULTS), regime="mcp")
    )
    reference = reference_rows(longform)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_length_stats(reference).to_csv(out_dir / "trace_length_stats.tsv", sep="\t", index=False)
    trace_length_summary(reference).to_csv(out_dir / "trace_length_summary.tsv", sep="\t", index=False)
    right_wrong_tokens(reference).to_csv(out_dir / "right_wrong_tokens_by_answerer.tsv", sep="\t", index=False)
