"""Aggregation and curation tables for citation integrity judgments."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable
from itertools import product
from pathlib import Path

import pandas as pd

from paper.otp_citation_audit.rubrics import CATEGORIES, CitationReport


def integrity_status(categories: list[str]) -> str:
    """Roll per-citation categories into the most severe trace status."""
    scoring = [category for category in categories if category != "skip"]
    if "completely_fabricated" in scoring:
        return "any_completely_fabricated"
    if "existing_pmid_fabricated_content" in scoring:
        return "any_existing_pmid_fabricated_content"
    if "similar_content_wrong_citation" in scoring:
        return "any_similar_content_wrong_citation"
    return "all_legitimate"


def condition_summaries(records: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Aggregate citation counts across model, regime, and answer outcome."""
    rows: list[dict[str, object]] = []
    for record in records:
        report = CitationReport.model_validate(record["report"])
        counts = Counter(report.category)
        status = integrity_status(report.category)
        rows.append(
            {
                "model": record["model"],
                "regime": record["regime"],
                "outcome": record["outcome"],
                "n_traces": 1,
                **{f"n_{category}": counts[category] for category in CATEGORIES if category != "skip"},
                "n_soft": counts["legitimate"] + counts["similar_content_wrong_citation"],
                "n_hard": counts["existing_pmid_fabricated_content"] + counts["completely_fabricated"],
                "n_skipped": counts["skip"],
                "n_all_legitimate": int(status == "all_legitimate"),
                "n_any_similar_content_wrong_citation": int(
                    status == "any_similar_content_wrong_citation"
                ),
                "n_any_existing_pmid_fabricated_content": int(
                    status == "any_existing_pmid_fabricated_content"
                ),
                "n_any_completely_fabricated": int(status == "any_completely_fabricated"),
            }
        )
    columns = [
        "n_traces",
        "n_legitimate",
        "n_similar_content_wrong_citation",
        "n_existing_pmid_fabricated_content",
        "n_completely_fabricated",
        "n_skipped",
        "n_soft",
        "n_hard",
        "n_all_legitimate",
        "n_any_similar_content_wrong_citation",
        "n_any_existing_pmid_fabricated_content",
        "n_any_completely_fabricated",
    ]
    complete_index = pd.MultiIndex.from_tuples(
        list(product(("haiku", "sonnet", "opus"), ("parametric", "mcp"), ("pass", "fail"))),
        names=["model", "regime", "outcome"],
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        empty = complete_index.to_frame(index=False)
        for column in columns:
            empty[column] = 0
        return empty
    for column in columns:
        if column not in frame:
            frame[column] = 0
    return frame.groupby(["model", "regime", "outcome"], dropna=False)[columns].sum().reindex(
        complete_index,
        fill_value=0,
    ).reset_index()


def aggregate_metrics(summary: pd.DataFrame) -> dict[str, object]:
    """Build compact overall, regime, and outcome citation totals."""
    count_columns = [column for column in summary.columns if column.startswith("n_")]

    def totals(frame: pd.DataFrame) -> dict[str, int | float]:
        values: dict[str, int | float] = {
            column: int(frame[column].sum()) for column in count_columns
        }
        scored = values.get("n_soft", 0) + values.get("n_hard", 0)
        values["hard_rate"] = values.get("n_hard", 0) / scored if scored else 0.0
        return values

    if summary.empty:
        return {"overall": {}, "by_regime": {}, "by_outcome": {}}
    return {
        "overall": totals(summary),
        "by_regime": {regime: totals(frame) for regime, frame in summary.groupby("regime")},
        "by_outcome": {outcome: totals(frame) for outcome, frame in summary.groupby("outcome")},
    }


def curation_rows(
    records: Iterable[dict[str, object]],
    selected_answers: dict[str, str],
) -> list[dict[str, object]]:
    """Flatten trace reports into one row per audited citation."""
    rows: list[dict[str, object]] = []
    for record in records:
        result_id = str(record["result_id"])
        report = CitationReport.model_validate(record["report"])
        status = integrity_status(report.category)
        answer = selected_answers.get(result_id, "")
        for index, values in enumerate(
            zip(
                report.citation_texts,
                report.category,
                report.matched_real_reference,
                report.evidence_url,
                report.reasoning,
                strict=True,
            ),
            start=1,
        ):
            citation, category, matched, url, reasoning = values
            position = answer.find(citation)
            excerpt = answer[max(0, position - 200) : position + len(citation) + 200] if position >= 0 else answer[:400]
            rows.append(
                {
                    "result_id": result_id,
                    "model": record["model"],
                    "regime": record["regime"],
                    "outcome": record["outcome"],
                    "citation_index": index,
                    "n_citations_in_trace": len(report.category),
                    "citation_text": citation,
                    "answer_excerpt": excerpt,
                    "category": category,
                    "matched_real_reference": matched,
                    "evidence_url": url,
                    "reasoning": reasoning,
                    "trace_integrity_status": status,
                    "curator_verdict": "",
                    "curator_agrees": "",
                    "curator_notes": "",
                }
            )
    return rows


def write_analysis(
    records: list[dict[str, object]],
    selected_answers: dict[str, str],
    output_dir: Path,
) -> None:
    """Write condition summaries, aggregate metrics, and curation CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = condition_summaries(records)
    summary.to_csv(output_dir / "condition_summaries.tsv", sep="\t", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(aggregate_metrics(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = curation_rows(records, selected_answers)
    fieldnames = list(rows[0]) if rows else ["result_id"]
    with (output_dir / "curation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
