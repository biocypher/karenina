"""Run the full citation screen, balanced selection, and web audit.

The default path calls GPT-OSS for screening and Claude Opus with web search
for citation investigation. Pass ``--reuse-stored-judgments`` for the explicit
offline path over archived stochastic judgments.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from karenina.benchmark import ResultsIOManager, evaluate_rubric_on_results
from karenina.schemas.verification import VerificationResult
from paper.common.bootstrap import bootstrap, input_path
from paper.config import (
    CITATION_ARCHIVED_JUDGMENTS,
    CITATION_ARCHIVED_SELECTED,
    CITATION_OUTPUT_DIR,
    OTP_MCP_RESULTS,
    OTP_PARAMETRIC_RESULTS,
)
from paper.otp_citation_audit.analysis import write_analysis
from paper.otp_citation_audit.config import (
    CLAUDE_MODELS,
    TARGET_PER_CONDITION,
    screener_model,
)
from paper.otp_citation_audit.rubrics import (
    AUDIT_TRAIT,
    SCREEN_TRAIT,
    CitationReport,
    CitationScreen,
    audit_rubric,
    reconstruct_schema,
    screen_rubric,
)
from paper.otp_citation_audit.selection import (
    CitationCandidate,
    balanced_sample,
    final_answer,
    is_canonical_claude_row,
    outcome,
    structural_skip_reason,
)

logger = logging.getLogger(__name__)

def _iter_results(path: Path) -> Iterator[VerificationResult]:
    return cast(Iterator[VerificationResult], ResultsIOManager.iter_from_json(path))


def _write_jsonl(path: Path, rows: Iterator[dict[str, object]] | list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path, *, skip_summary: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if skip_summary and row.get("kind") == "composition_summary":
                continue
            rows.append(row)
    return rows


def _selected_payload(candidate: CitationCandidate) -> dict[str, object]:
    return {
        "result_id": candidate.result_id,
        "question_id": candidate.result.metadata.question_id,
        "model": candidate.model,
        "regime": candidate.regime,
        "outcome": candidate.outcome,
        "replicate": candidate.result.metadata.replicate,
        "final_answer_text": candidate.final_answer_text,
        "n_citations": candidate.n_citations,
    }


def _fresh_judgments(
    output_dir: Path,
    *,
    limit: int | None,
    audit_model_name: str,
    workers: int,
    screen_only: bool = False,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    sources: list[tuple[str, VerificationResult]] = []
    for regime, path in (
        ("parametric", input_path(OTP_PARAMETRIC_RESULTS)),
        ("mcp", input_path(OTP_MCP_RESULTS)),
    ):
        sources.extend(
            (regime, result)
            for result in _iter_results(path)
            if is_canonical_claude_row(result) and structural_skip_reason(result) is None
        )
    regime_by_id = {result.metadata.result_id: regime for regime, result in sources}
    result_by_id = {result.metadata.result_id: result for _regime, result in sources}
    results = [result for _regime, result in sources]
    if limit is not None:
        results = results[:limit]
    logger.warning("Calling GPT-OSS to screen %d stored Claude answers", len(results))
    screen_judgments = evaluate_rubric_on_results(
        results,
        screen_rubric(),
        screener_model(),
        text_selector=final_answer,
        collapse_parser_siblings=True,
        max_workers=workers,
    )
    candidates: list[CitationCandidate] = []
    screen_rows: list[dict[str, object]] = []
    for judgment in screen_judgments:
        if judgment.error:
            raise RuntimeError(f"Citation screen failed for {judgment.key}: {judgment.error}")
        result_id = judgment.representative_result_id
        if result_id is None:
            raise RuntimeError("Citation screen judgment has no source result id")
        screen = cast(CitationScreen, reconstruct_schema(judgment.scores, SCREEN_TRAIT, CitationScreen))
        source = result_by_id[result_id]
        regime = regime_by_id[result_id]
        screen_rows.append({"result_id": result_id, "regime": regime, **screen.model_dump()})
        if screen.has_explicit_citation:
            candidates.append(
                CitationCandidate(
                    result=source,
                    model=CLAUDE_MODELS[source.metadata.answering.model_name],
                    regime=regime,
                    outcome=outcome(source),
                    final_answer_text=final_answer(source),
                    n_citations=screen.n_citations,
                )
            )
    _write_jsonl(output_dir / "screen_judgments.jsonl", screen_rows)
    selected = balanced_sample(
        candidates,
        target_per_condition=TARGET_PER_CONDITION,
    )
    selected_rows = [_selected_payload(candidate) for candidate in selected]
    _write_jsonl(output_dir / "selected.jsonl", selected_rows)
    selected_answers = {
        candidate.result_id: candidate.final_answer_text
        for candidate in selected
    }
    if screen_only:
        logger.warning("Screen-only mode selected %d answers and skipped the web audit", len(selected))
        return [], selected_answers

    logger.warning("Calling %s with web search for %d selected answers", audit_model_name, len(selected))
    audit_judgments = evaluate_rubric_on_results(
        [candidate.result for candidate in selected],
        audit_rubric(audit_model_name),
        screener_model(),
        text_selector=final_answer,
        collapse_parser_siblings=True,
        max_workers=workers,
    )
    selected_by_id = {candidate.result_id: candidate for candidate in selected}
    records: list[dict[str, object]] = []
    for judgment in audit_judgments:
        if judgment.error:
            raise RuntimeError(f"Citation audit failed for {judgment.key}: {judgment.error}")
        result_id = judgment.representative_result_id
        if result_id is None:
            raise RuntimeError("Citation audit judgment has no source result id")
        report = cast(CitationReport, reconstruct_schema(judgment.scores, AUDIT_TRAIT, CitationReport))
        candidate = selected_by_id[result_id]
        records.append(
            {
                "result_id": result_id,
                "model": candidate.model,
                "regime": candidate.regime,
                "outcome": candidate.outcome,
                "report": report.model_dump(),
                "judge_model": audit_model_name,
            }
        )
    _write_jsonl(output_dir / "citation_judgments.jsonl", records)
    return records, selected_answers


def run(
    output_dir: Path,
    *,
    reuse_stored_judgments: bool,
    limit: int | None = None,
    audit_model_name: str = "claude-opus-4-6",
    workers: int = 1,
    screen_only: bool = False,
) -> None:
    """Run fresh citation judgments or explicitly reuse their archived forms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if reuse_stored_judgments:
        selected = _load_jsonl(
            input_path(CITATION_ARCHIVED_SELECTED),
            skip_summary=True,
        )
        records = _load_jsonl(input_path(CITATION_ARCHIVED_JUDGMENTS))
        for row in selected:
            if row.get("regime") == "nomcp":
                row["regime"] = "parametric"
        for row in records:
            if row.get("regime") == "nomcp":
                row["regime"] = "parametric"
        selected_answers = {str(row["result_id"]): str(row.get("final_answer_text") or "") for row in selected}
    else:
        records, selected_answers = _fresh_judgments(
            output_dir,
            limit=limit,
            audit_model_name=audit_model_name,
            workers=workers,
            screen_only=screen_only,
        )
    write_analysis(records, selected_answers, output_dir / "analysis")


def main() -> None:
    """Parse command-line options and run the complete citation audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-stored-judgments", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=CITATION_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audit-model", default="claude-opus-4-6")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Run the GPT-OSS screen without the separate Claude web audit.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    bootstrap(args.verbose)
    run(
        args.output_dir,
        reuse_stored_judgments=args.reuse_stored_judgments,
        limit=args.limit,
        audit_model_name=args.audit_model,
        workers=args.workers,
        screen_only=args.screen_only,
    )


if __name__ == "__main__":
    main()
