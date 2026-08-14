"""Minimal example of citation screening and audit.

The full experiment performs structural exclusions, balanced sampling,
manifests, and paper-table generation. This teaching script keeps only the
Karenina stored-result workflow:

    load results -> attach screen rubric -> evaluate -> select -> audit

Both rubric stages call models. The first calls the configured GPT-OSS screen.
The second calls a web-enabled agentic judge.

Run from the Karenina repository root:

    uv run python -m paper.otp_citation_audit.simplified \
        --limit 5
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from karenina.benchmark import ResultsIOManager, evaluate_rubric_on_results
from karenina.schemas.verification import VerificationResult
from paper.common.bootstrap import bootstrap, input_path
from paper.config import OTP_MCP_RESULTS
from paper.otp_citation_audit.config import screener_model
from paper.otp_citation_audit.rubrics import (
    SCREEN_TRAIT,
    CitationScreen,
    audit_rubric,
    reconstruct_schema,
    screen_rubric,
)
from paper.otp_citation_audit.selection import final_answer


def main() -> None:
    """Screen and investigate a small set of stored answers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    bootstrap()
    results_path = args.results or input_path(OTP_MCP_RESULTS)

    # ## 1. Load validated stored results
    # ResultsIOManager validates each streamed object as a
    # validated VerificationResult, not an untyped JSON dictionary.
    iterator = cast(Iterator[VerificationResult], ResultsIOManager.iter_from_json(results_path))
    rows = list(iterator)[: args.limit]

    # ## 2. Attach and run the screening rubric
    # text_selector ensures the judge sees only the final answer, not tool
    # results that happen to contain paper identifiers.
    screens = list(
        evaluate_rubric_on_results(
            rows,
            screen_rubric(),
            screener_model(),
            text_selector=final_answer,
            max_workers=1,
        )
    )
    selected_ids: set[str] = set()
    for judgment in screens:
        if judgment.error or not judgment.scores:
            raise RuntimeError(judgment.error or "Citation screen returned no verdict")
        screen = cast(CitationScreen, reconstruct_schema(judgment.scores, SCREEN_TRAIT, CitationScreen))
        if screen.has_explicit_citation and judgment.representative_result_id is not None:
            selected_ids.add(judgment.representative_result_id)
    selected = [row for row in rows if row.metadata.result_id in selected_ids]
    print(f"[1] loaded {len(rows)} validated stored answers")
    print(f"[2] screened {len(screens)} answers, selected {len(selected)} with citations")

    # ## 3. Attach and run the agentic audit rubric
    # AgenticRubricTrait routes through the TaskEval-backed evaluator and
    # gives its configured agent access to web search.
    audits = list(
        evaluate_rubric_on_results(
            selected,
            audit_rubric(),
            screener_model(),
            text_selector=final_answer,
            max_workers=1,
        )
    )
    for judgment in audits:
        if judgment.error or not judgment.scores:
            raise RuntimeError(judgment.error or "Citation audit returned no verdict")
    print(f"[3] audited {len(audits)} citation-bearing answers with web search")
    print(
        "\nPublic Karenina flow:\n"
        "  ResultsIOManager.iter_from_json load validated result rows\n"
        "  Rubric                         describe each judgment\n"
        "  evaluate_rubric_on_results     evaluate saved answers via TaskEval"
    )


if __name__ == "__main__":
    main()
