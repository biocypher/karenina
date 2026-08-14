"""Synthetic tests for citation audit selection and analysis."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    ModelIdentity,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from paper.otp_citation_audit import run as citation_run
from paper.otp_citation_audit.analysis import condition_summaries, curation_rows, integrity_status, write_analysis
from paper.otp_citation_audit.rubrics import CitationReport, CitationScreen
from paper.otp_citation_audit.selection import (
    CitationCandidate,
    balanced_sample,
    final_answer,
    is_canonical_claude_row,
    structural_skip_reason,
)


def _result(
    question_id: str,
    *,
    outcome: str = "pass",
    answer: str = "Smith et al. (2020) supports this.",
    role: str = "assistant",
    replicate: int = 1,
) -> VerificationResult:
    answering = ModelIdentity(interface="langchain", model_name="claude-haiku-4-5-20251001")
    parsing = ModelIdentity(interface="langchain", model_name="claude-opus-4-6")
    timestamp = datetime.now(UTC).isoformat()
    failure = (
        Failure(category=FailureCategory.CONTENT, stage="VerifyTemplate", reason="wrong") if outcome == "fail" else None
    )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="template",
            question_text="Question",
            answering=answering,
            parsing=parsing,
            replicate=replicate,
            run_name="run",
            execution_time=1.0,
            timestamp=timestamp,
            failure=failure,
            result_id=VerificationResultMetadata.compute_result_id(
                question_id=question_id,
                answering=answering,
                parsing=parsing,
                replicate=replicate,
                timestamp=timestamp,
            ),
        ),
        template=VerificationResultTemplate(
            raw_llm_response=answer,
            trace_messages=[{"role": role, "content": answer}],
        ),
    )


def _report(category: str = "legitimate") -> dict[str, object]:
    counts = {
        "n_legitimate": 0,
        "n_similar_content_wrong_citation": 0,
        "n_existing_pmid_fabricated_content": 0,
        "n_completely_fabricated": 0,
        "n_skipped": 0,
    }
    count_name = "n_skipped" if category == "skip" else f"n_{category}"
    counts[count_name] = 1
    return {
        "citation_texts": ["Smith et al. (2020)"],
        "category": [category],
        "matched_real_reference": ["Smith et al. 2020"],
        "evidence_url": ["https://doi.org/10.1/example"],
        "reasoning": ["The DOI and title resolve."],
        **counts,
    }


@pytest.mark.unit
class TestCitationSchemas:
    def test_screen_quality_is_constrained_in_the_json_schema(self) -> None:
        schema = CitationScreen.model_json_schema()

        assert schema["properties"]["citation_quality"]["enum"] == [
            "clear",
            "ambiguous",
        ]

    def test_screen_requires_boolean_count_consistency(self) -> None:
        with pytest.raises(ValidationError, match="n_citations"):
            CitationScreen(
                has_explicit_citation=False,
                n_citations=1,
                citation_quality="clear",
            )

    def test_report_requires_parallel_lists_and_exact_counts(self) -> None:
        report = CitationReport.model_validate(_report())
        assert report.n_legitimate == 1

        invalid = _report()
        invalid["reasoning"] = []
        with pytest.raises(ValidationError, match="equal lengths"):
            CitationReport.model_validate(invalid)


@pytest.mark.unit
class TestCitationSelection:
    def test_screen_only_skips_the_agentic_audit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = _result("q1")
        calls = 0

        monkeypatch.setattr(citation_run, "input_path", Path)
        monkeypatch.setattr(
            citation_run,
            "_iter_results",
            lambda path: iter([source]) if "nomcp" in str(path) else iter([]),
        )

        def evaluate(*_args: object, **_kwargs: object) -> list[SimpleNamespace]:
            nonlocal calls
            calls += 1
            return [
                SimpleNamespace(
                    error=None,
                    key="q1",
                    representative_result_id=source.metadata.result_id,
                    scores={
                        "published_paper_citation_screen.has_explicit_citation": True,
                        "published_paper_citation_screen.n_citations": 1,
                        "published_paper_citation_screen.citation_quality": "clear",
                        "published_paper_citation_screen.notes": "",
                    },
                )
            ]

        monkeypatch.setattr(citation_run, "evaluate_rubric_on_results", evaluate)

        records, selected_answers = citation_run._fresh_judgments(
            tmp_path,
            limit=1,
            audit_model_name="unused",
            workers=1,
            screen_only=True,
        )

        assert calls == 1
        assert records == []
        assert selected_answers == {
            source.metadata.result_id: "Smith et al. (2020) supports this."
        }

    def test_extracts_only_final_assistant_and_detects_tool_tail(self) -> None:
        answer = _result("q1")
        tool_tail = _result("q2", role="tool")

        assert final_answer(answer) == "Smith et al. (2020) supports this."
        assert structural_skip_reason(tool_tail) == "no_ai_final_message"
        assert is_canonical_claude_row(answer) is True

    def test_balances_pass_and_fail_with_question_diversity(self) -> None:
        candidates: list[CitationCandidate] = []
        for status in ("pass", "fail"):
            for index in range(4):
                result = _result(f"{status}-{index}", outcome=status, replicate=index + 1)
                candidates.append(
                    CitationCandidate(
                        result=result,
                        model="haiku",
                        regime="mcp",
                        outcome=status,
                        final_answer_text=final_answer(result),
                        n_citations=4 - index,
                    )
                )

        selected = balanced_sample(candidates, target_per_condition=6)

        assert len(selected) == 6
        assert sum(row.outcome == "pass" for row in selected) == 3
        assert sum(row.outcome == "fail" for row in selected) == 3
        assert len({row.result.metadata.question_id for row in selected}) == 6

    def test_excludes_answers_above_configured_citation_cap(self) -> None:
        result = _result("too-many")
        candidate = CitationCandidate(
            result=result,
            model="haiku",
            regime="mcp",
            outcome="pass",
            final_answer_text=final_answer(result),
            n_citations=11,
        )

        assert balanced_sample([candidate], target_per_condition=12) == []


@pytest.mark.unit
class TestCitationAnalysis:
    def test_aggregates_categories_and_builds_curation_rows(self, tmp_path: Path) -> None:
        records: list[dict[str, object]] = [
            {
                "result_id": "r1",
                "model": "haiku",
                "regime": "mcp",
                "outcome": "pass",
                "report": _report("existing_pmid_fabricated_content"),
            }
        ]

        summary = condition_summaries(records)
        rows = curation_rows(records, {"r1": "Smith et al. (2020) supports this."})
        write_analysis(records, {"r1": "Smith et al. (2020) supports this."}, tmp_path)

        assert len(summary) == 12
        observed = summary[
            (summary["model"] == "haiku")
            & (summary["regime"] == "mcp")
            & (summary["outcome"] == "pass")
        ].iloc[0]
        assert observed["n_hard"] == 1
        assert observed["n_any_existing_pmid_fabricated_content"] == 1
        assert rows[0]["trace_integrity_status"] == "any_existing_pmid_fabricated_content"
        assert integrity_status(["legitimate"]) == "all_legitimate"
        written = pd.read_csv(tmp_path / "condition_summaries.tsv", sep="\t")
        assert len(written) == 12
        assert written["n_existing_pmid_fabricated_content"].sum() == 1
