"""Tests for missing-final-message analysis."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from paper.otp_response_characterization.analyses.missing_final_message import (
    CLASSIFICATIONS,
    MissingFinalJudgment,
    summarize_judgments,
    validate_source_joins,
)


def _judgment(
    classification: str = CLASSIFICATIONS[0],
    sibling_ids: list[str] | None = None,
) -> MissingFinalJudgment:
    ids = sibling_ids or ["result-1"]
    return MissingFinalJudgment(
        representative_result_id=ids[0],
        sibling_result_ids=ids,
        question_id="q1",
        question_text="Question?",
        reference_answer="Answer",
        answering_model="answerer",
        replicate=1,
        raw_trace_sha256=hashlib.sha256(b"trace").hexdigest(),
        classification=classification,
    )


def _source(result_id: str = "result-1") -> VerificationResult:
    timestamp = datetime.now(UTC).isoformat()
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q1",
            template_id="template",
            question_text="Question?",
            answering=ModelIdentity(interface="openai_endpoint", model_name="answerer"),
            parsing=ModelIdentity(interface="openai_endpoint", model_name="judge"),
            execution_time=0.1,
            timestamp=timestamp,
            result_id=result_id,
            replicate=1,
        ),
        template=VerificationResultTemplate(raw_llm_response="trace"),
    )


@pytest.mark.unit
class TestMissingFinalMessage:
    """Validate joins, class partitions, and summary arithmetic."""

    def test_source_join_accepts_parser_siblings(self) -> None:
        judgments = [_judgment(sibling_ids=["result-1", "result-2"])]
        joined = validate_source_joins(judgments, [_source("result-1"), _source("result-2")])
        assert joined == judgments

    def test_unknown_class_fails(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            validate_source_joins([_judgment("mystery")], [_source()])

    def test_duplicate_sibling_claim_fails(self) -> None:
        duplicate = _judgment()
        with pytest.raises(ValueError, match="multiple judgments"):
            validate_source_joins([duplicate, duplicate], [_source()])

    def test_summary_partitions_unique_and_sibling_rows(self) -> None:
        rows = [
            _judgment(CLASSIFICATIONS[0], ["a", "b"]),
            _judgment(CLASSIFICATIONS[1], ["c"]),
            _judgment(CLASSIFICATIONS[2], ["d", "e", "f"]),
        ]
        summary, by_answerer = summarize_judgments(rows)
        assert summary["unique_traces"].sum() == 3
        assert summary["sibling_rows"].sum() == 6
        assert summary["unique_fraction"].sum() == pytest.approx(1.0)
        assert summary["sibling_row_fraction"].sum() == pytest.approx(1.0)
        assert by_answerer["unique_traces"].sum() == 3
