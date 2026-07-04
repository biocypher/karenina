"""Tests for VerificationResultMetadata shape after failure-state harmonization."""

import pytest
from pydantic import ValidationError

from karenina.schemas.results.caveat import Caveat
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification.result_components import VerificationResultMetadata
from tests.schemas._metadata_factory import make_metadata


@pytest.mark.unit
class TestVerificationResultMetadata:
    def test_pass_defaults(self):
        md = make_metadata()
        assert md.failure is None
        assert md.caveats == []

    def test_failure_roundtrip(self):
        f = Failure(
            category=FailureCategory.TIMEOUT,
            stage="generate_answer",
            reason="budget exhausted",
        )
        md = make_metadata(failure=f, caveats=[Caveat.RETRIES_USED])
        assert md.failure is not None
        assert md.failure.category is FailureCategory.TIMEOUT
        assert Caveat.RETRIES_USED in md.caveats

    def test_legacy_field_rejected(self):
        """extra='forbid' ensures old payloads fail loud."""
        with pytest.raises(ValidationError):
            VerificationResultMetadata.model_validate(
                {
                    "question_id": "q1",
                    "template_id": "t1",
                    "question_text": "How many kidneys?",
                    "answering": {"interface": "openai", "model_name": "x"},
                    "parsing": {"interface": "openai", "model_name": "x"},
                    "execution_time": 1.0,
                    "timestamp": "2026-04-15T10:00:00Z",
                    "result_id": "r1",
                    "completed_without_errors": True,  # removed field
                }
            )
