"""Tests for the canonical result row identity key."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from karenina.schemas.results import ResultRowKey
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity


def _result() -> VerificationResult:
    answering = ModelIdentity(
        interface="openai_endpoint",
        model_name="answerer",
        tools=["otp"],
    )
    parsing = ModelIdentity(interface="openai_endpoint", model_name="judge")
    timestamp = datetime.now(UTC).isoformat()
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q1",
            template_id="template_hash",
            question_text="What is 2+2?",
            answering=answering,
            parsing=parsing,
            execution_time=0.5,
            timestamp=timestamp,
            replicate=2,
            result_id=VerificationResultMetadata.compute_result_id(
                question_id="q1",
                answering=answering,
                parsing=parsing,
                timestamp=timestamp,
                replicate=2,
            ),
            run_name="test_run",
        ),
        template=VerificationResultTemplate(raw_llm_response="four"),
    )


@pytest.mark.unit
class TestResultRowKey:
    def test_from_result_uses_canonical_keys(self) -> None:
        result = _result()
        key = ResultRowKey.from_result(result)
        assert key.question_id == result.metadata.question_id
        assert key.answering_key == result.metadata.answering.canonical_key
        assert key.parsing_key == result.metadata.parsing.canonical_key
        assert key.replicate == result.metadata.replicate
        assert key.run_name == result.metadata.run_name

    def test_trace_identity_drops_parser_and_run(self) -> None:
        key = ResultRowKey.from_result(_result())
        assert key.trace_identity == (key.question_id, key.answering_key, key.replicate)

    def test_is_hashable_and_frozen(self) -> None:
        key = ResultRowKey.from_result(_result())
        assert key in {key}
        attribute = "question_id"
        with pytest.raises(AttributeError):
            setattr(key, attribute, "other")
