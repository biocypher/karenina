"""Tests for VerificationResultMetadata failure + caveats + warnings fields."""

from __future__ import annotations

import pytest

from karenina.schemas.results.caveat import Caveat
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import VerificationResultMetadata


def _make_metadata(**overrides) -> VerificationResultMetadata:
    """Create a minimal VerificationResultMetadata with required fields."""
    defaults = {
        "question_id": "q1",
        "template_id": "tpl1",
        "failure": None,
        "caveats": [],
        "question_text": "What?",
        "answering": ModelIdentity(model_name="test", interface="openai"),
        "parsing": ModelIdentity(model_name="test", interface="openai"),
        "execution_time": 1.0,
        "timestamp": "2026-01-01T00:00:00Z",
        "result_id": "abcdef1234567890",
    }
    defaults.update(overrides)
    return VerificationResultMetadata(**defaults)


@pytest.mark.unit
class TestVerificationResultMetadataFailure:
    def test_default_failure_none(self) -> None:
        meta = _make_metadata()
        assert meta.failure is None

    def test_explicit_failure_roundtrip(self) -> None:
        """Field set to a Failure survives model_dump/model_validate."""
        failure = Failure(
            category=FailureCategory.CONNECTION,
            stage="generate_answer",
            reason="connection error",
        )
        meta = _make_metadata(failure=failure)
        assert meta.failure is not None
        assert meta.failure.category is FailureCategory.CONNECTION

        dumped = meta.model_dump()
        assert dumped["failure"]["category"] == "connection"

        restored = VerificationResultMetadata.model_validate(dumped)
        assert restored.failure is not None
        assert restored.failure.category is FailureCategory.CONNECTION

    def test_default_caveats_empty(self) -> None:
        meta = _make_metadata()
        assert meta.caveats == []

    def test_caveats_roundtrip(self) -> None:
        """Caveats list survives model_dump/model_validate."""
        meta = _make_metadata(caveats=[Caveat.RETRIES_USED, Caveat.PARTIAL_CONTENT])
        assert meta.caveats == [Caveat.RETRIES_USED, Caveat.PARTIAL_CONTENT]

        dumped = meta.model_dump()
        assert dumped["caveats"] == ["retries_used", "partial_content"]

        restored = VerificationResultMetadata.model_validate(dumped)
        assert restored.caveats == [Caveat.RETRIES_USED, Caveat.PARTIAL_CONTENT]

    def test_default_warnings_empty(self) -> None:
        meta = _make_metadata()
        assert meta.warnings == []

    def test_warnings_roundtrip(self) -> None:
        """Warnings list survives model_dump/model_validate."""
        meta = _make_metadata(warnings=["warn 1", "warn 2"])
        assert meta.warnings == ["warn 1", "warn 2"]

        dumped = meta.model_dump()
        assert dumped["warnings"] == ["warn 1", "warn 2"]

        restored = VerificationResultMetadata.model_validate(dumped)
        assert restored.warnings == ["warn 1", "warn 2"]

    def test_partial_content_default_none(self) -> None:
        meta = _make_metadata()
        assert meta.partial_content is None

    def test_partial_content_roundtrip(self) -> None:
        """partial_content survives model_dump/model_validate."""
        meta = _make_metadata(partial_content="some partial output")
        assert meta.partial_content == "some partial output"

        dumped = meta.model_dump()
        assert dumped["partial_content"] == "some partial output"

        restored = VerificationResultMetadata.model_validate(dumped)
        assert restored.partial_content == "some partial output"
