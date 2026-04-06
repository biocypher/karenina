"""Tests for VerificationResultMetadata transient error flag."""

from __future__ import annotations

import pytest

from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import VerificationResultMetadata


def _make_metadata(**overrides) -> VerificationResultMetadata:
    """Create a minimal VerificationResultMetadata with required fields."""
    defaults = {
        "question_id": "q1",
        "template_id": "tpl1",
        "completed_without_errors": True,
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
class TestVerificationResultMetadataTransientFlag:
    def test_default_is_transient_false(self) -> None:
        meta = _make_metadata()
        assert meta.is_transient_error is False

    def test_explicit_transient_true_roundtrip(self) -> None:
        """Field set to True survives model_dump/model_validate."""
        meta = _make_metadata(is_transient_error=True)
        assert meta.is_transient_error is True

        dumped = meta.model_dump()
        assert dumped["is_transient_error"] is True

        restored = VerificationResultMetadata.model_validate(dumped)
        assert restored.is_transient_error is True
