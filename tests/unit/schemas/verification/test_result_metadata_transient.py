"""Tests for VerificationResultMetadata error_category and warnings fields."""

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
class TestVerificationResultMetadataErrorCategory:
    def test_default_error_category_none(self) -> None:
        meta = _make_metadata()
        assert meta.error_category is None

    def test_explicit_error_category_roundtrip(self) -> None:
        """Field set to a category string survives model_dump/model_validate."""
        meta = _make_metadata(error_category="connection")
        assert meta.error_category == "connection"

        dumped = meta.model_dump()
        assert dumped["error_category"] == "connection"

        restored = VerificationResultMetadata.model_validate(dumped)
        assert restored.error_category == "connection"

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
