"""Tests for VerifiedField factory and VerificationMeta."""

import pytest
from pydantic import BaseModel

from karenina.schemas.entities.primitives import BooleanMatch, ExactMatch, TraceRegex
from karenina.schemas.entities.verified_field import VerificationMeta, VerifiedField


@pytest.mark.unit
class TestVerifiedFieldFactory:
    """Test that VerifiedField creates Pydantic FieldInfo with metadata."""

    def test_creates_field_with_metadata(self):
        """VerifiedField stores verification metadata in json_schema_extra."""

        class MyAnswer(BaseModel):
            target: str = VerifiedField(
                description="The protein target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )

        field_info = MyAnswer.model_fields["target"]
        assert field_info.description == "The protein target"
        extra = field_info.json_schema_extra
        assert "__verification__" in extra

    def test_metadata_round_trips(self):
        """VerificationMeta can be reconstructed from stored dict."""

        class MyAnswer(BaseModel):
            target: str = VerifiedField(
                description="The protein target",
                ground_truth="BCL2",
                verify_with=ExactMatch(normalize=["lowercase"]),
                weight=0.5,
                extraction_hint="Use HGNC symbol",
            )

        extra = MyAnswer.model_fields["target"].json_schema_extra
        meta = VerificationMeta.model_validate(extra["__verification__"])
        assert meta.ground_truth == "BCL2"
        assert meta.weight == 0.5
        assert meta.extraction_hint == "Use HGNC symbol"

    def test_description_required(self):
        """VerifiedField requires description parameter."""
        with pytest.raises(TypeError):
            VerifiedField(ground_truth="BCL2", verify_with=ExactMatch())

    def test_preserves_user_json_schema_extra(self):
        """User-provided json_schema_extra is merged, not overwritten."""

        class MyAnswer(BaseModel):
            target: str = VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
                json_schema_extra={"custom_key": "custom_value"},
            )

        extra = MyAnswer.model_fields["target"].json_schema_extra
        assert extra["custom_key"] == "custom_value"
        assert "__verification__" in extra

    def test_bool_field(self):
        """VerifiedField works with boolean fields."""

        class MyAnswer(BaseModel):
            is_approved: bool = VerifiedField(
                description="FDA approved",
                ground_truth=True,
                verify_with=BooleanMatch(),
            )

        extra = MyAnswer.model_fields["is_approved"].json_schema_extra
        meta = VerificationMeta.model_validate(extra["__verification__"])
        assert meta.ground_truth is True

    def test_trace_field(self):
        """VerifiedField works with trace primitives."""

        class MyAnswer(BaseModel):
            has_citations: bool = VerifiedField(
                description="Has citations",
                ground_truth=True,
                verify_with=TraceRegex(pattern=r"\[\d+\]"),
            )

        extra = MyAnswer.model_fields["has_citations"].json_schema_extra
        meta = VerificationMeta.model_validate(extra["__verification__"])
        assert meta.verify_with["type"] == "TraceRegex"
