"""Tests for VerifiedField factory and VerificationMeta."""

import pytest
from pydantic import BaseModel, Field

from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.composition import AllOf, AnyOf, FieldCheck
from karenina.schemas.entities.primitives import (
    BooleanMatch,
    ExactMatch,
    NumericTolerance,
    TraceRegex,
)
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


# --- Integration tests: BaseAnswer + VerifiedField ---


@pytest.mark.unit
class TestBaseAnswerGetVerifiedFields:
    """Test _get_verified_fields classmethod."""

    def test_extracts_verified_fields(self):
        class MyAnswer(BaseAnswer):
            target: str = VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )

        verified = MyAnswer._get_verified_fields()
        assert "target" in verified
        assert verified["target"].ground_truth == "BCL2"

    def test_ignores_plain_fields(self):
        class MyAnswer(BaseAnswer):
            target: str = VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )
            notes: str = Field(description="extra notes", default="")

        verified = MyAnswer._get_verified_fields()
        assert "target" in verified
        assert "notes" not in verified

    def test_empty_for_classic_template(self):
        class MyAnswer(BaseAnswer):
            target: str = Field(description="target", default="")

        verified = MyAnswer._get_verified_fields()
        assert verified == {}


@pytest.mark.unit
class TestBaseAnswerAutoVerify:
    """Test auto-generated verify() method."""

    def test_all_fields_pass(self):
        class MyAnswer(BaseAnswer):
            target: str = VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )
            is_approved: bool = VerifiedField(
                description="approved",
                ground_truth=True,
                verify_with=BooleanMatch(),
            )

        answer = MyAnswer(target="bcl2", is_approved=True)
        assert answer.verify() is True

    def test_one_field_fails(self):
        class MyAnswer(BaseAnswer):
            target: str = VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )

        answer = MyAnswer(target="TP53")
        assert answer.verify() is False

    def test_with_composition_strategy(self):
        class MyAnswer(BaseAnswer):
            target: str = VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )
            dose: float = VerifiedField(
                description="dose",
                ground_truth=5.0,
                verify_with=NumericTolerance(tolerance=0.1, mode="relative"),
            )
            unit: str = VerifiedField(
                description="unit",
                ground_truth="nM",
                verify_with=ExactMatch(),
            )

            class VerificationStrategy:
                verify_strategy = AllOf(
                    conditions=[
                        FieldCheck(field="target"),
                        AnyOf(conditions=[FieldCheck(field="dose"), FieldCheck(field="unit")]),
                    ]
                )

        # target correct, dose wrong, unit correct: should pass (AnyOf)
        answer = MyAnswer(target="BCL2", dose=99.0, unit="nM")
        assert answer.verify() is True

    def test_classic_template_has_no_verify(self):
        """Classic templates without VerifiedField do not get auto-generated verify()."""

        class MyAnswer(BaseAnswer):
            target: str = Field(description="target", default="")

        answer = MyAnswer(target="BCL2")
        assert not hasattr(answer, "verify")

    def test_custom_verify_overrides(self):
        """User-defined verify() shadows the auto-generated one."""

        class MyAnswer(BaseAnswer):
            target: str = VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )

            def verify(self) -> bool:
                return self.target == "CUSTOM"

        answer = MyAnswer(target="CUSTOM")
        assert answer.verify() is True
        answer2 = MyAnswer(target="BCL2")
        assert answer2.verify() is False


@pytest.mark.unit
class TestBaseAnswerAutoVerifyGranular:
    """Test auto-generated verify_granular() method."""

    def test_all_pass_returns_1(self):
        class MyAnswer(BaseAnswer):
            a: str = VerifiedField(description="a", ground_truth="x", verify_with=ExactMatch(), weight=1.0)
            b: str = VerifiedField(description="b", ground_truth="y", verify_with=ExactMatch(), weight=1.0)

        answer = MyAnswer(a="x", b="y")
        assert answer.verify_granular() == 1.0

    def test_half_pass_returns_half(self):
        class MyAnswer(BaseAnswer):
            a: str = VerifiedField(description="a", ground_truth="x", verify_with=ExactMatch(), weight=1.0)
            b: str = VerifiedField(description="b", ground_truth="y", verify_with=ExactMatch(), weight=1.0)

        answer = MyAnswer(a="x", b="wrong")
        assert answer.verify_granular() == 0.5

    def test_weighted_scoring(self):
        class MyAnswer(BaseAnswer):
            a: str = VerifiedField(description="a", ground_truth="x", verify_with=ExactMatch(), weight=3.0)
            b: str = VerifiedField(description="b", ground_truth="y", verify_with=ExactMatch(), weight=1.0)

        answer = MyAnswer(a="x", b="wrong")
        assert answer.verify_granular() == 0.75  # 3 / 4


@pytest.mark.unit
class TestBaseAnswerAutoGroundTruth:
    """Test auto-generated ground_truth() method."""

    def test_sets_correct(self):
        class MyAnswer(BaseAnswer):
            target: str = VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )

        answer = MyAnswer(target="anything")
        assert answer.correct == {"target": "BCL2"}


@pytest.mark.unit
class TestBaseAnswerTraceFields:
    """Test trace field handling."""

    def test_trace_field_with_raw_trace(self):
        class MyAnswer(BaseAnswer):
            has_citations: bool = VerifiedField(
                description="has citations",
                ground_truth=True,
                verify_with=TraceRegex(pattern=r"\[\d+\]"),
            )

        answer = MyAnswer(has_citations=False)
        answer._raw_trace = "See [1] and [2]"
        assert answer.verify() is True

    def test_trace_field_absent_with_ground_truth_false(self):
        class MyAnswer(BaseAnswer):
            no_profanity: bool = VerifiedField(
                description="no profanity",
                ground_truth=False,
                verify_with=TraceRegex(pattern=r"damn|hell"),
            )

        answer = MyAnswer(no_profanity=False)
        answer._raw_trace = "This is a clean response"
        assert answer.verify() is True

    def test_trace_field_without_raw_trace_raises(self):
        class MyAnswer(BaseAnswer):
            has_citations: bool = VerifiedField(
                description="has citations",
                ground_truth=True,
                verify_with=TraceRegex(pattern=r"\[\d+\]"),
            )

        answer = MyAnswer(has_citations=False)
        with pytest.raises(ValueError, match="requires _raw_trace"):
            answer.verify()

    def test_mixed_parsed_and_trace(self):
        class MyAnswer(BaseAnswer):
            target: str = VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )
            has_citations: bool = VerifiedField(
                description="citations",
                ground_truth=True,
                verify_with=TraceRegex(pattern=r"\[\d+\]"),
            )

        answer = MyAnswer(target="BCL2", has_citations=False)
        answer._raw_trace = "BCL2 is the target [1]"
        assert answer.verify() is True
