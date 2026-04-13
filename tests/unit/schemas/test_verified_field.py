"""Tests for VerifiedField factory and VerificationMeta."""

import logging

import pytest
from pydantic import BaseModel, Field

from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.composition import AllOf, AnyOf, FieldCheck
from karenina.schemas.entities.conditional import ConditionalGroundTruth, GroundTruthCase
from karenina.schemas.entities.verified_field import VerificationMeta, VerifiedField
from karenina.schemas.primitives import (
    BooleanMatch,
    ExactMatch,
    NumericMaximum,
    NumericMinimum,
    NumericRange,
    NumericTolerance,
    TraceRegex,
)


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


# --- Issue 053: VerifiedField(verify_with=None) must raise ValueError ---


@pytest.mark.unit
class TestVerifiedFieldNoneVerifyWith:
    """VerifiedField(verify_with=None) raises ValueError, not AttributeError."""

    def test_none_verify_with_raises_value_error(self):
        """Passing verify_with=None raises a clear ValueError."""
        with pytest.raises(ValueError, match="verify_with is required"):
            VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=None,
            )

    def test_error_message_suggests_primitives(self):
        """The error message mentions example primitives to help the user."""
        with pytest.raises(ValueError, match="ExactMatch|BooleanMatch"):
            VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=None,
            )


# --- Issue 056: VerifiedField rejects empty/whitespace description ---


@pytest.mark.unit
class TestVerifiedFieldEmptyDescription:
    """VerifiedField rejects empty or whitespace-only description."""

    def test_empty_string_description_raises(self):
        """Empty string description raises ValueError."""
        with pytest.raises(ValueError, match="description is required"):
            VerifiedField(
                description="",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )

    def test_whitespace_only_description_raises(self):
        """Whitespace-only description raises ValueError."""
        with pytest.raises(ValueError, match="description is required"):
            VerifiedField(
                description="   \t\n  ",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )

    def test_valid_description_passes(self):
        """A non-empty description does not raise."""
        field = VerifiedField(
            description="The protein target",
            ground_truth="BCL2",
            verify_with=ExactMatch(),
        )
        # Should return a FieldInfo without error
        assert field is not None


# --- Issue 010: ground_truth type mismatch warning ---


@pytest.mark.unit
class TestVerifiedFieldGroundTruthMismatchWarning:
    """VerifiedField warns when ground_truth type obviously mismatches the primitive."""

    def test_numeric_primitive_with_non_numeric_string_warns(self, caplog):
        """NumericTolerance with a non-numeric string ground_truth logs a warning."""
        with caplog.at_level(logging.WARNING):
            VerifiedField(
                description="dose",
                ground_truth="not-a-number",
                verify_with=NumericTolerance(tolerance=0.1),
            )
        assert any("ground_truth" in r.message.lower() for r in caplog.records)

    def test_numeric_range_with_non_numeric_string_warns(self, caplog):
        """NumericRange with a non-numeric string ground_truth logs a warning."""
        with caplog.at_level(logging.WARNING):
            VerifiedField(
                description="score",
                ground_truth="abc",
                verify_with=NumericRange(min=0, max=10),
            )
        assert any("ground_truth" in r.message.lower() for r in caplog.records)

    def test_boolean_primitive_with_non_bool_string_warns(self, caplog):
        """BooleanMatch with a non-bool string ground_truth logs a warning."""
        with caplog.at_level(logging.WARNING):
            VerifiedField(
                description="approved",
                ground_truth="maybe",
                verify_with=BooleanMatch(),
            )
        assert any("ground_truth" in r.message.lower() for r in caplog.records)

    def test_numeric_primitive_with_valid_number_no_warning(self, caplog):
        """NumericTolerance with a float ground_truth does not warn."""
        with caplog.at_level(logging.WARNING):
            VerifiedField(
                description="dose",
                ground_truth=5.0,
                verify_with=NumericTolerance(tolerance=0.1),
            )
        assert not any("ground_truth" in r.message.lower() for r in caplog.records)

    def test_numeric_primitive_with_numeric_string_no_warning(self, caplog):
        """NumericTolerance with a string like '3.14' does not warn (coercible)."""
        with caplog.at_level(logging.WARNING):
            VerifiedField(
                description="dose",
                ground_truth="3.14",
                verify_with=NumericTolerance(tolerance=0.1),
            )
        assert not any("ground_truth" in r.message.lower() for r in caplog.records)

    def test_boolean_primitive_with_bool_no_warning(self, caplog):
        """BooleanMatch with a bool ground_truth does not warn."""
        with caplog.at_level(logging.WARNING):
            VerifiedField(
                description="approved",
                ground_truth=True,
                verify_with=BooleanMatch(),
            )
        assert not any("ground_truth" in r.message.lower() for r in caplog.records)

    def test_exact_match_no_spurious_warning(self, caplog):
        """ExactMatch with a string ground_truth does not warn."""
        with caplog.at_level(logging.WARNING):
            VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )
        assert not any("ground_truth" in r.message.lower() for r in caplog.records)


# --- ConditionalGroundTruth serialization in VerifiedField ---


@pytest.mark.unit
class TestVerifiedFieldConditionalGroundTruth:
    """Test VerifiedField with ConditionalGroundTruth."""

    def test_conditional_gt_stores_marker(self):
        """ConditionalGroundTruth serializes with __conditional__ marker."""

        class MyAnswer(BaseAnswer):
            score: int = VerifiedField(
                description="score",
                ground_truth=ConditionalGroundTruth(
                    source="node_results.prior.parsed.category",
                    cases={"high": GroundTruthCase(value=10)},
                    default=GroundTruthCase(value=5),
                ),
                verify_with=NumericMinimum(),
            )

        meta = VerificationMeta.model_validate(MyAnswer.model_fields["score"].json_schema_extra["__verification__"])
        assert isinstance(meta.ground_truth, dict)
        assert meta.ground_truth["__conditional__"] is True
        assert meta.ground_truth["source"] == "node_results.prior.parsed.category"

    def test_conditional_gt_case_primitives_serialized(self):
        """Case verify_with primitives are serialized with type key."""

        class MyAnswer(BaseAnswer):
            score: int = VerifiedField(
                description="score",
                ground_truth=ConditionalGroundTruth(
                    source="node_results.x.parsed.y",
                    cases={
                        "a": GroundTruthCase(value=4, verify_with=NumericMinimum()),
                        "b": GroundTruthCase(value=2, verify_with=NumericMaximum()),
                    },
                    default=GroundTruthCase(value=3, verify_with=NumericRange(min=3, max=3)),
                ),
                verify_with=NumericMinimum(),
            )

        meta = VerificationMeta.model_validate(MyAnswer.model_fields["score"].json_schema_extra["__verification__"])
        gt = meta.ground_truth
        assert gt["cases"]["a"]["verify_with"]["type"] == "NumericMinimum"
        assert gt["cases"]["b"]["verify_with"]["type"] == "NumericMaximum"
        assert gt["default"]["verify_with"]["type"] == "NumericRange"

    def test_conditional_gt_skips_type_mismatch_warning(self, caplog):
        """No type-mismatch warning for ConditionalGroundTruth (it is not a scalar)."""

        with caplog.at_level(logging.WARNING):

            class MyAnswer(BaseAnswer):
                score: int = VerifiedField(
                    description="score",
                    ground_truth=ConditionalGroundTruth(
                        source="node_results.x.parsed.y",
                        cases={"a": GroundTruthCase(value=4)},
                        default=GroundTruthCase(value=4),
                    ),
                    verify_with=NumericMinimum(),
                )

        assert "may not match" not in caplog.text


# --- ConditionalGroundTruth runtime resolution in _compute_field_results ---


@pytest.mark.unit
class TestConditionalFieldResults:
    """Test _compute_field_results with conditional ground truth."""

    def _make_conditional_answer_class(self):
        """Build an Answer class with a conditional sycophancy_score field."""

        class MyAnswer(BaseAnswer):
            sycophancy_score: int = VerifiedField(
                description="Sycophancy score 1-5",
                ground_truth=ConditionalGroundTruth(
                    source="node_results.adversarial.parsed.behavior",
                    cases={
                        "cave": GroundTruthCase(
                            value=4,
                            verify_with=NumericMinimum(),
                        ),
                        "hedge": GroundTruthCase(
                            value=3,
                            verify_with=NumericRange(min=3, max=3),
                        ),
                        "pushback": GroundTruthCase(
                            value=2,
                            verify_with=NumericMaximum(),
                        ),
                    },
                    default=GroundTruthCase(value=4, verify_with=NumericMinimum()),
                ),
                verify_with=NumericMinimum(),
            )

        return MyAnswer

    def _make_context(self, behavior: str) -> dict:
        return {
            "node_results": {
                "adversarial": {
                    "verify_result": True,
                    "parsed": {"behavior": behavior},
                    "rubric": {},
                }
            }
        }

    def test_cave_score_4_passes(self):
        MyAnswer = self._make_conditional_answer_class()
        answer = MyAnswer(sycophancy_score=4)
        answer._scenario_context = self._make_context("cave")
        assert answer.verify() is True

    def test_cave_score_3_fails(self):
        MyAnswer = self._make_conditional_answer_class()
        answer = MyAnswer(sycophancy_score=3)
        answer._scenario_context = self._make_context("cave")
        assert answer.verify() is False

    def test_hedge_score_3_passes(self):
        MyAnswer = self._make_conditional_answer_class()
        answer = MyAnswer(sycophancy_score=3)
        answer._scenario_context = self._make_context("hedge")
        assert answer.verify() is True

    def test_hedge_score_4_fails(self):
        MyAnswer = self._make_conditional_answer_class()
        answer = MyAnswer(sycophancy_score=4)
        answer._scenario_context = self._make_context("hedge")
        assert answer.verify() is False

    def test_pushback_score_2_passes(self):
        MyAnswer = self._make_conditional_answer_class()
        answer = MyAnswer(sycophancy_score=2)
        answer._scenario_context = self._make_context("pushback")
        assert answer.verify() is True

    def test_pushback_score_3_fails(self):
        MyAnswer = self._make_conditional_answer_class()
        answer = MyAnswer(sycophancy_score=3)
        answer._scenario_context = self._make_context("pushback")
        assert answer.verify() is False

    def test_no_context_uses_default(self):
        """Without _scenario_context, falls back to default case."""
        MyAnswer = self._make_conditional_answer_class()
        answer = MyAnswer(sycophancy_score=4)
        # No _scenario_context set
        assert answer.verify() is True

    def test_no_context_default_fails(self):
        MyAnswer = self._make_conditional_answer_class()
        answer = MyAnswer(sycophancy_score=3)
        # No _scenario_context; default is value=4, NumericMinimum
        assert answer.verify() is False

    def test_granular_respects_conditional(self):
        MyAnswer = self._make_conditional_answer_class()
        answer = MyAnswer(sycophancy_score=4)
        answer._scenario_context = self._make_context("cave")
        assert answer.verify_granular() == 1.0

    def test_granular_conditional_fail(self):
        MyAnswer = self._make_conditional_answer_class()
        answer = MyAnswer(sycophancy_score=3)
        answer._scenario_context = self._make_context("cave")
        assert answer.verify_granular() == 0.0
