"""Unit tests for regex-only template pipeline support.

Tests cover:
- is_regex_only_template() detection logic
- Template validation (verify() optional for regex-only)
- ParseTemplateStage fast path (skips LLM for regex-only templates)
- VerifyTemplateStage without evaluator (regex-only verification)
- End-to-end StageOrchestrator run with regex-only template
"""

import pytest
from pydantic import Field

from karenina.benchmark.verification.stages import (
    ArtifactKeys,
    ParseTemplateStage,
    VerificationContext,
    VerifyTemplateStage,
)
from karenina.benchmark.verification.utils.template_parsing_helpers import (
    is_regex_only_template,
)
from karenina.benchmark.verification.utils.template_validation import (
    validate_answer_template,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import BaseAnswer

# =============================================================================
# Test Answer Classes
# =============================================================================

REGEX_ONLY_TEMPLATE_CODE = """\
from karenina.schemas.entities import BaseAnswer

class Answer(BaseAnswer):
    def model_post_init(self, __context):
        self.regex = {
            "has_number": {
                "pattern": r"\\\\d+",
                "expected": "42",
                "match_type": "contains",
            }
        }
"""

FIELD_TEMPLATE_NO_VERIFY_CODE = """\
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    target: str = Field(description="The target protein")
"""


class RegexOnlyAnswer(BaseAnswer):
    """Template with no user-defined fields, only regex validation."""

    def model_post_init(self, __context: object) -> None:
        self.regex = {
            "has_number": {
                "pattern": r"\d+",
                "expected": "42",
                "match_type": "contains",
            }
        }

    def verify(self) -> bool:
        return True


class FieldAnswer(BaseAnswer):
    """Template with user-defined fields."""

    drug_target: str = Field(description="The drug target protein")

    def verify(self) -> bool:
        return self.drug_target.lower() == "egfr"


class RegexOnlyNoVerifyAnswer(BaseAnswer):
    """Regex-only template without a custom verify() method."""

    def model_post_init(self, __context: object) -> None:
        self.regex = {
            "check_yes": {
                "pattern": r"\byes\b",
                "expected": 1,
                "match_type": "count",
            }
        }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_model_config() -> ModelConfig:
    """Return a minimal ModelConfig for testing."""
    return ModelConfig(
        id="test-model",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.0,
    )


@pytest.fixture
def regex_only_context(minimal_model_config: ModelConfig) -> VerificationContext:
    """Return a VerificationContext set up for regex-only template testing."""
    return VerificationContext(
        question_id="test-regex-only-1",
        template_id="regex-only-hash",
        question_text="What is the answer?",
        template_code=REGEX_ONLY_TEMPLATE_CODE,
        answering_model=minimal_model_config,
        parsing_model=minimal_model_config,
    )


# =============================================================================
# is_regex_only_template() Tests
# =============================================================================


@pytest.mark.unit
class TestIsRegexOnlyTemplate:
    """Test is_regex_only_template() detection logic."""

    def test_positive_no_user_fields(self) -> None:
        """Template with no user-defined fields returns True."""
        assert is_regex_only_template(RegexOnlyAnswer) is True

    def test_negative_has_user_fields(self) -> None:
        """Template with user-defined fields returns False."""
        assert is_regex_only_template(FieldAnswer) is False

    def test_positive_bare_base_answer(self) -> None:
        """BaseAnswer itself has no user fields."""
        assert is_regex_only_template(BaseAnswer) is True

    def test_positive_no_verify_method(self) -> None:
        """Regex-only template without verify() is still regex-only."""
        assert is_regex_only_template(RegexOnlyNoVerifyAnswer) is True


# =============================================================================
# Template Validation Tests
# =============================================================================


@pytest.mark.unit
class TestRegexOnlyValidation:
    """Test that validate_answer_template allows regex-only templates without verify()."""

    def test_regex_only_without_verify_passes(self) -> None:
        """Regex-only template without verify() passes validation."""
        is_valid, error_msg, answer_cls = validate_answer_template(REGEX_ONLY_TEMPLATE_CODE)
        assert is_valid is True
        assert error_msg is None
        assert answer_cls is not None

    def test_field_template_without_verify_fails(self) -> None:
        """Template with fields but no verify() fails validation."""
        is_valid, error_msg, _answer_cls = validate_answer_template(FIELD_TEMPLATE_NO_VERIFY_CODE)
        assert is_valid is False
        assert "verify" in error_msg


# =============================================================================
# ParseTemplateStage Regex-Only Fast Path Tests
# =============================================================================


@pytest.mark.unit
class TestParseTemplateRegexOnlyFastPath:
    """Test that ParseTemplateStage skips LLM parsing for regex-only templates."""

    def test_fast_path_sets_correct_artifacts(
        self,
        regex_only_context: VerificationContext,
    ) -> None:
        """Regex-only fast path sets parsed_answer, evaluator=None, and model string."""
        regex_only_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "The answer is 42.")
        regex_only_context.set_artifact(ArtifactKeys.ANSWER, RegexOnlyAnswer)
        regex_only_context.set_artifact(ArtifactKeys.RAW_ANSWER, RegexOnlyAnswer)

        stage = ParseTemplateStage()
        stage.execute(regex_only_context)

        # Verify artifacts
        parsed = regex_only_context.get_artifact(ArtifactKeys.PARSED_ANSWER)
        assert parsed is not None
        assert isinstance(parsed, RegexOnlyAnswer)
        assert regex_only_context.get_artifact(ArtifactKeys.TEMPLATE_EVALUATOR) is None
        assert regex_only_context.get_artifact(ArtifactKeys.PARSING_MODEL_STR) == "regex_only (no LLM)"
        assert regex_only_context.get_artifact(ArtifactKeys.DEEP_JUDGMENT_PERFORMED) is False

        # Verify result fields
        assert regex_only_context.get_result_field(ArtifactKeys.TRACE_EXTRACTION_ERROR) is None

    def test_fast_path_no_error(
        self,
        regex_only_context: VerificationContext,
    ) -> None:
        """Regex-only fast path completes without errors."""
        regex_only_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "42")
        regex_only_context.set_artifact(ArtifactKeys.ANSWER, RegexOnlyAnswer)
        regex_only_context.set_artifact(ArtifactKeys.RAW_ANSWER, RegexOnlyAnswer)

        stage = ParseTemplateStage()
        stage.execute(regex_only_context)

        assert regex_only_context.error is None
        assert regex_only_context.completed_without_errors is True

    def test_fast_path_not_triggered_for_field_templates(
        self,
        regex_only_context: VerificationContext,
    ) -> None:
        """Templates with fields do NOT trigger the fast path."""
        regex_only_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "EGFR")
        regex_only_context.set_artifact(ArtifactKeys.ANSWER, FieldAnswer)
        regex_only_context.set_artifact(ArtifactKeys.RAW_ANSWER, FieldAnswer)

        stage = ParseTemplateStage()
        # This will fail because we don't have a real parsing model,
        # but the point is it does NOT take the fast path
        stage.execute(regex_only_context)

        # Should NOT have the regex-only marker
        assert regex_only_context.get_artifact(ArtifactKeys.PARSING_MODEL_STR) != "regex_only (no LLM)"


# =============================================================================
# VerifyTemplateStage Without Evaluator Tests
# =============================================================================


@pytest.mark.unit
class TestVerifyTemplateWithoutEvaluator:
    """Test VerifyTemplateStage when evaluator is None (regex-only path)."""

    def test_regex_matches_verify_result_true(
        self,
        regex_only_context: VerificationContext,
    ) -> None:
        """Regex match succeeds → verify_result=True."""
        parsed = RegexOnlyAnswer()
        regex_only_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        regex_only_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "The answer is 42.")
        regex_only_context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, None)

        stage = VerifyTemplateStage()
        stage.execute(regex_only_context)

        assert regex_only_context.get_artifact(ArtifactKeys.VERIFY_RESULT) is True
        assert regex_only_context.get_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT) is True
        regex_results = regex_only_context.get_artifact(ArtifactKeys.REGEX_VERIFICATION_RESULTS)
        assert regex_results["success"] is True
        assert regex_only_context.error is None

    def test_regex_no_match_verify_result_false(
        self,
        regex_only_context: VerificationContext,
    ) -> None:
        """Regex doesn't match → verify_result=False."""
        parsed = RegexOnlyAnswer()
        regex_only_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        regex_only_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "No numbers here.")
        regex_only_context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, None)

        stage = VerifyTemplateStage()
        stage.execute(regex_only_context)

        assert regex_only_context.get_artifact(ArtifactKeys.VERIFY_RESULT) is False
        regex_results = regex_only_context.get_artifact(ArtifactKeys.REGEX_VERIFICATION_RESULTS)
        assert regex_results["success"] is False

    def test_without_verify_method_defaults_to_true(
        self,
        regex_only_context: VerificationContext,
    ) -> None:
        """Regex-only template without verify() defaults field_verification to True."""
        parsed = RegexOnlyNoVerifyAnswer()
        regex_only_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        regex_only_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "yes")
        regex_only_context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, None)

        stage = VerifyTemplateStage()
        stage.execute(regex_only_context)

        assert regex_only_context.get_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT) is True
        assert regex_only_context.get_artifact(ArtifactKeys.VERIFY_RESULT) is True

    def test_stores_result_fields(
        self,
        regex_only_context: VerificationContext,
    ) -> None:
        """Verify all result fields are set correctly for regex-only path."""
        parsed = RegexOnlyAnswer()
        regex_only_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        regex_only_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "42 is the answer")
        regex_only_context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, None)

        stage = VerifyTemplateStage()
        stage.execute(regex_only_context)

        assert regex_only_context.get_result_field(ArtifactKeys.VERIFY_RESULT) is True
        assert regex_only_context.get_result_field(ArtifactKeys.REGEX_VALIDATIONS_PERFORMED) is True
        assert regex_only_context.get_result_field(ArtifactKeys.REGEX_OVERALL_SUCCESS) is True
        assert regex_only_context.get_result_field(ArtifactKeys.REGEX_EXTRACTION_RESULTS) is not None

    def test_extracts_regex_matches(
        self,
        regex_only_context: VerificationContext,
    ) -> None:
        """Verify regex extraction results contain the actual matches."""
        parsed = RegexOnlyAnswer()
        regex_only_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        regex_only_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "42 and 99")
        regex_only_context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, None)

        stage = VerifyTemplateStage()
        stage.execute(regex_only_context)

        extraction = regex_only_context.get_artifact(ArtifactKeys.REGEX_EXTRACTION_RESULTS)
        assert "has_number" in extraction
        assert "42" in extraction["has_number"]
        assert "99" in extraction["has_number"]


# =============================================================================
# End-to-End: ParseTemplate + VerifyTemplate with Regex-Only Template
# =============================================================================


@pytest.mark.unit
class TestRegexOnlyEndToEnd:
    """Test regex-only template through ParseTemplate → VerifyTemplate."""

    def test_parse_then_verify_succeeds(
        self,
        regex_only_context: VerificationContext,
    ) -> None:
        """Full parse→verify flow for regex-only template with matching response."""
        regex_only_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "The answer is 42.")
        regex_only_context.set_artifact(ArtifactKeys.ANSWER, RegexOnlyAnswer)
        regex_only_context.set_artifact(ArtifactKeys.RAW_ANSWER, RegexOnlyAnswer)

        # Run ParseTemplate
        parse_stage = ParseTemplateStage()
        parse_stage.execute(regex_only_context)

        assert regex_only_context.error is None
        assert regex_only_context.get_artifact(ArtifactKeys.TEMPLATE_EVALUATOR) is None

        # Run VerifyTemplate
        verify_stage = VerifyTemplateStage()
        assert verify_stage.should_run(regex_only_context) is True
        verify_stage.execute(regex_only_context)

        assert regex_only_context.error is None
        assert regex_only_context.get_artifact(ArtifactKeys.VERIFY_RESULT) is True

    def test_parse_then_verify_fails(
        self,
        regex_only_context: VerificationContext,
    ) -> None:
        """Full parse→verify flow for regex-only template with non-matching response."""
        regex_only_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "No numbers here.")
        regex_only_context.set_artifact(ArtifactKeys.ANSWER, RegexOnlyAnswer)
        regex_only_context.set_artifact(ArtifactKeys.RAW_ANSWER, RegexOnlyAnswer)

        # Run ParseTemplate
        parse_stage = ParseTemplateStage()
        parse_stage.execute(regex_only_context)
        assert regex_only_context.error is None

        # Run VerifyTemplate
        verify_stage = VerifyTemplateStage()
        verify_stage.execute(regex_only_context)

        assert regex_only_context.error is None
        assert regex_only_context.get_artifact(ArtifactKeys.VERIFY_RESULT) is False
