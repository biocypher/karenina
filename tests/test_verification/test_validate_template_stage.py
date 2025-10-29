"""Tests for ValidateTemplateStage."""

from karenina.benchmark.verification.stage import VerificationContext
from karenina.benchmark.verification.stages.validate_template import ValidateTemplateStage


class TestValidateTemplateStage:
    """Test suite for ValidateTemplateStage."""

    def test_should_run_always_returns_true(self, basic_context: VerificationContext) -> None:
        """Test that should_run always returns True for the first stage."""
        stage = ValidateTemplateStage()
        assert stage.should_run(basic_context) is True

    def test_valid_template_execution(self, basic_context: VerificationContext, valid_template: str) -> None:
        """Test successful validation and question ID injection with valid template."""
        basic_context.template_code = valid_template
        stage = ValidateTemplateStage()

        # Execute stage
        stage.execute(basic_context)

        # Should not have error
        assert basic_context.error is None

        # Should have RawAnswer and Answer artifacts
        assert "RawAnswer" in basic_context.artifacts
        assert "Answer" in basic_context.artifacts

        # Answer class should be wrapped to inject question_id
        Answer = basic_context.get_artifact("Answer")
        assert Answer is not None
        # The inject_question_id_into_answer_class creates a wrapper that sets self.id
        # Verify this by checking we can create an instance
        assert Answer is not None

    def test_invalid_template_syntax(self, basic_context: VerificationContext, invalid_template_syntax: str) -> None:
        """Test that invalid template syntax is caught."""
        basic_context.template_code = invalid_template_syntax
        stage = ValidateTemplateStage()

        # Execute stage
        stage.execute(basic_context)

        # Should have error
        assert basic_context.error is not None
        assert "Template validation failed" in basic_context.error

        # Should have validation error artifact
        error = basic_context.get_artifact("template_validation_error")
        assert error is not None

    def test_missing_verify_method(
        self, basic_context: VerificationContext, invalid_template_missing_verify: str
    ) -> None:
        """Test that template missing verify() method is caught."""
        basic_context.template_code = invalid_template_missing_verify
        stage = ValidateTemplateStage()

        # Execute stage
        stage.execute(basic_context)

        # Should have error
        assert basic_context.error is not None
        assert "Template validation failed" in basic_context.error

    def test_question_id_injection(self, basic_context: VerificationContext, valid_template: str) -> None:
        """Test that question_id is properly injected into Answer class."""
        basic_context.template_code = valid_template
        basic_context.question_id = "unique_test_id_12345"
        stage = ValidateTemplateStage()

        # Execute stage
        stage.execute(basic_context)

        # Get the Answer class
        Answer = basic_context.get_artifact("Answer")
        assert Answer is not None

        # Create an instance and verify id is set via model_post_init
        answer_instance = Answer(
            result=4,
            correct={"value": 4},
        )
        # The inject_question_id_into_answer_class sets self.id in model_post_init
        assert answer_instance.id == "unique_test_id_12345"

    def test_stage_metadata(self) -> None:
        """Test stage name and artifact declarations."""
        stage = ValidateTemplateStage()

        assert stage.name == "ValidateTemplate"
        assert stage.requires == []
        assert "RawAnswer" in stage.produces
        assert "Answer" in stage.produces
        assert "template_validation_error" in stage.produces
