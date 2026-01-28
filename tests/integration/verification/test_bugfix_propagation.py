"""Integration tests for bugfix propagation in the verification pipeline.

These tests verify that bugfixes stages-022 (verify_granular_result) and stages-023
(llm_trait_labels) properly propagate values through the verification pipeline.

Test scenarios:
- stages-022: verify_granular_result populated for multi-attribute templates
- stages-023: llm_trait_labels populated for literal kind LLM traits

These tests use mock stages to isolate the bugfixes being tested without
requiring full LLM calls.
"""

import pytest
from pydantic import Field

from karenina.benchmark.verification.stages import (
    FinalizeResultStage,
    VerificationContext,
    VerifyTemplateStage,
)
from karenina.schemas.domain import BaseAnswer, LLMRubricTrait, Rubric
from karenina.schemas.workflow import ModelConfig

# =============================================================================
# Test Answer Classes for verify_granular_result Tests
# =============================================================================


class SingleAttributeAnswer(BaseAnswer):
    """Single-attribute answer - should NOT have verify_granular()."""

    value: str = Field(description="A single string value")

    def model_post_init(self, __context):
        self.correct = {"value": "expected"}

    def verify(self) -> bool:
        return self.value.strip().lower() == self.correct["value"]


class ThreeAttributeAnswer(BaseAnswer):
    """Three-attribute answer - should have verify_granular()."""

    name: str = Field(description="A person's name")
    age: int = Field(description="A person's age")
    city: str = Field(description="A person's city")

    def model_post_init(self, __context):
        self.correct = {"name": "alice", "age": 30, "city": "boston"}

    def verify(self) -> bool:
        return (
            self.name.strip().lower() == self.correct["name"]
            and self.age == self.correct["age"]
            and self.city.strip().lower() == self.correct["city"]
        )

    def verify_granular(self) -> float:
        """Return fraction of attributes that are correct."""
        correct_count = 0
        total_count = 3

        if self.name.strip().lower() == self.correct["name"]:
            correct_count += 1
        if self.age == self.correct["age"]:
            correct_count += 1
        if self.city.strip().lower() == self.correct["city"]:
            correct_count += 1

        return correct_count / total_count


# =============================================================================
# Mock Evaluator for Testing
# =============================================================================


class MockFieldResult:
    """Mock field verification result."""

    def __init__(self, success: bool, error: str | None = None):
        self.success = success
        self.error = error


class MockRegexResult:
    """Mock regex verification result."""

    def __init__(self, success: bool = True, error: str | None = None):
        self.success = success
        self.error = error
        self.results = {}
        self.details = {}
        self.extraction_results = {}


class MockTemplateEvaluator:
    """Mock template evaluator for testing VerifyTemplateStage."""

    def __init__(self, field_success: bool = True):
        self._field_success = field_success

    def verify_fields(self, parsed_answer):  # noqa: ARG002
        """Return mock field verification result."""
        return MockFieldResult(self._field_success)

    def verify_regex(self, parsed_answer, raw_llm_response):  # noqa: ARG002
        """Return mock regex verification result."""
        return MockRegexResult(success=True)


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
def context_for_verify_template(minimal_model_config: ModelConfig) -> VerificationContext:
    """Return a VerificationContext configured for VerifyTemplateStage testing."""
    return VerificationContext(
        question_id="test-question-1",
        template_id="template-hash-123",
        question_text="What is the person's name, age, and city?",
        template_code="# template code placeholder",
        answering_model=minimal_model_config,
        parsing_model=minimal_model_config,
    )


@pytest.fixture
def context_for_rubric(minimal_model_config: ModelConfig) -> VerificationContext:
    """Return a VerificationContext configured for rubric evaluation testing."""
    rubric = Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="sentiment",
                description="Classify the sentiment",
                kind="literal",
                classes={
                    "negative": "Negative sentiment",
                    "neutral": "Neutral sentiment",
                    "positive": "Positive sentiment",
                },
                higher_is_better=True,
            )
        ]
    )
    return VerificationContext(
        question_id="test-question-1",
        template_id="template-hash-123",
        question_text="How was the service?",
        template_code="# template code placeholder",
        answering_model=minimal_model_config,
        parsing_model=minimal_model_config,
        rubric=rubric,
    )


# =============================================================================
# stages-022: verify_granular_result Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestVerifyGranularResultBugfix:
    """Test that verify_granular_result is properly populated (stages-022 bugfix)."""

    def test_granular_result_populated_all_correct(self, context_for_verify_template: VerificationContext):
        """Verify verify_granular_result is 1.0 when all attributes are correct."""
        # Create answer where all 3 attributes are correct
        answer = ThreeAttributeAnswer(name="Alice", age=30, city="Boston")

        # Set up context with required artifacts
        context_for_verify_template.set_artifact("parsed_answer", answer)
        context_for_verify_template.set_artifact("raw_llm_response", "Name: Alice, Age: 30, City: Boston")
        context_for_verify_template.set_artifact("template_evaluator", MockTemplateEvaluator(field_success=True))

        # Execute the stage
        stage = VerifyTemplateStage()
        assert stage.should_run(context_for_verify_template) is True
        stage.execute(context_for_verify_template)

        # Verify verify_granular_result is set correctly
        granular_result = context_for_verify_template.get_result_field("verify_granular_result")
        assert granular_result is not None, "verify_granular_result should be populated"
        assert granular_result == 1.0, "All attributes correct should give 1.0"

    def test_granular_result_populated_partial_correct(self, context_for_verify_template: VerificationContext):
        """Verify verify_granular_result reflects partial correctness (2/3)."""
        # Create answer where only 2 of 3 attributes are correct
        # name="Alice" (correct), age=30 (correct), city="New York" (wrong - should be Boston)
        answer = ThreeAttributeAnswer(name="Alice", age=30, city="New York")

        # Set up context with required artifacts
        context_for_verify_template.set_artifact("parsed_answer", answer)
        context_for_verify_template.set_artifact("raw_llm_response", "Name: Alice, Age: 30, City: New York")
        context_for_verify_template.set_artifact("template_evaluator", MockTemplateEvaluator(field_success=False))

        # Execute the stage
        stage = VerifyTemplateStage()
        stage.execute(context_for_verify_template)

        # Verify verify_granular_result reflects 2/3 correct
        granular_result = context_for_verify_template.get_result_field("verify_granular_result")
        assert granular_result is not None, "verify_granular_result should be populated"
        assert abs(granular_result - (2 / 3)) < 0.001, f"Expected 2/3, got {granular_result}"

    def test_granular_result_populated_none_correct(self, context_for_verify_template: VerificationContext):
        """Verify verify_granular_result is 0.0 when no attributes are correct."""
        # Create answer where no attributes are correct
        answer = ThreeAttributeAnswer(name="Bob", age=25, city="Miami")

        # Set up context with required artifacts
        context_for_verify_template.set_artifact("parsed_answer", answer)
        context_for_verify_template.set_artifact("raw_llm_response", "Name: Bob, Age: 25, City: Miami")
        context_for_verify_template.set_artifact("template_evaluator", MockTemplateEvaluator(field_success=False))

        # Execute the stage
        stage = VerifyTemplateStage()
        stage.execute(context_for_verify_template)

        # Verify verify_granular_result is 0.0
        granular_result = context_for_verify_template.get_result_field("verify_granular_result")
        assert granular_result is not None, "verify_granular_result should be populated"
        assert granular_result == 0.0, f"Expected 0.0, got {granular_result}"

    def test_granular_result_not_set_for_single_attribute(self, context_for_verify_template: VerificationContext):
        """Verify verify_granular_result is NOT set for single-attribute templates."""
        # Create single-attribute answer (no verify_granular method)
        answer = SingleAttributeAnswer(value="expected")

        # Set up context with required artifacts
        context_for_verify_template.set_artifact("parsed_answer", answer)
        context_for_verify_template.set_artifact("raw_llm_response", "The value is: expected")
        context_for_verify_template.set_artifact("template_evaluator", MockTemplateEvaluator(field_success=True))

        # Execute the stage
        stage = VerifyTemplateStage()
        stage.execute(context_for_verify_template)

        # Verify verify_granular_result is NOT set (single attribute has no verify_granular method)
        granular_result = context_for_verify_template.get_result_field("verify_granular_result")
        assert granular_result is None, "Single-attribute templates should not have verify_granular_result"

    def test_granular_result_propagated_to_final_result(self, context_for_verify_template: VerificationContext):
        """Verify verify_granular_result propagates to VerificationResult via FinalizeResultStage."""
        # Create answer where all attributes are correct
        answer = ThreeAttributeAnswer(name="Alice", age=30, city="Boston")

        # Set up context with required artifacts
        context_for_verify_template.set_artifact("parsed_answer", answer)
        context_for_verify_template.set_artifact("raw_llm_response", "Name: Alice, Age: 30, City: Boston")
        context_for_verify_template.set_artifact("template_evaluator", MockTemplateEvaluator(field_success=True))

        # Execute VerifyTemplateStage first
        verify_stage = VerifyTemplateStage()
        verify_stage.execute(context_for_verify_template)

        # Set required result fields for FinalizeResultStage
        context_for_verify_template.set_result_field("timestamp", "2024-01-01 12:00:00")
        context_for_verify_template.set_result_field("execution_time", 1.0)

        # Execute FinalizeResultStage
        finalize_stage = FinalizeResultStage()
        finalize_stage.execute(context_for_verify_template)

        # Verify the final result contains verify_granular_result
        final_result = context_for_verify_template.get_artifact("final_result")
        assert final_result is not None, "final_result should be produced"
        assert final_result.template is not None, "template component should exist"
        assert final_result.template.verify_granular_result == 1.0, (
            "verify_granular_result should propagate to final result"
        )


# =============================================================================
# stages-023: llm_trait_labels Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestLLMTraitLabelsBugfix:
    """Test that llm_trait_labels is properly propagated (stages-023 bugfix)."""

    def test_llm_trait_labels_propagated_to_final_result(self, context_for_rubric: VerificationContext):
        """Verify llm_trait_labels propagates to VerificationResult via FinalizeResultStage."""
        # Simulate rubric evaluation setting the labels and scores
        context_for_rubric.set_result_field("verify_rubric", {"sentiment": 2})
        context_for_rubric.set_result_field("llm_trait_labels", {"sentiment": "positive"})
        context_for_rubric.set_result_field("rubric_evaluation_strategy", "batch")
        context_for_rubric.set_result_field("timestamp", "2024-01-01 12:00:00")
        context_for_rubric.set_result_field("execution_time", 1.0)

        # Execute FinalizeResultStage
        finalize_stage = FinalizeResultStage()
        finalize_stage.execute(context_for_rubric)

        # Verify the final result contains llm_trait_labels
        final_result = context_for_rubric.get_artifact("final_result")
        assert final_result is not None, "final_result should be produced"
        assert final_result.rubric is not None, "rubric component should exist"
        assert final_result.rubric.llm_trait_labels is not None, "llm_trait_labels should propagate"
        assert final_result.rubric.llm_trait_labels == {"sentiment": "positive"}, (
            "llm_trait_labels should contain correct values"
        )

    def test_llm_trait_labels_with_multiple_traits(self, minimal_model_config: ModelConfig):
        """Verify llm_trait_labels works with multiple literal traits."""
        # Create rubric with multiple literal traits
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="sentiment",
                    kind="literal",
                    classes={"negative": "Neg", "neutral": "Neu", "positive": "Pos"},
                    higher_is_better=True,
                ),
                LLMRubricTrait(
                    name="formality",
                    kind="literal",
                    classes={"casual": "Casual", "formal": "Formal"},
                    higher_is_better=False,
                ),
            ]
        )

        context = VerificationContext(
            question_id="test-question-1",
            template_id="template-hash-123",
            question_text="Test question",
            template_code="# placeholder",
            answering_model=minimal_model_config,
            parsing_model=minimal_model_config,
            rubric=rubric,
        )

        # Simulate rubric evaluation results
        context.set_result_field("verify_rubric", {"sentiment": 2, "formality": 1})
        context.set_result_field(
            "llm_trait_labels",
            {"sentiment": "positive", "formality": "formal"},
        )
        context.set_result_field("rubric_evaluation_strategy", "batch")
        context.set_result_field("timestamp", "2024-01-01 12:00:00")
        context.set_result_field("execution_time", 1.0)

        # Execute FinalizeResultStage
        finalize_stage = FinalizeResultStage()
        finalize_stage.execute(context)

        # Verify the final result contains all llm_trait_labels
        final_result = context.get_artifact("final_result")
        assert final_result.rubric.llm_trait_labels == {
            "sentiment": "positive",
            "formality": "formal",
        }

    def test_llm_trait_labels_none_when_no_literal_traits(self, minimal_model_config: ModelConfig):
        """Verify llm_trait_labels is None when no literal traits are evaluated."""
        # Create rubric with only boolean traits (no literal traits)
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="clarity",
                    description="Is the response clear?",
                    kind="boolean",
                    higher_is_better=True,
                )
            ]
        )

        context = VerificationContext(
            question_id="test-question-1",
            template_id="template-hash-123",
            question_text="Test question",
            template_code="# placeholder",
            answering_model=minimal_model_config,
            parsing_model=minimal_model_config,
            rubric=rubric,
        )

        # Simulate rubric evaluation results (no labels for boolean traits)
        context.set_result_field("verify_rubric", {"clarity": 1})
        # llm_trait_labels is NOT set (only set for literal traits)
        context.set_result_field("rubric_evaluation_strategy", "batch")
        context.set_result_field("timestamp", "2024-01-01 12:00:00")
        context.set_result_field("execution_time", 1.0)

        # Execute FinalizeResultStage
        finalize_stage = FinalizeResultStage()
        finalize_stage.execute(context)

        # Verify llm_trait_labels is None for non-literal traits
        final_result = context.get_artifact("final_result")
        assert final_result.rubric.llm_trait_labels is None, "No labels for boolean traits"

    def test_llm_trait_labels_with_error_state(self, context_for_rubric: VerificationContext):
        """Verify llm_trait_labels handles error state (score=-1, label=[ERROR:...])."""
        # Simulate rubric evaluation with an error
        context_for_rubric.set_result_field("verify_rubric", {"sentiment": -1})
        context_for_rubric.set_result_field(
            "llm_trait_labels",
            {"sentiment": "[ERROR: Invalid classification]"},
        )
        context_for_rubric.set_result_field("rubric_evaluation_strategy", "batch")
        context_for_rubric.set_result_field("timestamp", "2024-01-01 12:00:00")
        context_for_rubric.set_result_field("execution_time", 1.0)

        # Execute FinalizeResultStage
        finalize_stage = FinalizeResultStage()
        finalize_stage.execute(context_for_rubric)

        # Verify the error state is preserved in labels
        final_result = context_for_rubric.get_artifact("final_result")
        assert final_result.rubric.llm_trait_labels == {"sentiment": "[ERROR: Invalid classification]"}

    def test_llm_trait_scores_and_labels_consistency(self, context_for_rubric: VerificationContext):
        """Verify llm_trait_scores and llm_trait_labels are consistent."""
        # Simulate consistent evaluation results
        context_for_rubric.set_result_field("verify_rubric", {"sentiment": 1})
        context_for_rubric.set_result_field("llm_trait_labels", {"sentiment": "neutral"})
        context_for_rubric.set_result_field("rubric_evaluation_strategy", "batch")
        context_for_rubric.set_result_field("timestamp", "2024-01-01 12:00:00")
        context_for_rubric.set_result_field("execution_time", 1.0)

        # Execute FinalizeResultStage
        finalize_stage = FinalizeResultStage()
        finalize_stage.execute(context_for_rubric)

        # Verify both scores and labels are present and consistent
        final_result = context_for_rubric.get_artifact("final_result")
        assert final_result.rubric.llm_trait_scores == {"sentiment": 1}
        assert final_result.rubric.llm_trait_labels == {"sentiment": "neutral"}
        # Score 1 should correspond to "neutral" (index 1 in the classes dict)
