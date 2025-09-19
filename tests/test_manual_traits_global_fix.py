"""Test manual traits evaluation in global context."""

from unittest.mock import Mock, patch

import pytest

from karenina.benchmark.models import ModelConfig, VerificationConfig
from karenina.benchmark.task_eval.task_eval import TaskEval
from karenina.schemas.rubric_class import ManualRubricTrait, Rubric


class TestManualTraitsGlobalEvaluation:
    """Test that manual traits are properly evaluated in global context."""

    @pytest.fixture
    def parsing_model(self):
        """Create parsing model configuration."""
        return ModelConfig(
            id="parsing-model",
            model_provider="openai",
            model_name="gpt-4",
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )

    @pytest.fixture
    def verification_config(self, parsing_model):
        """Create verification configuration."""
        return VerificationConfig(
            parsing_models=[parsing_model],
            parsing_only=True,
        )

    def test_manual_traits_evaluated_in_global_context(self, verification_config):
        """Test that manual traits are evaluated in global evaluation context."""

        # Define test callable
        def has_code_snippet(text: str) -> bool:
            return "```" in text

        def word_count_check(text: str) -> bool:
            return len(text.split()) > 10

        # Create TaskEval with callable registry
        task = TaskEval(task_id="test_global_manual_traits")
        task.register_callable("has_code", has_code_snippet)
        task.register_callable("long_text", word_count_check)

        # Add question without answer template (to test standalone rubric evaluation)
        task.add_question(
            {
                "id": "q1",
                "question": "How do you write a Python function?",
                "raw_answer": "Use def keyword",
            }
        )

        # Log response with code
        task.log("""
        To write a Python function, you use the def keyword:

        ```python
        def my_function():
            return "Hello World"
        ```

        This creates a reusable block of code.
        """)

        # Create rubric with manual traits
        manual_traits = [
            ManualRubricTrait(name="has_code_block", callable_name="has_code"),
            ManualRubricTrait(name="detailed_response", callable_name="long_text"),
            ManualRubricTrait(name="mentions_def", pattern=r"\bdef\b", case_sensitive=False),
            ManualRubricTrait(name="no_errors", pattern=r"\berror\b", invert_result=True),
        ]

        rubric = Rubric(manual_traits=manual_traits)
        task.add_rubric(rubric)

        # Mock LLM initialization (no LLM traits so it won't be called)
        with patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified") as mock_init:
            mock_llm = Mock()
            mock_init.return_value = mock_llm

            # Evaluate global context
            result = task.evaluate(verification_config)

        # Check that manual traits were evaluated at the global level
        assert "q1" in result.global_eval.question_verification
        verification_results = result.global_eval.question_verification["q1"]
        assert len(verification_results) > 0

        # Check global rubric scores (standalone evaluation)
        global_rubric_scores = result.global_eval.rubric_scores
        assert "has_code_block" in global_rubric_scores
        assert "detailed_response" in global_rubric_scores
        assert "mentions_def" in global_rubric_scores
        assert "no_errors" in global_rubric_scores

        # Verify the evaluations are correct
        assert global_rubric_scores["has_code_block"] is True  # Contains code block
        assert global_rubric_scores["detailed_response"] is True  # More than 10 words
        assert global_rubric_scores["mentions_def"] is True  # Contains "def"
        assert global_rubric_scores["no_errors"] is True  # No "error" mentioned (inverted)

    def test_mixed_llm_and_manual_traits_global(self, verification_config):
        """Test mixed LLM and manual traits in global evaluation."""
        from karenina.schemas.rubric_class import RubricTrait

        # Define test callable
        def contains_python(text: str) -> bool:
            return "python" in text.lower()

        # Create TaskEval with callable registry
        task = TaskEval(task_id="test_mixed_global")
        task.register_callable("mentions_python", contains_python)

        # Add question
        task.add_question(
            {
                "id": "mixed_q1",
                "question": "Explain Python functions",
                "raw_answer": "Functions are reusable code blocks",
            }
        )

        task.log("Python functions are defined using the def keyword and allow code reuse.")

        # Create rubric with both LLM and manual traits
        llm_traits = [
            RubricTrait(name="accuracy", description="Is accurate?", kind="boolean"),
        ]

        manual_traits = [
            ManualRubricTrait(name="mentions_language", callable_name="mentions_python"),
            ManualRubricTrait(name="has_keyword_def", pattern=r"\bdef\b"),
        ]

        rubric = Rubric(traits=llm_traits, manual_traits=manual_traits)
        task.add_rubric(rubric)

        # Mock LLM response for LLM traits
        with patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified") as mock_init:
            mock_llm = Mock()
            mock_llm.invoke.return_value.content = '{"accuracy": true}'
            mock_init.return_value = mock_llm

            # Evaluate
            result = task.evaluate(verification_config)

        # Check both types of traits were evaluated
        global_rubric_scores = result.global_eval.rubric_scores

        # Manual traits should be evaluated
        assert global_rubric_scores["mentions_language"] is True
        assert global_rubric_scores["has_keyword_def"] is True

        # LLM traits should also be evaluated
        assert global_rubric_scores["accuracy"] is True

    def test_manual_traits_error_handling_global(self, verification_config):
        """Test error handling for manual traits in global context."""
        # Create TaskEval with missing callable
        task = TaskEval(task_id="test_error_global")

        task.add_question(
            {
                "id": "error_q1",
                "question": "Test question",
                "raw_answer": "Test answer",
            }
        )

        task.log("Test response text")

        # Create rubric with manual trait that references missing callable
        manual_traits = [
            ManualRubricTrait(name="missing_callable", callable_name="nonexistent_function"),
            ManualRubricTrait(name="valid_pattern", pattern=r"test", case_sensitive=False),
        ]

        rubric = Rubric(manual_traits=manual_traits)
        task.add_rubric(rubric)

        # Mock LLM
        with patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified") as mock_init:
            mock_llm = Mock()
            mock_init.return_value = mock_llm

            # Evaluate (should handle error gracefully)
            result = task.evaluate(verification_config)

        # Check error handling
        global_rubric_scores = result.global_eval.rubric_scores
        assert global_rubric_scores["missing_callable"] is None  # Failed evaluation
        assert global_rubric_scores["valid_pattern"] is True  # Successful evaluation
