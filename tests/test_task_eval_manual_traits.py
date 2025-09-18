"""Tests for TaskEval integration with ManualRubricTrait."""

from unittest.mock import Mock, patch

import pytest

from karenina.benchmark.models import ModelConfig, VerificationConfig
from karenina.benchmark.task_eval.task_eval import TaskEval
from karenina.schemas.rubric_class import ManualRubricTrait, Rubric, RubricTrait


class TestTaskEvalManualTraits:
    """Test TaskEval functionality with ManualRubricTrait."""

    @pytest.fixture
    def mock_parsing_model(self):
        """Create a mock parsing model configuration."""
        return ModelConfig(
            id="parsing-model",
            model_provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )

    @pytest.fixture
    def verification_config(self, mock_parsing_model):
        """Create verification configuration."""
        return VerificationConfig(
            parsing_models=[mock_parsing_model],
            parsing_only=True,
        )

    @pytest.fixture
    def sample_callable_registry(self):
        """Create sample callable functions for testing."""

        def has_code_block(text: str) -> bool:
            return "```" in text

        def word_count_over_20(text: str) -> bool:
            return len(text.split()) > 20

        def contains_python(text: str) -> bool:
            return "python" in text.lower()

        return {
            "has_code_block": has_code_block,
            "word_count_over_20": word_count_over_20,
            "contains_python": contains_python,
        }

    def test_task_eval_with_callable_registry(self, sample_callable_registry):
        """Test TaskEval initialization with callable registry."""
        task = TaskEval(
            task_id="test_task",
            callable_registry=sample_callable_registry,
        )

        assert task.callable_registry == sample_callable_registry
        assert "has_code_block" in task.callable_registry

    def test_register_callable_in_task_eval(self):
        """Test registering callable functions in TaskEval."""
        task = TaskEval(task_id="test_task")

        def test_func(text: str) -> bool:
            return "test" in text

        task.register_callable("test_func", test_func)

        assert "test_func" in task.callable_registry
        assert task.callable_registry["test_func"] == test_func

    def test_register_callable_invalid_signature(self):
        """Test error when registering callable with wrong signature."""
        task = TaskEval(task_id="test_task")

        def bad_func() -> bool:
            return True

        with pytest.raises(ValueError, match="must have exactly one parameter"):
            task.register_callable("bad_func", bad_func)

    @patch("karenina.benchmark.verification.runner.run_single_model_verification")
    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_with_manual_traits_only(
        self,
        mock_init_model,
        mock_verification,
        verification_config,
        sample_callable_registry,
    ):
        """Test evaluation with only manual traits (no LLM evaluation)."""
        # Mock LLM initialization
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # Mock verification result
        mock_verification.return_value = Mock(
            verify_result=True,
            parsed_gt_response="4",
            parsed_llm_response="4",
            execution_time=0.1,
            success=True,
            error=None,
        )

        # Create TaskEval with callable registry
        task = TaskEval(
            task_id="test_manual_traits",
            callable_registry=sample_callable_registry,
        )

        # Add question and logs
        task.add_question(
            {
                "id": "q1",
                "question": "How do you implement a function in Python?",
                "raw_answer": "Use def keyword",
                "answer_template": "def function_name():",
            }
        )

        task.log(
            "You can implement a function in Python using the def keyword like this:\n```python\ndef my_function():\n    pass\n```"
        )

        # Create rubric with only manual traits
        manual_traits = [
            ManualRubricTrait(name="has_code", callable_name="has_code_block"),
            ManualRubricTrait(name="mentions_python", callable_name="contains_python"),
            ManualRubricTrait(name="short_response", callable_name="word_count_over_20", invert_result=True),
        ]

        rubric = Rubric(manual_traits=manual_traits)
        task.add_rubric(rubric)

        # Evaluate
        result = task.evaluate(verification_config)

        # Check that manual traits were evaluated
        assert "q1" in result.global_eval.question_verification
        verification_results = result.global_eval.question_verification["q1"]
        assert len(verification_results) > 0

        verify_rubric = verification_results[0]["verify_rubric"]
        assert verify_rubric["has_code"] is True  # Contains code block
        assert verify_rubric["mentions_python"] is True  # Mentions Python
        assert verify_rubric["short_response"] is False  # More than 20 words (inverted)

    @patch("karenina.benchmark.task_eval.task_eval.run_single_model_verification")
    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_evaluate_with_mixed_traits(
        self,
        mock_init_model,
        mock_verification,
        verification_config,
        sample_callable_registry,
    ):
        """Test evaluation with both LLM and manual traits."""
        # Mock LLM initialization and responses
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm
        mock_llm.invoke.return_value.content = '{"accuracy": true, "clarity": 4}'

        # Mock verification result
        mock_verification.return_value = Mock(
            verify_result=True,
            parsed_gt_response="correct",
            parsed_llm_response="correct",
            execution_time=0.1,
            success=True,
            error=None,
        )

        # Create TaskEval with callable registry
        task = TaskEval(
            task_id="test_mixed_traits",
            callable_registry=sample_callable_registry,
        )

        # Add question and logs
        task.add_question(
            {
                "id": "q1",
                "question": "Explain Python functions",
                "raw_answer": "Functions are reusable code blocks",
                "answer_template": "Functions are...",
            }
        )

        task.log("Functions in Python are reusable blocks of code defined with the def keyword.")

        # Create rubric with both LLM and manual traits
        llm_traits = [
            RubricTrait(name="accuracy", description="Is accurate?", kind="boolean"),
            RubricTrait(name="clarity", description="Clarity score", kind="score", min_score=1, max_score=5),
        ]

        manual_traits = [
            ManualRubricTrait(name="mentions_def", pattern=r"\bdef\b"),
            ManualRubricTrait(name="long_explanation", callable_name="word_count_over_20"),
        ]

        rubric = Rubric(traits=llm_traits, manual_traits=manual_traits)
        task.add_rubric(rubric)

        # Evaluate
        result = task.evaluate(verification_config)

        # Check that both trait types were evaluated
        verify_rubric = result.global_eval.question_verification["q1"][0]["verify_rubric"]

        # Manual traits
        assert verify_rubric["mentions_def"] is True  # Contains "def"
        assert verify_rubric["long_explanation"] is False  # Less than 20 words

        # LLM traits
        assert verify_rubric["accuracy"] is True
        assert verify_rubric["clarity"] == 4

    @patch("karenina.benchmark.task_eval.task_eval.run_single_model_verification")
    @patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified")
    def test_step_evaluation_with_manual_traits(
        self,
        mock_init_model,
        mock_verification,
        verification_config,
        sample_callable_registry,
    ):
        """Test step-specific evaluation with manual traits."""
        # Mock LLM initialization
        mock_llm = Mock()
        mock_init_model.return_value = mock_llm

        # Mock verification result
        mock_verification.return_value = Mock(
            verify_result=True,
            parsed_gt_response="correct",
            parsed_llm_response="correct",
            execution_time=0.1,
            success=True,
            error=None,
        )

        # Create TaskEval with callable registry
        task = TaskEval(
            task_id="test_step_manual",
            callable_registry=sample_callable_registry,
        )

        # Add step-specific question and logs
        task.add_question(
            {
                "id": "step1_q1",
                "question": "Write a Python function",
                "raw_answer": "def func(): pass",
                "answer_template": "def func():",
            },
            step_id="step1",
        )

        task.log("```python\ndef hello():\n    print('Hello')\n```", step_id="step1")

        # Create step-specific rubric with manual traits
        manual_traits = [
            ManualRubricTrait(name="has_code_block", callable_name="has_code_block"),
            ManualRubricTrait(name="contains_python_keyword", callable_name="contains_python"),
        ]

        rubric = Rubric(manual_traits=manual_traits)
        task.add_rubric(rubric, step_id="step1")

        # Evaluate specific step
        result = task.evaluate(verification_config, step_id="step1")

        # Check step evaluation results
        assert "step1_q1" in result.per_step["step1"].question_verification
        verify_rubric = result.per_step["step1"].question_verification["step1_q1"][0]["verify_rubric"]

        assert verify_rubric["has_code_block"] is True
        assert verify_rubric["contains_python_keyword"] is True

    def test_fallback_evaluation_with_manual_traits(self, verification_config, sample_callable_registry):
        """Test fallback evaluation (no answer template) with manual traits."""
        # Create TaskEval with callable registry
        task = TaskEval(
            task_id="test_fallback_manual",
            callable_registry=sample_callable_registry,
        )

        # Add question without answer template (triggers fallback)
        task.add_question(
            {
                "id": "q1",
                "question": "What is Python?",
                "raw_answer": "A programming language",
            }
        )

        task.log("Python is a high-level programming language used for various applications.")

        # Create rubric with manual traits
        manual_traits = [
            ManualRubricTrait(name="mentions_language", pattern=r"language"),
            ManualRubricTrait(name="detailed_response", callable_name="word_count_over_20"),
        ]

        rubric = Rubric(manual_traits=manual_traits)
        task.add_rubric(rubric)

        # Evaluate (should use fallback path)
        with patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified") as mock_init_model:
            mock_llm = Mock()
            mock_init_model.return_value = mock_llm

            result = task.evaluate(verification_config)

        # Check that manual traits were evaluated in fallback mode
        verify_rubric = result.global_eval.question_verification["q1"][0]["verify_rubric"]
        assert verify_rubric["mentions_language"] is True
        assert verify_rubric["detailed_response"] is False  # Less than 20 words

    def test_error_handling_with_manual_traits(self, verification_config):
        """Test error handling when manual trait evaluation fails."""
        # Create TaskEval with missing callable
        task = TaskEval(task_id="test_error_manual")

        # Add question and logs
        task.add_question(
            {
                "id": "q1",
                "question": "Test question",
                "raw_answer": "Test answer",
            }
        )

        task.log("Test response")

        # Create rubric with manual trait that references missing callable
        manual_traits = [
            ManualRubricTrait(name="missing_func", callable_name="nonexistent_function"),
            ManualRubricTrait(name="valid_pattern", pattern=r"test"),
        ]

        rubric = Rubric(manual_traits=manual_traits)
        task.add_rubric(rubric)

        # Evaluate (should handle error gracefully)
        with patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified") as mock_init_model:
            mock_llm = Mock()
            mock_init_model.return_value = mock_llm

            result = task.evaluate(verification_config)

        # Check that error was handled gracefully
        verify_rubric = result.global_eval.question_verification["q1"][0]["verify_rubric"]
        assert verify_rubric["missing_func"] is None  # Failed evaluation
        assert verify_rubric["valid_pattern"] is True  # Successful evaluation

    def test_global_and_step_manual_traits_together(self, verification_config, sample_callable_registry):
        """Test evaluation with both global and step-specific manual traits."""
        # Create TaskEval with callable registry
        task = TaskEval(
            task_id="test_global_step_manual",
            callable_registry=sample_callable_registry,
        )

        # Add global question and rubric
        task.add_question(
            {
                "id": "global_q1",
                "question": "General Python question",
                "raw_answer": "Python answer",
            }
        )

        task.log("Python is great for coding")

        global_manual_traits = [
            ManualRubricTrait(name="mentions_python_global", callable_name="contains_python"),
        ]
        global_rubric = Rubric(manual_traits=global_manual_traits)
        task.add_rubric(global_rubric)

        # Add step-specific question and rubric
        task.add_question(
            {
                "id": "step_q1",
                "question": "Code example",
                "raw_answer": "Code",
            },
            step_id="coding",
        )

        task.log("```python\nprint('Hello')\n```", step_id="coding")

        step_manual_traits = [
            ManualRubricTrait(name="has_code_step", callable_name="has_code_block"),
        ]
        step_rubric = Rubric(manual_traits=step_manual_traits)
        task.add_rubric(step_rubric, step_id="coding")

        # Evaluate global (should automatically evaluate steps too)
        with patch("karenina.benchmark.verification.rubric_evaluator.init_chat_model_unified") as mock_init_model:
            mock_llm = Mock()
            mock_init_model.return_value = mock_llm

            result = task.evaluate(verification_config)

        # Check global evaluation
        global_rubric_result = result.global_eval.question_verification["global_q1"][0]["verify_rubric"]
        assert global_rubric_result["mentions_python_global"] is True

        # Check step evaluation
        step_rubric_result = result.per_step["coding"].question_verification["step_q1"][0]["verify_rubric"]
        assert step_rubric_result["has_code_step"] is True
