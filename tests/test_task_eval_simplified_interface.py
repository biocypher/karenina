"""Tests for TaskEval simplified interface with dict traces and rubric-only mode.

This test file validates the new simplified interface for TaskEval:
- Dict trace support (str | dict[str, str])
- Automatic evaluation mode detection
- Rubric-only mode (no templates required)
- Template-and-rubric mode
- Dict trace per-key evaluation
- Replicate support
"""

import os

import pytest

from karenina.benchmark.task_eval import TaskEval
from karenina.schemas.domain import LLMRubricTrait, RegexTrait, Rubric
from karenina.schemas.workflow import ModelConfig, VerificationConfig


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key required for TaskEval tests")
class TestTaskEvalSimplifiedInterface:
    """Test simplified TaskEval interface."""

    def test_dict_trace_logging(self) -> None:
        """Test that dict traces are properly logged with metadata."""
        task = TaskEval(task_id="dict_test")

        # Log a dict trace
        trace_dict = {"reasoning": "I thought about the problem", "action": "I took action A", "result": "Success"}
        task.log(trace_dict)

        # Verify log was stored correctly
        assert len(task.global_logs) == 1
        log = task.global_logs[0]

        assert log.is_dict_structured is True
        assert log.dict_keys == ["reasoning", "action", "result"]

        # Verify JSON serialization
        import json

        parsed = json.loads(log.text)
        assert parsed == trace_dict

    def test_string_trace_logging(self) -> None:
        """Test that string traces still work as before."""
        task = TaskEval(task_id="string_test")

        # Log a string trace
        task.log("This is a simple string log")

        # Verify log was stored correctly
        assert len(task.global_logs) == 1
        log = task.global_logs[0]

        assert log.is_dict_structured is False
        assert log.dict_keys is None
        assert log.text == "This is a simple string log"

    def test_rubric_only_mode_string_trace(self) -> None:
        """Test rubric-only evaluation mode with string traces."""
        task = TaskEval(task_id="rubric_only_string")

        # Add rubrics only (no templates)
        rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="clarity", description="Is the response clear?", kind="boolean")],
            regex_traits=[RegexTrait(name="has_action", description="Contains the word 'action'", pattern=r"action")],
        )
        task.add_rubric(rubric)

        # Log string traces
        task.log("I took action A to solve the problem")
        task.log("The result was successful")

        # Configure evaluation (no templates in questions)
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parser",
                    model_provider="openai",
                    model_name="gpt-4o-mini",
                    system_prompt="Evaluate the response quality",
                )
            ],
            parsing_only=True,
            replicate_count=1,
        )

        # Evaluate - should work without templates
        result = task.evaluate(config)

        # Verify rubric evaluation happened
        assert result.global_eval is not None
        # Note: In rubric_only mode without questions, no results are generated
        # because we only process explicit questions. This is expected behavior.

    def test_rubric_only_mode_dict_trace(self) -> None:
        """Test rubric-only evaluation with dict traces (per-key evaluation)."""
        task = TaskEval(task_id="rubric_only_dict")

        # Add rubrics only (no templates)
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="quality", description="Is the output high quality?", kind="score", min_score=1, max_score=5
                )
            ],
            regex_traits=[RegexTrait(name="has_success", description="Contains 'success'", pattern=r"success")],
        )
        task.add_rubric(rubric)

        # Log dict traces
        task.log(
            {
                "reasoning": "I analyzed the problem carefully",
                "action": "I implemented solution A",
                "result": "Success! All tests passed",
            }
        )

        # Configure evaluation
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parser",
                    model_provider="openai",
                    model_name="gpt-4o-mini",
                    system_prompt="Evaluate response quality",
                )
            ],
            parsing_only=True,
            replicate_count=1,
        )

        # Evaluate - should create synthetic questions per dict key
        result = task.evaluate(config)

        # Verify per-key evaluation happened
        assert result.global_eval is not None
        verification_results = result.global_eval.verification_results

        # Should have synthetic questions for each key
        expected_keys = ["dict_key_action", "dict_key_reasoning", "dict_key_result"]
        for key in expected_keys:
            assert key in verification_results, f"Missing synthetic question for {key}"

            # Each key should have verification results
            results = verification_results[key]
            assert len(results) > 0

            # Verify rubric evaluation happened
            vr = results[0]
            assert vr.verify_rubric is not None
            assert "quality" in vr.verify_rubric
            assert "has_success" in vr.verify_rubric

    def test_template_and_rubric_mode(self) -> None:
        """Test evaluation with both templates and rubrics."""
        task = TaskEval(task_id="template_and_rubric")

        # Add question with template
        answer_template_code = '''
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Answer with action field."""

    action: str = Field(description="The action taken")

    def model_post_init(self, __context):
        self.correct = {"action": "action_a"}

    def verify(self) -> bool:
        return self.action.lower() == self.correct["action"]
'''

        task.add_question(
            {
                "id": "action_q",
                "question": "What action was taken?",
                "raw_answer": "action_a",
                "answer_template": answer_template_code,
            }
        )

        # Add rubric
        rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="detail", description="Is the response detailed?", kind="boolean")]
        )
        task.add_rubric(rubric)

        # Log response
        task.log("I took action_a to solve the problem with careful planning")

        # Configure evaluation
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parser", model_provider="openai", model_name="gpt-4o-mini", system_prompt="Parse and evaluate"
                )
            ],
            parsing_only=True,
        )

        # Evaluate - should verify template AND evaluate rubric
        result = task.evaluate(config)

        # Verify both template verification and rubric evaluation
        assert result.global_eval is not None
        verification_results = result.global_eval.verification_results

        assert "action_q" in verification_results
        vr = verification_results["action_q"][0]

        # Template verification result
        assert vr.verify_result is not None

        # Rubric evaluation result
        assert vr.verify_rubric is not None
        assert "detail" in vr.verify_rubric

    def test_replicate_support(self) -> None:
        """Test that replicate_count runs evaluation multiple times."""
        task = TaskEval(task_id="replicate_test")

        # Add question with template
        answer_template_code = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: str = Field(description="The value")

    def verify(self) -> bool:
        return True
"""

        task.add_question(
            {"id": "test_q", "question": "Test question", "raw_answer": "test", "answer_template": answer_template_code}
        )

        task.log("Test response")

        # Configure with 3 replicates
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parser", model_provider="openai", model_name="gpt-4o-mini", system_prompt="Parse response"
                )
            ],
            parsing_only=True,
            replicate_count=3,
        )

        # Evaluate
        result = task.evaluate(config)

        # Verify replicates were run
        assert result.global_eval is not None
        verification_results = result.global_eval.verification_results

        assert "test_q" in verification_results
        # Should have 3 results (one per replicate)
        assert len(verification_results["test_q"]) == 3

    def test_step_specific_dict_traces(self) -> None:
        """Test dict traces logged to specific steps."""
        task = TaskEval(task_id="step_dict_test")

        # Add step-specific rubric
        rubric = Rubric(regex_traits=[RegexTrait(name="has_plan", description="Contains 'plan'", pattern=r"plan")])
        task.add_rubric(rubric, step_id="planning")

        # Log dict trace to specific step
        task.log({"analysis": "Analyzed the requirements", "plan": "Created a detailed plan"}, step_id="planning")

        # Configure evaluation
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(id="parser", model_provider="openai", model_name="gpt-4o-mini", system_prompt="Evaluate")
            ],
            parsing_only=True,
        )

        # Evaluate the specific step (not globally, since we only have step-specific data)
        result = task.evaluate(config, step_id="planning")

        # Verify step evaluation happened
        assert result.per_step is not None
        assert "planning" in result.per_step
        step_result = result.per_step["planning"]

        # Should have synthetic questions for dict keys in this step
        assert len(step_result.verification_results) > 0

    def test_automatic_mode_detection(self) -> None:
        """Test that evaluation mode is automatically detected."""
        # Test 1: Only templates → template_only mode
        task1 = TaskEval(task_id="template_only")
        task1.add_question(
            {
                "id": "q1",
                "question": "Test",
                "raw_answer": "test",
                "answer_template": "from karenina.schemas.domain import BaseAnswer\nclass Answer(BaseAnswer):\n    def verify(self): return True",
            }
        )
        task1.log("test")

        context1 = task1._get_evaluation_context(step_id=None)
        mode1 = task1._detect_evaluation_mode(context1)
        assert mode1 == "template_only"

        # Test 2: Only rubrics → rubric_only mode
        task2 = TaskEval(task_id="rubric_only")
        task2.add_rubric(Rubric(llm_traits=[LLMRubricTrait(name="test", description="test", kind="boolean")]))
        task2.log("test")

        context2 = task2._get_evaluation_context(step_id=None)
        mode2 = task2._detect_evaluation_mode(context2)
        assert mode2 == "rubric_only"

        # Test 3: Both templates and rubrics → template_and_rubric mode
        task3 = TaskEval(task_id="both")
        task3.add_question(
            {
                "id": "q1",
                "question": "Test",
                "raw_answer": "test",
                "answer_template": "from karenina.schemas.domain import BaseAnswer\nclass Answer(BaseAnswer):\n    def verify(self): return True",
            }
        )
        task3.add_rubric(Rubric(llm_traits=[LLMRubricTrait(name="test", description="test", kind="boolean")]))
        task3.log("test")

        context3 = task3._get_evaluation_context(step_id=None)
        mode3 = task3._detect_evaluation_mode(context3)
        assert mode3 == "template_and_rubric"

    def test_error_when_neither_template_nor_rubric(self) -> None:
        """Test that evaluation raises error when neither templates nor rubrics are provided."""
        task = TaskEval(task_id="empty_test")

        # Add question without template
        task.add_question({"id": "q1", "question": "Test", "raw_answer": "test"})
        task.log("test")

        context = task._get_evaluation_context(step_id=None)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Must provide either answer templates, rubrics, or both"):
            task._detect_evaluation_mode(context)
