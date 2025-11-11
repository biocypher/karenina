"""Tests for TaskEval functionality."""

import pytest

from karenina.benchmark.task_eval import TaskEval
from karenina.schemas import ModelConfig, VerificationConfig
from karenina.schemas.domain import Question, Rubric, RubricTrait


class TestTaskEvalBasics:
    """Test basic TaskEval functionality."""

    def test_init_default(self) -> None:
        """Test TaskEval initialization with defaults."""
        task = TaskEval()

        assert task.task_id is None
        assert task.metadata == {}
        assert task.global_logs == []
        assert task.step_logs == {}
        assert task.global_questions == []
        assert task.step_questions == {}
        assert task.global_rubrics == []
        assert task.step_rubrics == {}

    def test_init_with_params(self) -> None:
        """Test TaskEval initialization with parameters."""
        metadata = {"agent": "test_agent", "run": 123}
        task = TaskEval(task_id="task_123", metadata=metadata)

        assert task.task_id == "task_123"
        assert task.metadata == metadata

    def test_log_global(self) -> None:
        """Test logging to global logs."""
        task = TaskEval()

        task.log("Test message", target="global", level="info")

        assert len(task.global_logs) == 1
        assert task.global_logs[0].text == "Test message"
        assert task.global_logs[0].level == "info"
        assert len(task.step_logs) == 0

    def test_log_step(self) -> None:
        """Test logging to step logs."""
        task = TaskEval()

        task.log("Step message", step_id="step1", target="step", level="warn")

        assert len(task.global_logs) == 0
        assert "step1" in task.step_logs
        assert len(task.step_logs["step1"]) == 1
        assert task.step_logs["step1"][0].text == "Step message"
        assert task.step_logs["step1"][0].level == "warn"

    def test_log_both(self) -> None:
        """Test logging to both global and step logs."""
        task = TaskEval()

        task.log("Both message", step_id="step1", target="both", level="error")

        assert len(task.global_logs) == 1
        assert task.global_logs[0].text == "Both message"
        assert "step1" in task.step_logs
        assert len(task.step_logs["step1"]) == 1
        assert task.step_logs["step1"][0].text == "Both message"

    def test_log_with_tags_and_payload(self) -> None:
        """Test logging with tags and payload."""
        task = TaskEval()
        tags = ["tag1", "tag2"]
        payload = {"key": "value", "number": 42}

        task.log("Message with extras", tags=tags, payload=payload)

        log_event = task.global_logs[0]
        assert log_event.tags == tags
        assert log_event.payload == payload


class TestTaskEvalQuestions:
    """Test question handling in TaskEval."""

    def test_add_question_dict_global(self) -> None:
        """Test adding question dict to global questions."""
        task = TaskEval()
        question_dict = {
            "id": "q1",
            "question": "What is 2+2?",
            "raw_answer": "4",
            "keywords": ["math"],
        }

        task.add_question(question_dict)

        assert len(task.global_questions) == 1
        assert task.global_questions[0] == question_dict

    def test_add_question_object_global(self) -> None:
        """Test adding Question object to global questions."""
        task = TaskEval()
        question_obj = Question(question="What is the capital of France?", raw_answer="Paris", tags=["geography"])

        task.add_question(question_obj)

        assert len(task.global_questions) == 1
        assert task.global_questions[0] == question_obj

    def test_add_question_step(self) -> None:
        """Test adding question to step-specific questions."""
        task = TaskEval()
        question_dict = {"id": "q1", "question": "Test question", "raw_answer": "Test answer"}

        task.add_question(question_dict, step_id="step1")

        assert len(task.global_questions) == 0
        assert "step1" in task.step_questions
        assert len(task.step_questions["step1"]) == 1
        assert task.step_questions["step1"][0] == question_dict

    def test_normalize_question_dict(self) -> None:
        """Test normalizing question dict (should pass through unchanged)."""
        task = TaskEval()
        question_dict = {
            "id": "q1",
            "question": "Test question",
            "raw_answer": "Test answer",
            "keywords": ["test"],
            "answer_template": "class Answer(BaseModel): pass",
        }

        normalized = task._normalize_question(question_dict)

        assert normalized == question_dict

    def test_normalize_question_object(self) -> None:
        """Test normalizing Question object to dict."""
        task = TaskEval()
        question_obj = Question(
            question="What is 2+2?",
            raw_answer="4",
            tags=["math", "basic"],
            few_shot_examples=[{"question": "1+1?", "answer": "2"}],
        )

        normalized = task._normalize_question(question_obj)

        expected = {
            "id": question_obj.id,  # Auto-generated MD5 hash
            "question": "What is 2+2?",
            "raw_answer": "4",
            "keywords": ["math", "basic"],
            "few_shot_examples": [{"question": "1+1?", "answer": "2"}],
            "answer_template": None,
        }
        assert normalized == expected


class TestTaskEvalRubrics:
    """Test rubric handling in TaskEval."""

    def test_add_rubric_global(self) -> None:
        """Test adding rubric to global rubrics."""
        task = TaskEval()
        rubric = Rubric(traits=[RubricTrait(name="clarity", description="Answer clarity", kind="score")])

        task.add_rubric(rubric)

        assert len(task.global_rubrics) == 1
        assert task.global_rubrics[0] == rubric

    def test_add_rubric_step(self) -> None:
        """Test adding rubric to step-specific rubrics."""
        task = TaskEval()
        rubric = Rubric(traits=[RubricTrait(name="accuracy", description="Answer accuracy", kind="boolean")])

        task.add_rubric(rubric, step_id="step1")

        assert len(task.global_rubrics) == 0
        assert "step1" in task.step_rubrics
        assert len(task.step_rubrics["step1"]) == 1
        assert task.step_rubrics["step1"][0] == rubric

    def test_merge_rubrics_empty(self) -> None:
        """Test merging empty rubrics list."""
        task = TaskEval()

        result = task._merge_rubrics([])

        assert result is None

    def test_merge_rubrics_single(self) -> None:
        """Test merging single rubric."""
        task = TaskEval()
        rubric = Rubric(traits=[RubricTrait(name="clarity", description="Answer clarity", kind="score")])

        result = task._merge_rubrics([rubric])

        assert result is not None
        assert len(result.traits) == 1
        assert result.traits[0].name == "clarity"

    def test_merge_rubrics_multiple(self) -> None:
        """Test merging multiple rubrics."""
        task = TaskEval()
        rubric1 = Rubric(traits=[RubricTrait(name="clarity", description="Answer clarity", kind="score")])
        rubric2 = Rubric(traits=[RubricTrait(name="accuracy", description="Answer accuracy", kind="boolean")])

        result = task._merge_rubrics([rubric1, rubric2])

        assert result is not None
        assert len(result.traits) == 2
        trait_names = [trait.name for trait in result.traits]
        assert "clarity" in trait_names
        assert "accuracy" in trait_names

    def test_merge_rubrics_duplicate_names(self) -> None:
        """Test merging rubrics with duplicate trait names raises an error."""
        task = TaskEval()
        rubric1 = Rubric(traits=[RubricTrait(name="clarity", description="First clarity", kind="score")])
        rubric2 = Rubric(traits=[RubricTrait(name="clarity", description="Second clarity", kind="boolean")])

        # Should raise ValueError due to duplicate trait names
        import pytest

        with pytest.raises(ValueError, match="Duplicate rubric trait names found"):
            task._merge_rubrics([rubric1, rubric2])


class TestTaskEvalSimplifiedLogging:
    """Test TaskEval simplified logging and evaluation with regular log() calls."""

    def test_evaluation_with_simple_logs(self) -> None:
        """Test that evaluation works with simple log() calls."""
        task = TaskEval(task_id="simple_eval_test")

        # Add question with answer template
        question = {
            "id": "math_q1",
            "question": "What is 10 + 5?",
            "raw_answer": "15",
            "answer_template": """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    result: int = Field(description="The result")

    def model_post_init(self, __context):
        self.correct = {"result": 15}

    def verify(self) -> bool:
        return self.result == 15
""",
        }
        task.add_question(question)

        # Use simple log() calls - this is the key improvement
        task.log("To solve 10 + 5, I add them: result=15")

        # Create config with parsing models only
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parser",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    system_prompt="Parse and evaluate responses",
                )
            ],
            parsing_only=True,
        )

        # Evaluate
        result = task.evaluate(config)

        # Verify evaluation works with simple logs
        assert result.task_id == "simple_eval_test"
        assert result.global_eval is not None

        # The evaluation should have processed the logged text using VerificationResult
        verification_results = result.global_eval.verification_results
        assert verification_results is not None
        assert "math_q1" in verification_results  # Question ID should be in results

        # Check the VerificationResult
        question_results = verification_results["math_q1"]
        assert len(question_results) == 1  # Should have one logged response
        vr = question_results[0]
        assert vr.completed_without_errors is True
        assert vr.raw_llm_response is not None

    def test_evaluation_missing_logs(self) -> None:
        """Test evaluation when no logs are available for a question."""
        task = TaskEval()

        # Add question with template but don't log anything
        question = {
            "id": "missing_q",
            "question": "What is 5 + 5?",
            "raw_answer": "10",
            "answer_template": """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    result: int = Field(description="The result")

    def model_post_init(self, __context):
        self.correct = {"result": 10}

    def verify(self) -> bool:
        return self.result == 10
""",
        }
        task.add_question(question)

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parser", model_provider="openai", model_name="gpt-4.1-mini", system_prompt="Parse responses"
                )
            ],
            parsing_only=True,
        )

        result = task.evaluate(config)

        # Should handle missing logs gracefully - with empty concatenated logs
        assert result.global_eval is not None
        verification_results = result.global_eval.verification_results
        # When there are no logs, concatenated_logs is empty string, but still gets evaluated
        # The verification will run but with empty input
        assert verification_results is not None


class TestTaskEvalIntegration:
    """Test TaskEval integration with verification pipeline."""

    def test_evaluate_global_basic(self) -> None:
        """Test basic global evaluation with logged agent output."""
        # Setup TaskEval
        task = TaskEval(task_id="test_task")

        # Add answer template for verification
        answer_template_code = '''
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Simple test answer."""
    result: str = Field(description="The answer")

    def model_post_init(self, __context):
        self.correct = {"result": "Test!"}

    def verify(self) -> bool:
        return self.result.strip() == self.correct["result"]
'''

        task.add_question(
            {"id": "q1", "question": "Test?", "raw_answer": "Test!", "answer_template": answer_template_code}
        )
        task.add_rubric(
            Rubric(
                traits=[
                    RubricTrait(name="accuracy", description="Is answer accurate", kind="boolean"),
                    RubricTrait(
                        name="clarity", description="Answer clarity 1-5", kind="score", min_score=1, max_score=5
                    ),
                ]
            )
        )

        # Log output (this is what TaskEval evaluates)
        # The logged output should contain the expected answer for verification to pass
        task.log("result: Test!")

        # Setup config
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parsing", model_provider="openai", model_name="gpt-4.1-mini", system_prompt="Parse prompt"
                )
            ],
            parsing_only=True,
            rubric_enabled=True,
            evaluation_mode="template_and_rubric",
        )

        # Evaluate
        result = task.evaluate(config)

        # Verify results
        assert result.task_id == "test_task"
        assert result.global_eval is not None

        # The evaluation should have processed the logged output
        verification_results = result.global_eval.verification_results
        assert "q1" in verification_results  # Question ID should be in results

        # Check the response result
        question_results = verification_results["q1"]
        assert len(question_results) == 1  # Should have one logged response
        vr = question_results[0]
        assert vr.verify_result is True
        assert vr.completed_without_errors is True

        # Rubric scores should be in verification result
        assert vr.verify_rubric is not None
        assert len(vr.verify_rubric) == 2  # Both traits evaluated

    def test_evaluate_step_specific(self) -> None:
        """Test step-specific evaluation with logged agent output."""
        # Setup TaskEval with step-specific question and rubric
        task = TaskEval()

        # Add answer template for verification
        answer_template_code = '''
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Simple test answer."""
    result: str = Field(description="The answer")

    def model_post_init(self, __context):
        self.correct = {"result": "Test!"}

    def verify(self) -> bool:
        return self.result.strip() == self.correct["result"]
'''

        task.add_question(
            {"id": "q1", "question": "Test?", "raw_answer": "Test!", "answer_template": answer_template_code},
            step_id="step1",
        )
        task.add_rubric(
            Rubric(
                traits=[
                    RubricTrait(name="accuracy", description="Answer accuracy", kind="boolean"),
                    RubricTrait(name="clarity", description="Answer clarity", kind="score", min_score=1, max_score=5),
                ]
            ),
            step_id="step1",
        )

        # Log output for this step (incorrect answer to test failure modes)
        task.log("Wrong answer here", step_id="step1")

        # Setup config
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parsing", model_provider="openai", model_name="gpt-4.1-mini", system_prompt="Parse prompt"
                )
            ],
            parsing_only=True,
            rubric_enabled=True,
            evaluation_mode="template_and_rubric",
        )

        # Evaluate step
        result = task.evaluate(config, step_id="step1")

        # Verify results
        assert result.global_eval is None
        assert "step1" in result.per_step
        step_eval = result.per_step["step1"]

        # Check the step-specific verification results
        verification_results = step_eval.verification_results
        assert "q1" in verification_results  # Question ID should be in results

        # Check the response result
        question_results = verification_results["q1"]
        assert len(question_results) == 1  # Should have one logged response
        vr = question_results[0]
        assert vr.verify_result is False  # Wrong answer
        assert vr.completed_without_errors is True  # Evaluation succeeded
        assert vr.verify_rubric is not None
        assert len(vr.verify_rubric) == 2  # Both traits evaluated

    def test_evaluate_no_questions(self) -> None:
        """Test evaluation with no questions and no rubrics raises ValueError."""
        task = TaskEval()
        config = VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="answering", model_provider="openai", model_name="gpt-4.1-mini", system_prompt="Test prompt"
                )
            ],
            parsing_models=[
                ModelConfig(
                    id="parsing", model_provider="openai", model_name="gpt-4.1-mini", system_prompt="Parse prompt"
                )
            ],
        )

        # With the new automatic mode detection, evaluating without questions or rubrics raises ValueError
        with pytest.raises(ValueError, match="Must provide either answer templates, rubrics, or both"):
            task.evaluate(config)

    def test_evaluate_global_triggers_step_evaluations(self) -> None:
        """Test that global evaluation automatically evaluates all available steps."""
        task = TaskEval()

        # Simple answer template for all questions
        answer_template_code = '''
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Simple test answer."""
    result: str = Field(description="The answer")

    def model_post_init(self, __context):
        self.correct = {"result": "answer"}

    def verify(self) -> bool:
        return "answer" in self.result.lower()
'''

        # Add global question and log
        task.add_question(
            {
                "id": "global_q",
                "question": "Global question?",
                "raw_answer": "Global answer",
                "answer_template": answer_template_code,
            }
        )
        task.log("Global response")

        # Add step-specific questions and logs
        task.add_question(
            {
                "id": "step1_q",
                "question": "Step 1 question?",
                "raw_answer": "Step 1 answer",
                "answer_template": answer_template_code,
            },
            step_id="step1",
        )
        task.log("Step 1 response", step_id="step1")

        task.add_question(
            {
                "id": "step2_q",
                "question": "Step 2 question?",
                "raw_answer": "Step 2 answer",
                "answer_template": answer_template_code,
            },
            step_id="step2",
        )
        task.log("Step 2 response", step_id="step2")

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parsing", model_provider="openai", model_name="gpt-4.1-mini", system_prompt="Parse prompt"
                )
            ],
            parsing_only=True,
        )

        # Evaluate globally (should trigger step evaluations automatically)
        result = task.evaluate(config)

        # Verify global evaluation was performed
        assert result.global_eval is not None
        assert "global_q" in result.global_eval.verification_results

        # Verify step evaluations were automatically triggered
        assert len(result.per_step) == 2
        assert "step1" in result.per_step
        assert "step2" in result.per_step

        # Verify step-specific evaluations contain correct questions
        assert "step1_q" in result.per_step["step1"].verification_results
        assert "step2_q" in result.per_step["step2"].verification_results

        # Verify step evaluations are isolated (step1 doesn't have step2's question)
        assert "step2_q" not in result.per_step["step1"].verification_results
        assert "step1_q" not in result.per_step["step2"].verification_results


class TestTaskEvalRubricEvaluation:
    """Test TaskEval proper rubric evaluation using RubricEvaluator."""

    def test_rubric_evaluation_with_answer_template(self) -> None:
        """Test that rubrics are properly evaluated using RubricEvaluator when answer template is available."""
        task = TaskEval(task_id="rubric_test")

        # Add question with answer template
        answer_template_code = '''
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Answer template for basic question."""

    result: str = Field(description="The answer")

    def model_post_init(self, __context):
        self.correct = {"result": "42"}

    def verify(self) -> bool:
        return self.result.strip() == self.correct["result"]
'''
        question = {
            "id": "test_q1",
            "question": "What is the meaning of life?",
            "raw_answer": "42",
            "answer_template": answer_template_code,
        }
        task.add_question(question)

        # Add rubric
        from karenina.schemas.domain import Rubric, RubricTrait

        rubric = Rubric(
            traits=[
                RubricTrait(name="accuracy", description="Is the answer accurate?", kind="boolean"),
                RubricTrait(name="clarity", description="Rate clarity 1-5", kind="score", min_score=1, max_score=5),
            ]
        )
        task.add_rubric(rubric)

        # Log a response
        task.log("The meaning of life is 42, as calculated by Deep Thought.")

        # Create config with openai interface for rubric evaluation tests
        # Skip if no API key available
        import os

        if not os.getenv("OPENAI_API_KEY"):
            import pytest

            pytest.skip("OpenAI API key not available for rubric evaluation test")

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test_parser",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    temperature=0.0,
                    system_prompt="Parse responses and evaluate rubrics",
                )
            ],
            parsing_only=True,
        )

        # Evaluate
        result = task.evaluate(config)

        # Verify evaluation succeeded
        assert result.task_id == "rubric_test"
        assert result.global_eval is not None

        # Check that rubric scores are present and properly evaluated
        verification_results = result.global_eval.verification_results
        assert "test_q1" in verification_results

        question_results = verification_results["test_q1"]
        assert len(question_results) == 1

        vr = question_results[0]
        assert vr.completed_without_errors is True

        # Check that rubric evaluation happened - verify_rubric should not be empty
        assert vr.verify_rubric is not None
        assert isinstance(vr.verify_rubric, dict)

    def test_evaluation_without_template(self) -> None:
        """Test rubric-only evaluation: questions without templates are evaluated with minimal template."""
        task = TaskEval(task_id="no_template_test")

        # Add question WITHOUT answer template (will use rubric_only mode)
        question = {"id": "simple_q", "question": "What is 2+2?", "raw_answer": "4"}
        task.add_question(question)

        # Add rubric
        from karenina.schemas.domain import Rubric, RubricTrait

        rubric = Rubric(
            traits=[
                RubricTrait(name="correctness", description="Is the math correct?", kind="boolean"),
            ]
        )
        task.add_rubric(rubric)

        # Log a response
        task.log("2 + 2 = 4")

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test_parser",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    temperature=0.0,
                    system_prompt="Parse responses and evaluate rubrics",
                )
            ],
            parsing_only=True,
        )

        # Evaluate
        result = task.evaluate(config)

        # Verify evaluation succeeded with rubric_only mode
        assert result.task_id == "no_template_test"
        assert result.global_eval is not None

        # With rubric_only mode, question WITHOUT template is now evaluated using minimal template
        verification_results = result.global_eval.verification_results
        assert "simple_q" in verification_results  # Question IS evaluated in rubric_only mode

        # Verify rubric evaluation happened
        vr = verification_results["simple_q"][0]
        assert vr.verify_rubric is not None
        assert "correctness" in vr.verify_rubric

    def test_rubric_conflict_detection(self) -> None:
        """Test that duplicate rubric trait names raise an error."""
        task = TaskEval(task_id="conflict_test")

        # Add two rubrics with the same trait name
        from karenina.schemas.domain import Rubric, RubricTrait

        rubric1 = Rubric(
            traits=[
                RubricTrait(name="accuracy", description="Is the answer accurate?", kind="boolean"),
            ]
        )
        rubric2 = Rubric(
            traits=[
                RubricTrait(
                    name="accuracy", description="Different accuracy definition", kind="score", min_score=1, max_score=5
                ),
            ]
        )

        # Add both rubrics
        task.add_rubric(rubric1)
        task.add_rubric(rubric2)

        # Add a question and log
        question = {"id": "conflict_q", "question": "Test question", "raw_answer": "Test"}
        task.add_question(question)
        task.log("Test response")

        # Create config
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test_parser",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    system_prompt="Parse responses",
                )
            ],
            parsing_only=True,
        )

        # Evaluation should raise an error due to conflicting trait names
        import pytest

        with pytest.raises(ValueError, match="Duplicate rubric trait names found"):
            task.evaluate(config)

    def test_rubric_trait_extraction_from_template(self) -> None:
        """Test that standalone rubrics work with regular answer templates."""
        task = TaskEval(task_id="template_extraction_test")

        # Add standalone rubric traits
        from karenina.schemas.domain import Rubric, RubricTrait

        rubric = Rubric(
            traits=[
                RubricTrait(name="completeness", description="Is the answer complete?", kind="boolean"),
                RubricTrait(name="quality", description="Quality score", kind="score", min_score=1, max_score=5),
            ]
        )
        task.add_rubric(rubric)

        # Create a simple answer template
        answer_template_code = '''
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Simple answer template."""

    result: str = Field(description="The answer result")

    def model_post_init(self, __context):
        self.correct = {"result": "expected"}

    def verify(self) -> bool:
        return self.result.strip() == self.correct["result"]
'''

        # Add a question with the template
        question = {
            "id": "extract_test",
            "question": "Test question with rubric",
            "raw_answer": "expected",
            "answer_template": answer_template_code,
        }
        task.add_question(question)
        task.log("result: expected")

        # Create config
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test_parser",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    system_prompt="Parse responses",
                )
            ],
            parsing_only=True,
        )

        # Evaluation should work without errors
        result = task.evaluate(config)

        # Check that the evaluation completed successfully
        assert result.global_eval is not None
        verification_results = result.global_eval.verification_results
        assert "extract_test" in verification_results

        # The rubric traits should have been evaluated
        vr = verification_results["extract_test"][0]
        assert vr.verify_rubric is not None
        assert isinstance(vr.verify_rubric, dict)
        assert len(vr.verify_rubric) == 2  # Both traits

    def test_multiple_questions_with_same_rubric(self) -> None:
        """Test that the same standalone rubric can be used with multiple questions."""
        task = TaskEval(task_id="multi_question_test")

        # Add a standalone rubric
        from karenina.schemas.domain import Rubric, RubricTrait

        rubric = Rubric(
            traits=[
                RubricTrait(name="completeness", description="Completeness check", kind="boolean"),
            ]
        )
        task.add_rubric(rubric)

        # Create a simple answer template
        answer_template_code = '''
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Simple answer."""

    result: str = Field(description="The answer result")

    def model_post_init(self, __context):
        self.correct = {"result": "test"}

    def verify(self) -> bool:
        return self.result.strip() == self.correct["result"]
'''

        # Add two questions with the same rubric
        question1 = {
            "id": "q1",
            "question": "Question 1",
            "raw_answer": "test",
            "answer_template": answer_template_code,
        }
        question2 = {
            "id": "q2",
            "question": "Question 2",
            "raw_answer": "test",
            "answer_template": answer_template_code,
        }
        task.add_question(question1)
        task.add_question(question2)
        task.log("result: test")

        # Create config
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test_parser",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    system_prompt="Parse responses",
                )
            ],
            parsing_only=True,
        )

        # Evaluation should work - same rubric can be used for multiple questions
        result = task.evaluate(config)

        # Check that both questions were evaluated with the rubric
        assert result.global_eval is not None
        verification_results = result.global_eval.verification_results
        assert "q1" in verification_results
        assert "q2" in verification_results

        # Both should have rubric scores
        assert verification_results["q1"][0].verify_rubric is not None
        assert verification_results["q2"][0].verify_rubric is not None
