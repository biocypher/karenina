"""Tests for TaskEval functionality."""

from unittest.mock import patch

from karenina.benchmark.models import ModelConfig, VerificationConfig, VerificationResult
from karenina.benchmark.task_eval import TaskEval
from karenina.schemas.question_class import Question
from karenina.schemas.rubric_class import Rubric, RubricTrait


class TestTaskEvalBasics:
    """Test basic TaskEval functionality."""

    def test_init_default(self):
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

    def test_init_with_params(self):
        """Test TaskEval initialization with parameters."""
        metadata = {"agent": "test_agent", "run": 123}
        task = TaskEval(task_id="task_123", metadata=metadata)

        assert task.task_id == "task_123"
        assert task.metadata == metadata

    def test_log_global(self):
        """Test logging to global logs."""
        task = TaskEval()

        task.log("Test message", target="global", level="info")

        assert len(task.global_logs) == 1
        assert task.global_logs[0].text == "Test message"
        assert task.global_logs[0].level == "info"
        assert len(task.step_logs) == 0

    def test_log_step(self):
        """Test logging to step logs."""
        task = TaskEval()

        task.log("Step message", step_id="step1", target="step", level="warn")

        assert len(task.global_logs) == 0
        assert "step1" in task.step_logs
        assert len(task.step_logs["step1"]) == 1
        assert task.step_logs["step1"][0].text == "Step message"
        assert task.step_logs["step1"][0].level == "warn"

    def test_log_both(self):
        """Test logging to both global and step logs."""
        task = TaskEval()

        task.log("Both message", step_id="step1", target="both", level="error")

        assert len(task.global_logs) == 1
        assert task.global_logs[0].text == "Both message"
        assert "step1" in task.step_logs
        assert len(task.step_logs["step1"]) == 1
        assert task.step_logs["step1"][0].text == "Both message"

    def test_log_with_tags_and_payload(self):
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

    def test_add_question_dict_global(self):
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

    def test_add_question_object_global(self):
        """Test adding Question object to global questions."""
        task = TaskEval()
        question_obj = Question(question="What is the capital of France?", raw_answer="Paris", tags=["geography"])

        task.add_question(question_obj)

        assert len(task.global_questions) == 1
        assert task.global_questions[0] == question_obj

    def test_add_question_step(self):
        """Test adding question to step-specific questions."""
        task = TaskEval()
        question_dict = {"id": "q1", "question": "Test question", "raw_answer": "Test answer"}

        task.add_question(question_dict, step_id="step1")

        assert len(task.global_questions) == 0
        assert "step1" in task.step_questions
        assert len(task.step_questions["step1"]) == 1
        assert task.step_questions["step1"][0] == question_dict

    def test_normalize_question_dict(self):
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

    def test_normalize_question_object(self):
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

    def test_add_rubric_global(self):
        """Test adding rubric to global rubrics."""
        task = TaskEval()
        rubric = Rubric(traits=[RubricTrait(name="clarity", description="Answer clarity", kind="score")])

        task.add_rubric(rubric)

        assert len(task.global_rubrics) == 1
        assert task.global_rubrics[0] == rubric

    def test_add_rubric_step(self):
        """Test adding rubric to step-specific rubrics."""
        task = TaskEval()
        rubric = Rubric(traits=[RubricTrait(name="accuracy", description="Answer accuracy", kind="boolean")])

        task.add_rubric(rubric, step_id="step1")

        assert len(task.global_rubrics) == 0
        assert "step1" in task.step_rubrics
        assert len(task.step_rubrics["step1"]) == 1
        assert task.step_rubrics["step1"][0] == rubric

    def test_merge_rubrics_empty(self):
        """Test merging empty rubrics list."""
        task = TaskEval()

        result = task._merge_rubrics([])

        assert result is None

    def test_merge_rubrics_single(self):
        """Test merging single rubric."""
        task = TaskEval()
        rubric = Rubric(traits=[RubricTrait(name="clarity", description="Answer clarity", kind="score")])

        result = task._merge_rubrics([rubric])

        assert result is not None
        assert len(result.traits) == 1
        assert result.traits[0].name == "clarity"

    def test_merge_rubrics_multiple(self):
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

    def test_merge_rubrics_duplicate_names(self):
        """Test merging rubrics with duplicate trait names (later overrides earlier)."""
        task = TaskEval()
        rubric1 = Rubric(traits=[RubricTrait(name="clarity", description="First clarity", kind="score")])
        rubric2 = Rubric(traits=[RubricTrait(name="clarity", description="Second clarity", kind="boolean")])

        result = task._merge_rubrics([rubric1, rubric2])

        assert result is not None
        assert len(result.traits) == 1
        assert result.traits[0].name == "clarity"
        assert result.traits[0].description == "Second clarity"
        assert result.traits[0].kind == "boolean"


class TestTaskEvalFailureModes:
    """Test failure mode extraction."""

    def test_extract_failure_modes_empty(self):
        """Test extracting failure modes from empty scores."""
        task = TaskEval()

        failure_modes = task._extract_failure_modes({})

        assert failure_modes == []

    def test_extract_failure_modes_boolean_pass(self):
        """Test extracting failure modes from passing boolean traits."""
        task = TaskEval()
        scores = {
            "accuracy": True,
            "completeness": True,
        }

        failure_modes = task._extract_failure_modes(scores)

        assert failure_modes == []

    def test_extract_failure_modes_boolean_fail(self):
        """Test extracting failure modes from failing boolean traits."""
        task = TaskEval()
        scores = {
            "accuracy": False,
            "completeness": True,
            "relevance": False,
        }

        failure_modes = task._extract_failure_modes(scores)

        assert len(failure_modes) == 2
        assert "Failed trait: accuracy" in failure_modes
        assert "Failed trait: relevance" in failure_modes

    def test_extract_failure_modes_score_pass(self):
        """Test extracting failure modes from passing score traits."""
        task = TaskEval()
        scores = {
            "clarity": 4,
            "depth": 5,
            "structure": 3,
        }

        failure_modes = task._extract_failure_modes(scores)

        assert failure_modes == []

    def test_extract_failure_modes_score_fail(self):
        """Test extracting failure modes from failing score traits."""
        task = TaskEval()
        scores = {
            "clarity": 2,
            "depth": 5,
            "structure": 1,
        }

        failure_modes = task._extract_failure_modes(scores)

        assert len(failure_modes) == 2
        assert "Low score trait: clarity (score: 2)" in failure_modes
        assert "Low score trait: structure (score: 1)" in failure_modes

    def test_extract_failure_modes_mixed(self):
        """Test extracting failure modes from mixed trait types."""
        task = TaskEval()
        scores = {
            "accuracy": False,
            "clarity": 2,
            "completeness": True,
            "depth": 4,
        }

        failure_modes = task._extract_failure_modes(scores)

        assert len(failure_modes) == 2
        assert "Failed trait: accuracy" in failure_modes
        assert "Low score trait: clarity (score: 2)" in failure_modes


class TestTaskEvalTemplateGeneration:
    """Test template generation functionality."""

    def test_generate_minimal_template(self):
        """Test generating a minimal template."""
        task = TaskEval()
        question_dict = {"raw_answer": "Paris", "question": "What is the capital of France?"}

        template = task._generate_minimal_template(question_dict)

        assert "class Answer(BaseModel):" in template
        assert "correct: str" in template
        assert '"Paris"' in template

    def test_prepare_for_verification_with_template(self):
        """Test preparing question with existing template."""
        task = TaskEval()
        question_dict = {"id": "q1", "question": "Test question", "answer_template": "class Answer(BaseModel): pass"}

        q_id, q_text, template = task._prepare_for_verification(question_dict)

        assert q_id == "q1"
        assert q_text == "Test question"
        assert template == "class Answer(BaseModel): pass"

    def test_prepare_for_verification_without_template(self):
        """Test preparing question without template (generates minimal)."""
        task = TaskEval()
        question_dict = {"id": "q1", "question": "Test question", "raw_answer": "Test answer"}

        q_id, q_text, template = task._prepare_for_verification(question_dict)

        assert q_id == "q1"
        assert q_text == "Test question"
        assert "class Answer(BaseModel):" in template
        assert '"Test answer"' in template


class TestTaskEvalIntegration:
    """Test TaskEval integration with verification pipeline."""

    @patch("karenina.benchmark.task_eval.task_eval.run_single_model_verification")
    def test_evaluate_global_basic(self, mock_verify):
        """Test basic global evaluation."""
        # Setup mock
        mock_result = VerificationResult(
            question_id="q1",
            success=True,
            question_text="Test question",
            raw_llm_response="Test response",
            answering_model="test-model",
            parsing_model="test-parser",
            execution_time=1.0,
            timestamp="2023-01-01T00:00:00",
            verify_result=True,
            verify_granular_result={"details": "test"},
            verify_rubric={"accuracy": True, "clarity": 4},
        )
        mock_verify.return_value = mock_result

        # Setup TaskEval
        task = TaskEval(task_id="test_task")
        task.add_question({"id": "q1", "question": "Test?", "raw_answer": "Test!"})
        task.add_rubric(
            Rubric(traits=[RubricTrait(name="accuracy", kind="boolean"), RubricTrait(name="clarity", kind="score")])
        )

        # Setup config
        config = VerificationConfig(
            answering_models=[
                ModelConfig(id="answering", model_provider="openai", model_name="gpt-4", system_prompt="Test prompt")
            ],
            parsing_models=[
                ModelConfig(id="parsing", model_provider="openai", model_name="gpt-4", system_prompt="Parse prompt")
            ],
            rubric_enabled=True,
        )

        # Evaluate
        result = task.evaluate(config)

        # Verify results
        assert result.task_id == "test_task"
        assert result.global_eval is not None
        assert result.global_eval.rubric_scores == {"accuracy": True, "clarity": 4}
        assert result.global_eval.question_verification["correct"] is True
        assert len(result.global_eval.failure_modes) == 0  # No failures

        # Verify verification was called correctly
        mock_verify.assert_called_once()
        call_args = mock_verify.call_args
        assert call_args[1]["question_id"] == "q1"
        assert call_args[1]["question_text"] == "Test?"

    @patch("karenina.benchmark.task_eval.task_eval.run_single_model_verification")
    def test_evaluate_step_specific(self, mock_verify):
        """Test step-specific evaluation."""
        # Setup mock
        mock_result = VerificationResult(
            question_id="q1",
            success=True,
            question_text="Test question",
            raw_llm_response="Test response",
            answering_model="test-model",
            parsing_model="test-parser",
            execution_time=1.0,
            timestamp="2023-01-01T00:00:00",
            verify_result=False,
            verify_rubric={"accuracy": False, "clarity": 2},
        )
        mock_verify.return_value = mock_result

        # Setup TaskEval
        task = TaskEval()
        task.add_question({"id": "q1", "question": "Test?", "raw_answer": "Test!"}, step_id="step1")
        task.add_rubric(
            Rubric(traits=[RubricTrait(name="accuracy", kind="boolean"), RubricTrait(name="clarity", kind="score")]),
            step_id="step1",
        )

        # Setup config
        config = VerificationConfig(
            answering_models=[
                ModelConfig(id="answering", model_provider="openai", model_name="gpt-4", system_prompt="Test prompt")
            ],
            parsing_models=[
                ModelConfig(id="parsing", model_provider="openai", model_name="gpt-4", system_prompt="Parse prompt")
            ],
            rubric_enabled=True,
        )

        # Evaluate step
        result = task.evaluate(config, step_id="step1")

        # Verify results
        assert result.global_eval is None
        assert "step1" in result.per_step
        step_eval = result.per_step["step1"]
        assert step_eval.rubric_scores == {"accuracy": False, "clarity": 2}
        assert step_eval.question_verification["correct"] is False
        assert len(step_eval.failure_modes) == 2  # Both traits failed

    def test_evaluate_no_questions(self):
        """Test evaluation with no questions."""
        task = TaskEval()
        config = VerificationConfig(
            answering_models=[
                ModelConfig(id="answering", model_provider="openai", model_name="gpt-4", system_prompt="Test prompt")
            ],
            parsing_models=[
                ModelConfig(id="parsing", model_provider="openai", model_name="gpt-4", system_prompt="Parse prompt")
            ],
        )

        result = task.evaluate(config)

        assert result.global_eval is not None
        assert result.global_eval.rubric_scores == {}
        assert result.global_eval.question_verification is None
        assert result.global_eval.failure_modes == []
