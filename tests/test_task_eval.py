"""Tests for TaskEval functionality."""

from karenina.benchmark.models import ModelConfig, VerificationConfig
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


class TestTaskEvalAgentOutputs:
    """Test TaskEval agent output logging and evaluation philosophy."""

    def test_log_agent_output_global(self):
        """Test logging agent output globally."""
        task = TaskEval()

        agent_response = "The answer is 42 because 20 + 22 = 42"
        task.log_agent_output(agent_output=agent_response, question_id="q1", output_type="answer")

        assert len(task.global_logs) == 1
        log = task.global_logs[0]
        assert log.text == agent_response
        assert log.question_id == "q1"
        assert log.is_agent_output is True
        assert log.output_type == "answer"
        assert log.level == "info"

    def test_log_agent_output_step(self):
        """Test logging agent output to specific step."""
        task = TaskEval()

        agent_response = "I understand the problem: calculate 15 + 23"
        task.log_agent_output(
            agent_output=agent_response, question_id="understanding_q1", step_id="understanding", target="step"
        )

        assert len(task.global_logs) == 0
        assert len(task.step_logs) == 1
        assert "understanding" in task.step_logs

        step_logs = task.step_logs["understanding"]
        assert len(step_logs) == 1
        assert step_logs[0].text == agent_response
        assert step_logs[0].is_agent_output is True
        assert step_logs[0].question_id == "understanding_q1"

    def test_log_agent_output_both_targets(self):
        """Test logging agent output to both global and step."""
        task = TaskEval()

        agent_response = "Final answer: 25"
        task.log_agent_output(agent_output=agent_response, question_id="final_q", step_id="execution", target="both")

        # Should appear in both global and step logs
        assert len(task.global_logs) == 1
        assert task.global_logs[0].text == agent_response
        assert task.global_logs[0].is_agent_output is True

        assert len(task.step_logs) == 1
        step_logs = task.step_logs["execution"]
        assert len(step_logs) == 1
        assert step_logs[0].text == agent_response
        assert step_logs[0].is_agent_output is True

    def test_find_agent_output_for_question(self):
        """Test finding logged agent output for a specific question."""
        task = TaskEval()

        # Log some regular logs and agent outputs
        task.log("Starting task", level="info")
        task.log_agent_output("Answer to Q1: 42", question_id="q1")
        task.log_agent_output("Answer to Q2: 24", question_id="q2")
        task.log("Task completed", level="info")

        # Test finding specific agent outputs (now returns lists)
        outputs_q1 = task._find_agent_outputs_for_question("q1", task.global_logs)
        outputs_q2 = task._find_agent_outputs_for_question("q2", task.global_logs)
        outputs_missing = task._find_agent_outputs_for_question("q3", task.global_logs)

        assert outputs_q1 == ["Answer to Q1: 42"]
        assert outputs_q2 == ["Answer to Q2: 24"]
        assert outputs_missing == []

    def test_evaluation_uses_logged_outputs(self):
        """Test that evaluation uses logged agent outputs, not new generation."""
        task = TaskEval(task_id="agent_run_001")

        # Add question
        question = {"id": "math_q1", "question": "What is 10 + 5?", "raw_answer": "15"}
        task.add_question(question)

        # Log agent output (this is the answer TaskEval should evaluate)
        agent_answer = "To solve 10 + 5, I add them: 10 + 5 = 15"
        task.log_agent_output(agent_answer, question_id="math_q1")

        # Create config with parsing models only
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="parser",
                    model_provider="openai",
                    model_name="gpt-4",
                    system_prompt="Parse and evaluate responses",
                )
            ],
            parsing_only=True,
        )

        # Evaluate (this should use the logged output, not generate new ones)
        result = task.evaluate(config)

        # Verify the evaluation found the logged output
        assert result.task_id == "agent_run_001"
        assert result.global_eval is not None

        # The evaluation should have processed the logged agent output
        verification = result.global_eval.question_verification
        assert verification is not None
        assert "math_q1" in verification  # Question ID should be in results

        # Check the first (and only) response for this question
        question_results = verification["math_q1"]
        assert len(question_results) == 1  # Should have one logged response
        response_result = question_results[0]
        assert response_result["success"] is True
        assert "agent_output" in response_result["details"]

    def test_evaluation_missing_agent_output(self):
        """Test evaluation when no agent output is logged for a question."""
        task = TaskEval()

        # Add question but don't log any agent output for it
        question = {"id": "missing_q", "question": "What is 5 + 5?", "raw_answer": "10"}
        task.add_question(question)

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(id="parser", model_provider="openai", model_name="gpt-4", system_prompt="Parse responses")
            ],
            parsing_only=True,
        )

        result = task.evaluate(config)

        # Should handle missing agent output gracefully
        assert result.global_eval is not None
        verification = result.global_eval.question_verification
        assert verification is not None
        assert "missing_q" in verification  # Question ID should be in results

        # Check the error response for missing output
        question_results = verification["missing_q"]
        assert len(question_results) == 1  # Should have one error result
        error_result = question_results[0]
        assert error_result["success"] is False
        assert "Missing agent output for question" in error_result["error"]


class TestTaskEvalIntegration:
    """Test TaskEval integration with verification pipeline."""

    def test_evaluate_global_basic(self):
        """Test basic global evaluation with logged agent output."""
        # Setup TaskEval
        task = TaskEval(task_id="test_task")
        task.add_question({"id": "q1", "question": "Test?", "raw_answer": "Test!"})
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

        # Log agent output (this is what TaskEval evaluates)
        task.log_agent_output("The answer is Test! which is correct and clear", question_id="q1")

        # Setup config
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(id="parsing", model_provider="openai", model_name="gpt-4", system_prompt="Parse prompt")
            ],
            parsing_only=True,
            rubric_enabled=True,
        )

        # Evaluate
        result = task.evaluate(config)

        # Verify results
        assert result.task_id == "test_task"
        assert result.global_eval is not None

        # The evaluation should have processed the logged output
        verification = result.global_eval.question_verification
        assert "q1" in verification  # Question ID should be in results

        # Check the response result
        question_results = verification["q1"]
        assert len(question_results) == 1  # Should have one logged response
        response_result = question_results[0]
        assert response_result["correct"] is True
        assert response_result["success"] is True

        # Rubric scores depend on the simplified evaluation logic
        assert len(result.global_eval.rubric_scores) == 2  # Both traits evaluated

    def test_evaluate_step_specific(self):
        """Test step-specific evaluation with logged agent output."""
        # Setup TaskEval with step-specific question and rubric
        task = TaskEval()
        task.add_question({"id": "q1", "question": "Test?", "raw_answer": "Test!"}, step_id="step1")
        task.add_rubric(
            Rubric(
                traits=[
                    RubricTrait(name="accuracy", description="Answer accuracy", kind="boolean"),
                    RubricTrait(name="clarity", description="Answer clarity", kind="score", min_score=1, max_score=5),
                ]
            ),
            step_id="step1",
        )

        # Log agent output for this step (incorrect answer to test failure modes)
        task.log_agent_output("Wrong answer here", question_id="q1", step_id="step1")

        # Setup config
        config = VerificationConfig(
            parsing_models=[
                ModelConfig(id="parsing", model_provider="openai", model_name="gpt-4", system_prompt="Parse prompt")
            ],
            parsing_only=True,
            rubric_enabled=True,
        )

        # Evaluate step
        result = task.evaluate(config, step_id="step1")

        # Verify results
        assert result.global_eval is None
        assert "step1" in result.per_step
        step_eval = result.per_step["step1"]

        # Check the step-specific verification results
        verification = step_eval.question_verification
        assert "q1" in verification  # Question ID should be in results

        # Check the response result
        question_results = verification["q1"]
        assert len(question_results) == 1  # Should have one logged response
        response_result = question_results[0]
        assert response_result["correct"] is False  # Wrong answer
        assert response_result["success"] is True  # Evaluation succeeded
        assert len(step_eval.rubric_scores) == 2  # Both traits evaluated
        assert len(step_eval.failure_modes) > 0  # Should have failure modes

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
        assert result.global_eval.question_verification == {}  # Empty dict, not None
        assert result.global_eval.failure_modes == []
