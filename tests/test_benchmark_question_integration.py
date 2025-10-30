"""Tests for Benchmark Question object integration."""

import tempfile
from pathlib import Path

import pytest

from karenina.benchmark import Benchmark
from karenina.schemas.domain import Question


class TestBenchmarkQuestionObjectMethods:
    """Test the new Question object methods in Benchmark."""

    @pytest.fixture
    def sample_benchmark(self) -> None:
        """Create a sample benchmark for testing."""
        benchmark = Benchmark.create(
            name="Test Benchmark", description="Test benchmark for Question object methods", creator="Test Suite"
        )

        # Add some test questions
        benchmark.add_question(question="What is 2+2?", raw_answer="4")
        # Set keywords separately
        q1_id = benchmark.get_question_ids()[-1]
        benchmark.update_question_metadata(q1_id, keywords=["math", "addition"])

        benchmark.add_question(
            question="What is the capital of France?",
            raw_answer="Paris",
            few_shot_examples=[{"question": "What is the capital of Italy?", "answer": "Rome"}],
        )
        # Set keywords separately
        q2_id = benchmark.get_question_ids()[-1]
        benchmark.update_question_metadata(q2_id, keywords=["geography", "capitals"])

        return benchmark

    def test_get_question_as_object(self, sample_benchmark) -> None:
        """Test getting a question as a Question object."""
        # Get question IDs
        question_ids = sample_benchmark.get_question_ids()
        assert len(question_ids) == 2

        # Get first question as object
        question_obj = sample_benchmark.get_question_as_object(question_ids[0])

        assert isinstance(question_obj, Question)
        assert question_obj.question in ["What is 2+2?", "What is the capital of France?"]
        assert question_obj.raw_answer in ["4", "Paris"]
        assert isinstance(question_obj.tags, list)

    def test_get_question_as_object_not_found(self, sample_benchmark) -> None:
        """Test getting non-existent question as object."""
        with pytest.raises(ValueError, match="Question not found"):
            sample_benchmark.get_question_as_object("nonexistent_id")

    def test_get_all_questions_as_objects(self, sample_benchmark) -> None:
        """Test getting all questions as Question objects."""
        question_objects = sample_benchmark.get_all_questions_as_objects()

        assert len(question_objects) == 2
        assert all(isinstance(q, Question) for q in question_objects)

        # Check that we have both questions
        question_texts = [q.question for q in question_objects]
        assert "What is 2+2?" in question_texts
        assert "What is the capital of France?" in question_texts

        # Check that tags are properly converted
        for q in question_objects:
            assert isinstance(q.tags, list)

    def test_add_question_from_object(self, sample_benchmark) -> None:
        """Test adding a question from a Question object."""
        # Create a new Question object
        new_question = Question(
            question="What is the largest planet?",
            raw_answer="Jupiter",
            tags=["astronomy", "planets"],
            few_shot_examples=[{"question": "What is the smallest planet?", "answer": "Mercury"}],
        )

        # Add it to the benchmark
        question_id = sample_benchmark.add_question_from_object(new_question)

        # Verify it was added
        assert question_id == new_question.id  # Should return the auto-generated ID
        assert question_id in sample_benchmark.get_question_ids()

        # Verify the content
        retrieved_dict = sample_benchmark.get_question(question_id)
        assert retrieved_dict["question"] == "What is the largest planet?"
        assert retrieved_dict["raw_answer"] == "Jupiter"
        assert retrieved_dict["keywords"] == ["astronomy", "planets"]
        assert retrieved_dict["few_shot_examples"] == [
            {"question": "What is the smallest planet?", "answer": "Mercury"}
        ]

    def test_add_question_from_object_with_metadata(self, sample_benchmark) -> None:
        """Test adding a question from object with additional metadata."""
        new_question = Question(
            question="What is photosynthesis?",
            raw_answer="The process by which plants make food from sunlight",
            tags=["biology", "plants"],
        )

        # Add with metadata
        question_id = sample_benchmark.add_question_from_object(
            new_question,
            author={"name": "Test Author", "email": "test@example.com"},
            sources=["Biology Textbook"],
            custom_metadata={"difficulty": "medium"},
        )

        # Verify metadata was added
        metadata = sample_benchmark.get_question_metadata(question_id)
        assert metadata["author"]["name"] == "Test Author"
        assert metadata["sources"] == ["Biology Textbook"]

        # Check custom property
        difficulty = sample_benchmark.get_question_custom_property(question_id, "difficulty")
        assert difficulty == "medium"

    def test_add_question_from_object_duplicate_id(self, sample_benchmark) -> None:
        """Test adding a question object that already exists."""
        # Create a Question object and add it
        from karenina.schemas.domain import Question

        question_obj = Question(question="Duplicate test question", raw_answer="Duplicate test answer")

        # Add it once
        sample_benchmark.add_question_from_object(question_obj)

        # Try to add the same Question object again (same ID)
        with pytest.raises(ValueError, match="Question with ID .* already exists"):
            sample_benchmark.add_question_from_object(question_obj)

    def test_add_question_from_object_invalid_type(self, sample_benchmark) -> None:
        """Test adding invalid object type."""
        with pytest.raises(ValueError, match="question_obj must be a Question instance"):
            sample_benchmark.add_question_from_object({"not": "a_question_object"})

    def test_round_trip_conversion(self, sample_benchmark) -> None:
        """Test converting dict -> object -> dict maintains consistency."""
        question_ids = sample_benchmark.get_question_ids()
        original_dict = sample_benchmark.get_question(question_ids[0])

        # Convert to object and back
        question_obj = sample_benchmark.get_question_as_object(question_ids[0])

        # Check that the object has the expected fields
        assert question_obj.question == original_dict["question"]
        assert question_obj.raw_answer == original_dict["raw_answer"]
        assert question_obj.tags == (original_dict.get("keywords") or [])
        assert question_obj.few_shot_examples == original_dict.get("few_shot_examples")

    def test_question_object_id_consistency(self) -> None:
        """Test that Question object IDs are consistent with benchmark storage."""
        # Create a Question object
        question = Question(question="Test question for ID consistency", raw_answer="Test answer")

        # Create benchmark and add the question
        benchmark = Benchmark.create("Test", "Test", "Test")
        added_id = benchmark.add_question_from_object(question)

        # The added ID should match the Question object's auto-generated ID
        assert added_id == question.id

        # Retrieving as object should have the same ID
        retrieved_obj = benchmark.get_question_as_object(added_id)
        assert retrieved_obj.id == question.id

    def test_empty_tags_handling(self, sample_benchmark) -> None:
        """Test handling of empty/None tags."""
        question = Question(
            question="Question with no tags",
            raw_answer="Answer",
            # tags defaults to []
        )

        question_id = sample_benchmark.add_question_from_object(question)
        retrieved_obj = sample_benchmark.get_question_as_object(question_id)

        # Should be converted to empty list
        assert retrieved_obj.tags == []

    def test_persistence_round_trip(self) -> None:
        """Test that Question objects survive save/load cycles."""
        # Create benchmark with Question object
        benchmark = Benchmark.create("Test", "Test", "Test")
        original_question = Question(
            question="Persistence test question",
            raw_answer="Persistence test answer",
            tags=["test", "persistence"],
            few_shot_examples=[{"question": "Example", "answer": "Example answer"}],
        )

        question_id = benchmark.add_question_from_object(original_question)

        # Save and reload
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonld", delete=False) as f:
            temp_path = Path(f.name)

        try:
            benchmark.save(temp_path)
            loaded_benchmark = Benchmark.load(temp_path)

            # Retrieve as object from loaded benchmark
            loaded_question = loaded_benchmark.get_question_as_object(question_id)

            # Should match original
            assert loaded_question.question == original_question.question
            assert loaded_question.raw_answer == original_question.raw_answer
            assert loaded_question.tags == original_question.tags
            assert loaded_question.few_shot_examples == original_question.few_shot_examples
            assert loaded_question.id == original_question.id

        finally:
            temp_path.unlink(missing_ok=True)


class TestTaskEvalBenchmarkIntegration:
    """Test TaskEval integration with Benchmark Question objects."""

    @pytest.fixture
    def sample_benchmark(self) -> None:
        """Create a sample benchmark for TaskEval testing."""
        benchmark = Benchmark.create("TaskEval Test", "Test", "Test")

        # Add questions with templates
        q1_id = benchmark.add_question(question="What is 5+3?", raw_answer="8")

        # Set a template for the question
        template_code = '''from pydantic import Field
from karenina.schemas.domain import BaseAnswer

class Answer(BaseAnswer):
    """Answer for math question."""
    result: int = Field(description="The numerical result")
    correct: int = Field(description="The correct answer", default=8)

    def verify(self) -> bool:
        return self.result == self.correct
'''
        benchmark.add_answer_template(q1_id, template_code)

        return benchmark

    def test_task_eval_with_benchmark_dict(self, sample_benchmark) -> None:
        """Test TaskEval using question dict from Benchmark."""
        from karenina.benchmark.task_eval import TaskEval

        task = TaskEval(task_id="benchmark_integration_test")

        # Get question as dict and add to TaskEval
        question_ids = sample_benchmark.get_question_ids()
        question_dict = sample_benchmark.get_question(question_ids[0])
        task.add_question(question_dict)

        # Should be stored correctly
        assert len(task.global_questions) == 1
        assert task.global_questions[0] == question_dict

    def test_task_eval_with_benchmark_object(self, sample_benchmark) -> None:
        """Test TaskEval using Question object from Benchmark."""
        from karenina.benchmark.task_eval import TaskEval

        task = TaskEval(task_id="benchmark_integration_test")

        # Get question as object and add to TaskEval
        question_ids = sample_benchmark.get_question_ids()
        question_obj = sample_benchmark.get_question_as_object(question_ids[0])
        task.add_question(question_obj)

        # Should be stored correctly
        assert len(task.global_questions) == 1
        assert task.global_questions[0] == question_obj

    def test_task_eval_with_benchmark_all_objects(self, sample_benchmark) -> None:
        """Test TaskEval using all Question objects from Benchmark."""
        from karenina.benchmark.task_eval import TaskEval

        task = TaskEval(task_id="benchmark_integration_test")

        # Get all questions as objects and add to TaskEval
        all_questions = sample_benchmark.get_all_questions_as_objects()
        for question_obj in all_questions:
            task.add_question(question_obj)

        # Should have all questions
        assert len(task.global_questions) == len(all_questions)
        assert all(isinstance(q, Question) for q in task.global_questions)

    def test_task_eval_normalization_from_benchmark(self, sample_benchmark) -> None:
        """Test that TaskEval correctly normalizes questions from Benchmark."""
        from karenina.benchmark.task_eval import TaskEval

        task = TaskEval()
        question_ids = sample_benchmark.get_question_ids()

        # Test dict normalization (should pass through)
        question_dict = sample_benchmark.get_question(question_ids[0])
        normalized_dict = task._normalize_question(question_dict)
        assert normalized_dict == question_dict

        # Test object normalization (should convert to dict)
        question_obj = sample_benchmark.get_question_as_object(question_ids[0])
        normalized_obj = task._normalize_question(question_obj)

        expected_fields = ["id", "question", "raw_answer", "keywords", "few_shot_examples", "answer_template"]
        assert all(field in normalized_obj for field in expected_fields)
        assert normalized_obj["id"] == question_obj.id
        assert normalized_obj["question"] == question_obj.question
        assert normalized_obj["raw_answer"] == question_obj.raw_answer
        assert normalized_obj["keywords"] == question_obj.tags
