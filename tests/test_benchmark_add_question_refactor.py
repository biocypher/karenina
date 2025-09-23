"""Backward compatibility tests for add_question refactoring.

These tests capture the current behavior of add_question to ensure
the refactoring maintains 100% backward compatibility.
"""

import tempfile
from pathlib import Path

from karenina.benchmark import Benchmark
from karenina.schemas.question_class import Question


class TestAddQuestionBackwardCompatibility:
    """Test all current add_question usage patterns to ensure backward compatibility."""

    def test_basic_kwargs_usage(self):
        """Test basic add_question with minimal kwargs."""
        benchmark = Benchmark("Test Benchmark")

        # Current basic usage
        q_id = benchmark.add_question(question="What is Python?", raw_answer="Python is a programming language")

        # Verify question was added
        assert q_id in benchmark
        question_data = benchmark.get_question(q_id)
        assert question_data["question"] == "What is Python?"
        assert question_data["raw_answer"] == "Python is a programming language"
        assert question_data["id"] == q_id
        assert question_data["finished"] is False
        assert question_data.get("answer_template") is not None  # Default template created

    def test_kwargs_with_template(self):
        """Test add_question with answer template provided."""
        benchmark = Benchmark("Test Benchmark")

        template_code = """class Answer(BaseAnswer):
    response: str = Field(description="The answer")

    def verify(self) -> bool:
        return self.response.lower() == "python"
"""

        q_id = benchmark.add_question(question="What language?", raw_answer="Python", answer_template=template_code)

        question_data = benchmark.get_question(q_id)
        assert question_data["answer_template"] == template_code
        assert question_data["finished"] is False  # Not finished by default

    def test_kwargs_with_custom_id(self):
        """Test add_question with custom question ID."""
        benchmark = Benchmark("Test Benchmark")

        q_id = benchmark.add_question(question="Custom ID test", raw_answer="Test answer", question_id="custom_id_123")

        assert q_id == "custom_id_123"
        assert "custom_id_123" in benchmark
        question_data = benchmark.get_question("custom_id_123")
        assert question_data["id"] == "custom_id_123"

    def test_kwargs_with_full_metadata(self):
        """Test add_question with all metadata fields."""
        benchmark = Benchmark("Test Benchmark")

        author_info = {"name": "Test Author", "email": "test@example.com"}
        sources_info = [{"url": "https://example.com", "title": "Test Source"}]
        custom_meta = {"difficulty": "easy", "category": "test"}
        few_shot = [{"question": "Example Q", "answer": "Example A"}]

        q_id = benchmark.add_question(
            question="Full metadata test",
            raw_answer="Full answer",
            question_id="meta_test",
            finished=True,
            author=author_info,
            sources=sources_info,
            custom_metadata=custom_meta,
            few_shot_examples=few_shot,
        )

        question_data = benchmark.get_question(q_id)
        assert question_data["finished"] is True
        assert question_data["author"] == author_info
        assert question_data["sources"] == sources_info
        assert question_data["custom_metadata"] == custom_meta
        assert question_data["few_shot_examples"] == few_shot

    def test_add_question_from_object_current_method(self):
        """Test current add_question_from_object method."""
        benchmark = Benchmark("Test Benchmark")

        question_obj = Question(
            question="Object test question", raw_answer="Object test answer", tags=["test", "object"]
        )

        q_id = benchmark.add_question_from_object(question_obj)

        # Should use the Question object's auto-generated ID
        assert q_id == question_obj.id
        assert q_id in benchmark

        question_data = benchmark.get_question(q_id)
        assert question_data["question"] == "Object test question"
        assert question_data["raw_answer"] == "Object test answer"

    def test_question_id_generation_consistency(self):
        """Test that question ID generation is consistent."""
        benchmark1 = Benchmark("Test 1")
        benchmark2 = Benchmark("Test 2")

        # Same question text should generate same ID when no ID provided
        q1_id = benchmark1.add_question("What is 2+2?", "4")
        q2_id = benchmark2.add_question("What is 2+2?", "4")

        assert q1_id == q2_id  # Should generate same ID from same question text

    def test_duplicate_question_id_handling(self):
        """Test how duplicate question IDs are handled."""
        benchmark = Benchmark("Test Benchmark")

        # Add question with custom ID
        q1_id = benchmark.add_question(
            question="First question", raw_answer="First answer", question_id="duplicate_test"
        )
        assert q1_id == "duplicate_test"

        # Try to add another with same question text (should get different ID)
        q2_id = benchmark.add_question(
            question="First question",  # Same text
            raw_answer="Second answer",
        )

        # Should generate different ID due to uniqueness check
        assert q2_id != q1_id
        # The current implementation generates URN-style IDs from question text
        assert q2_id.startswith("urn:uuid:question-")

    def test_empty_and_none_values(self):
        """Test behavior with empty and None values."""
        benchmark = Benchmark("Test Benchmark")

        # Test with None values for optional parameters
        q_id = benchmark.add_question(
            question="Test with Nones",
            raw_answer="Test answer",
            answer_template=None,  # Should create default template
            author=None,
            sources=None,
            custom_metadata=None,
        )

        question_data = benchmark.get_question(q_id)
        assert question_data["answer_template"] is not None  # Default template created
        assert question_data.get("author") is None
        assert question_data.get("sources") is None
        assert question_data.get("custom_metadata") is None

    def test_cache_consistency(self):
        """Test that internal cache stays consistent with operations."""
        benchmark = Benchmark("Test Benchmark")

        # Add question
        q_id = benchmark.add_question("Cache test", "Test answer")

        # Verify cache contains question
        assert q_id in benchmark._questions_cache
        assert benchmark._questions_cache[q_id]["question"] == "Cache test"

        # Use the specific method for marking finished (not general metadata update)
        benchmark.mark_finished(q_id)

        # Cache should be updated
        updated_question = benchmark.get_question(q_id)
        assert updated_question["finished"] is True
        assert benchmark._questions_cache[q_id]["finished"] is True

    def test_save_load_consistency(self):
        """Test that saved and loaded benchmarks maintain consistency."""
        original_benchmark = Benchmark("Save/Load Test")

        # Add various questions
        q1_id = original_benchmark.add_question("Q1", "A1")
        q2_id = original_benchmark.add_question("Q2", "A2", finished=True, author={"name": "Test Author"})

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jsonld", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            original_benchmark.save(tmp_path)

            # Load benchmark
            loaded_benchmark = Benchmark.load(tmp_path)

            # Verify all questions preserved
            assert len(loaded_benchmark) == len(original_benchmark)
            assert q1_id in loaded_benchmark
            assert q2_id in loaded_benchmark

            # Verify question data preserved
            q1_data_orig = original_benchmark.get_question(q1_id)
            q1_data_loaded = loaded_benchmark.get_question(q1_id)
            assert q1_data_orig["question"] == q1_data_loaded["question"]
            assert q1_data_orig["raw_answer"] == q1_data_loaded["raw_answer"]

            q2_data_orig = original_benchmark.get_question(q2_id)
            q2_data_loaded = loaded_benchmark.get_question(q2_id)
            assert q2_data_orig["finished"] == q2_data_loaded["finished"]
            assert q2_data_orig["author"] == q2_data_loaded["author"]

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_batch_operations(self):
        """Test batch question operations work correctly."""
        benchmark = Benchmark("Batch Test")

        questions_data = [
            {"question": "Batch Q1", "raw_answer": "Batch A1"},
            {"question": "Batch Q2", "raw_answer": "Batch A2", "finished": True},
            {"question": "Batch Q3", "raw_answer": "Batch A3", "author": {"name": "Batch Author"}},
        ]

        question_ids = benchmark.add_questions_batch(questions_data)

        assert len(question_ids) == 3
        for q_id in question_ids:
            assert q_id in benchmark

        # Verify individual questions
        q1_data = benchmark.get_question(question_ids[0])
        assert q1_data["question"] == "Batch Q1"
        assert q1_data["finished"] is False

        q2_data = benchmark.get_question(question_ids[1])
        assert q2_data["finished"] is True

        q3_data = benchmark.get_question(question_ids[2])
        assert q3_data["author"]["name"] == "Batch Author"


class TestQuestionObjectCompatibility:
    """Test Question object usage patterns with current implementation."""

    def test_question_object_id_consistency(self):
        """Test that Question object IDs are consistent."""
        q1 = Question(question="Test question", raw_answer="Test answer")
        q2 = Question(question="Test question", raw_answer="Test answer")

        # Same question text should generate same ID
        assert q1.id == q2.id

    def test_question_object_properties(self):
        """Test Question object property access."""
        question = Question(
            question="What is AI?",
            raw_answer="Artificial Intelligence",
            tags=["ai", "technology"],
            few_shot_examples=[{"question": "What is ML?", "answer": "Machine Learning"}],
        )

        assert question.question == "What is AI?"
        assert question.raw_answer == "Artificial Intelligence"
        assert question.tags == ["ai", "technology"]
        assert len(question.few_shot_examples) == 1
        assert question.few_shot_examples[0]["question"] == "What is ML?"

    def test_question_object_with_benchmark(self):
        """Test current Question object integration with benchmark."""
        benchmark = Benchmark("Question Object Test")

        question = Question(
            question="Object integration test", raw_answer="Integration answer", tags=["test", "integration"]
        )

        # Use current add_question_from_object method
        q_id = benchmark.add_question_from_object(question)

        # Verify integration
        assert q_id == question.id
        retrieved_question = benchmark.get_question_as_object(q_id)
        assert retrieved_question.question == question.question
        assert retrieved_question.raw_answer == question.raw_answer
        assert retrieved_question.tags == question.tags


class TestTaskEvalCompatibility:
    """Test TaskEval compatibility with both dict and Question objects."""

    def test_task_eval_with_dict(self):
        """Test TaskEval accepts question dictionaries."""
        from karenina.benchmark.task_eval import TaskEval

        task = TaskEval(task_id="dict_test")

        question_dict = {"id": "dict_q1", "question": "TaskEval dict test", "raw_answer": "Dict answer"}

        # Should accept dict without error
        task.add_question(question_dict)

        # Verify question added
        assert len(task.global_questions) == 1
        assert task.global_questions[0] == question_dict

    def test_task_eval_with_question_object(self):
        """Test TaskEval accepts Question objects."""
        from karenina.benchmark.task_eval import TaskEval

        task = TaskEval(task_id="object_test")

        question_obj = Question(question="TaskEval object test", raw_answer="Object answer")

        # Should accept Question object without error
        task.add_question(question_obj)

        # Verify question added
        assert len(task.global_questions) == 1
        assert task.global_questions[0] == question_obj
