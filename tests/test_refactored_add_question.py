"""Tests for the refactored add_question method."""

import pytest

from karenina.benchmark import Benchmark
from karenina.schemas.domain import Question


class TestRefactoredAddQuestion:
    """Test the new transparent add_question method."""

    def test_add_question_with_question_object(self) -> None:
        """Test new functionality: passing Question object directly."""
        benchmark = Benchmark("Test Benchmark")

        # Create Question object
        question_obj = Question(
            question="What is machine learning?",
            raw_answer="Machine learning is a subset of AI",
            tags=["ml", "ai"],
            few_shot_examples=[{"question": "What is AI?", "answer": "Artificial Intelligence"}],
        )

        # Use new functionality
        q_id = benchmark.add_question(question_obj)

        # Verify it works correctly
        assert q_id == question_obj.id  # Should use Question object's ID
        assert q_id in benchmark

        # Verify question data
        question_data = benchmark.get_question(q_id)
        assert question_data["question"] == "What is machine learning?"
        assert question_data["raw_answer"] == "Machine learning is a subset of AI"
        assert question_data["few_shot_examples"] == [{"question": "What is AI?", "answer": "Artificial Intelligence"}]

    def test_add_question_with_question_object_and_metadata(self) -> None:
        """Test Question object with additional metadata."""
        benchmark = Benchmark("Test Benchmark")

        question_obj = Question(
            question="Explain neural networks",
            raw_answer="Neural networks are computing systems inspired by biological neural networks",
            tags=["neural", "networks"],
        )

        # Pass Question object with additional metadata
        q_id = benchmark.add_question(
            question_obj, finished=True, author={"name": "ML Expert"}, custom_metadata={"difficulty": "intermediate"}
        )

        # Verify all data is preserved
        question_data = benchmark.get_question(q_id)
        assert question_data["question"] == "Explain neural networks"
        assert question_data["finished"] is True
        assert question_data["author"]["name"] == "ML Expert"
        assert question_data["custom_metadata"]["difficulty"] == "intermediate"

    def test_add_question_with_question_object_override_id(self) -> None:
        """Test that custom question_id overrides Question object's ID."""
        benchmark = Benchmark("Test Benchmark")

        question_obj = Question(
            question="What is deep learning?", raw_answer="Deep learning is a subset of machine learning"
        )

        # Override the Question object's auto-generated ID
        custom_id = "custom_deep_learning_id"
        q_id = benchmark.add_question(question_obj, question_id=custom_id)

        # Should use the custom ID, not the Question object's ID
        assert q_id == custom_id
        assert q_id != question_obj.id
        assert custom_id in benchmark

    def test_add_question_backward_compatibility_kwargs(self) -> None:
        """Test that traditional kwargs usage still works."""
        benchmark = Benchmark("Test Benchmark")

        # Traditional usage should work exactly as before
        q_id = benchmark.add_question(
            question="What is Python?",
            raw_answer="Python is a programming language",
            finished=True,
            author={"name": "Python Expert"},
        )

        # Verify it works as expected
        assert q_id in benchmark
        question_data = benchmark.get_question(q_id)
        assert question_data["question"] == "What is Python?"
        assert question_data["raw_answer"] == "Python is a programming language"
        assert question_data["finished"] is True

    def test_add_question_validation_errors(self) -> None:
        """Test proper validation for invalid inputs."""
        benchmark = Benchmark("Test Benchmark")

        # Should raise error when question is string but raw_answer is None
        with pytest.raises(ValueError, match="raw_answer is required when question is a string"):
            benchmark.add_question("What is this?")

        # Should raise error for invalid question type
        with pytest.raises(TypeError):
            benchmark.add_question(123)  # Invalid type

    def test_question_object_id_consistency(self) -> None:
        """Test that Question objects generate consistent IDs."""
        benchmark1 = Benchmark("Test 1")
        benchmark2 = Benchmark("Test 2")

        # Same Question content should generate same ID
        q1 = Question(question="What is AI?", raw_answer="Artificial Intelligence")
        q2 = Question(question="What is AI?", raw_answer="Artificial Intelligence")

        id1 = benchmark1.add_question(q1)
        id2 = benchmark2.add_question(q2)

        assert id1 == id2  # Same content should generate same ID
        assert id1 == q1.id == q2.id

    def test_question_object_with_few_shot_override(self) -> None:
        """Test that explicit few_shot_examples override Question object's examples."""
        benchmark = Benchmark("Test Benchmark")

        question_obj = Question(
            question="Test question",
            raw_answer="Test answer",
            few_shot_examples=[{"question": "Original", "answer": "Original"}],
        )

        # Override few-shot examples
        override_examples = [{"question": "Override", "answer": "Override"}]
        q_id = benchmark.add_question(question_obj, few_shot_examples=override_examples)

        question_data = benchmark.get_question(q_id)
        assert question_data["few_shot_examples"] == override_examples
        assert question_data["few_shot_examples"] != question_obj.few_shot_examples

    def test_mixed_usage_patterns(self) -> None:
        """Test that both usage patterns can be used in the same benchmark."""
        benchmark = Benchmark("Mixed Usage Test")

        # Add question with kwargs (traditional)
        q1_id = benchmark.add_question("Traditional question", "Traditional answer")

        # Add question with Question object (new)
        q_obj = Question(question="Object question", raw_answer="Object answer")
        q2_id = benchmark.add_question(q_obj)

        # Both should work and coexist
        assert q1_id in benchmark
        assert q2_id in benchmark
        assert len(benchmark) == 2

        # Verify both questions are correct
        q1_data = benchmark.get_question(q1_id)
        q2_data = benchmark.get_question(q2_id)

        assert q1_data["question"] == "Traditional question"
        assert q2_data["question"] == "Object question"

    def test_question_object_tags_handling(self) -> None:
        """Test that Question object tags are properly handled as keywords."""
        benchmark = Benchmark("Test Benchmark")

        question_obj = Question(
            question="Tagged question",
            raw_answer="Tagged answer",
            tags=["tag1", "tag2", None, "tag3"],  # Include None to test filtering
        )

        q_id = benchmark.add_question(question_obj)

        # Check that tags were converted to keywords and None values filtered
        # The tags should be stored as keywords in the benchmark
        # Note: This depends on the internal implementation, but we can check
        # that the Question object as retrieved has the right tags
        retrieved_obj = benchmark.get_question_as_object(q_id)
        assert "tag1" in retrieved_obj.tags
        assert "tag2" in retrieved_obj.tags
        assert "tag3" in retrieved_obj.tags
        assert None not in retrieved_obj.tags

    def test_integration_with_existing_methods(self) -> None:
        """Test that new functionality integrates with existing benchmark methods."""
        benchmark = Benchmark("Integration Test")

        # Add question with new method
        question_obj = Question(question="Integration test question", raw_answer="Integration test answer")
        q_id = benchmark.add_question(question_obj)

        # Test existing methods still work - has_template returns False for default templates
        assert not benchmark.has_template(q_id)  # Should have default template (which returns False)
        assert not benchmark.get_question_metadata(q_id)["finished"]  # Not finished by default

        # Test marking finished
        benchmark.mark_finished(q_id)
        assert benchmark.get_question_metadata(q_id)["finished"]

        # Test get_question_as_object
        retrieved_obj = benchmark.get_question_as_object(q_id)
        assert retrieved_obj.question == question_obj.question
        assert retrieved_obj.raw_answer == question_obj.raw_answer

        # Test that a template was created (check via cache since get_template rejects default templates)
        template = benchmark._questions_cache[q_id].get("answer_template")
        assert template is not None
        assert "class Answer(BaseAnswer)" in template

        # Verify it's recognized as a default template
        question_text = benchmark._questions_cache[q_id].get("question", "")
        assert benchmark._template_manager._is_default_template(template, question_text)
