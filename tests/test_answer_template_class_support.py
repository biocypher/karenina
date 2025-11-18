"""Tests for Answer class support in add_question method."""

import inspect

import pytest
from pydantic import Field

from karenina.benchmark import Benchmark
from karenina.schemas.domain import BaseAnswer, LLMRubricTrait, Question


class TestAnswerClassSupport:
    """Test the new Answer class support in add_question."""

    def test_add_question_with_file_based_answer_class(self) -> None:
        """Test passing a file-based Answer class to add_question."""
        benchmark = Benchmark("Test Benchmark")

        # Define a file-based Answer class
        class TestFileAnswer(BaseAnswer):
            """Test answer class defined in file."""

            value: int = Field(description="Test value")

            def verify(self) -> bool:
                return self.value == 42

        # Add question with Answer class
        q_id = benchmark.add_question(question="What is 6 * 7?", raw_answer="42", answer_template=TestFileAnswer)

        # Verify the question was added
        assert q_id in benchmark
        question_data = benchmark.get_question(q_id)

        # Verify the template was converted to source code
        template_code = question_data["answer_template"]
        assert template_code is not None
        assert "class TestFileAnswer(BaseAnswer):" in template_code
        assert 'value: int = Field(description="Test value")' in template_code
        assert "def verify(self) -> bool:" in template_code
        assert "return self.value == 42" in template_code

    def test_add_question_with_answer_class_preserves_source(self) -> None:
        """Test that Answer classes with _source_code are handled correctly."""
        benchmark = Benchmark("Test Benchmark")

        # Create a class and manually set source code (simulating exec-created class)
        class DynamicAnswer(BaseAnswer):
            """Dynamically created answer."""

            result: str = Field(description="The result")

            def verify(self) -> bool:
                return self.result == "correct"

        # Manually set source code to simulate AnswerWithID pattern
        custom_source = '''from pydantic import Field

class Answer(BaseAnswer):
    """Custom answer template."""
    result: str = Field(description="The result")

    def verify(self) -> bool:
        return self.result == "correct"
'''
        DynamicAnswer._source_code = custom_source

        # Add question with Answer class that has custom source
        q_id = benchmark.add_question(
            question="What is the correct answer?", raw_answer="correct", answer_template=DynamicAnswer
        )

        # Verify the custom source code was used
        question_data = benchmark.get_question(q_id)
        template_code = question_data["answer_template"]
        assert template_code == custom_source
        assert "class Answer(BaseAnswer):" in template_code

    def test_add_question_with_answer_class_inheritance_validation(self) -> None:
        """Test that non-BaseAnswer classes are rejected."""
        benchmark = Benchmark("Test Benchmark")

        # Define a class that doesn't inherit from BaseAnswer
        class InvalidAnswer:
            def __init__(self) -> None:
                pass

        # Should raise TypeError
        with pytest.raises(TypeError, match="answer_template class must inherit from BaseAnswer"):
            benchmark.add_question(question="Test question", raw_answer="Test answer", answer_template=InvalidAnswer)

    def test_add_question_with_answer_class_no_source_error(self) -> None:
        """Test error handling when source code cannot be extracted."""
        benchmark = Benchmark("Test Benchmark")

        # Create a class that simulates failed source extraction
        class NoSourceAnswer(BaseAnswer):
            """Answer class with no accessible source."""

            value: bool = Field(description="Test value")

            def verify(self) -> bool:
                return self.value

        # Mock the get_source_code method to return None and inspect.getsource to fail
        original_get_source = NoSourceAnswer.get_source_code

        def mock_get_source() -> None:
            return None

        NoSourceAnswer.get_source_code = classmethod(mock_get_source)

        # Mock inspect.getsource to raise OSError
        original_getsource = inspect.getsource

        def mock_getsource(obj) -> None:
            raise OSError("source not available")

        inspect.getsource = mock_getsource

        try:
            # Should raise ValueError with descriptive message
            with pytest.raises(ValueError, match="Could not extract source code from Answer class NoSourceAnswer"):
                benchmark.add_question(
                    question="Test question", raw_answer="Test answer", answer_template=NoSourceAnswer
                )
        finally:
            # Restore original functions
            NoSourceAnswer.get_source_code = original_get_source
            inspect.getsource = original_getsource

    def test_add_question_with_answer_class_complex_definition(self) -> None:
        """Test Answer class with complex definition including class variables."""
        benchmark = Benchmark("Test Benchmark")
        from typing import ClassVar

        class ComplexAnswer(BaseAnswer):
            """Complex answer class with various features."""

            # Class variable
            DEFAULT_THRESHOLD: ClassVar[float] = 0.8

            # Multiple field types
            score: float = Field(description="Confidence score", ge=0.0, le=1.0)
            prediction: str = Field(description="The prediction")
            metadata: dict = Field(description="Additional metadata", default_factory=dict)

            def verify(self) -> bool:
                """Complex verification logic."""
                if self.score < self.DEFAULT_THRESHOLD:
                    return False
                return self.prediction in ["positive", "negative", "neutral"]

            @classmethod
            def create_positive(cls, score: float):
                """Factory method for positive predictions."""
                return cls(score=score, prediction="positive")

        # Add question with complex Answer class
        q_id = benchmark.add_question(
            question="What is the sentiment?", raw_answer="positive", answer_template=ComplexAnswer
        )

        # Verify the complex definition was preserved
        question_data = benchmark.get_question(q_id)
        template_code = question_data["answer_template"]

        assert "DEFAULT_THRESHOLD: ClassVar[float] = 0.8" in template_code
        assert "score: float = Field(" in template_code
        assert "def verify(self) -> bool:" in template_code
        assert "@classmethod" in template_code
        assert "def create_positive(cls, score: float):" in template_code

    def test_add_question_with_question_object_and_answer_class(self) -> None:
        """Test using both Question object and Answer class together."""
        benchmark = Benchmark("Test Benchmark")

        # Create Question object
        question_obj = Question(
            question="What programming language is this?",
            raw_answer="Python",
            tags=["programming", "language"],
        )

        # Create Answer class
        class LanguageAnswer(BaseAnswer):
            """Answer for programming language questions."""

            language: str = Field(description="The programming language")
            confidence: float = Field(description="Confidence level", ge=0.0, le=1.0)

            def verify(self) -> bool:
                return self.language.lower() == "python" and self.confidence > 0.5

        # Add question using both Question object and Answer class
        q_id = benchmark.add_question(question=question_obj, answer_template=LanguageAnswer, finished=True)

        # Verify everything works together
        assert q_id == question_obj.id
        question_data = benchmark.get_question(q_id)

        # Question data from Question object
        assert question_data["question"] == "What programming language is this?"
        assert question_data["raw_answer"] == "Python"
        assert question_data["finished"] is True

        # Template from Answer class
        template_code = question_data["answer_template"]
        assert "class LanguageAnswer(BaseAnswer):" in template_code
        assert "language: str = Field(" in template_code

    def test_add_question_answer_class_with_traditional_parameters(self) -> None:
        """Test Answer class with traditional string question and raw_answer."""
        benchmark = Benchmark("Test Benchmark")

        class MathAnswer(BaseAnswer):
            """Mathematical calculation answer."""

            operation: str = Field(description="The mathematical operation")
            operand1: int = Field(description="First operand")
            operand2: int = Field(description="Second operand")
            result: int = Field(description="The calculation result")

            def verify(self) -> bool:
                if self.operation == "multiply":
                    return self.result == self.operand1 * self.operand2
                return False

        # Add question with traditional parameters + Answer class
        q_id = benchmark.add_question(
            question="What is 8 * 9?",
            raw_answer="72",
            answer_template=MathAnswer,
            finished=True,
            author={"name": "Math Teacher"},
            custom_metadata={"subject": "arithmetic", "difficulty": "easy"},
        )

        # Verify all data is preserved
        question_data = benchmark.get_question(q_id)
        assert question_data["question"] == "What is 8 * 9?"
        assert question_data["raw_answer"] == "72"
        assert question_data["finished"] is True
        assert question_data["author"]["name"] == "Math Teacher"
        assert question_data["custom_metadata"]["subject"] == "arithmetic"

        # Verify Answer class was converted
        template_code = question_data["answer_template"]
        assert "class MathAnswer(BaseAnswer):" in template_code
        assert "operation: str = Field(" in template_code

    def test_add_question_string_template_still_works(self) -> None:
        """Test that string templates still work as before."""
        benchmark = Benchmark("Test Benchmark")

        template_code = '''from pydantic import Field

class Answer(BaseAnswer):
    """String template answer."""
    response: str = Field(description="The response")

    def verify(self) -> bool:
        return self.response == "expected"
'''

        # Add question with string template (existing functionality)
        q_id = benchmark.add_question(question="Test question", raw_answer="expected", answer_template=template_code)

        # Verify string template is preserved as-is
        question_data = benchmark.get_question(q_id)
        assert question_data["answer_template"] == template_code

    def test_add_question_none_template_creates_default(self) -> None:
        """Test that None template still creates default template."""
        benchmark = Benchmark("Test Benchmark")

        # Add question with None template (existing functionality)
        q_id = benchmark.add_question(question="What is the default behavior?", raw_answer="Creates default template")

        # Verify default template was created
        question_data = benchmark.get_question(q_id)
        template_code = question_data["answer_template"]
        assert template_code is not None
        assert "class Answer(BaseAnswer):" in template_code
        assert "response: str = Field(" in template_code
        assert "# TODO: Implement verification logic" in template_code

    def test_add_question_non_class_type_raises_error(self) -> None:
        """Test that non-class types raise appropriate errors."""
        benchmark = Benchmark("Test Benchmark")

        # Test with various invalid types
        invalid_types = [123, ["list"], {"dict": "value"}, lambda x: x, object()]

        for invalid_type in invalid_types:
            with pytest.raises((TypeError, ValueError)):
                benchmark.add_question(question="Test question", raw_answer="Test answer", answer_template=invalid_type)

    def test_benchmark_add_question_delegates_correctly(self) -> None:
        """Test that Benchmark.add_question correctly delegates to QuestionManager."""
        benchmark = Benchmark("Test Benchmark")

        class DelegationAnswer(BaseAnswer):
            """Answer to test delegation."""

            delegated: bool = Field(description="Was this delegated correctly")

            def verify(self) -> bool:
                return self.delegated

        # Add question through Benchmark (should delegate to QuestionManager)
        q_id = benchmark.add_question(
            question="Does delegation work?", raw_answer="Yes", answer_template=DelegationAnswer
        )

        # Verify it worked the same as direct QuestionManager usage
        question_data = benchmark.get_question(q_id)
        template_code = question_data["answer_template"]
        assert "class DelegationAnswer(BaseAnswer):" in template_code
        assert "delegated: bool = Field(" in template_code

    def test_answer_with_id_wrapper_pattern_compatibility(self) -> None:
        """Test compatibility with AnswerWithID wrapper pattern."""
        benchmark = Benchmark("Test Benchmark")

        # Simulate AnswerWithID wrapper by creating a class with preserved source
        class OriginalAnswer(BaseAnswer):
            """Original answer class."""

            original_field: str = Field(description="Original field")

            def verify(self) -> bool:
                return True

        # Simulate AnswerWithID wrapper (like inject_question_id_into_answer_class does)
        class AnswerWithID(OriginalAnswer):
            def model_post_init(self, __context) -> None:
                if hasattr(super(), "model_post_init"):
                    super().model_post_init(__context)
                self.id = "test_question_id"

        # Preserve the original source code (as AnswerWithID pattern does)
        original_source = '''from pydantic import Field

class Answer(BaseAnswer):
    """Original answer template."""
    original_field: str = Field(description="Original field")

    def verify(self) -> bool:
        return True
'''
        AnswerWithID._source_code = original_source

        # Add question with AnswerWithID (should use preserved source)
        q_id = benchmark.add_question(
            question="Does AnswerWithID work?", raw_answer="Yes", answer_template=AnswerWithID
        )

        # Verify the original source was used, not the wrapper
        question_data = benchmark.get_question(q_id)
        template_code = question_data["answer_template"]
        assert template_code == original_source
        assert "class Answer(BaseAnswer):" in template_code
        assert "model_post_init" not in template_code  # Wrapper method not included


class TestAnswerClassIntegration:
    """Integration tests for Answer class support with other benchmark features."""

    def test_answer_class_with_rubrics(self) -> None:
        """Test Answer class support with question rubrics."""
        benchmark = Benchmark("Test Benchmark")

        class RubricAnswer(BaseAnswer):
            """Answer class for rubric testing."""

            quality: str = Field(description="Quality of the answer")

            def verify(self) -> bool:
                return self.quality in ["excellent", "good", "fair"]

        # Add question with Answer class and rubric
        q_id = benchmark.add_question(
            question="How is the quality?", raw_answer="excellent", answer_template=RubricAnswer
        )

        # Add question-specific rubric

        rubric_trait = LLMRubricTrait(
            name="Quality Assessment",
            description="Assess the quality of the answer",
            kind="score",
            min_score=1,
            max_score=5,
        )
        benchmark.add_question_rubric_trait(q_id, rubric_trait)

        # Verify both Answer class and rubric work together
        question_data = benchmark.get_question(q_id)
        assert "class RubricAnswer(BaseAnswer):" in question_data["answer_template"]
        assert question_data.get("question_rubric") is not None

    def test_answer_class_with_batch_operations(self) -> None:
        """Test Answer class support with batch question operations."""
        benchmark = Benchmark("Test Benchmark")

        class BatchAnswer(BaseAnswer):
            """Answer class for batch testing."""

            batch_id: int = Field(description="Batch identifier")

            def verify(self) -> bool:
                return self.batch_id > 0

        # Create multiple questions with same Answer class
        questions_data = [
            {"question": f"Batch question {i}", "raw_answer": f"Answer {i}", "answer_template": BatchAnswer}
            for i in range(1, 4)
        ]

        # Add questions in batch (note: this tests individual add_question calls)
        question_ids = []
        for data in questions_data:
            q_id = benchmark.add_question(**data)
            question_ids.append(q_id)

        # Verify all questions have the Answer class template
        for q_id in question_ids:
            question_data = benchmark.get_question(q_id)
            template_code = question_data["answer_template"]
            assert "class BatchAnswer(BaseAnswer):" in template_code
            assert "batch_id: int = Field(" in template_code

    def test_answer_class_with_export_operations(self) -> None:
        """Test Answer class support with benchmark export operations."""
        benchmark = Benchmark("Test Benchmark")

        class ExportAnswer(BaseAnswer):
            """Answer class for export testing."""

            export_data: dict = Field(description="Data for export", default_factory=dict)

            def verify(self) -> bool:
                return isinstance(self.export_data, dict)

        # Add question with Answer class
        q_id = benchmark.add_question(
            question="Can this be exported?", raw_answer="Yes, it can", answer_template=ExportAnswer, finished=True
        )

        # Test various export formats
        summary = benchmark.get_summary()
        assert summary["question_count"] == 1
        assert summary["finished_count"] == 1

        # Test dictionary export
        dict_export = benchmark.to_dict()
        question_data = next(q for q in dict_export["questions"] if q["id"] == q_id)
        assert "class ExportAnswer(BaseAnswer):" in question_data["answer_template"]

        # Test CSV export
        csv_export = benchmark.to_csv()
        assert q_id in csv_export
        assert "Can this be exported?" in csv_export
