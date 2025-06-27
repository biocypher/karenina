import pytest
from pydantic import ValidationError

from karenina.schemas.answer_class import BaseAnswer
from karenina.schemas.question_class import Question


def test_question_schema():
    """Test Question schema validation."""
    # Test valid question
    question = Question(
        id="test123",
        question="Test question?",
        raw_answer="Test answer",
        tags=["tag1", "tag2"],
    )
    assert question.id == "test123"
    assert question.question == "Test question?"
    assert question.raw_answer == "Test answer"
    assert question.tags == ["tag1", "tag2"]

    # Test with empty tags
    question = Question(
        id="test123",
        question="Test question?",
        raw_answer="Test answer",
        tags=[],
    )
    assert question.tags == []

    # Test with None tags
    question = Question(
        id="test123",
        question="Test question?",
        raw_answer="Test answer",
        tags=[None],
    )
    assert question.tags == [None]

    # Test validation errors
    with pytest.raises(ValidationError):
        Question(
            id="test123",
            question="",  # Empty question
            raw_answer="Test answer",
            tags=[],
        )

    with pytest.raises(ValidationError):
        Question(
            id="test123",
            question="Test question?",
            raw_answer="",  # Empty answer
            tags=[],
        )


def test_base_answer_schema():
    """Test BaseAnswer schema validation."""

    # Test that BaseAnswer allows extra fields
    class TestAnswer(BaseAnswer):
        value: str

    answer = TestAnswer(value="test")
    assert answer.value == "test"

    # Test that extra fields are allowed
    answer = TestAnswer(value="test", extra_field="extra")
    assert answer.value == "test"
    assert answer.extra_field == "extra"
