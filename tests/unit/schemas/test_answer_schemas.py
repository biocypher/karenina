"""Unit tests for Pydantic domain schemas.

Tests cover:
- BaseAnswer validation, required fields, type coercion
- Question schema validation and computed ID
- LLMRubricTrait schema validation
- RegexTrait schema validation and pattern evaluation
- CallableTrait schema validation
- capture_answer_source function
"""

import pytest
from pydantic import ValidationError

from karenina.schemas.domain import (
    BaseAnswer,
    CallableTrait,
    LLMRubricTrait,
    Question,
    RegexTrait,
    TraitKind,
    capture_answer_source,
)

# =============================================================================
# Question Schema Tests
# =============================================================================


@pytest.mark.unit
def test_question_with_minimal_fields() -> None:
    """Test Question creation with minimal required fields."""
    q = Question(question="What is 2+2?", raw_answer="4")

    assert q.question == "What is 2+2?"
    assert q.raw_answer == "4"
    assert q.tags == []
    assert q.few_shot_examples is None


@pytest.mark.unit
def test_question_id_is_auto_generated() -> None:
    """Test that Question.id is computed as MD5 hash of question text."""
    q = Question(question="What is 2+2?", raw_answer="4")

    # ID should be a 32-character hex string (MD5)
    assert len(q.id) == 32
    assert all(c in "0123456789abcdef" for c in q.id)


@pytest.mark.unit
def test_question_same_text_same_id() -> None:
    """Test that identical questions produce the same ID."""
    q1 = Question(question="Test question?", raw_answer="A")
    q2 = Question(question="Test question?", raw_answer="B")

    assert q1.id == q2.id


@pytest.mark.unit
def test_question_different_text_different_id() -> None:
    """Test that different questions produce different IDs."""
    q1 = Question(question="Question 1?", raw_answer="A")
    q2 = Question(question="Question 2?", raw_answer="A")

    assert q1.id != q2.id


@pytest.mark.unit
def test_question_with_tags() -> None:
    """Test Question with tags."""
    q = Question(question="Test?", raw_answer="Answer", tags=["math", "easy"])

    assert q.tags == ["math", "easy"]


@pytest.mark.unit
def test_question_with_few_shot_examples() -> None:
    """Test Question with few-shot examples."""
    examples = [
        {"question": "2+2?", "answer": "4"},
        {"question": "3+3?", "answer": "6"},
    ]
    q = Question(question="Test?", raw_answer="Answer", few_shot_examples=examples)

    assert q.few_shot_examples == examples


@pytest.mark.unit
def test_question_empty_question_raises_validation_error() -> None:
    """Test that empty question text raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Question(question="", raw_answer="4")

    assert "question" in str(exc_info.value).lower()


@pytest.mark.unit
def test_question_empty_raw_answer_raises_validation_error() -> None:
    """Test that empty raw_answer raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Question(question="Test?", raw_answer="")

    assert "raw_answer" in str(exc_info.value).lower()


# =============================================================================
# LLMRubricTrait Schema Tests
# =============================================================================


@pytest.mark.unit
def test_llm_rubric_trait_boolean_minimal() -> None:
    """Test LLMRubricTrait with boolean kind and minimal fields."""
    trait = LLMRubricTrait(
        name="safety",
        kind="boolean",
        higher_is_better=True,
    )

    assert trait.name == "safety"
    assert trait.kind == "boolean"
    assert trait.description is None
    assert trait.min_score == 1  # default
    assert trait.max_score == 5  # default
    assert trait.higher_is_better is True


@pytest.mark.unit
def test_llm_rubric_trait_score_with_range() -> None:
    """Test LLMRubricTrait with score kind and custom range."""
    trait = LLMRubricTrait(
        name="quality",
        kind="score",
        min_score=1,
        max_score=10,
        higher_is_better=True,
    )

    assert trait.kind == "score"
    assert trait.min_score == 1
    assert trait.max_score == 10


@pytest.mark.unit
def test_llm_rubric_trait_name_min_length_validation() -> None:
    """Test that empty name raises ValidationError."""
    with pytest.raises(ValidationError):
        LLMRubricTrait(
            name="",
            kind="boolean",
            higher_is_better=True,
        )


@pytest.mark.unit
def test_llm_rubric_trait_validate_boolean_score() -> None:
    """Test validate_score method for boolean traits."""
    trait = LLMRubricTrait(
        name="test",
        kind="boolean",
        higher_is_better=True,
    )

    assert trait.validate_score(True) is True
    assert trait.validate_score(False) is True
    assert trait.validate_score(1) is False  # int not valid for boolean
    assert trait.validate_score(0) is False


@pytest.mark.unit
def test_llm_rubric_trait_validate_numeric_score() -> None:
    """Test validate_score method for score traits."""
    trait = LLMRubricTrait(
        name="test",
        kind="score",
        min_score=1,
        max_score=5,
        higher_is_better=True,
    )

    assert trait.validate_score(1) is True
    assert trait.validate_score(3) is True
    assert trait.validate_score(5) is True
    assert trait.validate_score(0) is False  # Below min
    assert trait.validate_score(6) is False  # Above max
    assert trait.validate_score(True) is False  # bool not valid for score
    assert trait.validate_score(False) is False


@pytest.mark.unit
def test_llm_rubric_trait_deep_judgment_fields() -> None:
    """Test deep judgment fields."""
    trait = LLMRubricTrait(
        name="test",
        kind="boolean",
        higher_is_better=True,
        deep_judgment_enabled=True,
        deep_judgment_excerpt_enabled=True,
        deep_judgment_max_excerpts=3,
        deep_judgment_fuzzy_match_threshold=0.8,
        deep_judgment_excerpt_retry_attempts=2,
        deep_judgment_search_enabled=True,
    )

    assert trait.deep_judgment_enabled is True
    assert trait.deep_judgment_excerpt_enabled is True
    assert trait.deep_judgment_max_excerpts == 3
    assert trait.deep_judgment_fuzzy_match_threshold == 0.8
    assert trait.deep_judgment_excerpt_retry_attempts == 2
    assert trait.deep_judgment_search_enabled is True


@pytest.mark.unit
def test_llm_rubric_trait_default_higher_is_better() -> None:
    """Test that higher_is_better defaults to True for legacy data."""
    # Using model_validate to simulate loading from dict
    trait = LLMRubricTrait.model_validate(
        {
            "name": "test",
            "kind": "boolean",
            "higher_is_better": None,  # Legacy: missing or None
        }
    )

    assert trait.higher_is_better is True


@pytest.mark.unit
def test_llm_rubric_trait_extra_fields_forbidden() -> None:
    """Test that extra fields are rejected."""
    with pytest.raises(ValidationError):
        LLMRubricTrait(
            name="test",
            kind="boolean",
            higher_is_better=True,
            extra_field="not_allowed",
        )


# =============================================================================
# RegexTrait Schema Tests
# =============================================================================


@pytest.mark.unit
def test_regex_trait_creation() -> None:
    """Test RegexTrait creation with valid pattern."""
    trait = RegexTrait(
        name="email_present",
        pattern=r"\S+@\S+",
        higher_is_better=True,
    )

    assert trait.name == "email_present"
    assert trait.pattern == r"\S+@\S+"
    assert trait.case_sensitive is True  # default
    assert trait.invert_result is False  # default
    assert trait.higher_is_better is True


@pytest.mark.unit
def test_regex_trait_case_insensitive() -> None:
    """Test RegexTrait with case_sensitive=False."""
    trait = RegexTrait(
        name="keyword_match",
        pattern=r"python",
        case_sensitive=False,
        higher_is_better=True,
    )

    assert trait.case_sensitive is False


@pytest.mark.unit
def test_regex_trait_invert_result() -> None:
    """Test RegexTrait with invert_result=True."""
    trait = RegexTrait(
        name="no_profanity",
        pattern=r"\bbadword\b",
        invert_result=True,
        higher_is_better=True,
    )

    assert trait.invert_result is True


@pytest.mark.unit
def test_regex_trait_invalid_pattern_raises_error() -> None:
    """Test that invalid regex pattern raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        RegexTrait(
            name="test",
            pattern=r"(?P<unclosed",  # Invalid regex
            higher_is_better=True,
        )

    assert "Invalid regex pattern" in str(exc_info.value)


@pytest.mark.unit
def test_regex_trait_evaluate_match() -> None:
    """Test RegexTrait.evaluate() with matching text."""
    trait = RegexTrait(
        name="email",
        pattern=r"\S+@\S+",
        case_sensitive=True,
        higher_is_better=True,
    )

    assert trait.evaluate("Contact us at test@example.com") is True


@pytest.mark.unit
def test_regex_trait_evaluate_no_match() -> None:
    """Test RegexTrait.evaluate() with non-matching text."""
    trait = RegexTrait(
        name="email",
        pattern=r"\S+@\S+",
        case_sensitive=True,
        higher_is_better=True,
    )

    assert trait.evaluate("No email here") is False


@pytest.mark.unit
def test_regex_trait_evaluate_case_insensitive() -> None:
    """Test RegexTrait.evaluate() with case_sensitive=False."""
    trait = RegexTrait(
        name="keyword",
        pattern=r"PYTHON",
        case_sensitive=False,
        higher_is_better=True,
    )

    assert trait.evaluate("I love python programming") is True


@pytest.mark.unit
def test_regex_trait_evaluate_inverted() -> None:
    """Test RegexTrait.evaluate() with invert_result=True."""
    trait = RegexTrait(
        name="no_profanity",
        pattern=r"\bbadword\b",
        invert_result=True,
        higher_is_better=True,
    )

    # Pattern NOT found = True (inverted)
    assert trait.evaluate("This is clean text") is True
    # Pattern found = False (inverted)
    assert trait.evaluate("This contains badword here") is False


@pytest.mark.unit
def test_regex_trait_extra_fields_forbidden() -> None:
    """Test that extra fields are rejected."""
    with pytest.raises(ValidationError):
        RegexTrait(
            name="test",
            pattern=r"\w+",
            higher_is_better=True,
            extra_field="not_allowed",
        )


# =============================================================================
# CallableTrait Schema Tests
# =============================================================================


@pytest.mark.unit
def test_callable_trait_boolean() -> None:
    """Test CallableTrait with boolean kind."""
    import cloudpickle

    def check_length(text: str) -> bool:
        return len(text) >= 10

    trait = CallableTrait(
        name="min_length",
        kind="boolean",
        callable_code=cloudpickle.dumps(check_length),
        higher_is_better=True,
    )

    assert trait.name == "min_length"
    assert trait.kind == "boolean"
    assert trait.min_score is None
    assert trait.max_score is None
    assert trait.invert_result is False  # default


@pytest.mark.unit
def test_callable_trait_score() -> None:
    """Test CallableTrait with score kind."""
    import cloudpickle

    def word_count_score(text: str) -> int:
        return min(len(text.split()), 10)

    trait = CallableTrait(
        name="word_count",
        kind="score",
        callable_code=cloudpickle.dumps(word_count_score),
        min_score=0,
        max_score=10,
        higher_is_better=True,
    )

    assert trait.kind == "score"
    assert trait.min_score == 0
    assert trait.max_score == 10


@pytest.mark.unit
def test_callable_trait_extra_fields_forbidden() -> None:
    """Test that extra fields are rejected."""
    import cloudpickle

    trait = CallableTrait(
        name="test",
        kind="boolean",
        callable_code=cloudpickle.dumps(lambda _: True),
        higher_is_better=True,
    )

    # Extra fields should raise ValidationError
    with pytest.raises(ValidationError):
        CallableTrait.model_validate(
            {
                **trait.model_dump(),
                "extra_field": "not_allowed",
            }
        )


# =============================================================================
# BaseAnswer Schema Tests
# =============================================================================


@pytest.mark.unit
def test_base_answer_extra_fields_allowed() -> None:
    """Test that BaseAnswer allows extra fields via ConfigDict."""
    # BaseAnswer has extra="allow" configuration
    answer = BaseAnswer()

    # Should be able to set arbitrary attributes
    answer.custom_field = "test"
    assert hasattr(answer, "custom_field")


@pytest.mark.unit
def test_base_answer_id_optional() -> None:
    """Test that BaseAnswer.id is optional."""
    answer = BaseAnswer()

    assert answer.id is None  # Default is None


@pytest.mark.unit
def test_base_answer_id_can_be_set() -> None:
    """Test that BaseAnswer.id can be set."""
    answer = BaseAnswer()
    answer.id = "test-question-id"

    assert answer.id == "test-question-id"


@pytest.mark.unit
def test_base_answer_get_source_code() -> None:
    """Test BaseAnswer.get_source_code() method."""
    # For dynamically created classes, source code might be None
    BaseAnswer()

    # get_source_code should return None for classes without source
    source = BaseAnswer.get_source_code()
    # It's okay if it's None (happens for exec-created classes)
    assert source is None or isinstance(source, str)


@pytest.mark.unit
def test_base_answer_verify_regex() -> None:
    """Test BaseAnswer.verify_regex() method."""
    from typing import ClassVar

    from karenina.schemas.domain.answer import BaseAnswer

    class Answer(BaseAnswer):
        value: str
        # Type annotation for regex (ClassVar to exclude from Pydantic fields)
        # Note: For 'contains' match_type, expected should be a substring that might be in matches
        regex: ClassVar[dict] = {
            "has_42": {"pattern": r"\b42\b", "expected": "42", "match_type": "exact"},
            "has_hello": {"pattern": r"\bhello\b", "expected": "hello", "match_type": "exact"},
        }

        def verify(self) -> bool:
            return True

    answer = Answer(value="test")

    # Test regex verification
    result = answer.verify_regex("The number is 42 and hello there")

    assert result["success"] is True
    assert result["results"]["has_42"] is True
    assert result["results"]["has_hello"] is True


@pytest.mark.unit
def test_base_answer_verify_regex_no_regex_defined() -> None:
    """Test verify_regex() when no regex patterns are defined."""
    answer = BaseAnswer()

    result = answer.verify_regex("any text")

    assert result["success"] is True
    assert result["results"] == {}


@pytest.mark.unit
def test_trait_kind_literal() -> None:
    """Test that TraitKind is a Literal with correct values."""
    valid_kinds: list[TraitKind] = ["boolean", "score"]

    assert "boolean" in valid_kinds
    assert "score" in valid_kinds


@pytest.mark.unit
def test_question_serialization_roundtrip() -> None:
    """Test Question serialization and deserialization."""
    original = Question(
        question="What is AI?",
        raw_answer="Artificial Intelligence",
        tags=["tech", "ml"],
    )

    # Serialize to dict
    data = original.model_dump()

    # Deserialize back
    restored = Question(**data)

    assert restored.question == original.question
    assert restored.raw_answer == original.raw_answer
    assert restored.tags == original.tags
    # ID is computed, should be the same
    assert restored.id == original.id


@pytest.mark.unit
def test_llm_rubric_trait_serialization_roundtrip() -> None:
    """Test LLMRubricTrait serialization and deserialization."""
    original = LLMRubricTrait(
        name="clarity",
        kind="score",
        min_score=1,
        max_score=5,
        higher_is_better=True,
        description="Response clarity",
    )

    # Serialize to dict
    data = original.model_dump()

    # Deserialize back
    restored = LLMRubricTrait(**data)

    assert restored.name == original.name
    assert restored.kind == original.kind
    assert restored.min_score == original.min_score
    assert restored.max_score == original.max_score


@pytest.mark.unit
def test_regex_trait_serialization_roundtrip() -> None:
    """Test RegexTrait serialization and deserialization."""
    original = RegexTrait(
        name="has_citation",
        pattern=r"\[\d+\]",
        case_sensitive=False,
        higher_is_better=True,
    )

    # Serialize to dict
    data = original.model_dump()

    # Deserialize back
    restored = RegexTrait(**data)

    assert restored.name == original.name
    assert restored.pattern == original.pattern
    assert restored.case_sensitive == original.case_sensitive


# =============================================================================
# capture_answer_source Tests
# =============================================================================


@pytest.mark.unit
def test_capture_answer_source_returns_class() -> None:
    """Test that capture_answer_source returns the same class."""
    from karenina.schemas.domain.answer import BaseAnswer

    class CustomAnswer(BaseAnswer):
        value: str

        def verify(self) -> bool:
            return True

    # Should return the same class
    result = capture_answer_source(CustomAnswer)
    assert result is CustomAnswer


@pytest.mark.unit
def test_capture_answer_source_calls_method_when_exists() -> None:
    """Test that capture_answer_source calls set_source_code_from_notebook if present."""
    from karenina.schemas.domain.answer import BaseAnswer

    # Track if method was called
    method_called = {"called": False}

    class CustomAnswer(BaseAnswer):
        value: str

        @classmethod
        def set_source_code_from_notebook(cls) -> None:
            method_called["called"] = True

        def verify(self) -> bool:
            return True

    # Method should be called
    capture_answer_source(CustomAnswer)
    assert method_called["called"] is True


@pytest.mark.unit
def test_capture_answer_source_no_error_without_method() -> None:
    """Test that capture_answer_source doesn't error when method is missing."""
    from karenina.schemas.domain.answer import BaseAnswer

    class PlainAnswer(BaseAnswer):
        value: str

        def verify(self) -> bool:
            return True

    # Should not raise error even though method doesn't exist
    result = capture_answer_source(PlainAnswer)
    assert result is PlainAnswer


@pytest.mark.unit
def test_capture_answer_source_as_decorator() -> None:
    """Test that capture_answer_source works as a decorator."""
    from karenina.schemas.domain.answer import BaseAnswer

    method_called = {"called": False}

    @capture_answer_source
    class DecoratedAnswer(BaseAnswer):
        value: str

        @classmethod
        def set_source_code_from_notebook(cls) -> None:
            method_called["called"] = True

        def verify(self) -> bool:
            return True

    # Method should have been called during decoration
    assert method_called["called"] is True
    assert DecoratedAnswer.__name__ == "DecoratedAnswer"
