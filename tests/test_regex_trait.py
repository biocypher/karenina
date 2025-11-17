"""Tests for RegexTrait functionality."""

import pytest
from pydantic import ValidationError

from karenina.schemas.domain import RegexTrait, Rubric


class TestRegexTrait:
    """Test RegexTrait schema and functionality."""

    def test_regex_trait_creation(self) -> None:
        """Test creating a regex trait."""
        trait = RegexTrait(
            name="contains_hello",
            description="Check if text contains 'hello'",
            pattern=r"\bhello\b",
            case_sensitive=False,
        )

        assert trait.name == "contains_hello"
        assert trait.description == "Check if text contains 'hello'"
        assert trait.pattern == r"\bhello\b"
        assert trait.case_sensitive is False
        assert trait.invert_result is False

    def test_regex_trait_minimal(self) -> None:
        """Test creating regex trait with minimal fields."""
        trait = RegexTrait(
            name="has_number",
            pattern=r"\d+",
        )

        assert trait.name == "has_number"
        assert trait.pattern == r"\d+"
        assert trait.description is None
        assert trait.case_sensitive is True  # Default
        assert trait.invert_result is False  # Default

    def test_invalid_regex_pattern(self) -> None:
        """Test validation of regex patterns."""
        with pytest.raises(ValidationError, match="Invalid regex pattern"):
            RegexTrait(
                name="bad_regex",
                pattern=r"[invalid",  # Unclosed bracket
            )

    def test_empty_name_validation(self) -> None:
        """Test that empty names are rejected."""
        with pytest.raises(ValidationError):
            RegexTrait(
                name="",
                pattern=r"test",
            )

    def test_regex_evaluation_case_sensitive(self) -> None:
        """Test regex evaluation with case sensitivity."""
        trait = RegexTrait(
            name="contains_hello",
            pattern=r"hello",
            case_sensitive=True,
        )

        # Should match exact case
        assert trait.evaluate("Say hello world") is True
        assert trait.evaluate("Say HELLO world") is False
        assert trait.evaluate("No match here") is False

    def test_regex_evaluation_case_insensitive(self) -> None:
        """Test regex evaluation without case sensitivity."""
        trait = RegexTrait(
            name="contains_hello",
            pattern=r"hello",
            case_sensitive=False,
        )

        # Should match any case
        assert trait.evaluate("Say hello world") is True
        assert trait.evaluate("Say HELLO world") is True
        assert trait.evaluate("Say HeLLo world") is True
        assert trait.evaluate("No match here") is False

    def test_regex_evaluation_with_invert(self) -> None:
        """Test regex evaluation with result inversion."""
        trait = RegexTrait(
            name="not_contains_error",
            pattern=r"error",
            invert_result=True,
        )

        # Should invert the result
        assert trait.evaluate("This is good") is True  # No error -> True (inverted)
        assert trait.evaluate("This is an error") is False  # Has error -> False (inverted)

    def test_regex_evaluation_word_boundaries(self) -> None:
        """Test regex with word boundaries."""
        trait = RegexTrait(
            name="has_word_test",
            pattern=r"\btest\b",
        )

        assert trait.evaluate("This is a test") is True
        assert trait.evaluate("testing is good") is False  # "testing" contains "test" but not as word
        assert trait.evaluate("contest") is False

    def test_regex_evaluation_email_pattern(self) -> None:
        """Test regex with email validation pattern."""
        trait = RegexTrait(
            name="contains_email",
            pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        )

        assert trait.evaluate("Contact me at test@example.com") is True
        assert trait.evaluate("Email: user.name+tag@domain.co.uk") is True
        assert trait.evaluate("No email here") is False
        assert trait.evaluate("Invalid: @example.com") is False

    def test_regex_evaluation_date_pattern(self) -> None:
        """Test regex with date pattern."""
        trait = RegexTrait(
            name="has_iso_date",
            pattern=r"\d{4}-\d{2}-\d{2}",
        )

        assert trait.evaluate("Today is 2025-01-17") is True
        assert trait.evaluate("2024-12-31 was last year") is True
        assert trait.evaluate("01/17/2025 is not ISO") is False

    def test_regex_evaluation_multiline(self) -> None:
        """Test regex with multiline text."""
        trait = RegexTrait(
            name="starts_with_error",
            pattern=r"^ERROR:",
        )

        assert trait.evaluate("ERROR: Something went wrong") is True
        assert trait.evaluate("INFO: All good\nERROR: Now broken") is False  # ^ matches start of string

    def test_regex_evaluation_empty_string(self) -> None:
        """Test regex evaluation with empty string."""
        trait = RegexTrait(
            name="not_empty",
            pattern=r".+",  # At least one character
        )

        assert trait.evaluate("") is False
        assert trait.evaluate("   ") is True  # Whitespace counts
        assert trait.evaluate("a") is True


class TestRubricWithRegexTraits:
    """Test Rubric functionality with regex traits."""

    def test_rubric_with_regex_traits(self) -> None:
        """Test creating rubric with regex traits."""
        from karenina.schemas.domain import LLMRubricTrait

        llm_trait = LLMRubricTrait(name="accuracy", description="Accurate?", kind="boolean")
        regex_trait = RegexTrait(name="has_number", pattern=r"\d+")

        rubric = Rubric(
            traits=[llm_trait],
            regex_traits=[regex_trait],
        )

        assert len(rubric.traits) == 1
        assert len(rubric.regex_traits) == 1
        assert set(rubric.get_trait_names()) == {"accuracy", "has_number"}
        assert rubric.get_llm_trait_names() == ["accuracy"]
        assert rubric.get_regex_trait_names() == ["has_number"]

    def test_rubric_validation_with_regex_traits(self) -> None:
        """Test validation of evaluation results with regex traits."""
        from karenina.schemas.domain import LLMRubricTrait

        llm_trait = LLMRubricTrait(name="score", kind="score", min_score=1, max_score=5)
        regex_trait = RegexTrait(name="has_keyword", pattern=r"keyword")

        rubric = Rubric(
            traits=[llm_trait],
            regex_traits=[regex_trait],
        )

        # Valid evaluation
        valid_eval = {"score": 4, "has_keyword": True}
        assert rubric.validate_evaluation(valid_eval) is True

        # Invalid - regex trait should be boolean
        invalid_eval = {"score": 4, "has_keyword": "yes"}
        assert rubric.validate_evaluation(invalid_eval) is False

        # Invalid - missing trait
        incomplete_eval = {"score": 4}
        assert rubric.validate_evaluation(incomplete_eval) is False

        # Invalid - unknown trait
        extra_eval = {"score": 4, "has_keyword": True, "unknown": False}
        assert rubric.validate_evaluation(extra_eval) is False


class TestMergeRubricsWithRegexTraits:
    """Test rubric merging with regex traits."""

    def test_merge_rubrics_with_regex_traits(self) -> None:
        """Test merging rubrics that contain regex traits."""
        from karenina.schemas.domain import LLMRubricTrait, merge_rubrics

        # Global rubric with both types
        global_llm = LLMRubricTrait(name="accuracy", kind="boolean")
        global_regex = RegexTrait(name="has_number", pattern=r"\d+")
        global_rubric = Rubric(traits=[global_llm], regex_traits=[global_regex])

        # Question rubric with both types
        question_llm = LLMRubricTrait(name="completeness", kind="score", min_score=1, max_score=5)
        question_regex = RegexTrait(name="has_date", pattern=r"\d{4}-\d{2}-\d{2}")
        question_rubric = Rubric(traits=[question_llm], regex_traits=[question_regex])

        # Merge rubrics
        merged = merge_rubrics(global_rubric, question_rubric)

        assert merged is not None
        assert len(merged.traits) == 2  # accuracy + completeness
        assert len(merged.regex_traits) == 2  # has_number + has_date
        assert set(merged.get_trait_names()) == {"accuracy", "completeness", "has_number", "has_date"}

    def test_merge_rubrics_name_conflicts_across_types(self) -> None:
        """Test that name conflicts are detected across trait types."""
        from karenina.schemas.domain import LLMRubricTrait, merge_rubrics

        # Create conflict: same name in LLM and regex traits
        global_llm = LLMRubricTrait(name="quality", kind="boolean")
        global_rubric = Rubric(traits=[global_llm])

        question_regex = RegexTrait(name="quality", pattern=r"good")
        question_rubric = Rubric(regex_traits=[question_regex])

        with pytest.raises(ValueError, match="Trait name conflicts"):
            merge_rubrics(global_rubric, question_rubric)

    def test_merge_empty_rubrics_with_regex_traits(self) -> None:
        """Test merging when one rubric has only regex traits."""
        from karenina.schemas.domain import merge_rubrics

        regex_trait = RegexTrait(name="has_keyword", pattern=r"important")
        regex_only_rubric = Rubric(regex_traits=[regex_trait])

        empty_rubric = Rubric()

        merged = merge_rubrics(regex_only_rubric, empty_rubric)
        assert merged is not None
        assert len(merged.traits) == 0
        assert len(merged.regex_traits) == 1
        assert merged.get_regex_trait_names() == ["has_keyword"]
