"""Tests for ManualRubricTrait functionality."""

import pytest
from pydantic import ValidationError

from karenina.schemas.rubric_class import ManualRubricTrait, Rubric


class TestManualRubricTrait:
    """Test ManualRubricTrait schema and functionality."""

    def test_regex_pattern_trait_creation(self) -> None:
        """Test creating a trait with regex pattern."""
        trait = ManualRubricTrait(
            name="contains_hello",
            description="Check if text contains 'hello'",
            pattern=r"\bhello\b",
            case_sensitive=False,
        )

        assert trait.name == "contains_hello"
        assert trait.description == "Check if text contains 'hello'"
        assert trait.pattern == r"\bhello\b"
        assert trait.callable_name is None
        assert trait.case_sensitive is False
        assert trait.invert_result is False

    def test_callable_trait_creation(self) -> None:
        """Test creating a trait with callable function."""
        trait = ManualRubricTrait(
            name="is_numeric",
            description="Check if text is numeric",
            callable_name="is_numeric_func",
            invert_result=True,
        )

        assert trait.name == "is_numeric"
        assert trait.callable_name == "is_numeric_func"
        assert trait.pattern is None
        assert trait.invert_result is True

    def test_mutually_exclusive_validation(self) -> None:
        """Test that pattern and callable_name are mutually exclusive."""
        # Both specified should fail
        with pytest.raises(ValidationError, match="Only one of 'pattern' or 'callable_name'"):
            ManualRubricTrait(
                name="test",
                pattern=r"test",
                callable_name="test_func",
            )

        # Neither specified should fail
        with pytest.raises(ValidationError, match="Either 'pattern' or 'callable_name' must be specified"):
            ManualRubricTrait(name="test")

    def test_invalid_regex_pattern(self) -> None:
        """Test validation of regex patterns."""
        with pytest.raises(ValidationError, match="Invalid regex pattern"):
            ManualRubricTrait(
                name="bad_regex",
                pattern=r"[invalid",  # Unclosed bracket
            )

    def test_empty_name_validation(self) -> None:
        """Test that empty names are rejected."""
        with pytest.raises(ValidationError):
            ManualRubricTrait(
                name="",
                pattern=r"test",
            )

    def test_regex_evaluation_case_sensitive(self) -> None:
        """Test regex evaluation with case sensitivity."""
        trait = ManualRubricTrait(
            name="contains_hello",
            pattern=r"hello",
            case_sensitive=True,
        )

        # Should match exact case
        assert trait.evaluate("Say hello world") is True
        assert trait.evaluate("Say HELLO world") is False

    def test_regex_evaluation_case_insensitive(self) -> None:
        """Test regex evaluation without case sensitivity."""
        trait = ManualRubricTrait(
            name="contains_hello",
            pattern=r"hello",
            case_sensitive=False,
        )

        # Should match any case
        assert trait.evaluate("Say hello world") is True
        assert trait.evaluate("Say HELLO world") is True
        assert trait.evaluate("Say HeLLo world") is True

    def test_regex_evaluation_with_invert(self) -> None:
        """Test regex evaluation with result inversion."""
        trait = ManualRubricTrait(
            name="not_contains_error",
            pattern=r"error",
            invert_result=True,
        )

        # Should invert the result
        assert trait.evaluate("This is good") is True  # No error -> True (inverted)
        assert trait.evaluate("This is an error") is False  # Has error -> False (inverted)

    def test_callable_evaluation(self) -> None:
        """Test evaluation using callable functions."""

        def is_numeric(text: str) -> bool:
            return text.strip().isdigit()

        trait = ManualRubricTrait(
            name="is_numeric",
            callable_name="is_numeric_func",
        )

        registry = {"is_numeric_func": is_numeric}

        assert trait.evaluate("123", registry) is True
        assert trait.evaluate("abc", registry) is False
        assert trait.evaluate("12.3", registry) is False

    def test_callable_evaluation_with_invert(self) -> None:
        """Test callable evaluation with result inversion."""

        def is_empty(text: str) -> bool:
            return len(text.strip()) == 0

        trait = ManualRubricTrait(
            name="has_content",
            callable_name="is_empty_func",
            invert_result=True,
        )

        registry = {"is_empty_func": is_empty}

        assert trait.evaluate("Hello", registry) is True  # Not empty -> True (inverted)
        assert trait.evaluate("", registry) is False  # Empty -> False (inverted)
        assert trait.evaluate("   ", registry) is False  # Whitespace only -> False (inverted)

    def test_callable_not_found_error(self) -> None:
        """Test error when callable is not found in registry."""
        trait = ManualRubricTrait(
            name="test",
            callable_name="missing_func",
        )

        with pytest.raises(RuntimeError, match="Failed to evaluate manual trait 'test'"):
            trait.evaluate("test", {})

        with pytest.raises(RuntimeError, match="Failed to evaluate manual trait 'test'"):
            trait.evaluate("test", None)

    def test_callable_invalid_return_type(self) -> None:
        """Test error when callable returns non-boolean."""

        def bad_func(text: str) -> str:
            return "not a boolean"

        trait = ManualRubricTrait(
            name="test",
            callable_name="bad_func",
        )

        registry = {"bad_func": bad_func}

        with pytest.raises(RuntimeError, match="Failed to evaluate manual trait 'test'"):
            trait.evaluate("test", registry)

    def test_evaluation_runtime_error_handling(self) -> None:
        """Test handling of runtime errors during evaluation."""

        def error_func(text: str) -> bool:
            raise ValueError("Something went wrong")

        trait = ManualRubricTrait(
            name="error_test",
            callable_name="error_func",
        )

        registry = {"error_func": error_func}

        with pytest.raises(RuntimeError, match="Failed to evaluate manual trait 'error_test'"):
            trait.evaluate("test", registry)


class TestRubricWithManualTraits:
    """Test Rubric functionality with manual traits."""

    def test_rubric_with_manual_traits(self) -> None:
        """Test creating rubric with manual traits."""
        from karenina.schemas.rubric_class import RubricTrait

        llm_trait = RubricTrait(name="accuracy", description="Accurate?", kind="boolean")
        manual_trait = ManualRubricTrait(name="has_number", pattern=r"\d+")

        rubric = Rubric(
            traits=[llm_trait],
            manual_traits=[manual_trait],
        )

        assert len(rubric.traits) == 1
        assert len(rubric.manual_traits) == 1
        assert rubric.get_trait_names() == ["accuracy", "has_number"]
        assert rubric.get_llm_trait_names() == ["accuracy"]
        assert rubric.get_manual_trait_names() == ["has_number"]

    def test_rubric_validation_with_manual_traits(self) -> None:
        """Test validation of evaluation results with manual traits."""
        from karenina.schemas.rubric_class import RubricTrait

        llm_trait = RubricTrait(name="score", kind="score", min_score=1, max_score=5)
        manual_trait = ManualRubricTrait(name="has_keyword", pattern=r"keyword")

        rubric = Rubric(
            traits=[llm_trait],
            manual_traits=[manual_trait],
        )

        # Valid evaluation
        valid_eval = {"score": 4, "has_keyword": True}
        assert rubric.validate_evaluation(valid_eval) is True

        # Invalid - manual trait should be boolean
        invalid_eval = {"score": 4, "has_keyword": "yes"}
        assert rubric.validate_evaluation(invalid_eval) is False

        # Invalid - missing trait
        incomplete_eval = {"score": 4}
        assert rubric.validate_evaluation(incomplete_eval) is False

        # Invalid - unknown trait
        extra_eval = {"score": 4, "has_keyword": True, "unknown": False}
        assert rubric.validate_evaluation(extra_eval) is False


class TestMergeRubricsWithManualTraits:
    """Test rubric merging with manual traits."""

    def test_merge_rubrics_with_manual_traits(self) -> None:
        """Test merging rubrics that contain manual traits."""
        from karenina.schemas.rubric_class import RubricTrait, merge_rubrics

        # Global rubric with both types
        global_llm = RubricTrait(name="accuracy", kind="boolean")
        global_manual = ManualRubricTrait(name="has_number", pattern=r"\d+")
        global_rubric = Rubric(traits=[global_llm], manual_traits=[global_manual])

        # Question rubric with both types
        question_llm = RubricTrait(name="completeness", kind="score")
        question_manual = ManualRubricTrait(name="has_date", pattern=r"\d{4}-\d{2}-\d{2}")
        question_rubric = Rubric(traits=[question_llm], manual_traits=[question_manual])

        # Merge rubrics
        merged = merge_rubrics(global_rubric, question_rubric)

        assert merged is not None
        assert len(merged.traits) == 2  # accuracy + completeness
        assert len(merged.manual_traits) == 2  # has_number + has_date
        assert merged.get_trait_names() == ["accuracy", "completeness", "has_number", "has_date"]

    def test_merge_rubrics_name_conflicts_across_types(self) -> None:
        """Test that name conflicts are detected across trait types."""
        from karenina.schemas.rubric_class import RubricTrait, merge_rubrics

        # Create conflict: same name in LLM and manual traits
        global_llm = RubricTrait(name="quality", kind="boolean")
        global_rubric = Rubric(traits=[global_llm])

        question_manual = ManualRubricTrait(name="quality", pattern=r"good")
        question_rubric = Rubric(manual_traits=[question_manual])

        with pytest.raises(ValueError, match="Trait name conflicts"):
            merge_rubrics(global_rubric, question_rubric)

    def test_merge_empty_rubrics_with_manual_traits(self) -> None:
        """Test merging when one rubric has only manual traits."""
        from karenina.schemas.rubric_class import merge_rubrics

        manual_trait = ManualRubricTrait(name="has_keyword", pattern=r"important")
        manual_only_rubric = Rubric(manual_traits=[manual_trait])

        empty_rubric = Rubric()

        merged = merge_rubrics(manual_only_rubric, empty_rubric)
        assert merged is not None
        assert len(merged.traits) == 0
        assert len(merged.manual_traits) == 1
        assert merged.get_manual_trait_names() == ["has_keyword"]
