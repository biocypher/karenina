"""Tests for CallableTrait functionality."""

import pytest
from pydantic import ValidationError

from karenina.schemas.domain import CallableTrait, Rubric


class TestCallableTraitCreation:
    """Test CallableTrait creation and validation."""

    def test_callable_trait_from_callable_boolean(self) -> None:
        """Test creating boolean callable trait from function."""

        def is_numeric(text: str) -> bool:
            return text.strip().isdigit()

        trait = CallableTrait.from_callable(
            name="is_numeric",
            func=is_numeric,
            kind="boolean",
            description="Check if text is numeric",
        )

        assert trait.name == "is_numeric"
        assert trait.description == "Check if text is numeric"
        assert trait.kind == "boolean"
        assert trait.min_score is None
        assert trait.max_score is None
        assert trait.invert_result is False
        assert trait.callable_code is not None

    def test_callable_trait_from_callable_score(self) -> None:
        """Test creating score callable trait from function."""

        def word_count_score(text: str) -> int:
            """Return word count as a score (1-5 scale)."""
            count = len(text.split())
            if count >= 100:
                return 5
            elif count >= 50:
                return 4
            elif count >= 20:
                return 3
            elif count >= 10:
                return 2
            else:
                return 1

        trait = CallableTrait.from_callable(
            name="word_count",
            func=word_count_score,
            kind="score",
            min_score=1,
            max_score=5,
            description="Word count score",
        )

        assert trait.name == "word_count"
        assert trait.kind == "score"
        assert trait.min_score == 1
        assert trait.max_score == 5

    def test_callable_trait_from_lambda_boolean(self) -> None:
        """Test creating boolean callable trait from lambda."""
        trait = CallableTrait.from_callable(
            name="has_long_words",
            func=lambda text: any(len(word) > 10 for word in text.split()),
            kind="boolean",
        )

        assert trait.name == "has_long_words"
        assert trait.kind == "boolean"

    def test_callable_trait_from_lambda_score(self) -> None:
        """Test creating score callable trait from lambda."""
        trait = CallableTrait.from_callable(
            name="sentence_count",
            func=lambda text: min(5, max(1, text.count(".") + 1)),
            kind="score",
            min_score=1,
            max_score=5,
        )

        assert trait.name == "sentence_count"
        assert trait.kind == "score"

    def test_callable_trait_invert_result(self) -> None:
        """Test creating callable trait with result inversion."""

        def is_empty(text: str) -> bool:
            return len(text.strip()) == 0

        trait = CallableTrait.from_callable(
            name="has_content",
            func=is_empty,
            kind="boolean",
            invert_result=True,
        )

        assert trait.invert_result is True

    def test_callable_trait_score_missing_min_max(self) -> None:
        """Test that score kind requires min/max scores."""
        with pytest.raises(ValueError, match="min_score and max_score are required"):
            CallableTrait.from_callable(
                name="test",
                func=lambda _: 5,
                kind="score",
                # Missing min_score and max_score
            )

    def test_callable_trait_score_invalid_range(self) -> None:
        """Test that min_score must be less than max_score."""
        with pytest.raises(ValueError, match="min_score .* must be less than max_score"):
            CallableTrait.from_callable(
                name="test",
                func=lambda _: 5,
                kind="score",
                min_score=5,
                max_score=1,  # Invalid: max < min
            )

    def test_callable_trait_boolean_with_scores_error(self) -> None:
        """Test that boolean kind rejects min/max scores."""
        with pytest.raises(ValueError, match="min_score and max_score should not be set"):
            CallableTrait.from_callable(
                name="test",
                func=lambda _: True,
                kind="boolean",
                min_score=1,
                max_score=5,  # These are not allowed for boolean
            )

    def test_empty_name_validation(self) -> None:
        """Test that empty names are rejected."""
        with pytest.raises(ValidationError):
            CallableTrait.from_callable(
                name="",
                func=lambda _: True,
                kind="boolean",
            )


class TestCallableTraitEvaluation:
    """Test CallableTrait evaluation."""

    def test_boolean_callable_evaluation(self) -> None:
        """Test evaluation of boolean callable trait."""

        def contains_keyword(text: str) -> bool:
            return "important" in text.lower()

        trait = CallableTrait.from_callable(
            name="has_keyword",
            func=contains_keyword,
            kind="boolean",
        )

        assert trait.evaluate("This is IMPORTANT information") is True
        assert trait.evaluate("This is not relevant") is False

    def test_score_callable_evaluation(self) -> None:
        """Test evaluation of score callable trait."""

        def word_count_score(text: str) -> int:
            count = len(text.split())
            if count >= 100:
                return 5
            elif count >= 50:
                return 4
            elif count >= 20:
                return 3
            elif count >= 10:
                return 2
            else:
                return 1

        trait = CallableTrait.from_callable(
            name="word_count",
            func=word_count_score,
            kind="score",
            min_score=1,
            max_score=5,
        )

        assert trait.evaluate("short") == 1
        assert trait.evaluate(" ".join(["word"] * 15)) == 2
        assert trait.evaluate(" ".join(["word"] * 25)) == 3
        assert trait.evaluate(" ".join(["word"] * 60)) == 4
        assert trait.evaluate(" ".join(["word"] * 120)) == 5

    def test_boolean_evaluation_with_invert(self) -> None:
        """Test boolean evaluation with result inversion."""

        def is_empty(text: str) -> bool:
            return len(text.strip()) == 0

        trait = CallableTrait.from_callable(
            name="has_content",
            func=is_empty,
            kind="boolean",
            invert_result=True,
        )

        assert trait.evaluate("Hello") is True  # Not empty -> True (inverted)
        assert trait.evaluate("") is False  # Empty -> False (inverted)
        assert trait.evaluate("   ") is False  # Whitespace only -> False (inverted)

    def test_score_evaluation_no_invert(self) -> None:
        """Test that score evaluation does not support inversion (only boolean does)."""

        def word_count(text: str) -> int:
            count = len(text.split())
            if count >= 100:
                return 5
            elif count >= 50:
                return 4
            elif count >= 20:
                return 3
            elif count >= 10:
                return 2
            else:
                return 1

        # invert_result is ignored for score kind
        trait = CallableTrait.from_callable(
            name="word_count",
            func=word_count,
            kind="score",
            min_score=1,
            max_score=5,
            invert_result=False,  # Only for boolean kind
        )

        # No inversion happens for scores
        assert trait.evaluate("short") == 1
        assert trait.evaluate(" ".join(["word"] * 15)) == 2
        assert trait.evaluate(" ".join(["word"] * 60)) == 4

    def test_evaluation_with_closure(self) -> None:
        """Test that closures are properly serialized and evaluated."""
        threshold = 50

        def above_threshold(text: str) -> bool:
            return len(text.split()) > threshold

        trait = CallableTrait.from_callable(
            name="long_text",
            func=above_threshold,
            kind="boolean",
        )

        assert trait.evaluate(" ".join(["word"] * 60)) is True
        assert trait.evaluate(" ".join(["word"] * 30)) is False

    def test_evaluation_with_complex_logic(self) -> None:
        """Test callable with complex validation logic."""

        def citation_score(text: str) -> int:
            """Score based on number and quality of citations."""
            import re

            # Count different citation formats
            bracketed = len(re.findall(r"\[\d+\]", text))
            parenthetical = len(re.findall(r"\([A-Z][a-z]+,\s*\d{4}\)", text))
            footnotes = len(re.findall(r"\^\d+", text))

            total = bracketed + parenthetical + footnotes

            if total >= 10:
                return 5
            elif total >= 5:
                return 4
            elif total >= 3:
                return 3
            elif total >= 1:
                return 2
            else:
                return 1

        trait = CallableTrait.from_callable(
            name="citation_quality",
            func=citation_score,
            kind="score",
            min_score=1,
            max_score=5,
        )

        assert trait.evaluate("No citations here") == 1
        assert trait.evaluate("One citation [1] here") == 2
        assert trait.evaluate("[1] and [2] and [3] citations") == 3
        assert trait.evaluate("[1][2][3][4][5] five citations") == 4

    def test_evaluation_type_mismatch_error(self) -> None:
        """Test error when callable returns wrong type for boolean kind."""

        def wrong_type(text: str) -> str:
            return "not a boolean"

        trait = CallableTrait.from_callable(
            name="test",
            func=wrong_type,
            kind="boolean",
        )

        with pytest.raises(RuntimeError, match="must return bool"):
            trait.evaluate("test")

    def test_evaluation_score_out_of_range_error(self) -> None:
        """Test error when callable returns score outside min/max range."""

        def bad_score(text: str) -> int:
            return 10  # Outside 1-5 range

        trait = CallableTrait.from_callable(
            name="test",
            func=bad_score,
            kind="score",
            min_score=1,
            max_score=5,
        )

        with pytest.raises(RuntimeError, match="is above maximum"):
            trait.evaluate("test")

    def test_evaluation_runtime_error_handling(self) -> None:
        """Test handling of runtime errors during evaluation."""

        def error_func(text: str) -> bool:
            raise ValueError("Something went wrong")

        trait = CallableTrait.from_callable(
            name="error_test",
            func=error_func,
            kind="boolean",
        )

        with pytest.raises(RuntimeError, match="Failed to evaluate callable trait 'error_test'"):
            trait.evaluate("test")


class TestCallableTraitSerialization:
    """Test CallableTrait serialization and deserialization."""

    def test_serialize_deserialize_function(self) -> None:
        """Test that functions are properly serialized and deserialized."""

        def check_length(text: str) -> bool:
            return len(text) > 100

        trait = CallableTrait.from_callable(
            name="long_text",
            func=check_length,
            kind="boolean",
        )

        # Verify it can be evaluated after creation
        assert trait.evaluate("x" * 150) is True
        assert trait.evaluate("short") is False

    def test_serialize_deserialize_lambda(self) -> None:
        """Test that lambdas are properly serialized and deserialized."""
        trait = CallableTrait.from_callable(
            name="has_numbers",
            func=lambda text: any(c.isdigit() for c in text),
            kind="boolean",
        )

        assert trait.evaluate("abc123") is True
        assert trait.evaluate("abc") is False

    def test_serialize_deserialize_with_imports(self) -> None:
        """Test that functions with imports work after deserialization."""

        def has_email(text: str) -> bool:
            import re

            return bool(re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text))

        trait = CallableTrait.from_callable(
            name="contains_email",
            func=has_email,
            kind="boolean",
        )

        assert trait.evaluate("Contact: test@example.com") is True
        assert trait.evaluate("No email here") is False


class TestRubricWithCallableTraits:
    """Test Rubric functionality with callable traits."""

    def test_rubric_with_callable_traits(self) -> None:
        """Test creating rubric with callable traits."""
        from karenina.schemas.domain import LLMRubricTrait

        llm_trait = LLMRubricTrait(name="accuracy", description="Accurate?", kind="boolean")
        callable_trait = CallableTrait.from_callable(
            name="word_count",
            func=lambda text: len(text.split()) > 50,
            kind="boolean",
        )

        rubric = Rubric(
            llm_traits=[llm_trait],
            callable_traits=[callable_trait],
        )

        assert len(rubric.llm_traits) == 1
        assert len(rubric.callable_traits) == 1
        assert set(rubric.get_trait_names()) == {"accuracy", "word_count"}
        assert rubric.get_llm_trait_names() == ["accuracy"]
        assert rubric.get_callable_trait_names() == ["word_count"]

    def test_rubric_validation_with_callable_boolean_traits(self) -> None:
        """Test validation of evaluation results with boolean callable traits."""
        from karenina.schemas.domain import LLMRubricTrait

        llm_trait = LLMRubricTrait(name="clarity", kind="score", min_score=1, max_score=5)
        callable_trait = CallableTrait.from_callable(
            name="has_citations",
            func=lambda text: "[" in text,
            kind="boolean",
        )

        rubric = Rubric(
            llm_traits=[llm_trait],
            callable_traits=[callable_trait],
        )

        # Valid evaluation
        valid_eval = {"clarity": 4, "has_citations": True}
        assert rubric.validate_evaluation(valid_eval) is True

        # Invalid - callable boolean trait should be boolean, not int
        invalid_eval = {"clarity": 4, "has_citations": 1}
        assert rubric.validate_evaluation(invalid_eval) is False

    # Note: Detailed validation testing is handled in test_rubric_schemas.py
    # This test just verifies basic rubric creation with callable score traits
    def test_rubric_with_callable_score_trait(self) -> None:
        """Test creating rubric with score callable trait."""
        from karenina.schemas.domain import LLMRubricTrait

        llm_trait = LLMRubricTrait(name="clarity", kind="score", min_score=1, max_score=5)
        callable_trait = CallableTrait.from_callable(
            name="readability",
            func=lambda text: min(5, max(1, len(text) // 20)),
            kind="score",
            min_score=1,
            max_score=5,
        )

        rubric = Rubric(
            llm_traits=[llm_trait],
            callable_traits=[callable_trait],
        )

        assert len(rubric.llm_traits) == 1
        assert len(rubric.callable_traits) == 1
        assert rubric.get_callable_trait_names() == ["readability"]


class TestMergeRubricsWithCallableTraits:
    """Test rubric merging with callable traits."""

    def test_merge_rubrics_with_callable_traits(self) -> None:
        """Test merging rubrics that contain callable traits."""
        from karenina.schemas.domain import LLMRubricTrait, merge_rubrics

        # Global rubric with both types
        global_llm = LLMRubricTrait(name="accuracy", kind="boolean")
        global_callable = CallableTrait.from_callable(
            name="word_count",
            func=lambda text: len(text.split()) > 50,
            kind="boolean",
        )
        global_rubric = Rubric(llm_traits=[global_llm], callable_traits=[global_callable])

        # Question rubric with both types
        question_llm = LLMRubricTrait(name="completeness", kind="score", min_score=1, max_score=5)
        question_callable = CallableTrait.from_callable(
            name="citation_count",
            func=lambda text: min(5, max(1, text.count("["))),
            kind="score",
            min_score=1,
            max_score=5,
        )
        question_rubric = Rubric(llm_traits=[question_llm], callable_traits=[question_callable])

        # Merge rubrics
        merged = merge_rubrics(global_rubric, question_rubric)

        assert merged is not None
        assert len(merged.traits) == 2  # accuracy + completeness
        assert len(merged.callable_traits) == 2  # word_count + citation_count
        assert set(merged.get_trait_names()) == {"accuracy", "completeness", "word_count", "citation_count"}

    def test_merge_rubrics_name_conflicts_across_types(self) -> None:
        """Test that name conflicts are detected across trait types."""
        from karenina.schemas.domain import LLMRubricTrait, merge_rubrics

        # Create conflict: same name in LLM and callable traits
        global_llm = LLMRubricTrait(name="quality", kind="boolean")
        global_rubric = Rubric(llm_traits=[global_llm])

        question_callable = CallableTrait.from_callable(
            name="quality",
            func=lambda _: True,
            kind="boolean",
        )
        question_rubric = Rubric(callable_traits=[question_callable])

        with pytest.raises(ValueError, match="Trait name conflicts"):
            merge_rubrics(global_rubric, question_rubric)
