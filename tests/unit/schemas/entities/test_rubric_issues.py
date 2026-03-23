"""Tests for rubric entity fixes and feature expansions.

Issue 127: CallableRubricTrait supports boolean, score, float, and literal kinds.
"""

import cloudpickle
import pytest
from pydantic import ValidationError

from karenina.schemas.entities.rubric import CallableRubricTrait


def _pickle(func: object) -> bytes:
    """Shorthand for cloudpickle.dumps."""
    return cloudpickle.dumps(func)


def _const_str(value: str):  # noqa: ARG005
    """Return a callable that ignores its input and returns a constant string."""
    return lambda _text: value


@pytest.mark.unit
class TestCallableRubricTraitKinds:
    """Issue 127: CallableRubricTrait supports boolean, score, and literal kinds."""

    def test_boolean_kind(self) -> None:
        """kind='boolean' with bool-returning callable works."""
        trait = CallableRubricTrait.from_callable(
            name="has_citations",
            func=lambda text: "cite" in text.lower(),
            kind="boolean",
        )
        assert trait.kind == "boolean"
        assert trait.evaluate("No links here") is False
        assert trait.evaluate("Please cite your sources") is True

    def test_score_kind_int(self) -> None:
        """kind='score' with int-returning callable works."""
        trait = CallableRubricTrait.from_callable(
            name="word_count_bucket",
            func=lambda text: min(len(text.split()), 5),
            kind="score",
            min_score=0,
            max_score=5,
        )
        assert trait.kind == "score"
        result = trait.evaluate("one two three")
        assert result == 3
        assert isinstance(result, int)

    def test_score_kind_float(self) -> None:
        """kind='score' with float-returning callable preserves float precision."""
        trait = CallableRubricTrait.from_callable(
            name="density",
            func=lambda text: len(text) / max(len(text.split()), 1),
            kind="score",
            min_score=0,
            max_score=100,
        )
        result = trait.evaluate("hello world")
        assert isinstance(result, float)
        assert result == pytest.approx(5.5)

    def test_literal_kind_requires_classes(self) -> None:
        """kind='literal' without classes raises ValidationError."""
        with pytest.raises(ValidationError):
            CallableRubricTrait(
                name="tone",
                kind="literal",
                callable_code=_pickle(_const_str("neutral")),
                higher_is_better=True,
            )

    def test_literal_kind_with_classes(self) -> None:
        """kind='literal' with classes field works."""
        classes = {"formal": "Academic tone", "casual": "Conversational tone", "neutral": "Balanced tone"}
        trait = CallableRubricTrait.from_callable(
            name="tone",
            func=lambda text: "formal" if "therefore" in text else "casual",
            kind="literal",
            classes=classes,
        )
        assert trait.kind == "literal"
        assert trait.classes == classes
        assert trait.min_score == 0
        assert trait.max_score == 2  # len(classes) - 1

    def test_literal_evaluate_returns_class_index(self) -> None:
        """Literal callable returns class index (int) from the callable's string return."""
        classes = {"positive": "Good", "negative": "Bad"}
        trait = CallableRubricTrait.from_callable(
            name="sentiment",
            func=lambda text: "positive" if "good" in text.lower() else "negative",
            kind="literal",
            classes=classes,
        )
        assert trait.evaluate("This is good") == 0
        assert trait.evaluate("This is bad") == 1

    def test_literal_evaluate_invalid_class_raises(self) -> None:
        """Literal callable returning unknown class label raises ValueError."""
        classes = {"positive": "Good", "negative": "Bad"}
        trait = CallableRubricTrait.from_callable(
            name="sentiment",
            func=_const_str("unknown_class"),
            kind="literal",
            classes=classes,
        )
        with pytest.raises(RuntimeError, match="not a valid class"):
            trait.evaluate("some text")

    def test_from_callable_literal_validates_classes_required(self) -> None:
        """from_callable with kind='literal' but no classes raises ValueError."""
        with pytest.raises(ValueError, match="classes.*required"):
            CallableRubricTrait.from_callable(
                name="test",
                func=_const_str("a"),
                kind="literal",
            )

    def test_from_callable_literal_validates_min_classes(self) -> None:
        """from_callable with kind='literal' and only 1 class raises ValueError."""
        with pytest.raises(ValidationError):
            CallableRubricTrait.from_callable(
                name="test",
                func=_const_str("only"),
                kind="literal",
                classes={"only": "The only class"},
            )
