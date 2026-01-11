"""Unit tests for CallableTrait evaluation rules.

Tests cover:
- CallableTrait creation and validation
- from_callable() classmethod (boolean and score)
- Function signature validation
- Score parameter validation
- Callable serialization (cloudpickle to bytes to base64)
- deserialize_callable() method
- evaluate() method for boolean traits
- evaluate() method for score traits
- Invert result functionality
- Error handling
"""

import base64

import cloudpickle
import pytest
from pydantic import ValidationError

from karenina.schemas.domain import CallableTrait

# =============================================================================
# CallableTrait Creation Tests
# =============================================================================


@pytest.mark.unit
def test_callable_trait_minimal_boolean() -> None:
    """Test CallableTrait with minimal boolean fields."""
    def func(text):
        return len(text) > 10
    code = cloudpickle.dumps(func)

    trait = CallableTrait(
        name="min_length",
        kind="boolean",
        callable_code=code,
        higher_is_better=True,
    )

    assert trait.name == "min_length"
    assert trait.kind == "boolean"
    assert trait.invert_result is False
    assert trait.description is None
    assert trait.min_score is None
    assert trait.max_score is None


@pytest.mark.unit
def test_callable_trait_minimal_score() -> None:
    """Test CallableTrait with minimal score fields."""
    def func(text):
        return min(len(text), 100)
    code = cloudpickle.dumps(func)

    trait = CallableTrait(
        name="length_score",
        kind="score",
        callable_code=code,
        min_score=0,
        max_score=100,
        higher_is_better=True,
    )

    assert trait.name == "length_score"
    assert trait.kind == "score"
    assert trait.min_score == 0
    assert trait.max_score == 100


@pytest.mark.unit
def test_callable_trait_with_all_fields() -> None:
    """Test CallableTrait with all fields."""
    def func(text):
        return len(text.split())
    code = cloudpickle.dumps(func)

    trait = CallableTrait(
        name="word_count",
        kind="score",
        callable_code=code,
        min_score=0,
        max_score=1000,
        invert_result=False,
        higher_is_better=True,
        description="Counts words in the text",
    )

    assert trait.name == "word_count"
    assert trait.description == "Counts words in the text"


@pytest.mark.unit
def test_callable_trait_default_invert_result() -> None:
    """Test invert_result defaults to False."""
    def func(text):
        return True
    code = cloudpickle.dumps(func)

    trait = CallableTrait(
        name="always_true",
        kind="boolean",
        callable_code=code,
        higher_is_better=True,
    )

    assert trait.invert_result is False


@pytest.mark.unit
def test_callable_trait_extra_fields_forbidden() -> None:
    """Test that extra fields are rejected."""
    def func(text):
        return True
    code = cloudpickle.dumps(func)

    with pytest.raises(ValidationError):
        CallableTrait(
            name="test",
            kind="boolean",
            callable_code=code,
            higher_is_better=True,
            extra_field="not_allowed",  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_callable_trait_kind_must_be_valid() -> None:
    """Test that kind must be 'boolean' or 'score'."""
    def func(text):
        return True
    code = cloudpickle.dumps(func)

    with pytest.raises(ValidationError):
        CallableTrait(
            name="test",
            kind="invalid",  # type: ignore[arg-type]
            callable_code=code,
            higher_is_better=True,
        )


# =============================================================================
# callable_code Serialization Tests
# =============================================================================


@pytest.mark.unit
def test_callable_trait_code_from_bytes() -> None:
    """Test callable_code can be provided as bytes."""
    def func(text):
        return "keyword" in text
    code = cloudpickle.dumps(func)

    trait = CallableTrait(
        name="has_keyword",
        kind="boolean",
        callable_code=code,
        higher_is_better=True,
    )

    assert isinstance(trait.callable_code, bytes)


@pytest.mark.unit
def test_callable_trait_code_from_base64_string() -> None:
    """Test callable_code can be provided as base64 string."""
    def func(text):
        return "test" in text
    code = cloudpickle.dumps(func)
    code_b64 = base64.b64encode(code).decode("ascii")

    trait = CallableTrait(
        name="has_test",
        kind="boolean",
        callable_code=code_b64,
        higher_is_better=True,
    )

    # Should deserialize to bytes internally
    assert isinstance(trait.callable_code, bytes)


@pytest.mark.unit
def test_callable_trait_serializes_to_base64() -> None:
    """Test that callable_code serializes to base64 string for JSON."""
    def func(text):
        return len(text)
    code = cloudpickle.dumps(func)

    trait = CallableTrait(
        name="length",
        kind="boolean",
        callable_code=code,
        higher_is_better=True,
    )

    # model_dump with serialization should give base64 string
    data = trait.model_dump()
    assert isinstance(data["callable_code"], str)

    # Should be valid base64
    decoded = base64.b64decode(data["callable_code"])
    assert decoded == code


@pytest.mark.unit
def test_callable_trait_invalid_code_type_raises_error() -> None:
    """Test that invalid callable_code type raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        CallableTrait(
            name="test",
            kind="boolean",
            callable_code=12345,  # type: ignore[arg-type]
            higher_is_better=True,
        )

    assert "callable_code must be bytes or base64 string" in str(exc_info.value)


# =============================================================================
# from_callable() Classmethod Tests
# =============================================================================


@pytest.mark.unit
def test_from_callable_boolean_simple() -> None:
    """Test from_callable() with simple boolean function."""
    def func(text):
        return len(text) > 50

    trait = CallableTrait.from_callable(
        name="long_enough",
        func=func,
        kind="boolean",
    )

    assert trait.name == "long_enough"
    assert trait.kind == "boolean"
    assert trait.higher_is_better is True  # Default
    assert trait.invert_result is False  # Default


@pytest.mark.unit
def test_from_callable_boolean_with_description() -> None:
    """Test from_callable() with description."""
    def func(text):
        return "important" in text.lower()

    trait = CallableTrait.from_callable(
        name="has_important",
        func=func,
        kind="boolean",
        description="Checks if text contains 'important' keyword",
    )

    assert trait.description == "Checks if text contains 'important' keyword"


@pytest.mark.unit
def test_from_callable_boolean_with_invert() -> None:
    """Test from_callable() with invert_result."""
    def func(text):
        return "error" in text.lower()

    trait = CallableTrait.from_callable(
        name="no_errors",
        func=func,
        kind="boolean",
        invert_result=True,
        higher_is_better=True,
    )

    assert trait.invert_result is True


@pytest.mark.unit
def test_from_callable_score_simple() -> None:
    """Test from_callable() with score function."""
    def func(text):
        return min(len(text), 100)

    trait = CallableTrait.from_callable(
        name="length_score",
        func=func,
        kind="score",
        min_score=0,
        max_score=100,
    )

    assert trait.kind == "score"
    assert trait.min_score == 0
    assert trait.max_score == 100


@pytest.mark.unit
def test_from_callable_score_with_bounds() -> None:
    """Test from_callable() with score bounds."""
    def func(text):
        return min(len(text.split()), 10)

    trait = CallableTrait.from_callable(
        name="word_score",
        func=func,
        kind="score",
        min_score=0,
        max_score=10,
        higher_is_better=True,
    )

    # Evaluate should respect bounds
    assert trait.evaluate("one two three") == 3


@pytest.mark.unit
def test_from_callable_lower_is_better() -> None:
    """Test from_callable() with higher_is_better=False."""
    def func(text):
        return text.count("error")

    trait = CallableTrait.from_callable(
        name="error_count",
        func=func,
        kind="score",
        min_score=0,
        max_score=100,
        higher_is_better=False,  # Lower is better (fewer errors)
    )

    assert trait.higher_is_better is False


@pytest.mark.unit
def test_from_callable_function_with_closure() -> None:
    """Test from_callable() with function using closure."""
    threshold = 100

    def count_function(text: str) -> bool:
        return len(text) > threshold

    trait = CallableTrait.from_callable(
        name="check_length",
        func=count_function,
        kind="boolean",
    )

    assert trait.evaluate("x" * 101) is True
    assert trait.evaluate("x" * 99) is False


@pytest.mark.unit
def test_from_callable_lambda_with_expression() -> None:
    """Test from_callable() with complex lambda."""
    def func(text):
        return sum(1 for c in text if c.isupper())

    trait = CallableTrait.from_callable(
        name="uppercase_count",
        func=func,
        kind="score",
        min_score=0,
        max_score=100,
    )

    assert trait.evaluate("ABC") == 3
    assert trait.evaluate("abc") == 0


# =============================================================================
# Function Signature Validation Tests
# =============================================================================


@pytest.mark.unit
def test_from_callable_no_parameters_raises_error() -> None:
    """Test that function with no parameters raises ValueError."""
    def no_params() -> bool:
        return True

    with pytest.raises(ValueError) as exc_info:
        CallableTrait.from_callable(
            name="bad",
            func=no_params,
            kind="boolean",
        )

    assert "exactly one parameter" in str(exc_info.value)


@pytest.mark.unit
def test_from_callable_two_parameters_raises_error() -> None:
    """Test that function with two parameters raises ValueError."""
    def two_params(text: str, extra: int) -> bool:
        return True

    with pytest.raises(ValueError) as exc_info:
        CallableTrait.from_callable(
            name="bad",
            func=two_params,
            kind="boolean",
        )

    assert "exactly one parameter" in str(exc_info.value)


# =============================================================================
# Score Parameter Validation Tests
# =============================================================================


@pytest.mark.unit
def test_from_callable_score_without_min_max_raises_error() -> None:
    """Test that score kind without min/max scores raises ValueError."""
    def func(text):
        return len(text)

    with pytest.raises(ValueError) as exc_info:
        CallableTrait.from_callable(
            name="bad",
            func=func,
            kind="score",
        )

    assert "min_score and max_score are required" in str(exc_info.value)


@pytest.mark.unit
def test_from_callable_score_only_min_raises_error() -> None:
    """Test that score kind with only min_score raises ValueError."""
    def func(text):
        return len(text)

    with pytest.raises(ValueError) as exc_info:
        CallableTrait.from_callable(
            name="bad",
            func=func,
            kind="score",
            min_score=0,
        )

    assert "min_score and max_score are required" in str(exc_info.value)


@pytest.mark.unit
def test_from_callable_score_only_max_raises_error() -> None:
    """Test that score kind with only max_score raises ValueError."""
    def func(text):
        return len(text)

    with pytest.raises(ValueError) as exc_info:
        CallableTrait.from_callable(
            name="bad",
            func=func,
            kind="score",
            max_score=100,
        )

    assert "min_score and max_score are required" in str(exc_info.value)


@pytest.mark.unit
def test_from_callable_min_equals_max_raises_error() -> None:
    """Test that min_score == max_score raises ValueError."""
    def func(text):
        return 50

    with pytest.raises(ValueError) as exc_info:
        CallableTrait.from_callable(
            name="bad",
            func=func,
            kind="score",
            min_score=50,
            max_score=50,
        )

    assert "must be less than max_score" in str(exc_info.value)


@pytest.mark.unit
def test_from_callable_min_greater_than_max_raises_error() -> None:
    """Test that min_score > max_score raises ValueError."""
    def func(text):
        return 50

    with pytest.raises(ValueError) as exc_info:
        CallableTrait.from_callable(
            name="bad",
            func=func,
            kind="score",
            min_score=100,
            max_score=50,
        )

    assert "must be less than max_score" in str(exc_info.value)


@pytest.mark.unit
def test_from_callable_boolean_with_scores_raises_error() -> None:
    """Test that boolean kind with scores raises ValueError."""
    def func(text):
        return True

    with pytest.raises(ValueError) as exc_info:
        CallableTrait.from_callable(
            name="bad",
            func=func,
            kind="boolean",
            min_score=0,
            max_score=1,
        )

    assert "min_score and max_score should not be set when kind='boolean'" in str(exc_info.value)


# =============================================================================
# deserialize_callable() Tests
# =============================================================================


@pytest.mark.unit
def test_deserialize_callable_returns_function() -> None:
    """Test deserialize_callable() returns the original function."""
    def original_func(text):
        return text.count("word")

    trait = CallableTrait.from_callable(
        name="word_counter",
        func=original_func,
        kind="boolean",
    )

    restored_func = trait.deserialize_callable()

    assert restored_func("word word word") == 3


@pytest.mark.unit
def test_deserialize_callable_preserves_function_logic() -> None:
    """Test deserialized function preserves original logic."""
    def complex_func(text: str) -> int:
        words = text.split()
        return len([w for w in words if len(w) > 5])

    trait = CallableTrait.from_callable(
        name="long_word_count",
        func=complex_func,
        kind="score",
        min_score=0,
        max_score=100,
    )

    restored = trait.deserialize_callable()
    # "The quick brown fox jumps" - "quick"=5, "brown"=5, "jumps"=5 -> none > 5
    assert restored("The quick brown fox jumps") == 0
    # "The jumping fox" - "jumping"=7 > 5
    assert restored("The jumping fox") == 1


@pytest.mark.unit
def test_deserialize_callable_with_closure() -> None:
    """Test deserializing function with closure variables."""
    multiplier = 2

    def multiply(text: str) -> int:
        return len(text) * multiplier

    trait = CallableTrait.from_callable(
        name="double_length",
        func=multiply,
        kind="score",
        min_score=0,
        max_score=1000,
    )

    restored = trait.deserialize_callable()
    assert restored("hello") == 10  # len("hello") = 5, 5 * 2 = 10


@pytest.mark.unit
def test_deserialize_callable_invalid_code_raises_error() -> None:
    """Test that invalid callable code raises RuntimeError."""
    # Create invalid "bytes" (not actually pickled data)
    invalid_code = b"not a pickle"

    trait = CallableTrait(
        name="bad",
        kind="boolean",
        callable_code=invalid_code,
        higher_is_better=True,
    )

    with pytest.raises(RuntimeError) as exc_info:
        trait.deserialize_callable()

    assert "Failed to deserialize" in str(exc_info.value)


# =============================================================================
# evaluate() Boolean Trait Tests
# =============================================================================


@pytest.mark.unit
def test_evaluate_boolean_returns_true() -> None:
    """Test evaluate() with boolean function returning True."""
    trait = CallableTrait.from_callable(
        name="is_long",
        func=lambda text: len(text) > 10,
        kind="boolean",
    )

    assert trait.evaluate("This is a long text") is True
    assert trait.evaluate("Short") is False


@pytest.mark.unit
def test_evaluate_boolean_returns_false() -> None:
    """Test evaluate() with boolean function returning False."""
    trait = CallableTrait.from_callable(
        name="has_keyword",
        func=lambda text: "forbidden" in text.lower(),
        kind="boolean",
    )

    assert trait.evaluate("This is safe") is False
    assert trait.evaluate("This is forbidden") is True


@pytest.mark.unit
def test_evaluate_boolean_invert_result() -> None:
    """Test evaluate() with invert_result=True for boolean."""
    trait = CallableTrait.from_callable(
        name="has_long_word",
        func=lambda text: any(len(w) > 5 for w in text.split()),
        kind="boolean",
        invert_result=True,
    )

    # Function returns True when any word is long (>5 chars)
    # Inverted: True becomes False, False becomes True
    assert trait.evaluate("hi there") is True  # No long words -> False -> inverted to True
    assert trait.evaluate("hi there verylongword") is False  # Has long word -> True -> inverted to False


@pytest.mark.unit
def test_evaluate_boolean_with_complex_logic() -> None:
    """Test evaluate() with complex boolean logic."""
    def has_both_keywords(text: str) -> bool:
        lower = text.lower()
        return "apple" in lower and "banana" in lower

    trait = CallableTrait.from_callable(
        name="has_both_fruits",
        func=has_both_keywords,
        kind="boolean",
    )

    assert trait.evaluate("I like apple and banana") is True
    assert trait.evaluate("I like apple") is False


@pytest.mark.unit
def test_evaluate_boolean_empty_string() -> None:
    """Test evaluate() boolean with empty string."""
    trait = CallableTrait.from_callable(
        name="is_nonempty",
        func=lambda text: len(text) > 0,
        kind="boolean",
    )

    assert trait.evaluate("") is False
    assert trait.evaluate("x") is True


@pytest.mark.unit
def test_evaluate_boolean_with_newlines() -> None:
    """Test evaluate() boolean with multiline text."""
    trait = CallableTrait.from_callable(
        name="has_multiple_lines",
        func=lambda text: text.count("\n") >= 2,
        kind="boolean",
    )

    assert trait.evaluate("Line 1\nLine 2\nLine 3") is True
    assert trait.evaluate("Single line") is False


@pytest.mark.unit
def test_evaluate_boolean_case_insensitive_check() -> None:
    """Test evaluate() boolean with case insensitive logic."""
    trait = CallableTrait.from_callable(
        name="has_error",
        func=lambda text: "error" in text.lower(),
        kind="boolean",
    )

    assert trait.evaluate("ERROR occurred") is True
    assert trait.evaluate("Error occurred") is True
    assert trait.evaluate("No issue") is False


@pytest.mark.unit
def test_evaluate_boolean_regex_like() -> None:
    """Test evaluate() boolean with regex-like pattern matching."""
    import re

    trait = CallableTrait.from_callable(
        name="matches_pattern",
        func=lambda text: bool(re.search(r"\d{4}", text)),
        kind="boolean",
    )

    assert trait.evaluate("Code: 1234") is True
    assert trait.evaluate("Code: 123") is False


# =============================================================================
# evaluate() Score Trait Tests
# =============================================================================


@pytest.mark.unit
def test_evaluate_score_returns_int() -> None:
    """Test evaluate() with score function returning int."""
    trait = CallableTrait.from_callable(
        name="length",
        func=lambda text: len(text),
        kind="score",
        min_score=0,
        max_score=1000,
    )

    assert trait.evaluate("hello") == 5
    assert trait.evaluate("") == 0


@pytest.mark.unit
def test_evaluate_score_returns_float_converts_to_int() -> None:
    """Test evaluate() converts float return to int."""
    trait = CallableTrait.from_callable(
        name="ratio",
        func=lambda text: len(text) / 2,
        kind="score",
        min_score=0,
        max_score=100,
    )

    result = trait.evaluate("hello")  # len=5, 5/2=2.5
    assert isinstance(result, int)
    assert result == 2


@pytest.mark.unit
def test_evaluate_score_within_bounds() -> None:
    """Test evaluate() with score within min/max bounds."""
    trait = CallableTrait.from_callable(
        name="score_1_to_10",
        func=lambda text: min(len(text), 10),
        kind="score",
        min_score=0,
        max_score=10,
    )

    assert trait.evaluate("hello") == 5
    assert trait.evaluate("hello world!") == 10


@pytest.mark.unit
def test_evaluate_score_at_lower_bound() -> None:
    """Test evaluate() with score at min_score."""
    trait = CallableTrait.from_callable(
        name="non_negative",
        func=lambda text: max(len(text), 0),
        kind="score",
        min_score=0,
        max_score=100,
    )

    assert trait.evaluate("") == 0


@pytest.mark.unit
def test_evaluate_score_at_upper_bound() -> None:
    """Test evaluate() with score at max_score."""
    trait = CallableTrait.from_callable(
        name="max_100",
        func=lambda text: min(len(text), 100),
        kind="score",
        min_score=0,
        max_score=100,
    )

    assert trait.evaluate("x" * 200) == 100


@pytest.mark.unit
def test_evaluate_score_complex_calculation() -> None:
    """Test evaluate() with complex score calculation."""
    def calculate_score(text: str) -> int:
        words = text.split()
        avg_length = sum(len(w) for w in words) / len(words) if words else 0
        return int(avg_length * 10)

    trait = CallableTrait.from_callable(
        name="avg_length_score",
        func=calculate_score,
        kind="score",
        min_score=0,
        max_score=100,
    )

    # "hi there": avg length = (2 + 5) / 2 = 3.5, 3.5 * 10 = 35
    assert trait.evaluate("hi there") == 35


@pytest.mark.unit
def test_evaluate_score_word_count() -> None:
    """Test evaluate() with word count score."""
    trait = CallableTrait.from_callable(
        name="word_count_score",
        func=lambda text: len(text.split()),
        kind="score",
        min_score=0,
        max_score=100,
    )

    assert trait.evaluate("one two three four") == 4
    assert trait.evaluate("single") == 1


@pytest.mark.unit
def test_evaluate_score_char_count() -> None:
    """Test evaluate() with character count score."""
    trait = CallableTrait.from_callable(
        name="char_count_score",
        func=lambda text: len(text),
        kind="score",
        min_score=0,
        max_score=1000,
    )

    assert trait.evaluate("hello") == 5


# =============================================================================
# Error Handling in evaluate() Tests
# =============================================================================


@pytest.mark.unit
def test_evaluate_boolean_returns_non_bool_raises_error() -> None:
    """Test that boolean trait returning non-bool raises RuntimeError."""
    trait = CallableTrait.from_callable(
        name="bad",
        func=lambda text: len(text),  # Returns int, not bool
        kind="boolean",
    )

    with pytest.raises(RuntimeError) as exc_info:
        trait.evaluate("test")

    assert "Failed to evaluate" in str(exc_info.value)
    assert "must return bool" in str(exc_info.value)


@pytest.mark.unit
def test_evaluate_score_returns_non_numeric_raises_error() -> None:
    """Test that score trait returning non-numeric raises RuntimeError."""
    trait = CallableTrait.from_callable(
        name="bad",
        func=lambda text: "not a number",  # type: ignore[return-value]
        kind="score",
        min_score=0,
        max_score=10,
    )

    with pytest.raises(RuntimeError) as exc_info:
        trait.evaluate("test")

    assert "Failed to evaluate" in str(exc_info.value)
    assert "must return int or float" in str(exc_info.value)


@pytest.mark.unit
def test_evaluate_score_below_min_raises_error() -> None:
    """Test that score below min_score raises RuntimeError."""
    trait = CallableTrait.from_callable(
        name="bad",
        func=lambda text: -1,
        kind="score",
        min_score=0,
        max_score=10,
    )

    with pytest.raises(RuntimeError) as exc_info:
        trait.evaluate("test")

    assert "Failed to evaluate" in str(exc_info.value)
    assert "below minimum" in str(exc_info.value)


@pytest.mark.unit
def test_evaluate_score_above_max_raises_error() -> None:
    """Test that score above max_score raises RuntimeError."""
    trait = CallableTrait.from_callable(
        name="bad",
        func=lambda text: 15,
        kind="score",
        min_score=0,
        max_score=10,
    )

    with pytest.raises(RuntimeError) as exc_info:
        trait.evaluate("test")

    assert "Failed to evaluate" in str(exc_info.value)
    assert "above maximum" in str(exc_info.value)


@pytest.mark.unit
def test_evaluate_runtime_error_propagates() -> None:
    """Test that runtime errors in callable are wrapped in RuntimeError."""
    def buggy_func(text: str) -> int:
        raise ValueError("Something went wrong!")

    trait = CallableTrait.from_callable(
        name="buggy",
        func=buggy_func,
        kind="score",
        min_score=0,
        max_score=10,
    )

    with pytest.raises(RuntimeError) as exc_info:
        trait.evaluate("test")

    assert "Failed to evaluate" in str(exc_info.value)


# =============================================================================
# higher_is_better Tests
# =============================================================================


@pytest.mark.unit
def test_callable_trait_higher_is_better_true() -> None:
    """Test CallableTrait with higher_is_better=True."""
    trait = CallableTrait.from_callable(
        name="more_is_better",
        func=lambda text: len(text),
        kind="score",
        min_score=0,
        max_score=100,
        higher_is_better=True,
    )

    assert trait.higher_is_better is True


@pytest.mark.unit
def test_callable_trait_higher_is_better_false() -> None:
    """Test CallableTrait with higher_is_better=False."""
    trait = CallableTrait.from_callable(
        name="less_is_better",
        func=lambda text: text.count("error"),
        kind="score",
        min_score=0,
        max_score=100,
        higher_is_better=False,
    )

    assert trait.higher_is_better is False


@pytest.mark.unit
def test_callable_trait_default_higher_is_better() -> None:
    """Test that higher_is_better defaults to True."""
    trait = CallableTrait.from_callable(
        name="default",
        func=lambda text: True,
        kind="boolean",
    )

    assert trait.higher_is_better is True


# =============================================================================
# Round-trip Serialization Tests
# =============================================================================


@pytest.mark.unit
def test_callable_trait_roundtrip() -> None:
    """Test that CallableTrait survives serialize/deserialize roundtrip."""
    def original_func(text: str) -> bool:
        return "success" in text.lower()

    original_trait = CallableTrait.from_callable(
        name="has_success",
        func=original_func,
        kind="boolean",
    )

    # Serialize to dict (simulates JSON export)
    data = original_trait.model_dump()

    # Deserialize
    restored_trait = CallableTrait(**data)

    # Function should work after roundtrip
    assert restored_trait.evaluate("SUCCESS!") is True
    assert restored_trait.evaluate("FAILED!") is False


@pytest.mark.unit
def test_callable_trait_json_roundtrip() -> None:
    """Test CallableTrait through JSON serialize/deserialize."""
    def score_func(text: str) -> int:
        return len(text.split())

    original = CallableTrait.from_callable(
        name="word_count",
        func=score_func,
        kind="score",
        min_score=0,
        max_score=100,
    )

    # Serialize to JSON
    json_str = original.model_dump_json()

    # Deserialize from JSON
    restored = CallableTrait.model_validate_json(json_str)

    # Should evaluate correctly
    assert restored.evaluate("one two three") == 3


@pytest.mark.unit
def test_callable_trait_description_preserved() -> None:
    """Test that description is preserved through roundtrip."""
    trait = CallableTrait.from_callable(
        name="test",
        func=lambda text: True,
        kind="boolean",
        description="This is a test trait",
    )

    data = trait.model_dump()
    restored = CallableTrait(**data)

    assert restored.description == "This is a test trait"


@pytest.mark.unit
def test_callable_trait_invert_result_preserved() -> None:
    """Test that invert_result is preserved through roundtrip."""
    trait = CallableTrait.from_callable(
        name="test",
        func=lambda text: True,
        kind="boolean",
        invert_result=True,
    )

    data = trait.model_dump()
    restored = CallableTrait(**data)

    assert restored.invert_result is True


# =============================================================================
# Legacy Defaults Tests
# =============================================================================


@pytest.mark.unit
def test_callable_trait_legacy_default_higher_is_better() -> None:
    """Test legacy data gets higher_is_better=True default."""
    # When loading from dict with None or missing higher_is_better
    trait = CallableTrait(
        name="test",
        kind="boolean",
        callable_code=cloudpickle.dumps(lambda t: True),
        higher_is_better=None,  # type: ignore[arg-type]
    )

    # Should default to True for legacy compatibility
    assert trait.higher_is_better is True
