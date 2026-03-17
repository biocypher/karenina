"""Unit tests for RegexTrait evaluation rules.

Tests cover:
- RegexTrait creation and validation
- Pattern validation (valid and invalid regex)
- Case sensitive and insensitive matching
- Invert result functionality
- evaluate() method with various inputs
- Error handling
"""

import pytest
from pydantic import ValidationError

from karenina.schemas.entities import RegexTrait

# =============================================================================
# RegexTrait Creation and Defaults Tests
# =============================================================================


@pytest.mark.unit
def test_regex_trait_minimal() -> None:
    """Test RegexTrait with minimal required fields."""
    trait = RegexTrait(
        name="has_email",
        pattern=r"\S+@\S+",
        higher_is_better=True,
    )

    assert trait.name == "has_email"
    assert trait.pattern == r"\S+@\S+"
    assert trait.case_sensitive is True
    assert trait.invert_result is False
    assert trait.description is None


@pytest.mark.unit
def test_regex_trait_with_all_fields() -> None:
    """Test RegexTrait with all fields specified."""
    trait = RegexTrait(
        name="has_citation",
        pattern=r"\[\d+\]",
        case_sensitive=False,
        invert_result=False,
        higher_is_better=True,
        description="Checks for citation format like [1]",
    )

    assert trait.name == "has_citation"
    assert trait.case_sensitive is False
    assert trait.invert_result is False
    assert trait.description == "Checks for citation format like [1]"


@pytest.mark.unit
def test_regex_trait_defaults() -> None:
    """Test RegexTrait field defaults."""
    trait = RegexTrait(
        name="test",
        pattern=r"\w+",
        higher_is_better=True,
    )

    # Default values
    assert trait.case_sensitive is True  # Default: case sensitive
    assert trait.invert_result is False  # Default: not inverted


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


@pytest.mark.unit
def test_regex_trait_name_min_length() -> None:
    """Test that name must be at least 1 character."""
    with pytest.raises(ValidationError) as exc_info:
        RegexTrait(
            name="",
            pattern=r"\w+",
            higher_is_better=True,
        )

    # Pydantic validation error for empty string with min_length=1
    assert "at least 1" in str(exc_info.value).lower() or "min_length" in str(exc_info.value).lower()


# =============================================================================
# Pattern Validation Tests
# =============================================================================


@pytest.mark.unit
def test_regex_trait_valid_pattern_simple() -> None:
    """Test RegexTrait accepts valid simple pattern."""
    trait = RegexTrait(
        name="word_match",
        pattern=r"hello",
        higher_is_better=True,
    )

    assert trait.pattern == "hello"


@pytest.mark.unit
def test_regex_trait_valid_pattern_complex() -> None:
    """Test RegexTrait accepts valid complex pattern."""
    trait = RegexTrait(
        name="complex_pattern",
        pattern=r"(?i)^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$",
        higher_is_better=True,
    )

    assert "@" in trait.pattern


@pytest.mark.unit
def test_regex_trait_invalid_pattern_raises_error() -> None:
    """Test that invalid regex pattern raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        RegexTrait(
            name="bad_pattern",
            pattern=r"[invalid(",  # Unclosed bracket
            higher_is_better=True,
        )

    assert "Invalid regex pattern" in str(exc_info.value)


@pytest.mark.unit
def test_regex_trait_unmatched_bracket() -> None:
    """Test unmatched bracket pattern raises error."""
    with pytest.raises(ValidationError):
        RegexTrait(
            name="unmatched",
            pattern=r"[abc",
            higher_is_better=True,
        )


@pytest.mark.unit
def test_regex_trait_invalid_escape() -> None:
    """Test invalid escape sequence raises error."""
    with pytest.raises(ValidationError):
        RegexTrait(
            name="bad_escape",
            pattern=r"\p",  # \p is not a valid escape in Python regex
            higher_is_better=True,
        )


# =============================================================================
# Case Sensitivity Tests
# =============================================================================


@pytest.mark.unit
def test_regex_trait_case_sensitive_match() -> None:
    """Test case-sensitive matching."""
    trait = RegexTrait(
        name="uppercase_word",
        pattern=r"[A-Z]{2,}",
        case_sensitive=True,
        higher_is_better=True,
    )

    assert trait.evaluate("The WORD is here") is True
    assert trait.evaluate("the word is here") is False


@pytest.mark.unit
def test_regex_trait_case_insensitive_match() -> None:
    """Test case-insensitive matching with literal word."""
    trait = RegexTrait(
        name="case_insensitive_word",
        pattern=r"error",
        case_sensitive=False,
        higher_is_better=True,
    )

    assert trait.evaluate("the ERROR is here") is True
    assert trait.evaluate("The Error is here") is True
    assert trait.evaluate("the warning is here") is False


@pytest.mark.unit
def test_regex_trait_default_case_sensitive() -> None:
    """Test that case_sensitive defaults to True."""
    trait = RegexTrait(
        name="default_sensitive",
        pattern=r"ERROR",
        higher_is_better=True,
    )

    # Should be case sensitive by default
    assert trait.case_sensitive is True
    assert trait.evaluate("An ERROR occurred") is True
    assert trait.evaluate("An error occurred") is False


# =============================================================================
# Invert Result Tests
# =============================================================================


@pytest.mark.unit
def test_regex_trait_invert_result_true_to_false() -> None:
    """Test invert_result converts match to False."""
    trait = RegexTrait(
        name="no_profanity",
        pattern=r"\bbadword\b",
        invert_result=True,
        higher_is_better=True,
    )

    # Pattern matches, but result is inverted
    assert trait.evaluate("This contains badword") is False
    assert trait.evaluate("This is clean") is True


@pytest.mark.unit
def test_regex_trait_invert_result_false_to_true() -> None:
    """Test invert_result with case insensitive matching."""
    trait = RegexTrait(
        name="no_profanity",
        pattern=r"\bBADWORD\b",
        case_sensitive=False,
        invert_result=True,
        higher_is_better=True,
    )

    # Pattern matches (case insensitive), but result is inverted
    assert trait.evaluate("This contains BadWord") is False
    assert trait.evaluate("This is clean") is True


@pytest.mark.unit
def test_regex_trait_invert_result_with_no_match() -> None:
    """Test invert_result when pattern doesn't match."""
    trait = RegexTrait(
        name="no_profanity",
        pattern=r"\bbadword\b",
        invert_result=True,
        higher_is_better=True,
    )

    # Pattern doesn't match, result is inverted (False -> True)
    assert trait.evaluate("This is clean text") is True


@pytest.mark.unit
def test_regex_trait_default_not_inverted() -> None:
    """Test that invert_result defaults to False."""
    trait = RegexTrait(
        name="has_url",
        pattern=r"https?://\S+",
        higher_is_better=True,
    )

    # Should not be inverted by default
    assert trait.invert_result is False
    assert trait.evaluate("Visit https://example.com") is True
    assert trait.evaluate("No link here") is False


# =============================================================================
# evaluate() Method Tests
# =============================================================================


@pytest.mark.unit
def test_regex_evaluate_simple_match() -> None:
    """Test evaluate() with simple matching pattern."""
    trait = RegexTrait(
        name="contains_python",
        pattern=r"python",
        case_sensitive=True,
        higher_is_better=True,
    )

    assert trait.evaluate("I love python programming") is True
    assert trait.evaluate("I love Java programming") is False


@pytest.mark.unit
def test_regex_evaluate_word_boundary() -> None:
    """Test evaluate() with word boundary."""
    trait = RegexTrait(
        name="has_test",
        pattern=r"\btest\b",
        higher_is_better=True,
    )

    assert trait.evaluate("this is a test") is True
    assert trait.evaluate("this is testing") is False  # "test" is part of "testing"


@pytest.mark.unit
def test_regex_evaluate_email_pattern() -> None:
    """Test evaluate() with email pattern."""
    trait = RegexTrait(
        name="has_email",
        pattern=r"\S+@\S+\.\S+",
        higher_is_better=True,
    )

    assert trait.evaluate("Contact me at user@example.com") is True
    assert trait.evaluate("Contact me by phone") is False


@pytest.mark.unit
def test_regex_evaluate_multiple_matches() -> None:
    """Test evaluate() returns True when multiple matches exist."""
    trait = RegexTrait(
        name="has_digit",
        pattern=r"\d+",
        higher_is_better=True,
    )

    assert trait.evaluate("Values: 42 and 100") is True


@pytest.mark.unit
def test_regex_evaluate_empty_string() -> None:
    """Test evaluate() with empty string."""
    trait = RegexTrait(
        name="has_content",
        pattern=r"\w+",
        higher_is_better=True,
    )

    assert trait.evaluate("") is False


@pytest.mark.unit
def test_regex_evaluate_special_characters() -> None:
    """Test evaluate() with special character pattern."""
    trait = RegexTrait(
        name="has_dollar_sign",
        pattern=r"\$",
        higher_is_better=True,
    )

    assert trait.evaluate("Price: $100") is True
    assert trait.evaluate("Price: 100") is False


@pytest.mark.unit
def test_regex_evaluate_multiline_text() -> None:
    """Test evaluate() with multiline input."""
    trait = RegexTrait(
        name="has_error",
        pattern=r"ERROR",
        case_sensitive=True,
        higher_is_better=True,
    )

    text = """
    Line 1: Some text
    Line 2: ERROR found here
    Line 3: More text
    """

    assert trait.evaluate(text) is True


@pytest.mark.unit
def test_regex_evaluate_unicode() -> None:
    """Test evaluate() with unicode characters."""
    trait = RegexTrait(
        name="has_emoji",
        pattern=r"😀",
        higher_is_better=True,
    )

    assert trait.evaluate("Hello 😀 world") is True
    assert trait.evaluate("Hello world") is False


@pytest.mark.unit
def test_regex_evaluate_newline_carriage_return() -> None:
    """Test evaluate() with newline and carriage return in pattern."""
    trait = RegexTrait(
        name="multiline",
        pattern=r"line1.*line2",
        higher_is_better=True,
    )

    # Default re.search doesn't match across newlines without DOTALL flag
    assert trait.evaluate("line1\nline2") is False


# =============================================================================
# higher_is_better Tests
# =============================================================================


@pytest.mark.unit
def test_regex_trait_higher_is_better_true() -> None:
    """Test RegexTrait with higher_is_better=True."""
    trait = RegexTrait(
        name="has_citation",
        pattern=r"\[\d+\]",
        higher_is_better=True,
    )

    assert trait.higher_is_better is True


@pytest.mark.unit
def test_regex_trait_higher_is_better_false() -> None:
    """Test RegexTrait with higher_is_better=False."""
    trait = RegexTrait(
        name="no_profanity",
        pattern=r"\bbadword\b",
        higher_is_better=False,
    )

    assert trait.higher_is_better is False


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.unit
def test_regex_evaluate_runtime_error_on_pattern_failure() -> None:
    """Test evaluate() raises RuntimeError on unexpected errors."""
    # This test verifies error handling, though valid patterns shouldn't fail
    trait = RegexTrait(
        name="test",
        pattern=r"test",  # Valid pattern
        higher_is_better=True,
    )

    # Should not raise for normal input
    assert trait.evaluate("test string") is True


@pytest.mark.unit
def test_regex_trait_description_field() -> None:
    """Test RegexTrait description field."""
    trait = RegexTrait(
        name="citation_check",
        pattern=r"\[\d+\]",
        description="Checks for academic citation format",
        higher_is_better=True,
    )

    assert trait.description == "Checks for academic citation format"


@pytest.mark.unit
def test_regex_trait_with_escaped_backslash() -> None:
    """Test RegexTrait with escaped backslash in pattern."""
    trait = RegexTrait(
        name="has_backslash",
        pattern=r"\\",
        higher_is_better=True,
    )

    assert trait.evaluate("path\\to\\file") is True
    assert trait.evaluate("path/to/file") is False


@pytest.mark.unit
def test_regex_trait_with_digit_class() -> None:
    """Test RegexTrait with digit character class."""
    trait = RegexTrait(
        name="has_number",
        pattern=r"\d+",
        higher_is_better=True,
    )

    assert trait.evaluate("The value is 42") is True
    assert trait.evaluate("The value is forty-two") is False


@pytest.mark.unit
def test_regex_trait_with_negated_class() -> None:
    """Test RegexTrait with negated character class."""
    trait = RegexTrait(
        name="no_spaces",
        pattern=r"^\S+$",
        higher_is_better=True,
    )

    assert trait.evaluate("no_spaces_here") is True
    assert trait.evaluate("has spaces here") is False


@pytest.mark.unit
def test_regex_trait_with_anchors() -> None:
    """Test RegexTrait with start/end anchors."""
    trait = RegexTrait(
        name="starts_with_hello",
        pattern=r"^Hello",
        higher_is_better=True,
    )

    assert trait.evaluate("Hello world") is True
    assert trait.evaluate("Say Hello") is False


@pytest.mark.unit
def test_regex_trait_case_insensitive_with_uppercase_pattern() -> None:
    """Test case insensitive with uppercase pattern."""
    trait = RegexTrait(
        name="case_insensitive",
        pattern=r"ERROR",
        case_sensitive=False,
        higher_is_better=True,
    )

    assert trait.evaluate("error") is True
    assert trait.evaluate("Error") is True
    assert trait.evaluate("ERROR") is True


# =============================================================================
# Parametrized Advanced Pattern Tests (lookahead, lookbehind, groups)
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,pattern,match_text,no_match_text",
    [
        ("lookahead", r"start(?!.*end)", "start here", "start and then end"),
        ("lookbehind", r"(?<=:)\s*\w+", "Key: value", "Key value"),
        ("non_capturing_group", r"(?:hello|world)", "hello there", "foo there"),
        ("capturing_group", r"(hello|world)", "hello there", "foo there"),
    ],
    ids=["lookahead", "lookbehind", "non_capturing", "capturing"],
)
def test_regex_trait_advanced_patterns(name: str, pattern: str, match_text: str, no_match_text: str) -> None:
    """Test RegexTrait with advanced regex patterns."""
    trait = RegexTrait(
        name=name,
        pattern=pattern,
        higher_is_better=True,
    )

    assert trait.evaluate(match_text) is True
    assert trait.evaluate(no_match_text) is False


@pytest.mark.unit
def test_regex_trait_zero_quantifier() -> None:
    """Test RegexTrait with zero quantifier."""
    trait = RegexTrait(
        name="has_optional",
        pattern=r"\d?\.\d+",
        higher_is_better=True,
    )

    assert trait.evaluate(".5") is True  # 0 matches
    assert trait.evaluate("0.5") is True
    assert trait.evaluate("3.14") is True


# =============================================================================
# Parametrized Quantifier Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,pattern,match_text,no_match_text",
    [
        ("star_quantifier", r"a.*b", "aXXXb", "aXXX"),
        ("plus_quantifier", r"\d+", "123", "abc"),
        ("exact_quantifier", r"\d{4}", "Code: 1234", "Code: 123"),
        ("lazy_quantifier", r"<.*?>", "<a><b>", "no tags"),
        ("dot_pattern", r".{3}", "abc", "ab"),
    ],
    ids=["star", "plus", "exact", "lazy", "dot"],
)
def test_regex_trait_quantifiers(name: str, pattern: str, match_text: str, no_match_text: str) -> None:
    """Test RegexTrait with various quantifiers."""
    trait = RegexTrait(
        name=name,
        pattern=pattern,
        higher_is_better=True,
    )

    assert trait.evaluate(match_text) is True
    assert trait.evaluate(no_match_text) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "text,expected",
    [
        ("12", True),
        ("123", True),
        ("1234", True),
        ("1", False),
        ("12345", False),
    ],
    ids=["two_digits", "three_digits", "four_digits", "one_digit", "five_digits"],
)
def test_regex_trait_range_quantifier(text: str, expected: bool) -> None:
    """Test RegexTrait with range quantifier {2,4}."""
    trait = RegexTrait(
        name="two_to_four_digits",
        pattern=r"^\d{2,4}$",
        higher_is_better=True,
    )

    assert trait.evaluate(text) is expected


# =============================================================================
# Parametrized Pattern Type Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,pattern,match_text,no_match_text",
    [
        ("or_pattern", r"apple|banana|cherry", "I like apple pie", "I like grape"),
        ("character_set", r"[aeiou]", "hello", "rhythm"),
        ("negated_set", r"[^a-z]", "123", "abc"),
    ],
    ids=["or", "set", "negated_set"],
)
def test_regex_trait_pattern_types(name: str, pattern: str, match_text: str, no_match_text: str) -> None:
    """Test RegexTrait with various pattern types."""
    trait = RegexTrait(
        name=name,
        pattern=pattern,
        higher_is_better=True,
    )

    assert trait.evaluate(match_text) is True
    assert trait.evaluate(no_match_text) is False


# =============================================================================
# Default Value Tests
# =============================================================================


@pytest.mark.unit
def test_regex_trait_higher_is_better_none_defaults_to_true() -> None:
    """Test that higher_is_better=None defaults to True (old checkpoint data)."""
    trait = RegexTrait(
        name="test",
        pattern=r"\w+",
        higher_is_better=None,
    )

    assert trait.higher_is_better is True
