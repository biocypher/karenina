"""Unit tests for verification exceptions.

Tests ExcerptNotFoundError used in deep-judgment excerpt validation.
"""

import pytest

from karenina.benchmark.verification.exceptions import ExcerptNotFoundError


@pytest.mark.unit
def test_excerpt_not_found_error_attributes() -> None:
    """Test that ExcerptNotFoundError stores all attributes correctly."""
    error = ExcerptNotFoundError(excerpt="hallucinated text", attribute="target_gene", similarity_score=0.3)

    assert error.excerpt == "hallucinated text"
    assert error.attribute == "target_gene"
    assert error.similarity_score == 0.3


@pytest.mark.unit
def test_excerpt_not_found_error_message() -> None:
    """Test that ExcerptNotFoundError formats message correctly."""
    error = ExcerptNotFoundError(excerpt="some text", attribute="result", similarity_score=0.45)

    assert str(error) == "Excerpt for attribute 'result' not found in trace (similarity score: 0.45)"


@pytest.mark.unit
def test_excerpt_not_found_error_low_score() -> None:
    """Test ExcerptNotFoundError with very low similarity score."""
    error = ExcerptNotFoundError(excerpt="completely different", attribute="value", similarity_score=0.0)

    assert error.similarity_score == 0.0
    assert "0.00" in str(error)


@pytest.mark.unit
def test_excerpt_not_found_error_high_score() -> None:
    """Test ExcerptNotFoundError with score just below threshold."""
    error = ExcerptNotFoundError(excerpt="almost match", attribute="field", similarity_score=0.74)

    assert error.similarity_score == 0.74
    assert "0.74" in str(error)


@pytest.mark.unit
def test_excerpt_not_found_error_is_exception() -> None:
    """Test that ExcerptNotFoundError can be raised and caught."""
    with pytest.raises(ExcerptNotFoundError) as exc_info:
        raise ExcerptNotFoundError("text", "attr", 0.5)

    error = exc_info.value
    assert error.excerpt == "text"
    assert error.attribute == "attr"
    assert error.similarity_score == 0.5


@pytest.mark.unit
def test_excerpt_not_found_error_rounding_in_message() -> None:
    """Test that similarity score is rounded to 2 decimal places in message."""
    error = ExcerptNotFoundError(excerpt="text", attribute="attr", similarity_score=0.666666)

    assert "0.67" in str(error)
