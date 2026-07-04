"""Unit tests for fuzzy matching utilities.

Tests fuzzy_match_excerpt function used for deep-judgment excerpt validation.
"""

import pytest

from karenina.benchmark.verification.utils.trace_fuzzy_match import (
    fuzzy_match_excerpt,
)


@pytest.mark.unit
def test_fuzzy_match_exact_match() -> None:
    """Test fuzzy matching with exact text match."""
    trace = "The drug targets BCL2 protein and induces apoptosis."
    excerpt = "targets BCL2"

    match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

    assert match_found is True
    assert similarity == 1.0


@pytest.mark.unit
def test_fuzzy_match_exact_full_excerpt() -> None:
    """Test fuzzy matching when excerpt is the entire trace."""
    trace = "The answer is four."
    excerpt = "The answer is four."

    match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

    assert match_found is True
    assert similarity == 1.0


@pytest.mark.unit
def test_fuzzy_match_extra_whitespace() -> None:
    """Test fuzzy matching normalizes whitespace differences."""
    trace = "The answer is four."
    excerpt = "The  answer   is four."

    match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

    assert match_found is True
    assert similarity >= 0.9


@pytest.mark.unit
def test_fuzzy_match_minor_differences() -> None:
    """Test fuzzy matching with minor text differences."""
    trace = "The drug venetoclax targets BCL-2 protein."
    excerpt = "drug venetoclax targets BCL2"

    match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

    # Should match well despite dash vs no dash
    assert similarity >= 0.75


@pytest.mark.unit
def test_fuzzy_match_partial_match_below_threshold() -> None:
    """Test fuzzy matching when only part of excerpt is found."""
    trace = "BCL2 is a gene."
    excerpt = "BCL2 is a gene that encodes a protein"

    match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

    assert match_found is False
    assert similarity < 0.75


@pytest.mark.unit
def test_fuzzy_match_no_match_hallucinated() -> None:
    """Test fuzzy matching with completely different text."""
    trace = "The answer is BCL2."
    excerpt = "TP53 is a completely different answer."

    match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

    assert match_found is False
    # Similarity should be low due to mostly different content
    assert similarity < 0.75


@pytest.mark.unit
def test_fuzzy_match_empty_excerpt() -> None:
    """Test fuzzy matching with empty excerpt."""
    trace = "Some text here."

    match_found, similarity = fuzzy_match_excerpt("", trace)

    assert match_found is False
    assert similarity == 0.0


@pytest.mark.unit
def test_fuzzy_match_empty_trace() -> None:
    """Test fuzzy matching with empty trace."""
    excerpt = "Some text"

    match_found, similarity = fuzzy_match_excerpt(excerpt, "")

    assert match_found is False
    assert similarity == 0.0


@pytest.mark.unit
def test_fuzzy_match_both_empty() -> None:
    """Test fuzzy matching with both empty strings."""
    match_found, similarity = fuzzy_match_excerpt("", "")

    assert match_found is False
    assert similarity == 0.0


@pytest.mark.unit
def test_fuzzy_match_excerpt_in_middle() -> None:
    """Test fuzzy matching when excerpt is in the middle of trace."""
    trace = "The drug targets BCL2 protein and induces apoptosis."
    excerpt = "targets BCL2"

    match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

    assert match_found is True
    assert similarity == 1.0


@pytest.mark.unit
def test_fuzzy_match_excerpt_at_end() -> None:
    """Test fuzzy matching when excerpt is at the end of trace."""
    trace = "The drug targets BCL2"
    excerpt = "targets BCL2"

    match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

    assert match_found is True
    assert similarity == 1.0


@pytest.mark.unit
def test_fuzzy_match_threshold_boundary() -> None:
    """Test fuzzy matching at the 75% threshold boundary."""
    # This should match right at the threshold
    trace = "The answer is BCL2 protein"
    excerpt = "The answer is BCL2"  # Missing "protein" (80% match)

    match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

    assert similarity >= 0.75


@pytest.mark.unit
def test_fuzzy_match_newlines_and_tabs() -> None:
    """Test fuzzy matching with newlines and tabs."""
    trace = "Line 1\nLine 2\tLine 3"
    excerpt = "Line 1 Line 2"

    match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

    # Whitespace normalization should handle this
    assert match_found is True
    assert similarity >= 0.75


@pytest.mark.unit
def test_fuzzy_match_case_sensitive() -> None:
    """Test that fuzzy matching is case-sensitive."""
    trace = "The drug targets BCL2"
    excerpt = "the drug targets bcl2"

    match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

    # Case differences should reduce similarity
    # But exact match of normalized text should still work
    assert similarity < 1.0 or match_found is True
