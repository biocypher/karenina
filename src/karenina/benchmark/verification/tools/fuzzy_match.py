"""Fuzzy matching utilities for deep-judgment excerpt validation.

This module provides fuzzy string matching functionality to verify that
extracted excerpts actually exist in the original trace, accounting for
minor whitespace and formatting differences.
"""

from difflib import SequenceMatcher


def fuzzy_match_excerpt(excerpt: str, trace: str) -> tuple[bool, float]:
    """Verify that an excerpt exists in the trace using fuzzy matching.

    This function uses the SequenceMatcher algorithm to find the best matching
    substring in the trace and calculate a similarity score. This accounts for
    minor whitespace differences and formatting variations while detecting
    hallucinated or significantly paraphrased excerpts.

    Args:
        excerpt: The extracted excerpt text to verify
        trace: The full raw trace to search within

    Returns:
        A tuple of (match_found, similarity_score) where:
        - match_found: True if similarity >= 75% threshold
        - similarity_score: Float between 0.0 and 1.0 indicating match quality
          - 1.0 = perfect match
          - 0.75+ = good match (minor differences)
          - 0.50-0.75 = partial match (significant differences)
          - <0.50 = poor match (likely hallucinated)

    Examples:
        >>> fuzzy_match_excerpt("exact text", "This contains exact text here")
        (True, 1.0)

        >>> fuzzy_match_excerpt("some  text", "some text")  # Extra whitespace
        (True, 0.90)

        >>> fuzzy_match_excerpt("hallucinated text", "completely different")
        (False, 0.1)
    """
    # Handle edge cases
    if not excerpt or not trace:
        return False, 0.0

    # Normalize whitespace for comparison
    # This allows matching despite different whitespace formatting
    excerpt_normalized = " ".join(excerpt.split())
    trace_normalized = " ".join(trace.split())

    # Empty after normalization
    if not excerpt_normalized or not trace_normalized:
        return False, 0.0

    # Find best matching substring using SequenceMatcher
    # This finds the longest contiguous matching subsequence
    matcher = SequenceMatcher(None, excerpt_normalized, trace_normalized)
    match = matcher.find_longest_match(0, len(excerpt_normalized), 0, len(trace_normalized))

    # No match found
    if match.size == 0:
        return False, 0.0

    # Calculate similarity as ratio of matched length to excerpt length
    # This penalizes excerpts that are only partially present
    similarity = match.size / len(excerpt_normalized)

    # Threshold of 75% - allows for minor differences but catches hallucinations
    match_found = similarity >= 0.75

    return match_found, similarity


def fuzzy_match_excerpt_with_context(excerpt: str, trace: str, context_chars: int = 50) -> tuple[bool, float, str]:
    """Fuzzy match with surrounding context extraction (for debugging).

    This is an extended version of fuzzy_match_excerpt that also returns
    the surrounding context from the trace where the match was found.
    Useful for debugging and understanding why matches succeeded or failed.

    Args:
        excerpt: The extracted excerpt text to verify
        trace: The full raw trace to search within
        context_chars: Number of characters to include before/after match

    Returns:
        A tuple of (match_found, similarity_score, context) where:
        - match_found: True if similarity >= 75% threshold
        - similarity_score: Float between 0.0 and 1.0
        - context: The matching text with surrounding context

    Examples:
        >>> match, score, ctx = fuzzy_match_excerpt_with_context(
        ...     "targets BCL-2",
        ...     "The drug venetoclax targets BCL-2 protein and induces apoptosis."
        ... )
        >>> match
        True
        >>> "targets BCL-2" in ctx
        True
    """
    # Use base function for core matching
    match_found, similarity = fuzzy_match_excerpt(excerpt, trace)

    if not match_found or similarity == 0.0:
        return match_found, similarity, ""

    # Find the match position to extract context
    excerpt_normalized = " ".join(excerpt.split())
    trace_normalized = " ".join(trace.split())

    matcher = SequenceMatcher(None, excerpt_normalized, trace_normalized)
    match = matcher.find_longest_match(0, len(excerpt_normalized), 0, len(trace_normalized))

    if match.size == 0:
        return match_found, similarity, ""

    # Extract context around the match
    start_pos = max(0, match.b - context_chars)
    end_pos = min(len(trace_normalized), match.b + match.size + context_chars)
    context = trace_normalized[start_pos:end_pos]

    # Add ellipses to indicate truncation
    if start_pos > 0:
        context = "..." + context
    if end_pos < len(trace_normalized):
        context = context + "..."

    return match_found, similarity, context
