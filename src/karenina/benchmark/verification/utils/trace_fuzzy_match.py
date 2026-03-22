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
        - match_found: True if similarity >= 75% threshold.
        - similarity_score: Float between 0.0 and 1.0 indicating match quality.
            1.0 = perfect match, 0.75+ = good match (minor differences),
            0.50-0.75 = partial match (significant differences),
            <0.50 = poor match (likely hallucinated).

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
