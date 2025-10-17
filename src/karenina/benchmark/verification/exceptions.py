"""Exceptions for deep-judgment verification feature.

This module defines custom exceptions for the deep-judgment multi-stage parsing system.
"""


class ExcerptNotFoundError(Exception):
    """Raised when an extracted excerpt doesn't exist in the trace.

    This exception is raised during excerpt validation when fuzzy matching
    fails to find the excerpt in the original trace, indicating either:
    1. LLM hallucinated the excerpt
    2. Excerpt was paraphrased rather than verbatim
    3. Whitespace/formatting differences too large

    Attributes:
        excerpt: The excerpt text that was not found
        attribute: The attribute name this excerpt was supposed to support
        similarity_score: The fuzzy match similarity score (0.0 to 1.0)
    """

    def __init__(self, excerpt: str, attribute: str, similarity_score: float):
        """Initialize ExcerptNotFoundError.

        Args:
            excerpt: The excerpt text that was not found
            attribute: The attribute name this excerpt was supposed to support
            similarity_score: The fuzzy match similarity score (0.0 to 1.0)
        """
        self.excerpt = excerpt
        self.attribute = attribute
        self.similarity_score = similarity_score
        super().__init__(
            f"Excerpt for attribute '{attribute}' not found in trace (similarity score: {similarity_score:.2f})"
        )
