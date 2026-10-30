"""Informational flags attached to a verification result regardless of verdict."""

from __future__ import annotations

from enum import Enum


class Caveat(str, Enum):
    """Non-fatal observations that accompany a verification run."""

    PARTIAL_CONTENT = "partial_content"
    EMBEDDING_OVERRIDE = "embedding_override"
    RETRIES_USED = "retries_used"
    PARSE_DECISION_MALFORMED = "parse_decision_malformed"
    # An agentic rubric judge's investigation raised, so at least one agentic
    # trait has no judgment. Set whatever failure category the row ends with.
    AGENTIC_INVESTIGATION_FAILED = "agentic_investigation_failed"
    # An agentic rubric judge's score extraction failed with a retryable error
    # (connection, timeout, rate limit, server error), so the trait has a trace
    # but no judgment. Non-retryable extraction failures (for example schema
    # validation) do not raise it and leave only a missing score.
    AGENTIC_EXTRACTION_FAILED = "agentic_extraction_failed"


__all__ = ["Caveat"]
