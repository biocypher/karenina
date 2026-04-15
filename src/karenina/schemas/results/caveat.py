"""Informational flags attached to a verification result regardless of verdict."""

from __future__ import annotations

from enum import Enum


class Caveat(str, Enum):
    """Non-fatal observations that accompany a verification run."""

    PARTIAL_CONTENT = "partial_content"
    EMBEDDING_OVERRIDE = "embedding_override"
    RETRIES_USED = "retries_used"


__all__ = ["Caveat"]
