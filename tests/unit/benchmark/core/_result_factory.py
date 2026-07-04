"""Shared factory for VerificationResult fixtures across benchmark core tests."""

from __future__ import annotations

from typing import Any

from karenina.schemas.verification.result import VerificationResult
from karenina.schemas.verification.result_components import VerificationResultMetadata


def make_result(metadata: VerificationResultMetadata, **overrides: Any) -> VerificationResult:
    """Build a minimal VerificationResult wrapping the given metadata.

    Args:
        metadata: The ``VerificationResultMetadata`` to attach to the result.
        **overrides: Additional ``VerificationResult`` fields to override.

    Returns:
        A ``VerificationResult`` with only the metadata populated by default.
    """
    base: dict[str, Any] = {
        "metadata": metadata,
        "template": None,
        "rubric": None,
        "deep_judgment": None,
        "deep_judgment_rubric": None,
    }
    base.update(overrides)
    return VerificationResult(**base)


__all__ = ["make_result"]
