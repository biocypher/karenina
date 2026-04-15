"""Shared factory for VerificationResultMetadata fixtures across schema tests."""

from __future__ import annotations

from typing import Any

from karenina.schemas.results.caveat import Caveat  # noqa: F401  (re-exported for convenience)
from karenina.schemas.results.failure import Failure, FailureCategory  # noqa: F401  (re-exported)
from karenina.schemas.verification.result_components import (
    ModelIdentity,
    VerificationResultMetadata,
)


def make_metadata(**overrides: Any) -> VerificationResultMetadata:
    """Build a VerificationResultMetadata with reasonable defaults for tests."""
    base: dict[str, Any] = {
        "question_id": "q1",
        "template_id": "t1",
        "question_text": "How many kidneys?",
        "answering": ModelIdentity(interface="openai", model_name="gpt-4o"),
        "parsing": ModelIdentity(interface="openai", model_name="gpt-4o"),
        "execution_time": 1.0,
        "timestamp": "2026-04-15T10:00:00Z",
        "result_id": "r1",
        "retry_counts": None,
        "failure": None,
        "caveats": [],
    }
    base.update(overrides)
    return VerificationResultMetadata(**base)


__all__ = ["Caveat", "Failure", "FailureCategory", "make_metadata"]
