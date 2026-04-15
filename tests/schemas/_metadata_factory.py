"""Shared factory for VerificationResultMetadata fixtures across schema tests."""

from __future__ import annotations

from typing import Any

from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import VerificationResultMetadata


def make_metadata(**overrides: Any) -> VerificationResultMetadata:
    """Build a VerificationResultMetadata with reasonable defaults.

    Task 3 will replace the legacy fields with ``failure`` / ``caveats``.
    Until then, this factory builds against the CURRENT schema (legacy
    fields present). Task 3 will update this factory alongside the schema
    edit to pass ``failure=None`` and ``caveats=[]``.
    """
    base: dict[str, Any] = {
        "question_id": "q1",
        "template_id": "t1",
        "completed_without_errors": True,
        "question_text": "How many kidneys?",
        "answering": ModelIdentity(interface="openai", model_name="gpt-4o"),
        "parsing": ModelIdentity(interface="openai", model_name="gpt-4o"),
        "execution_time": 1.0,
        "timestamp": "2026-04-15T10:00:00Z",
        "result_id": "r1",
        "retry_counts": None,
    }
    base.update(overrides)
    return VerificationResultMetadata(**base)


__all__ = ["make_metadata"]
