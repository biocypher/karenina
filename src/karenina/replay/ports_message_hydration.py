"""Rehydrate stored trace_messages dicts into port Message objects."""

from __future__ import annotations

from typing import Any

from karenina.ports.messages import Message


def hydrate_trace_messages(raw: list[dict[str, Any]]) -> list[Message]:
    """Convert a list of Message.to_dict() outputs back to Message objects."""
    return [Message.from_dict(item) for item in raw]
