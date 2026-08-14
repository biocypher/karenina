"""Public helpers for formatting stored structured traces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from karenina.benchmark.verification.prompts.trace.abstention import ABSTENTION_DETECTION_SYS
from karenina.benchmark.verification.utils.trace_formatting import messages_to_raw_trace
from karenina.ports.messages import Message


def format_trace_messages(messages: Iterable[Message | Mapping[str, Any]]) -> str:
    """Format stored message records using Karenina's canonical trace format.

    Args:
        messages: Live ``Message`` objects or dictionaries from a stored
            verification result's ``template.trace_messages`` field.

    Returns:
        The canonical raw trace, including assistant text, tool calls, and tool
        results.
    """
    hydrated = [message if isinstance(message, Message) else Message.from_dict(dict(message)) for message in messages]
    return messages_to_raw_trace(hydrated)


def abstention_detection_instruction() -> str:
    """Return Karenina's standard instruction for explicit abstention checks."""
    return ABSTENTION_DETECTION_SYS


__all__ = ["abstention_detection_instruction", "format_trace_messages"]
