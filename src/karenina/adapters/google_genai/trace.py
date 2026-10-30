"""Trace building for the google_genai adapter.

The agent loop builds ``trace_messages`` turn by turn (assistant messages,
tool results and the synthetic server-side tool pairs). ``raw_trace`` comes
from the shared ``messages_to_raw_trace``, so the delimiters match every
other adapter, plus one ``[Note: ...]`` line per event that ended or shaped
the run (turn limit, timeout, finish reasons, search budget).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from karenina.ports import Message


def build_raw_trace(messages: list[Message], notes: Iterable[str] = ()) -> str:
    """Build the raw trace string with run notes appended.

    Args:
        messages: The run's trace messages.
        notes: Notes to append, each rendered as ``[Note: <note>]``.

    Returns:
        The raw trace.
    """
    from karenina.benchmark.verification.utils.trace_formatting import messages_to_raw_trace

    blocks = [messages_to_raw_trace(messages)]
    blocks.extend(f"[Note: {note}]" for note in notes)
    return "\n\n".join(block for block in blocks if block)
