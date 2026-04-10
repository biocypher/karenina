"""Handover utilities for multi-agent scenario context routing.

Provides TaggedMessage for tracking agent identity on messages,
format_transcript for rendering tagged message history as a labeled
text transcript, and apply_handover for dispatching between transcript
strategies and callable handovers on edge transitions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from karenina.ports.messages import (
    ContentType,
    Message,
    TextContent,
    ToolResultContent,
    ToolUseContent,
)
from karenina.scenario.trace_materialization import materialize_trace
from karenina.schemas.scenario.state import ScenarioState
from karenina.schemas.scenario.types import ScenarioEdge


@dataclass
class TaggedMessage:
    """A Message paired with the agent identity that produced it."""

    message: Message
    agent_id: str


def format_transcript(tagged_messages: list[TaggedMessage]) -> str:
    """Format tagged messages into a labeled transcript.

    Each content block gets a label: [agent_id:role:content_type] for
    agent messages, [__user__] for scenario prompts. ThinkingContent
    blocks are excluded.

    Args:
        tagged_messages: The full tagged message history.

    Returns:
        A formatted transcript string. Empty string if no messages.
    """
    if not tagged_messages:
        return ""

    lines: list[str] = []
    for tm in tagged_messages:
        for block in tm.message.content:
            if block.type == ContentType.THINKING:
                continue

            label = (
                "[__user__]"
                if tm.agent_id == "__user__"
                else f"[{tm.agent_id}:{tm.message.role.value}:{block.type.value}]"
            )

            if isinstance(block, TextContent):
                lines.append(f"{label} {block.text}")
            elif isinstance(block, ToolUseContent):
                args = ", ".join(f"{k}={v!r}" for k, v in block.input.items())
                lines.append(f"{label} {block.name}({args})")
            elif isinstance(block, ToolResultContent):
                lines.append(f"{label} {block.content}")

    return "\n".join(lines)


TRANSCRIPT_SEPARATOR = "\n\n---\n\n"


def apply_handover(
    edge: ScenarioEdge,
    tagged_messages: list[TaggedMessage],
    state: ScenarioState,
    question_text: str,
    turn_dir: Path | None = None,
) -> tuple[str, list[Message]] | None:
    """Apply a handover strategy from an edge.

    Args:
        edge: The edge that was followed (may have handover config).
        tagged_messages: The full tagged message history.
        state: The current scenario state.
        question_text: The target node's question text.
        turn_dir: Pre-created per-turn workspace directory for
            transcript_materialize. Traces are written to
            ``turn_dir/traces/``. Falls back to a temporary directory
            when None.

    Returns:
        None if the edge has no handover.
        Otherwise a tuple of (modified_question_text, conversation_history).
        For transcript strategies: question text includes the transcript,
        conversation_history is empty.
        For callable strategies: question text is unchanged,
        conversation_history is whatever the callable returns.
    """
    if edge.handover_callable is not None:
        history = edge.handover_callable(tagged_messages, state)
        return question_text, history

    if edge.handover is None:
        return None

    transcript = format_transcript(tagged_messages)

    if edge.handover == "transcript_prepend":
        combined = transcript + TRANSCRIPT_SEPARATOR + question_text
        return combined, []

    if edge.handover == "transcript_append":
        combined = question_text + TRANSCRIPT_SEPARATOR + transcript
        return combined, []

    if edge.handover == "transcript_materialize":
        # Build a unique question_id for the filename. Include a content
        # hash so concurrent scenarios don't overwrite each other's traces.
        content_hash = hashlib.sha256(transcript.encode()).hexdigest()[:10]
        question_id = f"{state.current_node}_{content_hash}_handover"
        # Materialize only the transcript; the question text is given to
        # the agent separately as its prompt, so including it in the file
        # would create confusing duplication. The trailing separator
        # triggers XML reformatting of tagged-message lines.
        trace_path = materialize_trace(
            question_text=transcript + TRANSCRIPT_SEPARATOR,
            conversation_history=None,
            trace_dir=turn_dir,
            question_id=question_id,
            scenario_turn=state.turn,
        )
        # Place the file-reading instruction BEFORE the question so the
        # agent reads the conversation context before seeing the task.
        preamble = (
            "A conversation trace from prior turns has been saved to a "
            "file. You MUST read this file before responding.\n\n"
            f"Trace file path: {trace_path}\n\n"
            "Read the file now using the Read tool. The file contains "
            "the prior conversation structured as XML turns with <turn>, "
            "<system_prompt>, <user>, and <assistant> elements. "
            "If Read truncates, keep calling Read with increasing offset "
            "until you reach the end. Large content blocks may have been "
            "offloaded to separate files in the artifacts/ subdirectory; "
            'when you see an XML element with offloaded="true", the '
            "element body contains the file path.\n\n"
            "After reading the full trace, carry out the task below.\n\n"
            "---\n\n"
        )
        enriched = preamble + question_text
        return enriched, []

    msg = f"Unknown handover strategy '{edge.handover}'"
    raise ValueError(msg)
