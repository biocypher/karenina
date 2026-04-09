"""Trace materialization utilities for scenario handover.

Provides functions to write conversation transcripts to files for
agentic answering models. Used by the ``transcript_materialize``
handover strategy and optionally by Stage 7b for coding-task traces.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from karenina.ports.messages import Message

logger = logging.getLogger(__name__)

DEFAULT_TRACE_TRUNCATION_THRESHOLD = 2000

# Regex matching tagged-message lines produced by format_transcript().
# Captures: group 1 = full tag (e.g. "__user__" or "agent:assistant:text"),
# group 2 = content after the tag on the same line.
TAG_RE = re.compile(
    r"^\[(__user__|[^\]]+:(?:system|user|assistant|tool):(?:text|tool_use|tool_result))\] ?(.*)",
)


def parse_transcript_entries(
    transcript: str,
) -> list[dict[str, str | None]]:
    """Parse tagged-message transcript into a list of entry dicts.

    Each entry has keys: role, agent_id, content_type, content.
    Multi-line content (continuation lines without a tag prefix)
    is appended to the previous entry.

    Args:
        transcript: Raw transcript text from format_transcript().

    Returns:
        List of parsed entries.
    """
    entries: list[dict[str, str | None]] = []
    current: dict[str, str | None] | None = None

    for line in transcript.split("\n"):
        match = TAG_RE.match(line)
        if match:
            if current is not None:
                entries.append(current)
            tag, content = match.group(1), match.group(2)
            if tag == "__user__":
                current = {
                    "role": "user",
                    "agent_id": None,
                    "content_type": "text",
                    "content": content,
                }
            else:
                # 3-part tag: agent_id:role:content_type
                parts = tag.rsplit(":", 2)
                agent_id, role, content_type = parts[0], parts[1], parts[2]
                current = {
                    "role": role,
                    "agent_id": agent_id,
                    "content_type": content_type,
                    "content": content,
                }
        elif current is not None:
            current["content"] = (current["content"] or "") + "\n" + line

    if current is not None:
        entries.append(current)

    return entries


def group_entries_into_turns(
    entries: list[dict[str, str | None]],
) -> list[dict[str, Any]]:
    """Group parsed entries into turns.

    Each ``[__user__]`` entry starts a new turn. System entries
    become the ``system_prompt`` of the following turn. All subsequent
    non-user, non-system entries become the assistant response blocks.

    Args:
        entries: Parsed transcript entries.

    Returns:
        List of turn dicts with ``system_prompt``, ``user_content``,
        ``agent_id``, and ``blocks`` (list of assistant entries).
    """
    turns: list[dict[str, Any]] = []
    current_turn: dict[str, Any] | None = None
    pending_system: str | None = None
    pending_system_agent: str | None = None

    for entry in entries:
        if entry["role"] == "system":
            pending_system = entry["content"]
            pending_system_agent = entry["agent_id"]
        elif entry["role"] == "user":
            if current_turn is not None:
                turns.append(current_turn)
            current_turn = {
                "system_prompt": pending_system,
                "system_agent": pending_system_agent,
                "user_content": entry["content"],
                "agent_id": None,
                "blocks": [],
            }
            pending_system = None
            pending_system_agent = None
        elif current_turn is not None:
            if current_turn["agent_id"] is None and entry["agent_id"]:
                current_turn["agent_id"] = entry["agent_id"]
            current_turn["blocks"].append(entry)

    if current_turn is not None:
        turns.append(current_turn)

    # Handle trailing system entry with no following user entry
    if pending_system is not None:
        turns.append(
            {
                "system_prompt": pending_system,
                "system_agent": pending_system_agent,
                "user_content": None,
                "agent_id": None,
                "blocks": [],
            }
        )

    return turns


def format_turns_as_xml(
    turns: list[dict[str, Any]],
    artifacts_dir: Path | None = None,
    truncation_threshold: int = DEFAULT_TRACE_TRUNCATION_THRESHOLD,
) -> str:
    """Format grouped turns into XML with nested elements.

    Produces ``<turn>``, ``<system_prompt>``, ``<user>``,
    ``<assistant>``, ``<text>``, ``<tool_call>``, and
    ``<tool_result>`` elements. System prompts appear inside
    ``<turn>`` before ``<user>``. Tool results carry a ``name``
    attribute from the most recent preceding tool call.

    Content blocks exceeding ``truncation_threshold`` are offloaded
    to numbered files in ``artifacts_dir`` when provided.

    Args:
        turns: Grouped turn dicts from ``group_entries_into_turns``.
        artifacts_dir: Directory for offloaded content. None disables
            offloading.
        truncation_threshold: Char count above which blocks are
            offloaded.

    Returns:
        XML-formatted string.
    """
    artifact_counter = 0

    def _maybe_offload(content: str, tag_type: str) -> tuple[str, bool]:
        nonlocal artifact_counter
        if artifacts_dir is None or len(content) <= truncation_threshold:
            return content, False

        artifact_counter += 1
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{tag_type}_{artifact_counter:03d}.txt"
        artifact_path = artifacts_dir / filename
        artifact_path.write_text(content, encoding="utf-8")

        reference = f"[Content offloaded: {len(content):,} chars]\nFile: {artifact_path}"
        return reference, True

    def _emit_block(
        parts: list[str],
        tag: str,
        attrs: str,
        content: str,
        tag_type: str,
    ) -> None:
        body, offloaded = _maybe_offload(content.strip(), tag_type)
        offload_attr = ' offloaded="true"' if offloaded else ""
        parts.append(f"    <{tag}{attrs}{offload_attr}>")
        for line in body.split("\n"):
            parts.append(f"      {line}")
        parts.append(f"    </{tag}>")

    parts: list[str] = []

    for idx, turn in enumerate(turns, 1):
        parts.append(f'<turn number="{idx}">')

        if turn.get("system_prompt"):
            sys_agent = turn.get("system_agent") or turn.get("agent_id") or "unknown"
            parts.append(f'  <system_prompt agent="{sys_agent}">')
            for line in (turn["system_prompt"] or "").strip().split("\n"):
                parts.append(f"    {line}")
            parts.append("  </system_prompt>")

        if turn.get("user_content") is not None:
            parts.append("  <user>")
            for line in turn["user_content"].strip().split("\n"):
                parts.append(f"    {line}")
            parts.append("  </user>")

        if turn["blocks"]:
            agent_id = turn["agent_id"] or "unknown"
            parts.append(f'  <assistant agent="{agent_id}">')

            last_tool_name = "unknown"
            for block in turn["blocks"]:
                ctype = block["content_type"]
                raw = (block["content"] or "").strip()

                if ctype == "text":
                    _emit_block(parts, "text", "", raw, "text")
                elif ctype == "tool_use":
                    paren_idx = raw.find("(")
                    if paren_idx > 0 and raw.endswith(")"):
                        tool_name = raw[:paren_idx]
                        tool_args = raw[paren_idx + 1 : -1]
                    else:
                        tool_name = "unknown"
                        tool_args = raw
                    last_tool_name = tool_name
                    _emit_block(
                        parts,
                        "tool_call",
                        f' name="{tool_name}"',
                        tool_args,
                        "tool_call",
                    )
                elif ctype == "tool_result":
                    _emit_block(
                        parts,
                        "tool_result",
                        f' name="{last_tool_name}"',
                        raw,
                        "tool_result",
                    )

            parts.append("  </assistant>")
        parts.append("</turn>")
        parts.append("")

    return "\n".join(parts)


TRANSCRIPT_SEPARATOR = "\n\n---\n\n"


def serialize_conversation_history(history: list[Message]) -> str:
    """Serialize a list of Message objects into labeled text blocks.

    Uses ``Message.text`` which joins TextContent blocks only.

    Args:
        history: Prior-turn messages from the scenario conversation.

    Returns:
        Formatted string with ``--- Role Message ---`` delimiters.
    """
    parts: list[str] = []
    for msg in history:
        role_label = msg.role.value.title()
        parts.append(f"--- {role_label} Message ---")
        parts.append(msg.text)
        parts.append("")
    return "\n".join(parts)


def reformat_transcript_as_xml(
    question_text: str,
    artifacts_dir: Path | None = None,
    truncation_threshold: int = DEFAULT_TRACE_TRUNCATION_THRESHOLD,
) -> str:
    """Reformat transcript_prepend question text into XML-structured turns.

    Detects whether ``question_text`` contains a ``transcript_prepend``
    pattern (``[__user__]`` tags followed by the separator). If so,
    parses the tagged-message format and reformats into nested XML.

    If the text does not match the transcript pattern, returns it
    unchanged.

    Args:
        question_text: The question prompt, potentially containing a
            prepended transcript.
        artifacts_dir: Directory for offloaded content files.
        truncation_threshold: Char count above which content blocks
            are offloaded.

    Returns:
        Reformatted text with XML structure, or the original text.
    """
    if TRANSCRIPT_SEPARATOR not in question_text or "[__user__]" not in question_text:
        return question_text

    transcript_part, eval_part = question_text.split(TRANSCRIPT_SEPARATOR, 1)

    entries = parse_transcript_entries(transcript_part)
    if not entries:
        return question_text

    turns = group_entries_into_turns(entries)
    xml = format_turns_as_xml(
        turns,
        artifacts_dir=artifacts_dir,
        truncation_threshold=truncation_threshold,
    )

    return xml + "\n\n" + eval_part


def materialize_trace(
    question_text: str,
    conversation_history: list[Message] | None,
    workspace_root: Path | None,
    question_id: str,
    scenario_turn: int | None = None,
) -> Path:
    """Write conversation context to a file for agent filesystem-tool access.

    Creates a trace file at ``<workspace_root>/.karenina/traces/`` with
    a KARENINA CONVERSATION TRACE header, XML-reformatted transcript,
    and optional conversation history.

    Args:
        question_text: The question (potentially with transcript_prepend).
        conversation_history: Prior-turn Message objects.
        workspace_root: Root workspace directory. Falls back to a
            temporary directory when None.
        question_id: Question identifier (sanitized for filesystem).
        scenario_turn: Optional turn index for multi-turn scenarios.

    Returns:
        Path to the written trace file.
    """
    if workspace_root is None:
        trace_dir = Path(tempfile.mkdtemp(prefix="karenina_traces_"))
    else:
        trace_dir = Path(workspace_root) / ".karenina" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", question_id)
    filename = f"{safe_id}_turn{scenario_turn}_trace.txt" if scenario_turn is not None else f"{safe_id}_trace.txt"
    trace_path = trace_dir / filename

    header = (
        "# =============================================================\n"
        "# KARENINA CONVERSATION TRACE\n"
        "# -------------------------------------------------------------\n"
        "# This file contains the prior conversation from a multi-turn\n"
        "# scenario, structured for file-tool access. It may include\n"
        "# XML-formatted turns with <turn>, <system_prompt>, <user>,\n"
        "# and <assistant> elements.\n"
        "#\n"
        "# When CONVERSATION HISTORY sections are present, they show\n"
        "# Message objects from prior scenario turns.\n"
        "#\n"
        "# Large content blocks may have been offloaded to separate\n"
        "# files in the artifacts/ subdirectory.\n"
        "# =============================================================\n\n"
    )

    # Resolve truncation threshold from env var with fallback.
    raw_threshold = os.environ.get("KARENINA_TRACE_TRUNCATION_THRESHOLD")
    if raw_threshold is not None:
        try:
            threshold = int(raw_threshold)
        except ValueError:
            logger.warning(
                "Invalid KARENINA_TRACE_TRUNCATION_THRESHOLD=%s, using default %d",
                raw_threshold,
                DEFAULT_TRACE_TRUNCATION_THRESHOLD,
            )
            threshold = DEFAULT_TRACE_TRUNCATION_THRESHOLD
    else:
        threshold = DEFAULT_TRACE_TRUNCATION_THRESHOLD
    artifacts_dir = trace_dir / "artifacts" if workspace_root else None

    sections: list[str] = [header]

    if question_text:
        formatted_text = reformat_transcript_as_xml(
            question_text,
            artifacts_dir=artifacts_dir,
            truncation_threshold=threshold,
        )
        sections.append(f"# QUESTION CONTEXT\n# =================\n{formatted_text}\n\n")

    if conversation_history:
        serialized = serialize_conversation_history(conversation_history)
        sections.append(
            "# CONVERSATION HISTORY (prior scenario turns)\n"
            "# ============================================\n"
            f"{serialized}\n"
        )

    trace_path.write_text("".join(sections), encoding="utf-8")
    logger.debug("Materialized conversation trace to %s", trace_path)
    return trace_path
