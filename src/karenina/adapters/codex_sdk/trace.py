"""Trace conversion for Codex SDK thread items.

Converts ``TurnResult.items`` into karenina's two trace formats:

- ``codex_items_to_raw_trace``: the canonical delimited string format shared
  by the langchain, claude_agent_sdk, and langchain_deep_agents adapters
  (``--- AI Message ---``, inline ``Tool Calls:`` blocks,
  ``--- Tool Message (call_id: ...) ---``, ``--- Thinking ---``).
- ``codex_items_to_trace_messages``: the structured list[dict] format
  consumed by the frontend TraceMessage component.

Both formats are derived from the unified Message conversion in
``messages.py`` so the two stay consistent by construction.
"""

from __future__ import annotations

from typing import Any

from karenina.ports import Role, ThinkingContent, ToolResultContent

from .messages import CodexMessageConverter

_converter = CodexMessageConverter()


def codex_items_to_raw_trace(
    items: list[Any],
    include_user_messages: bool = False,
) -> str:
    """Convert Codex thread items to the canonical raw_trace string.

    Args:
        items: ``TurnResult.items`` entries.
        include_user_messages: If True, include user messages in the trace.
            Defaults to False to match the other agent adapters.

    Returns:
        Formatted trace string with ``---`` delimiters.

    Example:
        >>> trace = codex_items_to_raw_trace(turn_result.items)
        >>> print(trace)
        --- Thinking ---
        The user wants a file created...
        <BLANKLINE>
        --- AI Message ---
        <BLANKLINE>
        Tool Calls:
          shell (call_item_1)
           Call ID: item_1
           Args: {'command': "echo hi > hello.txt"}
        <BLANKLINE>
        --- Tool Message (call_id: item_1) ---
        hi
    """
    parts: list[str] = []
    for message in _converter.from_provider(items):
        if message.role == Role.SYSTEM:
            continue

        if message.role == Role.USER:
            if include_user_messages and message.text:
                parts.append(f"--- Human Message ---\n{message.text}")
            continue

        if message.role == Role.ASSISTANT:
            thinking = "\n".join(c.thinking for c in message.content if isinstance(c, ThinkingContent))
            if thinking:
                parts.append(f"--- Thinking ---\n{thinking}")

            text = message.text
            tool_calls = message.tool_calls
            if text or tool_calls:
                content_parts: list[str] = []
                if text:
                    content_parts.append(text)
                if tool_calls:
                    content_parts.append("\nTool Calls:")
                    for tc in tool_calls:
                        content_parts.append(f"  {tc.name} (call_{tc.id})")
                        content_parts.append(f"   Call ID: {tc.id}")
                        if tc.input:
                            content_parts.append(f"   Args: {tc.input}")
                parts.append("--- AI Message ---\n" + "\n".join(content_parts))
            continue

        if message.role == Role.TOOL:
            for block in message.content:
                if isinstance(block, ToolResultContent):
                    content = block.content
                    if block.is_error:
                        content = f"[ERROR] {content}"
                    parts.append(f"--- Tool Message (call_id: {block.tool_use_id}) ---\n{content}")

    return "\n\n".join(parts)


def codex_items_to_trace_messages(
    items: list[Any],
    include_user_messages: bool = False,
) -> list[dict[str, Any]]:
    """Convert Codex thread items to the structured TraceMessage list.

    Args:
        items: ``TurnResult.items`` entries.
        include_user_messages: If True, include user messages.

    Returns:
        List of TraceMessage dicts (role, content, block_index, plus
        optional tool_calls, tool_result, thinking) matching the format
        produced by the other agent adapters.
    """
    result: list[dict[str, Any]] = []
    block_index = 0
    for message in _converter.from_provider(items):
        if message.role == Role.SYSTEM:
            continue
        if message.role == Role.USER and not include_user_messages:
            continue
        entry = message.to_dict()
        entry["block_index"] = block_index
        result.append(entry)
        block_index += 1
    return result


def extract_final_response(items: list[Any]) -> str | None:
    """Reconstruct the final assistant response from thread items.

    Codex's own ``TurnResult.final_response`` can be empty even on success
    (observed against vLLM, see specs/codex_sdk_findings.md), so this
    helper walks items in reverse and returns the last non-empty
    agentMessage text.

    Args:
        items: ``TurnResult.items`` entries.

    Returns:
        The last non-empty assistant text, or None.
    """
    for message in reversed(_converter.from_provider(items)):
        if message.role == Role.ASSISTANT and message.text:
            return message.text
    return None
