"""Codex SDK message converter.

Converts between karenina's unified Message types and Codex SDK thread
items. Like the Claude Agent SDK, codex takes a string prompt per turn and
a thread-level ``base_instructions`` system prompt rather than a message
array.

Output conversion walks ``TurnResult.items`` (ThreadItem objects). Items
are discriminated by their ``type`` string rather than isinstance checks,
so this module imports cleanly and works on simple stand-ins without the
openai-codex package installed.

Item mapping:
    userMessage      -> Message.user
    reasoning        -> assistant Message with ThinkingContent
    agentMessage     -> assistant Message with TextContent
    commandExecution -> assistant ToolUseContent(name="shell") plus a
                        paired Message.tool_result sharing the item id,
                        is_error from the exit code
    mcpToolCall      -> ToolUseContent(name="mcp__<server>__<tool>") plus
                        paired tool result (is_error from error/status)
    fileChange       -> ToolUseContent(name="apply_patch") plus paired
                        tool result summarizing the changes
    webSearch        -> ToolUseContent(name="web_search") plus paired
                        tool result (codex does not surface result text)

Unknown item types are skipped with a debug log.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from karenina.ports import (
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)

logger = logging.getLogger(__name__)

SHELL_TOOL_NAME = "shell"
APPLY_PATCH_TOOL_NAME = "apply_patch"
WEB_SEARCH_TOOL_NAME = "web_search"

# Per-message caps for serialized multi-turn history. Long shell outputs
# replayed verbatim every turn would dominate the prompt, so tool results
# and tool arguments are truncated with an explicit marker.
HISTORY_TOOL_RESULT_MAX_CHARS = 2000
HISTORY_TOOL_ARGS_MAX_CHARS = 500

_FAILED_STATUSES = frozenset({"failed", "declined"})


def _truncate_for_history(text: str, max_chars: int) -> str:
    """Truncate history content to max_chars with an explicit marker."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated, {len(text)} chars total]"


def unwrap_item(item: Any) -> Any:
    """Unwrap a ThreadItem RootModel to its inner typed item."""
    return item.root if hasattr(item, "root") else item


def item_type(item: Any) -> str | None:
    """Return the ``type`` discriminator string of a thread item."""
    inner = unwrap_item(item)
    raw_type = getattr(inner, "type", None)
    if raw_type is None:
        return None
    return str(getattr(raw_type, "value", raw_type))


def _enum_value(value: Any) -> Any:
    """Return the wire value for enum-like objects, passthrough otherwise."""
    return getattr(value, "value", value)


def _stringify(value: Any) -> str:
    """Best-effort string rendering for tool results and arguments."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True, exclude_none=True, mode="json")
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _user_message_text(inner: Any) -> str:
    """Extract text from a userMessage item's content list."""
    parts: list[str] = []
    for entry in getattr(inner, "content", None) or []:
        text = getattr(entry, "text", None)
        if text is None and isinstance(entry, dict):
            text = entry.get("text")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


class CodexMessageConverter:
    """Convert between unified Message and Codex SDK thread items.

    Input conversion: SYSTEM messages are extracted separately for
    ``thread_start(base_instructions=...)``. For single-turn input (no
    assistant or tool messages) USER messages are joined into a plain
    prompt string, matching the Claude SDK converter. When the message
    list carries multi-turn history (scenario re-invocation passes the
    full prior conversation each turn, and codex has no session resume),
    prior turns are serialized into a role-labeled transcript ending with
    the latest user message as the current question.

    Example:
        >>> converter = CodexMessageConverter()
        >>> messages = [Message.system("Be helpful"), Message.user("Hello")]
        >>> converter.to_prompt_string(messages)
        'Hello'
        >>> converter.extract_system_prompt(messages)
        'Be helpful'
    """

    def to_prompt_string(self, messages: list[Message]) -> str:
        """Build the turn input string for ``thread.turn()``.

        Single-turn input (no ASSISTANT or TOOL messages) joins USER
        messages with blank lines, byte-identical to the previous
        behavior. Multi-turn history is rendered as a role-labeled
        transcript (see _render_history_transcript).

        Args:
            messages: List of unified Message objects.

        Returns:
            Prompt string suitable for ``thread.turn()``.
        """
        has_history = any(m.role in (Role.ASSISTANT, Role.TOOL) for m in messages)
        if not has_history:
            user_parts = [m.text for m in messages if m.role == Role.USER]
            return "\n\n".join(user_parts) if user_parts else ""
        return self._render_history_transcript(messages)

    def _render_history_transcript(self, messages: list[Message]) -> str:
        """Serialize multi-turn history into a role-labeled transcript.

        Prior turns appear under a "Conversation so far:" header with
        ``User:`` and ``Assistant:`` labels. Tool activity is summarized
        as "Assistant ran tool <name> with arguments: ..." and
        "Tool result (<id>): ..." lines. ThinkingContent is skipped
        because internal reasoning should not be replayed as context.
        Tool arguments and tool result content are truncated per message
        (HISTORY_TOOL_ARGS_MAX_CHARS / HISTORY_TOOL_RESULT_MAX_CHARS)
        so long shell outputs in history cannot blow up the prompt. The
        latest USER message is rendered last under "Current user
        message:" as the question to answer now.

        Args:
            messages: Unified messages including prior-turn history.

        Returns:
            Transcript string for the codex turn input.
        """
        non_system = [m for m in messages if m.role != Role.SYSTEM]

        # The trailing USER message is the current question. When the list
        # does not end with a user message (unusual, but possible when a
        # caller passes raw history), everything renders as history.
        current: Message | None = None
        history = non_system
        if non_system and non_system[-1].role == Role.USER:
            current = non_system[-1]
            history = non_system[:-1]

        lines: list[str] = ["Conversation so far:", ""]
        for message in history:
            if message.role == Role.USER:
                lines.append(f"User: {message.text}")
                lines.append("")
            elif message.role == Role.ASSISTANT:
                emitted = False
                if message.text:
                    lines.append(f"Assistant: {message.text}")
                    emitted = True
                for tool_call in message.tool_calls:
                    args = _truncate_for_history(_stringify(tool_call.input), HISTORY_TOOL_ARGS_MAX_CHARS)
                    lines.append(f"Assistant ran tool {tool_call.name} with arguments: {args}")
                    emitted = True
                if emitted:
                    lines.append("")
            elif message.role == Role.TOOL:
                emitted = False
                for block in message.content:
                    if isinstance(block, ToolResultContent):
                        content = _truncate_for_history(block.content, HISTORY_TOOL_RESULT_MAX_CHARS)
                        prefix = "Tool result (error)" if block.is_error else "Tool result"
                        lines.append(f"{prefix} ({block.tool_use_id}): {content}")
                        emitted = True
                if emitted:
                    lines.append("")

        if current is not None:
            lines.append("Current user message:")
            lines.append("")
            lines.append(current.text)

        return "\n".join(lines).strip()

    def extract_system_prompt(self, messages: list[Message]) -> str | None:
        """Extract SYSTEM messages for ``thread_start(base_instructions=...)``.

        Args:
            messages: List of unified Message objects.

        Returns:
            System prompt string, or None when no system messages exist.
        """
        system_msgs = [m for m in messages if m.role == Role.SYSTEM]
        if system_msgs:
            return "\n\n".join(m.text for m in system_msgs)
        return None

    def from_provider(self, items: list[Any]) -> list[Message]:
        """Convert Codex thread items to unified Messages.

        Args:
            items: ``TurnResult.items`` entries (ThreadItem or inner items).

        Returns:
            List of unified Message objects in item order. Tool-style items
            expand to an assistant tool-use message plus a paired tool
            result message sharing the item id.
        """
        result: list[Message] = []
        for item in items:
            result.extend(self._convert_item(item))
        return result

    def _convert_item(self, item: Any) -> list[Message]:
        """Convert one thread item to zero or more unified Messages."""
        inner = unwrap_item(item)
        kind = item_type(item)

        if kind == "userMessage":
            text = _user_message_text(inner)
            return [Message.user(text)] if text else []

        if kind == "reasoning":
            thinking = "\n\n".join(getattr(inner, "content", None) or [])
            if not thinking:
                thinking = "\n\n".join(getattr(inner, "summary", None) or [])
            if not thinking:
                return []
            return [Message(role=Role.ASSISTANT, content=[ThinkingContent(thinking=thinking)])]

        if kind == "agentMessage":
            text = getattr(inner, "text", "") or ""
            return [Message(role=Role.ASSISTANT, content=[TextContent(text=text)])] if text else []

        if kind == "commandExecution":
            return self._convert_command_execution(inner)

        if kind == "mcpToolCall":
            return self._convert_mcp_tool_call(inner)

        if kind == "fileChange":
            return self._convert_file_change(inner)

        if kind == "webSearch":
            return self._convert_web_search(inner)

        logger.debug("Skipping unsupported codex thread item type: %s", kind)
        return []

    def _tool_pair(
        self,
        *,
        item_id: str,
        name: str,
        tool_input: dict[str, Any],
        result_content: str,
        is_error: bool,
    ) -> list[Message]:
        """Build the assistant tool-use plus tool-result message pair."""
        return [
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseContent(id=item_id, name=name, input=tool_input)],
            ),
            Message.tool_result(tool_use_id=item_id, content=result_content, is_error=is_error),
        ]

    def _convert_command_execution(self, inner: Any) -> list[Message]:
        exit_code = getattr(inner, "exit_code", None)
        status = _enum_value(getattr(inner, "status", None))
        is_error = (exit_code is not None and exit_code != 0) or status in _FAILED_STATUSES
        return self._tool_pair(
            item_id=str(getattr(inner, "id", "")),
            name=SHELL_TOOL_NAME,
            tool_input={"command": getattr(inner, "command", "") or ""},
            result_content=getattr(inner, "aggregated_output", None) or "",
            is_error=is_error,
        )

    def _convert_mcp_tool_call(self, inner: Any) -> list[Message]:
        server = getattr(inner, "server", "") or "unknown"
        tool = getattr(inner, "tool", "") or "unknown"
        arguments = getattr(inner, "arguments", None)
        tool_input = arguments if isinstance(arguments, dict) else {"arguments": _stringify(arguments)}

        error = getattr(inner, "error", None)
        status = _enum_value(getattr(inner, "status", None))
        is_error = error is not None or status in _FAILED_STATUSES
        if error is not None:
            result_content = getattr(error, "message", None) or _stringify(error)
        else:
            result_content = _stringify(getattr(inner, "result", None))

        return self._tool_pair(
            item_id=str(getattr(inner, "id", "")),
            name=f"mcp__{server}__{tool}",
            tool_input=tool_input,
            result_content=result_content,
            is_error=is_error,
        )

    def _convert_file_change(self, inner: Any) -> list[Message]:
        changes = getattr(inner, "changes", None) or []
        change_summaries = [
            {
                "path": getattr(change, "path", "") or "",
                "kind": str(_enum_value(getattr(change, "kind", "")) or ""),
            }
            for change in changes
        ]
        status = _enum_value(getattr(inner, "status", None))
        result_lines = [f"{entry['kind']} {entry['path']}".strip() for entry in change_summaries]
        result_content = "\n".join(line for line in result_lines if line) or f"status: {status}"
        return self._tool_pair(
            item_id=str(getattr(inner, "id", "")),
            name=APPLY_PATCH_TOOL_NAME,
            tool_input={"changes": change_summaries},
            result_content=result_content,
            is_error=status in _FAILED_STATUSES,
        )

    def _convert_web_search(self, inner: Any) -> list[Message]:
        return self._tool_pair(
            item_id=str(getattr(inner, "id", "")),
            name=WEB_SEARCH_TOOL_NAME,
            tool_input={"query": getattr(inner, "query", "") or ""},
            result_content="",
            is_error=False,
        )
