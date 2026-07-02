"""Trace extraction from LangGraph agent results.

Converts LangGraph BaseMessage lists into karenina's raw trace format:
a delimited string for backward compatibility with existing infrastructure
(regex highlighting, database storage, trace validation).

Also exposes a structured trace_messages builder that returns the
list[dict] shape consumed by the frontend TraceMessage component and
preserves per-call usage_metadata on assistant entries.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _extract_ai_usage_metadata(msg: Any) -> dict[str, int] | None:
    """Extract per-call usage metadata from a LangChain AIMessage.

    Prefers AIMessage.usage_metadata (the modern LangChain standard with
    input_tokens / output_tokens keys). Falls back to
    response_metadata.token_usage (older LangChain shape with
    prompt_tokens / completion_tokens) and renames the keys.

    Mirrors the per-message extraction logic in
    `langchain_deep_agents/usage.py:extract_deep_agents_usage` but
    returns the per-message dict instead of aggregating.

    Args:
        msg: A LangGraph AIMessage instance.

    Returns:
        Dict with input_tokens / output_tokens (and any cache fields
        propagated from usage_metadata when present), or None when no
        usage information is reported.
    """
    usage_meta = getattr(msg, "usage_metadata", None)
    if usage_meta and isinstance(usage_meta, dict):
        per_call: dict[str, int] = {
            "input_tokens": int(usage_meta.get("input_tokens", 0) or 0),
            "output_tokens": int(usage_meta.get("output_tokens", 0) or 0),
        }
        # If the prompt-cache middleware is in use, LangChain may expose
        # Anthropic-shaped cache fields here. Forward them when present.
        cache_read = usage_meta.get("cache_read_input_tokens")
        if cache_read is not None:
            per_call["cache_read_input_tokens"] = int(cache_read)
        cache_creation = usage_meta.get("cache_creation_input_tokens")
        if cache_creation is not None:
            per_call["cache_creation_input_tokens"] = int(cache_creation)
        return per_call

    resp_meta = getattr(msg, "response_metadata", None)
    if resp_meta and isinstance(resp_meta, dict):
        token_usage = resp_meta.get("token_usage")
        if isinstance(token_usage, dict) and ("prompt_tokens" in token_usage or "completion_tokens" in token_usage):
            return {
                "input_tokens": int(token_usage.get("prompt_tokens", 0) or 0),
                "output_tokens": int(token_usage.get("completion_tokens", 0) or 0),
            }

    return None


def deep_agents_messages_to_raw_trace(
    messages: list[Any],
    include_user_messages: bool = False,
) -> str:
    """Convert LangGraph messages to raw trace string format.

    Produces the canonical karenina trace format shared by the langchain,
    claude_agent_sdk, and manual adapters: one ``--- AI Message ---``
    section per turn carrying the text followed by an inline ``Tool Calls:``
    block, and ``--- Tool Message (call_id: <id>) ---`` for each tool
    result. Thinking content surfaces as a separate ``--- Thinking ---``
    section before the AI Message for that turn.

    Args:
        messages: List of LangGraph BaseMessage objects.
        include_user_messages: If True, include HumanMessage in trace.

    Returns:
        Formatted trace string with ``---`` delimiters compatible with
        regex highlighting, database storage, and the trace-evaluation
        infrastructure shared with the other adapters.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    if not messages:
        return ""

    parts: list[str] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue

        if isinstance(msg, HumanMessage):
            if include_user_messages:
                parts.append(f"--- Human Message ---\n{msg.content}")
            continue

        if isinstance(msg, AIMessage):
            thinking = _extract_thinking_content(msg)
            if thinking:
                parts.append(f"--- Thinking ---\n{thinking}")

            text = _extract_ai_text(msg)
            tool_calls = getattr(msg, "tool_calls", None) or []

            if text or tool_calls:
                content_parts: list[str] = []
                if text:
                    content_parts.append(text)
                if tool_calls:
                    content_parts.append("\nTool Calls:")
                    for tc in tool_calls:
                        tool_name = tc.get("name", "unknown")
                        tool_id = tc.get("id", "unknown")
                        tool_args = tc.get("args", {})
                        content_parts.append(f"  {tool_name} (call_{tool_id})")
                        content_parts.append(f"   Call ID: {tool_id}")
                        if tool_args:
                            content_parts.append(f"   Args: {tool_args}")
                parts.append("--- AI Message ---\n" + "\n".join(content_parts))

        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            tool_call_id = getattr(msg, "tool_call_id", "unknown") or "unknown"
            parts.append(f"--- Tool Message (call_id: {tool_call_id}) ---\n{content}")

    return "\n\n".join(parts)


def _extract_thinking_content(msg: Any) -> str:
    """Extract reasoning / thinking content from a LangChain AIMessage.

    LangChain has no single canonical location for chain-of-thought content
    across providers. Check, in priority order:

    1. ``additional_kwargs["reasoning_content"]`` — vLLM's OpenAI-compatible
       endpoint emits ``delta.reasoning_content`` for thinking models
       (qwen3, deepseek-r1, etc.); recent langchain_openai surfaces it here.
    2. ``additional_kwargs["reasoning"]`` — alternate naming used by some
       providers / older LangChain wrappers.
    3. Anthropic-style content blocks with ``type == "thinking"`` inside
       ``msg.content`` when the model is invoked through an Anthropic
       wrapper. The ``thinking`` key holds the text.

    Returns an empty string when none of these surface a reasoning trace,
    so the caller can skip the ``--- Thinking ---`` section entirely.
    """
    additional = getattr(msg, "additional_kwargs", None) or {}
    if isinstance(additional, dict):
        for key in ("reasoning_content", "reasoning"):
            value = additional.get(key)
            if isinstance(value, str) and value.strip():
                return value

    content = getattr(msg, "content", None)
    if isinstance(content, list):
        thinking_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                text = block.get("thinking") or ""
                if isinstance(text, str) and text.strip():
                    thinking_parts.append(text)
        if thinking_parts:
            return "\n".join(thinking_parts)

    return ""


def _extract_ai_text(msg: Any) -> str:
    """Extract text content from an AIMessage.

    Handles both simple string content and list content (mixed text/tool_use blocks).

    Args:
        msg: A LangGraph AIMessage instance.

    Returns:
        Concatenated text content from the message.
    """
    if isinstance(msg.content, str):
        return msg.content

    if isinstance(msg.content, list):
        text_parts = []
        for block in msg.content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block["text"])
        return "\n".join(text_parts)

    return ""


def deep_agents_messages_to_trace_messages(
    messages: list[Any],
    include_user_messages: bool = False,
) -> list[dict[str, Any]]:
    """Convert LangGraph messages to structured TraceMessage list.

    Produces the structured list[dict] format consumed by the frontend
    TraceMessage component. Each assistant entry preserves per-call
    usage_metadata when the underlying AIMessage reports it (so
    post-hoc audits of token usage / cache hits per turn remain possible).

    Args:
        messages: List of LangGraph BaseMessage objects.
        include_user_messages: If True, include HumanMessage entries. Defaults
            to False to match the existing raw_trace behavior.

    Returns:
        List of TraceMessage dicts with role, content, block_index, and
        optional tool_calls, tool_result, and usage_metadata fields.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    if not messages:
        return []

    result: list[dict[str, Any]] = []
    block_index = 0

    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue

        if isinstance(msg, HumanMessage):
            if include_user_messages:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content:
                    result.append(
                        {
                            "role": "user",
                            "content": content,
                            "block_index": block_index,
                        }
                    )
                    block_index += 1
            continue

        if isinstance(msg, AIMessage):
            text = _extract_ai_text(msg)
            thinking = _extract_thinking_content(msg)
            tool_calls: list[dict[str, Any]] = []
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        {
                            "id": tc.get("id", ""),
                            "name": tc.get("name", "unknown"),
                            "input": tc.get("args", {}) if isinstance(tc.get("args"), dict) else {},
                        }
                    )

            if text or tool_calls or thinking:
                trace_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": text,
                    "block_index": block_index,
                }

                if tool_calls:
                    trace_msg["tool_calls"] = tool_calls

                if thinking:
                    # Mirror the CSDK structured-trace shape so downstream
                    # consumers (frontend trace component, evaluators) can
                    # treat reasoning identically across adapters.
                    trace_msg["thinking"] = {"thinking": thinking}

                usage_metadata = _extract_ai_usage_metadata(msg)
                if usage_metadata is not None:
                    trace_msg["usage_metadata"] = usage_metadata

                result.append(trace_msg)
                block_index += 1
            continue

        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            tool_use_id = getattr(msg, "tool_call_id", "") or ""
            result.append(
                {
                    "role": "tool",
                    "content": content,
                    "block_index": block_index,
                    "tool_result": {
                        "tool_use_id": tool_use_id,
                        "is_error": False,
                    },
                }
            )
            block_index += 1

    return result
