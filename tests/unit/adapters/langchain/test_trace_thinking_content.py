"""Tests for trace formatting of Anthropic thinking + block-list content.

Regression: ``_format_langchain_message`` and
``langchain_messages_to_trace_messages`` used to call ``str(msg.content)``
on AIMessages, producing Python dict repr for list-structured content
(Anthropic extended thinking, vision, tool_use mixes). The adapter now
extracts visible text via ``extract_text_from_lc_content`` so thinking
blocks stay out of the trace.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from karenina.adapters.langchain.trace import (
    _format_langchain_message,
    langchain_messages_to_raw_trace,
    langchain_messages_to_trace_messages,
)


@pytest.mark.unit
class TestTraceFormattingWithThinkingContent:
    """Trace formatting must strip thinking blocks and emit clean text."""

    def test_str_content_unchanged(self) -> None:
        """Plain str AIMessage content passes through as-is (non-thinking providers)."""
        msg = AIMessage(content="The answer is 42.")
        formatted = _format_langchain_message(msg)
        assert formatted == "--- AI Message ---\nThe answer is 42."

    def test_thinking_plus_text_extracts_text_only(self) -> None:
        """Anthropic thinking blocks are dropped; only the text block reaches the trace."""
        msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "Let me reason...", "signature": "sig"},
                {"type": "text", "text": "The answer is yes."},
            ]
        )
        formatted = _format_langchain_message(msg)
        assert formatted == "--- AI Message ---\nThe answer is yes."
        assert "thinking" not in formatted.lower()
        assert "signature" not in formatted

    def test_text_prefix_plus_tool_call_keeps_text(self) -> None:
        """Prefix text before a tool call is captured; tool calls append via tool_calls branch."""
        msg = AIMessage(
            content=[{"type": "text", "text": "Let me search."}],
            tool_calls=[{"id": "t1", "name": "search", "args": {"q": "x"}}],
        )
        formatted = _format_langchain_message(msg)
        assert "Let me search." in formatted
        assert "Tool Calls:" in formatted
        assert "search (call_t1)" in formatted

    def test_empty_content_no_body(self) -> None:
        """Empty content produces just the header (matches prior behavior)."""
        msg = AIMessage(content="")
        formatted = _format_langchain_message(msg)
        assert formatted == "--- AI Message ---"

    def test_raw_trace_integration_strips_thinking(self) -> None:
        """End-to-end: full trace string drops thinking content for thinking-enabled messages."""
        msgs = [
            AIMessage(
                content=[
                    {"type": "thinking", "thinking": "internal chain"},
                    {"type": "text", "text": "visible answer"},
                ]
            )
        ]
        trace = langchain_messages_to_raw_trace(msgs)
        assert "visible answer" in trace
        assert "internal chain" not in trace

    def test_structured_trace_strips_thinking(self) -> None:
        """Structured trace_messages content is text-only for thinking-enabled AIMessages."""
        msgs = [
            AIMessage(
                content=[
                    {"type": "thinking", "thinking": "internal chain"},
                    {"type": "text", "text": "visible answer"},
                ]
            )
        ]
        structured = langchain_messages_to_trace_messages(msgs)
        assert len(structured) == 1
        assert structured[0]["role"] == "assistant"
        assert structured[0]["content"] == "visible answer"
