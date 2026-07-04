"""Conformance tests for trace format validation.

Validates that trace output from any adapter uses the correct delimiters
and structure expected by the karenina infrastructure (regex highlighting,
database storage, frontend display).
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core", reason="langchain-core not installed")

from karenina.adapters.langchain_deep_agents.trace import deep_agents_messages_to_raw_trace


@pytest.mark.unit
class TestTraceFormatConformance:
    """Verify trace format correctness for Deep Agents adapter."""

    def test_ai_message_uses_standard_delimiter(self):
        """raw_trace must use '--- AI Message ---' for assistant responses."""
        from langchain_core.messages import AIMessage

        messages = [AIMessage(content="Hello")]
        trace = deep_agents_messages_to_raw_trace(messages)
        assert "--- AI Message ---" in trace

    def test_tool_call_uses_standard_delimiter(self):
        """raw_trace must place tool calls inline inside ``--- AI Message ---``
        with a ``Tool Calls:`` block, matching the canonical format used by
        the langchain, claude_agent_sdk, and manual adapters."""
        from langchain_core.messages import AIMessage

        messages = [
            AIMessage(
                content="Searching...",
                tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "c1"}],
            ),
        ]
        trace = deep_agents_messages_to_raw_trace(messages)
        assert "--- AI Message ---" in trace
        assert "Tool Calls:" in trace
        # No standalone --- Tool Call --- section; everything stays in the AI Message
        assert "--- Tool Call ---" not in trace

    def test_tool_result_uses_standard_delimiter(self):
        """raw_trace must use ``--- Tool Message (call_id: <id>) ---`` for
        tool outputs, matching the canonical format used by the other adapters."""
        from langchain_core.messages import ToolMessage

        messages = [ToolMessage(content="result data", tool_call_id="c1")]
        trace = deep_agents_messages_to_raw_trace(messages)
        assert "--- Tool Message (call_id: c1) ---" in trace
        assert "--- Tool Result ---" not in trace

    def test_empty_messages_produce_empty_trace(self):
        """Empty message list must produce empty string, not an error."""
        assert deep_agents_messages_to_raw_trace([]) == ""

    def test_trace_preserves_message_content(self):
        """Content from messages must appear in the trace string."""
        from langchain_core.messages import AIMessage

        messages = [AIMessage(content="The capital of France is Paris.")]
        trace = deep_agents_messages_to_raw_trace(messages)
        assert "Paris" in trace

    def test_tool_call_includes_tool_name(self):
        """Tool call sections must include the tool name."""
        from langchain_core.messages import AIMessage

        messages = [
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {"q": "test"}, "id": "c1"}],
            ),
        ]
        trace = deep_agents_messages_to_raw_trace(messages)
        assert "web_search" in trace

    def test_multi_turn_trace_ordering(self):
        """Multiple messages should appear in order in the trace."""
        from langchain_core.messages import AIMessage, ToolMessage

        messages = [
            AIMessage(
                content="First response",
                tool_calls=[{"name": "tool1", "args": {}, "id": "c1"}],
            ),
            ToolMessage(content="tool result", tool_call_id="c1"),
            AIMessage(content="Second response"),
        ]
        trace = deep_agents_messages_to_raw_trace(messages)

        first_pos = trace.index("First response")
        tool_pos = trace.index("tool result")
        second_pos = trace.index("Second response")
        assert first_pos < tool_pos < second_pos
