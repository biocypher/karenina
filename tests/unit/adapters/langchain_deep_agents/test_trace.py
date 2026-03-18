"""Tests for trace extraction from Deep Agents results."""

from __future__ import annotations

import pytest

from karenina.adapters.langchain_deep_agents.trace import deep_agents_messages_to_raw_trace


@pytest.mark.unit
class TestDeepAgentsTraceExtraction:
    def test_ai_message_produces_ai_delimiter(self):
        from langchain_core.messages import AIMessage

        messages = [AIMessage(content="Hello world")]
        trace = deep_agents_messages_to_raw_trace(messages)
        assert "--- AI Message ---" in trace
        assert "Hello world" in trace

    def test_tool_call_produces_tool_delimiter(self):
        from langchain_core.messages import AIMessage, ToolMessage

        messages = [
            AIMessage(
                content="Let me search.",
                tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "call_1"}],
            ),
            ToolMessage(content="search result", tool_call_id="call_1"),
        ]
        trace = deep_agents_messages_to_raw_trace(messages)
        assert "--- AI Message ---" in trace
        assert "--- Tool Call ---" in trace
        assert "--- Tool Result ---" in trace
        assert "search" in trace

    def test_empty_messages_returns_empty_string(self):
        trace = deep_agents_messages_to_raw_trace([])
        assert trace == ""

    def test_multiple_ai_messages(self):
        from langchain_core.messages import AIMessage

        messages = [AIMessage(content="First"), AIMessage(content="Second")]
        trace = deep_agents_messages_to_raw_trace(messages)
        assert trace.count("--- AI Message ---") == 2
