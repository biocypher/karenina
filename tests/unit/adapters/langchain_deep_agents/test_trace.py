"""Tests for trace extraction from Deep Agents results."""

from __future__ import annotations

import pytest

from karenina.adapters.langchain_deep_agents.trace import (
    deep_agents_messages_to_raw_trace,
    deep_agents_messages_to_trace_messages,
)


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


@pytest.mark.unit
class TestDeepAgentsThinkingExtraction:
    """Parity with CSDK: thinking / reasoning content must surface in the
    trace when the underlying provider attaches it. vLLM's OpenAI-compatible
    endpoint emits delta.reasoning_content for thinking models (qwen3,
    deepseek-r1, etc.), which recent langchain_openai surfaces under
    ``AIMessage.additional_kwargs["reasoning_content"]``."""

    def test_reasoning_content_renders_thinking_section(self):
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="The answer is 42.",
            additional_kwargs={"reasoning_content": "Let me think... 2+40=42."},
        )
        trace = deep_agents_messages_to_raw_trace([msg])
        assert "--- Thinking ---" in trace
        assert "Let me think... 2+40=42." in trace
        # Thinking appears before the AI Message text.
        assert trace.index("--- Thinking ---") < trace.index("--- AI Message ---")

    def test_alternate_reasoning_key_supported(self):
        """Some providers/LangChain wrappers use the bare key ``reasoning``."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="Done.",
            additional_kwargs={"reasoning": "Chain of thought goes here."},
        )
        trace = deep_agents_messages_to_raw_trace([msg])
        assert "--- Thinking ---" in trace
        assert "Chain of thought goes here." in trace

    def test_anthropic_style_thinking_block_in_content_list(self):
        """When the model is invoked through an Anthropic wrapper, thinking
        arrives as a content block with ``type == 'thinking'``."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "Reasoning step one."},
                {"type": "text", "text": "Final answer."},
            ],
        )
        trace = deep_agents_messages_to_raw_trace([msg])
        assert "--- Thinking ---" in trace
        assert "Reasoning step one." in trace
        assert "Final answer." in trace

    def test_thinking_only_message_still_renders(self):
        """A turn with only reasoning (no text, no tool calls) — e.g. a
        deliberative turn before a tool call comes in a separate message —
        must still produce a Thinking section in the trace."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "Pondering options."},
        )
        trace = deep_agents_messages_to_raw_trace([msg])
        assert "--- Thinking ---" in trace
        assert "Pondering options." in trace

    def test_no_reasoning_means_no_thinking_section(self):
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="Just an answer.")
        trace = deep_agents_messages_to_raw_trace([msg])
        assert "--- Thinking ---" not in trace

    def test_blank_reasoning_string_is_ignored(self):
        """Whitespace-only reasoning_content (some endpoints emit empty
        strings on non-thinking turns) must NOT produce an empty section."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="Reply.",
            additional_kwargs={"reasoning_content": "   "},
        )
        trace = deep_agents_messages_to_raw_trace([msg])
        assert "--- Thinking ---" not in trace

    def test_structured_trace_messages_attach_thinking_metadata(self):
        """Parity with CSDK structured trace: thinking surfaces under the
        ``thinking`` key on the assistant entry, alongside content/tool_calls."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="The answer is 42.",
            additional_kwargs={"reasoning_content": "Computed via 2+40."},
            tool_calls=[{"name": "submit", "args": {"a": 1}, "id": "call_x"}],
        )
        trace_msgs = deep_agents_messages_to_trace_messages([msg])
        assert len(trace_msgs) == 1
        entry = trace_msgs[0]
        assert entry["role"] == "assistant"
        assert entry["content"] == "The answer is 42."
        assert entry["thinking"] == {"thinking": "Computed via 2+40."}
        assert entry["tool_calls"][0]["name"] == "submit"
