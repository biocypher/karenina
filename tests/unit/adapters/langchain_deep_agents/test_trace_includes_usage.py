"""Tests that deep_agents_messages_to_trace_messages preserves per-call usage.

Fix E (karenina infrastructure plan): the DA structured trace converter must
propagate the per-message ``usage_metadata`` that LangChain AIMessages carry
(or fall back to ``response_metadata.token_usage`` for adapters that haven't
adopted the modern usage_metadata standard) so that per-turn token /
cache accounting survives into ``template.trace_messages``.
"""

from __future__ import annotations

import pytest


def _to_trace(messages):
    from karenina.adapters.langchain_deep_agents.trace import (
        deep_agents_messages_to_trace_messages,
    )

    return deep_agents_messages_to_trace_messages(messages)


@pytest.mark.unit
class TestTraceUsageMetadataDA:
    def test_ai_message_with_modern_usage_metadata_propagates(self) -> None:
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="Hello world",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )

        trace = _to_trace([msg])
        assert len(trace) == 1
        assert trace[0]["role"] == "assistant"
        assert trace[0]["content"] == "Hello world"
        assert trace[0]["usage_metadata"] == {
            "input_tokens": 100,
            "output_tokens": 50,
        }

    def test_ai_message_with_response_metadata_token_usage_is_renamed(self) -> None:
        """Older LangChain adapters expose token counts via
        ``response_metadata.token_usage`` using OpenAI-style keys
        (prompt_tokens / completion_tokens). The converter must rename
        those keys to input_tokens / output_tokens so downstream consumers
        see a uniform shape.
        """
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="From an older provider",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 20,
                }
            },
        )

        trace = _to_trace([msg])
        assert trace[0]["usage_metadata"] == {
            "input_tokens": 80,
            "output_tokens": 20,
        }

    def test_ai_message_usage_metadata_preferred_over_response_metadata(self) -> None:
        """When both are present, usage_metadata wins (it's the modern source)."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="x",
            usage_metadata={
                "input_tokens": 5,
                "output_tokens": 7,
                "total_tokens": 12,
            },
            response_metadata={"token_usage": {"prompt_tokens": 999, "completion_tokens": 999}},
        )

        trace = _to_trace([msg])
        assert trace[0]["usage_metadata"] == {
            "input_tokens": 5,
            "output_tokens": 7,
        }

    def test_ai_message_without_any_usage_omits_usage_metadata(self) -> None:
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="no usage info")
        trace = _to_trace([msg])

        assert len(trace) == 1
        assert "usage_metadata" not in trace[0]

    def test_cache_fields_in_usage_metadata_propagate(self) -> None:
        """If the prompt-cache middleware exposes Anthropic-shaped cache
        fields via AIMessage.usage_metadata, they survive into the trace.
        """
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="cached",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 25,
            },
        )

        trace = _to_trace([msg])
        assert trace[0]["usage_metadata"] == {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 25,
        }

    def test_tool_call_message_emits_assistant_with_tool_calls_and_usage(self) -> None:
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="Searching",
            tool_calls=[{"id": "call_1", "name": "search", "args": {"q": "BCL2"}}],
            usage_metadata={
                "input_tokens": 30,
                "output_tokens": 6,
                "total_tokens": 36,
            },
        )

        trace = _to_trace([msg])
        assert trace[0]["role"] == "assistant"
        assert trace[0]["tool_calls"] == [{"id": "call_1", "name": "search", "input": {"q": "BCL2"}}]
        assert trace[0]["usage_metadata"] == {"input_tokens": 30, "output_tokens": 6}

    def test_tool_message_does_not_get_usage_metadata(self) -> None:
        from langchain_core.messages import AIMessage, ToolMessage

        messages = [
            AIMessage(
                content="step",
                tool_calls=[{"id": "c1", "name": "s", "args": {}}],
                usage_metadata={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            ),
            ToolMessage(content="result", tool_call_id="c1"),
        ]

        trace = _to_trace(messages)
        # Find tool entry
        tool_entries = [t for t in trace if t["role"] == "tool"]
        assert len(tool_entries) == 1
        assert "usage_metadata" not in tool_entries[0]

    def test_empty_messages_returns_empty_list(self) -> None:
        assert _to_trace([]) == []
