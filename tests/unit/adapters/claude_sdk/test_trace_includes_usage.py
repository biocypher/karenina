"""Tests that sdk_messages_to_trace_messages preserves per-call usage_metadata.

Fix E (karenina infrastructure plan): trace converters must propagate the
per-message ``usage`` payload that the Claude Agent SDK attaches to each
``AssistantMessage`` so that ``template.trace_messages[*].usage_metadata``
contains per-turn token / cache accounting (and not just the aggregate).
"""

from __future__ import annotations

import pytest


def _to_trace(messages):
    from karenina.adapters.claude_agent_sdk.trace import sdk_messages_to_trace_messages

    return sdk_messages_to_trace_messages(messages)


@pytest.mark.unit
class TestTraceUsageMetadataCSDK:
    def test_assistant_message_with_usage_propagates_input_output_tokens(self) -> None:
        pytest.importorskip("claude_agent_sdk.types")
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock

        messages = [
            AssistantMessage(
                model="claude-haiku-4-5",
                content=[TextBlock(text="The answer is 42.")],
                usage={"input_tokens": 123, "output_tokens": 45},
            ),
        ]

        trace = _to_trace(messages)
        assert len(trace) == 1
        assert trace[0]["role"] == "assistant"
        assert "usage_metadata" in trace[0]
        assert trace[0]["usage_metadata"] == {
            "input_tokens": 123,
            "output_tokens": 45,
        }

    def test_cache_fields_present_propagate_when_reported(self) -> None:
        pytest.importorskip("claude_agent_sdk.types")
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock

        messages = [
            AssistantMessage(
                model="claude-haiku-4-5",
                content=[TextBlock(text="cached answer")],
                usage={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 800,
                    "cache_creation_input_tokens": 50,
                },
            ),
        ]

        trace = _to_trace(messages)
        assert trace[0]["usage_metadata"] == {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 50,
        }

    def test_partial_cache_fields_only_include_reported_ones(self) -> None:
        """When only one cache field is reported, the other must not be added.

        This distinguishes 'not reported' from a real zero — crucial for
        provenance during post-hoc audits.
        """
        pytest.importorskip("claude_agent_sdk.types")
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock

        messages = [
            AssistantMessage(
                model="claude-haiku-4-5",
                content=[TextBlock(text="partial cache info")],
                usage={
                    "input_tokens": 60,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 0,
                },
            ),
        ]

        trace = _to_trace(messages)
        usage_metadata = trace[0]["usage_metadata"]
        assert usage_metadata["input_tokens"] == 60
        assert usage_metadata["output_tokens"] == 10
        # cache_read_input_tokens was explicitly reported as 0 → keep it.
        assert usage_metadata["cache_read_input_tokens"] == 0
        # cache_creation_input_tokens was NOT reported → must be absent.
        assert "cache_creation_input_tokens" not in usage_metadata

    def test_assistant_message_without_usage_omits_usage_metadata(self) -> None:
        pytest.importorskip("claude_agent_sdk.types")
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock

        # usage defaults to None on AssistantMessage construction
        messages = [
            AssistantMessage(
                model="claude-haiku-4-5",
                content=[TextBlock(text="no usage payload")],
            ),
        ]

        trace = _to_trace(messages)
        assert len(trace) == 1
        assert trace[0]["role"] == "assistant"
        assert "usage_metadata" not in trace[0]

    def test_assistant_message_with_empty_usage_dict_omits_usage_metadata(self) -> None:
        """An empty usage dict is falsy -> treated as 'not reported'."""
        pytest.importorskip("claude_agent_sdk.types")
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock

        messages = [
            AssistantMessage(
                model="claude-haiku-4-5",
                content=[TextBlock(text="empty usage")],
                usage={},
            ),
        ]

        trace = _to_trace(messages)
        assert "usage_metadata" not in trace[0]
