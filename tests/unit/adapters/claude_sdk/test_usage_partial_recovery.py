"""Tests for partial-usage recovery from SDK AssistantMessage lists.

These exercise the fallback path used when a ClaudeSDKClient run is
cancelled mid-stream (e.g. by asyncio.wait_for) and no aggregate
ResultMessage is ever emitted. The collected AssistantMessage objects
still carry per-call usage dicts, so we aggregate them ourselves rather
than reporting zero tokens.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from karenina.adapters.claude_agent_sdk import (
    extract_sdk_usage_from_messages,
)
from karenina.adapters.claude_agent_sdk.usage import (
    backfill_assistant_output_tokens,
    collapse_partial_assistant_messages,
)
from karenina.ports import UsageMetadata


def _assistant_message(usage: dict | None) -> MagicMock:
    """Build a mock that ``isinstance(msg, AssistantMessage)`` accepts.

    Populates the attributes that downstream trace/usage helpers touch so the
    mock can flow through ``_build_agent_result`` without AttributeError.
    """
    from claude_agent_sdk import AssistantMessage

    msg = MagicMock(spec=AssistantMessage)
    msg.usage = usage
    msg.content = []  # raw_trace builder iterates content blocks
    msg.model = None
    msg.parent_tool_use_id = None
    return msg


def _non_assistant_message() -> MagicMock:
    """A non-AssistantMessage to verify we filter it out."""
    msg = MagicMock()
    msg.usage = {"input_tokens": 9999, "output_tokens": 9999}
    return msg


class TestExtractSdkUsageFromMessages:
    def test_returns_zero_for_empty_list(self) -> None:
        usage = extract_sdk_usage_from_messages([])
        assert isinstance(usage, UsageMetadata)
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cache_read_tokens is None
        assert usage.cache_creation_tokens is None

    def test_sums_input_and_output_tokens(self) -> None:
        messages = [
            _assistant_message({"input_tokens": 100, "output_tokens": 20}),
            _assistant_message({"input_tokens": 150, "output_tokens": 30}),
            _assistant_message({"input_tokens": 200, "output_tokens": 40}),
        ]
        usage = extract_sdk_usage_from_messages(messages)
        assert usage.input_tokens == 450
        assert usage.output_tokens == 90
        assert usage.total_tokens == 540

    def test_propagates_model_parameter(self) -> None:
        usage = extract_sdk_usage_from_messages([], model="qwen-122b")
        assert usage.model == "qwen-122b"

    def test_aggregates_cache_tokens_when_reported(self) -> None:
        messages = [
            _assistant_message(
                {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 50,
                    "cache_creation_input_tokens": 5,
                }
            ),
            _assistant_message(
                {
                    "input_tokens": 80,
                    "output_tokens": 15,
                    "cache_read_input_tokens": 30,
                }
            ),
        ]
        usage = extract_sdk_usage_from_messages(messages)
        assert usage.input_tokens == 180
        assert usage.output_tokens == 35
        assert usage.cache_read_tokens == 80
        assert usage.cache_creation_tokens == 5

    def test_cache_fields_stay_none_when_never_reported(self) -> None:
        """If no message carries cache tokens, the field stays None (not 0).

        Distinguishes "endpoint did not report cache info" from "endpoint
        reported zero cache hits".
        """
        messages = [
            _assistant_message({"input_tokens": 50, "output_tokens": 10}),
            _assistant_message({"input_tokens": 60, "output_tokens": 12}),
        ]
        usage = extract_sdk_usage_from_messages(messages)
        assert usage.cache_read_tokens is None
        assert usage.cache_creation_tokens is None

    def test_skips_non_assistant_messages(self) -> None:
        """UserMessage, ResultMessage, tool results etc. must not be summed."""
        messages = [
            _assistant_message({"input_tokens": 100, "output_tokens": 20}),
            _non_assistant_message(),  # has usage dict but isn't AssistantMessage
            _assistant_message({"input_tokens": 50, "output_tokens": 10}),
        ]
        usage = extract_sdk_usage_from_messages(messages)
        assert usage.input_tokens == 150
        assert usage.output_tokens == 30

    def test_tolerates_missing_usage_on_some_messages(self) -> None:
        messages = [
            _assistant_message({"input_tokens": 100, "output_tokens": 20}),
            _assistant_message(None),
            _assistant_message({}),
            _assistant_message({"input_tokens": 50, "output_tokens": 10}),
        ]
        usage = extract_sdk_usage_from_messages(messages)
        assert usage.input_tokens == 150
        assert usage.output_tokens == 30

    def test_handles_none_values_in_usage_dict(self) -> None:
        """Some SDK versions populate fields with explicit None on missing data."""
        messages = [
            _assistant_message(
                {"input_tokens": None, "output_tokens": None}
            ),
            _assistant_message({"input_tokens": 75, "output_tokens": 15}),
        ]
        usage = extract_sdk_usage_from_messages(messages)
        assert usage.input_tokens == 75
        assert usage.output_tokens == 15


def _assistant_message_with_id(
    message_id: str | None,
    usage: dict | None,
) -> MagicMock:
    msg = _assistant_message(usage)
    msg.message_id = message_id
    return msg


def _stream_event(event: dict) -> MagicMock:
    from claude_agent_sdk import StreamEvent

    se = MagicMock(spec=StreamEvent)
    se.event = event
    return se


class TestCollapsePartialAssistantMessages:
    """The CLI subprocess emits one AssistantMessage per content_block_stop,
    each carrying only the single block that just finished, all sharing a
    message_id. Merge the content lists into one AssistantMessage per turn
    so the trace keeps thinking + text + tool_use, and iterations / input
    tokens reflect real LLM calls instead of per-block emissions."""

    def test_merges_content_blocks_across_partials(self) -> None:
        """Each partial carries ONE block; merge them all onto the last
        emission, preserving source order (thinking, text, tool_use)."""
        partial_1 = _assistant_message_with_id("msg_a", {"input_tokens": 100, "output_tokens": 0})
        partial_1.content = ["thinking_block"]
        partial_2 = _assistant_message_with_id("msg_a", {"input_tokens": 100, "output_tokens": 0})
        partial_2.content = ["text_block"]
        partial_3 = _assistant_message_with_id("msg_a", {"input_tokens": 100, "output_tokens": 0})
        partial_3.content = ["tool_use_block"]
        result = collapse_partial_assistant_messages([partial_1, partial_2, partial_3])
        assert len(result) == 1
        assert result[0] is partial_3  # the last emission is the survivor
        assert result[0].content == ["thinking_block", "text_block", "tool_use_block"]

    def test_preserves_distinct_message_ids(self) -> None:
        a1 = _assistant_message_with_id("msg_a", {"input_tokens": 50, "output_tokens": 0})
        a1.content = ["a_thinking"]
        a2 = _assistant_message_with_id("msg_a", {"input_tokens": 50, "output_tokens": 0})
        a2.content = ["a_tool_use"]
        b1 = _assistant_message_with_id("msg_b", {"input_tokens": 60, "output_tokens": 0})
        b1.content = ["b_thinking"]
        b2 = _assistant_message_with_id("msg_b", {"input_tokens": 60, "output_tokens": 0})
        b2.content = ["b_tool_use"]
        result = collapse_partial_assistant_messages([a1, a2, b1, b2])
        assert len(result) == 2
        assert result[0] is a2
        assert result[0].content == ["a_thinking", "a_tool_use"]
        assert result[1] is b2
        assert result[1].content == ["b_thinking", "b_tool_use"]

    def test_passes_through_non_assistant_messages(self) -> None:
        from claude_agent_sdk import StreamEvent

        a1 = _assistant_message_with_id("msg_a", {"input_tokens": 100, "output_tokens": 0})
        a1.content = ["block_1"]
        a2 = _assistant_message_with_id("msg_a", {"input_tokens": 100, "output_tokens": 0})
        a2.content = ["block_2"]
        se = MagicMock(spec=StreamEvent)
        user = _non_assistant_message()
        result = collapse_partial_assistant_messages([se, a1, user, a2])
        # se and user stay; a1 dropped; a2 stays at its position with merged content
        assert result == [se, user, a2]
        assert a2.content == ["block_1", "block_2"]

    def test_keeps_messages_without_message_id(self) -> None:
        """Defensive: if the SDK ever returns AssistantMessages without an id,
        we must not silently drop them by treating them all as one bucket."""
        a1 = _assistant_message_with_id(None, {"input_tokens": 100, "output_tokens": 5})
        a2 = _assistant_message_with_id(None, {"input_tokens": 200, "output_tokens": 7})
        result = collapse_partial_assistant_messages([a1, a2])
        assert len(result) == 2
        assert result[0] is a1
        assert result[1] is a2

    def test_returns_empty_for_empty_input(self) -> None:
        assert collapse_partial_assistant_messages([]) == []

    def test_collapse_then_aggregate_avoids_double_count(self) -> None:
        """End-to-end: without collapse, two partials with same input_tokens=100
        would sum to 200. After collapse, sum is 100."""
        a1 = _assistant_message_with_id("msg_a", {"input_tokens": 100, "output_tokens": 0})
        a1.content = ["text_block"]
        a2 = _assistant_message_with_id("msg_a", {"input_tokens": 100, "output_tokens": 7})
        a2.content = ["tool_use_block"]
        collapsed = collapse_partial_assistant_messages([a1, a2])
        usage = extract_sdk_usage_from_messages(collapsed)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 7

    def test_handles_empty_content_lists(self) -> None:
        """A partial may carry an empty content list (e.g. for the initial
        emission before any content_block_stop). It must still be merged
        without raising and the merged content stays empty."""
        a1 = _assistant_message_with_id("msg_a", {"input_tokens": 100, "output_tokens": 0})
        a1.content = []
        a2 = _assistant_message_with_id("msg_a", {"input_tokens": 100, "output_tokens": 0})
        a2.content = []
        result = collapse_partial_assistant_messages([a1, a2])
        assert len(result) == 1
        assert result[0].content == []


class TestBackfillAssistantOutputTokens:
    """vLLM and other non-canonical Anthropic endpoints defer output_tokens to
    the streaming message_delta event. With include_partial_messages=True the
    SDK surfaces those as StreamEvent objects, which this helper folds back
    into AssistantMessage.usage so downstream aggregation sees real counts."""

    def test_backfills_output_tokens_from_message_delta(self) -> None:
        msg = _assistant_message_with_id("msg_abc", {"input_tokens": 100, "output_tokens": 0})
        events = [
            _stream_event({"type": "message_start", "message": {"id": "msg_abc", "usage": {"input_tokens": 100, "output_tokens": 0}}}),
            _stream_event({"type": "content_block_delta", "delta": {}}),
            _stream_event({"type": "message_delta", "usage": {"input_tokens": 100, "output_tokens": 42}}),
            _stream_event({"type": "message_stop"}),
        ]
        backfill_assistant_output_tokens([*events, msg])
        assert msg.usage == {"input_tokens": 100, "output_tokens": 42}

    def test_correlates_multiple_messages_by_message_id(self) -> None:
        msg_a = _assistant_message_with_id("msg_a", {"input_tokens": 50, "output_tokens": 0})
        msg_b = _assistant_message_with_id("msg_b", {"input_tokens": 70, "output_tokens": 0})
        events = [
            _stream_event({"type": "message_start", "message": {"id": "msg_a"}}),
            _stream_event({"type": "message_delta", "usage": {"output_tokens": 11}}),
            _stream_event({"type": "message_start", "message": {"id": "msg_b"}}),
            _stream_event({"type": "message_delta", "usage": {"output_tokens": 22}}),
        ]
        backfill_assistant_output_tokens([*events, msg_a, msg_b])
        assert msg_a.usage["output_tokens"] == 11
        assert msg_b.usage["output_tokens"] == 22

    def test_does_not_clobber_non_zero_output_tokens(self) -> None:
        """Canonical Anthropic puts real output_tokens on AssistantMessage.usage
        directly; the delta value would be redundant but must not overwrite a
        non-zero existing count."""
        msg = _assistant_message_with_id("msg_x", {"input_tokens": 100, "output_tokens": 99})
        events = [
            _stream_event({"type": "message_start", "message": {"id": "msg_x"}}),
            _stream_event({"type": "message_delta", "usage": {"output_tokens": 42}}),
        ]
        backfill_assistant_output_tokens([*events, msg])
        assert msg.usage["output_tokens"] == 99

    def test_noop_when_no_stream_events_present(self) -> None:
        """Without include_partial_messages=True there are no StreamEvent
        records; the helper must leave AssistantMessages untouched."""
        msg = _assistant_message_with_id("msg_y", {"input_tokens": 100, "output_tokens": 0})
        backfill_assistant_output_tokens([msg])
        assert msg.usage["output_tokens"] == 0

    def test_noop_when_message_id_unknown_to_delta_map(self) -> None:
        msg = _assistant_message_with_id("msg_unmatched", {"input_tokens": 100, "output_tokens": 0})
        events = [
            _stream_event({"type": "message_start", "message": {"id": "msg_other"}}),
            _stream_event({"type": "message_delta", "usage": {"output_tokens": 42}}),
        ]
        backfill_assistant_output_tokens([*events, msg])
        assert msg.usage["output_tokens"] == 0

    def test_preserves_input_tokens_and_cache_fields(self) -> None:
        msg = _assistant_message_with_id(
            "msg_full",
            {
                "input_tokens": 100,
                "output_tokens": 0,
                "cache_read_input_tokens": 50,
                "cache_creation_input_tokens": 20,
            },
        )
        events = [
            _stream_event({"type": "message_start", "message": {"id": "msg_full"}}),
            _stream_event({"type": "message_delta", "usage": {"output_tokens": 30}}),
        ]
        backfill_assistant_output_tokens([*events, msg])
        assert msg.usage["input_tokens"] == 100
        assert msg.usage["output_tokens"] == 30
        assert msg.usage["cache_read_input_tokens"] == 50
        assert msg.usage["cache_creation_input_tokens"] == 20

    def test_aggregate_picks_up_backfilled_output_tokens(self) -> None:
        """End-to-end: after backfill, extract_sdk_usage_from_messages sums correctly."""
        msg_a = _assistant_message_with_id("msg_a", {"input_tokens": 50, "output_tokens": 0})
        msg_b = _assistant_message_with_id("msg_b", {"input_tokens": 70, "output_tokens": 0})
        all_messages = [
            _stream_event({"type": "message_start", "message": {"id": "msg_a"}}),
            _stream_event({"type": "message_delta", "usage": {"output_tokens": 11}}),
            msg_a,
            _stream_event({"type": "message_start", "message": {"id": "msg_b"}}),
            _stream_event({"type": "message_delta", "usage": {"output_tokens": 22}}),
            msg_b,
        ]
        backfill_assistant_output_tokens(all_messages)
        usage = extract_sdk_usage_from_messages(all_messages)
        assert usage.input_tokens == 120
        assert usage.output_tokens == 33


class TestBuildAgentResultUsesFallbackOnPartialTrace:
    """When ResultMessage is absent, _build_agent_result must use the per-message fallback.

    This is the regression scenario from the qwen122 / glm51 BixBench runs:
    csdk's wall-clock timeouts produced 0-token AgentResults despite tens of
    iterations because the SDK never emits ResultMessage on cancellation.
    """

    @pytest.fixture
    def adapter(self):
        from karenina.adapters.claude_agent_sdk.agent import (
            ClaudeSDKAgentAdapter,
        )
        from karenina.schemas.config.models import ModelConfig

        config = ModelConfig(
            id="answering",
            model_name="qwen3.5-122b-a10b",
            interface="claude_agent_sdk",
        )
        return ClaudeSDKAgentAdapter(config)

    def test_fallback_aggregates_when_result_message_is_none(self, adapter) -> None:
        messages = [
            _assistant_message({"input_tokens": 1000, "output_tokens": 50}),
            _assistant_message({"input_tokens": 1200, "output_tokens": 60}),
            _assistant_message({"input_tokens": 900, "output_tokens": 40}),
        ]
        result = adapter._build_agent_result(
            messages,
            result_message=None,
            limit_reached=False,
            timeout_reached=True,
        )
        assert result.usage.input_tokens == 3100
        assert result.usage.output_tokens == 150
        assert result.usage.total_tokens == 3250
        assert result.timeout_reached is True

    def test_uses_result_message_when_present(self, adapter) -> None:
        """The fallback must NOT override a real ResultMessage."""
        messages = [
            _assistant_message({"input_tokens": 1000, "output_tokens": 50}),
        ]
        result_message = MagicMock()
        result_message.usage = {"input_tokens": 9999, "output_tokens": 9999}
        result_message.total_cost_usd = 0.123
        result_message.num_turns = 5
        result_message.session_id = "sess-x"
        result_message.subtype = ""

        result = adapter._build_agent_result(
            messages,
            result_message=result_message,
            limit_reached=False,
            timeout_reached=False,
        )
        assert result.usage.input_tokens == 9999
        assert result.usage.output_tokens == 9999
        assert result.usage.cost_usd == 0.123
