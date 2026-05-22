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
