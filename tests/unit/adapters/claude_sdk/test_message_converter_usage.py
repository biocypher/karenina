"""Tests that ClaudeSDKMessageConverter.from_provider populates usage_metadata.

Fix E (extension): the production trace path in claude_agent_sdk/agent.py
builds ``trace_messages`` via ``self._converter.from_provider(...)`` rather
than the direct-to-dict ``sdk_messages_to_trace_messages`` helper. This test
asserts that the converter attaches per-call ``usage_metadata`` to the
returned ``Message`` so that ``Message.to_dict()`` emits it into
``template.trace_messages[*]``.
"""

from __future__ import annotations

import pytest

from karenina.adapters.claude_agent_sdk import ClaudeSDKMessageConverter
from karenina.ports import Role


@pytest.mark.unit
class TestClaudeSDKConverterUsageMetadata:
    def test_assistant_message_carries_usage_metadata(self) -> None:
        pytest.importorskip("claude_agent_sdk.types")
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock

        sdk_messages = [
            AssistantMessage(
                model="claude-haiku-4-5",
                content=[TextBlock(text="The answer is 42.")],
                usage={"input_tokens": 123, "output_tokens": 45},
            ),
        ]

        converter = ClaudeSDKMessageConverter()
        result = converter.from_provider(sdk_messages)

        assert len(result) == 1
        msg = result[0]
        assert msg.role == Role.ASSISTANT
        assert msg.usage_metadata == {"input_tokens": 123, "output_tokens": 45}

    def test_to_dict_emits_usage_metadata_after_conversion(self) -> None:
        """End-to-end: converter populates field, to_dict emits it."""
        pytest.importorskip("claude_agent_sdk.types")
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock

        sdk_messages = [
            AssistantMessage(
                model="claude-haiku-4-5",
                content=[TextBlock(text="hi")],
                usage={"input_tokens": 7, "output_tokens": 3},
            ),
        ]

        converter = ClaudeSDKMessageConverter()
        [msg] = converter.from_provider(sdk_messages)
        out = msg.to_dict()

        assert out["role"] == "assistant"
        assert out["usage_metadata"] == {"input_tokens": 7, "output_tokens": 3}

    def test_cache_fields_propagate_when_reported(self) -> None:
        pytest.importorskip("claude_agent_sdk.types")
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock

        sdk_messages = [
            AssistantMessage(
                model="claude-haiku-4-5",
                content=[TextBlock(text="cached")],
                usage={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 800,
                    "cache_creation_input_tokens": 50,
                },
            ),
        ]

        converter = ClaudeSDKMessageConverter()
        [msg] = converter.from_provider(sdk_messages)

        assert msg.usage_metadata == {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 50,
        }

    def test_partial_cache_fields_only_include_reported(self) -> None:
        """A reported zero must be preserved; missing fields must stay absent."""
        pytest.importorskip("claude_agent_sdk.types")
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock

        sdk_messages = [
            AssistantMessage(
                model="claude-haiku-4-5",
                content=[TextBlock(text="partial")],
                usage={
                    "input_tokens": 60,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 0,
                },
            ),
        ]

        converter = ClaudeSDKMessageConverter()
        [msg] = converter.from_provider(sdk_messages)

        assert msg.usage_metadata is not None
        assert msg.usage_metadata["cache_read_input_tokens"] == 0
        assert "cache_creation_input_tokens" not in msg.usage_metadata

    def test_no_usage_yields_none_usage_metadata(self) -> None:
        pytest.importorskip("claude_agent_sdk.types")
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock

        sdk_messages = [
            AssistantMessage(
                model="claude-haiku-4-5",
                content=[TextBlock(text="no usage")],
            ),
        ]

        converter = ClaudeSDKMessageConverter()
        [msg] = converter.from_provider(sdk_messages)

        assert msg.usage_metadata is None
        # to_dict must omit the key entirely
        assert "usage_metadata" not in msg.to_dict()

    def test_empty_usage_dict_yields_none(self) -> None:
        pytest.importorskip("claude_agent_sdk.types")
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock

        sdk_messages = [
            AssistantMessage(
                model="claude-haiku-4-5",
                content=[TextBlock(text="empty usage")],
                usage={},
            ),
        ]

        converter = ClaudeSDKMessageConverter()
        [msg] = converter.from_provider(sdk_messages)

        assert msg.usage_metadata is None
        assert "usage_metadata" not in msg.to_dict()
