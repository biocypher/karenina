"""Tests that Message.to_dict() emits the usage_metadata field when set.

Fix E (extension): the unified ``Message`` carries an optional
``usage_metadata`` dict so that per-call token / cache accounting is
preserved end-to-end into ``template.trace_messages[*]``.
"""

from __future__ import annotations

import pytest

from karenina.ports import Message, Role, TextContent


@pytest.mark.unit
class TestMessageToDictUsageMetadata:
    def test_to_dict_emits_usage_metadata_when_set(self) -> None:
        msg = Message(
            role=Role.ASSISTANT,
            content=[TextContent(text="42")],
            usage_metadata={"input_tokens": 100, "output_tokens": 20},
        )

        out = msg.to_dict()

        assert out["role"] == "assistant"
        assert out["content"] == "42"
        assert out["usage_metadata"] == {"input_tokens": 100, "output_tokens": 20}

    def test_to_dict_omits_usage_metadata_when_none(self) -> None:
        msg = Message(role=Role.ASSISTANT, content=[TextContent(text="hi")])

        out = msg.to_dict()

        assert "usage_metadata" not in out

    def test_to_dict_emits_usage_metadata_with_cache_fields(self) -> None:
        msg = Message(
            role=Role.ASSISTANT,
            content=[TextContent(text="cached")],
            usage_metadata={
                "input_tokens": 50,
                "output_tokens": 10,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 0,
            },
        )

        out = msg.to_dict()

        assert out["usage_metadata"] == {
            "input_tokens": 50,
            "output_tokens": 10,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 0,
        }

    def test_to_dict_emits_empty_usage_metadata_dict(self) -> None:
        """An explicit empty dict is distinct from None and should be emitted."""
        msg = Message(
            role=Role.ASSISTANT,
            content=[TextContent(text="weird")],
            usage_metadata={},
        )

        out = msg.to_dict()

        # Explicit empty dict is still 'set' (not None) → emit it.
        assert out["usage_metadata"] == {}

    def test_default_usage_metadata_is_none(self) -> None:
        msg = Message.assistant("hi")
        assert msg.usage_metadata is None

    def test_from_dict_preserves_usage_metadata(self) -> None:
        data = {
            "role": "assistant",
            "content": "cached answer",
            "block_index": 0,
            "usage_metadata": {
                "input_tokens": 50,
                "output_tokens": 10,
                "cache_read_input_tokens": 800,
            },
        }

        msg = Message.from_dict(data)

        assert msg.usage_metadata == {
            "input_tokens": 50,
            "output_tokens": 10,
            "cache_read_input_tokens": 800,
        }
        assert msg.to_dict()["usage_metadata"] == data["usage_metadata"]
