"""Tests for issue 140: Message.from_dict() duplicates content for tool messages.

When deserializing a tool message, from_dict() creates both a TextContent
block (from the "content" field) and a ToolResultContent block (from the
"tool_result" field) with the same text. This causes the .text property
to return content that should only be in the ToolResultContent block.
"""

import pytest

from karenina.ports.messages import (
    Message,
    Role,
    TextContent,
    ToolResultContent,
)


@pytest.mark.unit
class TestFromDictToolMessageDuplication:
    """Tests for issue 140: tool message content duplication in from_dict()."""

    def test_tool_message_roundtrip_no_duplicate_content(self) -> None:
        """A tool message should not have TextContent after from_dict() roundtrip.

        to_dict() puts the tool result content in the "content" field for
        serialization. from_dict() should not create a redundant TextContent
        block from it when a tool_result is present.
        """
        original = Message.tool_result(
            tool_use_id="call_123",
            content="Tool output text",
            is_error=False,
        )

        serialized = original.to_dict()
        restored = Message.from_dict(serialized)

        # Should have only ToolResultContent, no TextContent
        text_blocks = [c for c in restored.content if isinstance(c, TextContent)]
        tool_blocks = [c for c in restored.content if isinstance(c, ToolResultContent)]

        assert len(text_blocks) == 0, (
            f"Expected no TextContent blocks, got {len(text_blocks)}: "
            f"from_dict() should skip TextContent when tool_result is present"
        )
        assert len(tool_blocks) == 1
        assert tool_blocks[0].content == "Tool output text"
        assert tool_blocks[0].tool_use_id == "call_123"

    def test_tool_message_text_property_empty_after_roundtrip(self) -> None:
        """The .text property on a tool message should be empty after roundtrip.

        Tool messages store content in ToolResultContent, not TextContent.
        The .text property should return empty string.
        """
        original = Message.tool_result(
            tool_use_id="call_456",
            content="Some result",
        )

        serialized = original.to_dict()
        restored = Message.from_dict(serialized)

        assert restored.text == ""
        assert restored.role == Role.TOOL

    def test_non_tool_message_still_gets_text_content(self) -> None:
        """Non-tool messages should still get TextContent from the content field."""
        data = {
            "role": "assistant",
            "content": "Hello world",
            "block_index": 0,
        }

        msg = Message.from_dict(data)

        text_blocks = [c for c in msg.content if isinstance(c, TextContent)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "Hello world"
