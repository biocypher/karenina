"""Tests for message conversion utilities in claude_tool adapter.

Tests extract_system_prompt, convert_to_anthropic, convert_from_anthropic_message,
convert_from_anthropic_messages, and build_system_with_cache.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from karenina.adapters.claude_tool.messages import (
    build_system_with_cache,
    convert_from_anthropic_message,
    convert_from_anthropic_messages,
    convert_to_anthropic,
    extract_system_prompt,
)
from karenina.ports import Message
from karenina.ports.messages import Role, TextContent, ToolResultContent, ToolUseContent


class TestExtractSystemPrompt:
    """Tests for extract_system_prompt function."""

    def test_extracts_single_system_prompt(self) -> None:
        """Test extraction of a single system prompt."""
        messages = [
            Message.system("You are a helpful assistant"),
            Message.user("Hello"),
        ]

        result = extract_system_prompt(messages)

        assert result == "You are a helpful assistant"

    def test_extracts_first_system_prompt(self) -> None:
        """Test extraction prioritizes first system message."""
        messages = [
            Message.system("First system"),
            Message.system("Second system"),
            Message.user("Hello"),
        ]

        result = extract_system_prompt(messages)

        assert result == "First system"

    def test_returns_none_when_no_system(self) -> None:
        """Test returns None when no system message present."""
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]

        result = extract_system_prompt(messages)

        assert result is None

    def test_returns_none_for_empty_list(self) -> None:
        """Test returns None for empty message list."""
        result = extract_system_prompt([])

        assert result is None

    def test_ignores_other_roles(self) -> None:
        """Test only extracts from system role messages."""
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi!"),
        ]

        result = extract_system_prompt(messages)

        assert result is None


class TestConvertToAnthropic:
    """Tests for convert_to_anthropic function."""

    def test_converts_user_message(self) -> None:
        """Test converting a simple user message."""
        messages = [Message.user("Hello, Claude!")]

        result = convert_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"type": "text", "text": "Hello, Claude!"}]

    def test_converts_assistant_message(self) -> None:
        """Test converting an assistant message."""
        messages = [Message.assistant("Hello, human!")]

        result = convert_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"type": "text", "text": "Hello, human!"}]

    def test_skips_system_messages(self) -> None:
        """Test that system messages are skipped (handled separately)."""
        messages = [
            Message.system("You are helpful"),
            Message.user("Hello"),
        ]

        result = convert_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_converts_conversation(self) -> None:
        """Test converting a multi-turn conversation."""
        messages = [
            Message.user("What is 2+2?"),
            Message.assistant("4"),
            Message.user("And 3+3?"),
        ]

        result = convert_to_anthropic(messages)

        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_converts_tool_use_content(self) -> None:
        """Test converting assistant message with tool use."""
        tool_use = ToolUseContent(
            id="tool_1",
            name="search",
            input={"query": "test"},
        )
        messages = [Message(role=Role.ASSISTANT, content=[tool_use])]

        result = convert_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]["content"][0]["type"] == "tool_use"
        assert result[0]["content"][0]["id"] == "tool_1"
        assert result[0]["content"][0]["name"] == "search"
        assert result[0]["content"][0]["input"] == {"query": "test"}

    def test_converts_tool_result_message(self) -> None:
        """Test converting tool result message."""
        tool_result = ToolResultContent(
            tool_use_id="tool_1",
            content="Search results here",
        )
        messages = [
            Message(role=Role.TOOL, content=[tool_result]),
        ]

        result = convert_to_anthropic(messages)

        # Tool results should be wrapped in a user message
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "tool_result"
        assert result[0]["content"][0]["tool_use_id"] == "tool_1"
        assert result[0]["content"][0]["content"] == "Search results here"

    def test_converts_tool_result_with_error(self) -> None:
        """Test converting tool result with error flag."""
        tool_result = ToolResultContent(
            tool_use_id="tool_1",
            content="Error occurred",
            is_error=True,
        )
        messages = [Message(role=Role.TOOL, content=[tool_result])]

        result = convert_to_anthropic(messages)

        assert result[0]["content"][0]["is_error"] is True

    def test_handles_empty_list(self) -> None:
        """Test handling empty message list."""
        result = convert_to_anthropic([])

        assert result == []

    def test_mixed_content_in_message(self) -> None:
        """Test message with both text and tool use content."""
        text_content = TextContent(text="Let me search for that")
        tool_use = ToolUseContent(id="tool_1", name="search", input={"q": "test"})
        messages = [
            Message(role=Role.ASSISTANT, content=[text_content, tool_use]),
        ]

        result = convert_to_anthropic(messages)

        assert len(result) == 1
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "tool_use"


class TestConvertFromAnthropicMessage:
    """Tests for convert_from_anthropic_message function."""

    def test_converts_text_response(self) -> None:
        """Test converting response with text content."""
        mock_response = MagicMock()
        mock_response.role = "assistant"
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello there!"
        mock_response.content = [mock_text_block]

        result = convert_from_anthropic_message(mock_response)

        assert result.role == Role.ASSISTANT
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "Hello there!"

    def test_converts_tool_use_response(self) -> None:
        """Test converting response with tool use."""
        mock_response = MagicMock()
        mock_response.role = "assistant"
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_123"
        mock_tool_block.name = "search"
        mock_tool_block.input = {"query": "test"}
        mock_response.content = [mock_tool_block]

        result = convert_from_anthropic_message(mock_response)

        assert result.role == Role.ASSISTANT
        assert len(result.content) == 1
        assert isinstance(result.content[0], ToolUseContent)
        assert result.content[0].id == "tool_123"
        assert result.content[0].name == "search"
        assert result.content[0].input == {"query": "test"}

    def test_converts_mixed_content_response(self) -> None:
        """Test converting response with mixed content types."""
        mock_response = MagicMock()
        mock_response.role = "assistant"

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Let me search"

        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_tool.id = "tool_1"
        mock_tool.name = "search"
        mock_tool.input = {}

        mock_response.content = [mock_text, mock_tool]

        result = convert_from_anthropic_message(mock_response)

        assert len(result.content) == 2
        assert isinstance(result.content[0], TextContent)
        assert isinstance(result.content[1], ToolUseContent)

    def test_handles_none_content(self) -> None:
        """Test handling response with None content."""
        mock_response = MagicMock()
        mock_response.role = "assistant"
        mock_response.content = None

        result = convert_from_anthropic_message(mock_response)

        assert result.role == Role.ASSISTANT
        assert result.content == []

    def test_handles_user_role(self) -> None:
        """Test handling user role in response (unusual but possible)."""
        mock_response = MagicMock()
        mock_response.role = "user"
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "User message"
        mock_response.content = [mock_text]

        result = convert_from_anthropic_message(mock_response)

        assert result.role == Role.USER


class TestConvertFromAnthropicMessages:
    """Tests for convert_from_anthropic_messages function."""

    def test_converts_multiple_messages(self) -> None:
        """Test converting a list of responses."""
        mock_resp1 = MagicMock()
        mock_resp1.role = "assistant"
        mock_text1 = MagicMock()
        mock_text1.type = "text"
        mock_text1.text = "First"
        mock_resp1.content = [mock_text1]

        mock_resp2 = MagicMock()
        mock_resp2.role = "assistant"
        mock_text2 = MagicMock()
        mock_text2.type = "text"
        mock_text2.text = "Second"
        mock_resp2.content = [mock_text2]

        result = convert_from_anthropic_messages([mock_resp1, mock_resp2])

        assert len(result) == 2
        assert result[0].content[0].text == "First"
        assert result[1].content[0].text == "Second"

    def test_handles_empty_list(self) -> None:
        """Test handling empty response list."""
        result = convert_from_anthropic_messages([])

        assert result == []


class TestBuildSystemWithCache:
    """Tests for build_system_with_cache function."""

    def test_builds_cached_system_prompt(self) -> None:
        """Test building system prompt with cache control."""
        result = build_system_with_cache("You are helpful")

        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "You are helpful"
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_returns_none_for_none_input(self) -> None:
        """Test returns None when system prompt is None."""
        result = build_system_with_cache(None)

        assert result is None

    def test_returns_none_for_empty_string(self) -> None:
        """Test returns None when system prompt is empty string."""
        result = build_system_with_cache("")

        assert result is None

    def test_preserves_complex_system_prompt(self) -> None:
        """Test preserves multi-line system prompts."""
        complex_prompt = """You are a helpful assistant.

Rules:
1. Be concise
2. Be accurate"""

        result = build_system_with_cache(complex_prompt)

        assert result is not None
        assert result[0]["text"] == complex_prompt
