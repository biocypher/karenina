"""Tests for ClaudeSDKMessageConverter.

Tests message conversion to/from SDK format.
"""

from __future__ import annotations

import pytest

from karenina.adapters.claude_agent_sdk import ClaudeSDKMessageConverter
from karenina.ports import Message


class TestClaudeSDKMessageConverter:
    """Tests for ClaudeSDKMessageConverter."""

    @pytest.fixture
    def converter(self) -> ClaudeSDKMessageConverter:
        """Create a converter instance."""
        return ClaudeSDKMessageConverter()

    # -------------------------------------------------------------------------
    # to_prompt_string tests
    # -------------------------------------------------------------------------

    def test_to_prompt_string_single_user_message(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test converting a single user message to prompt string."""
        messages = [Message.user("What is BCL2?")]
        result = converter.to_prompt_string(messages)

        assert result == "What is BCL2?"

    def test_to_prompt_string_multiple_user_messages(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test converting multiple user messages to prompt string."""
        messages = [
            Message.user("First question"),
            Message.user("Second question"),
        ]
        result = converter.to_prompt_string(messages)

        assert result == "First question\n\nSecond question"

    def test_to_prompt_string_ignores_system_messages(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test that system messages are not included in prompt string."""
        messages = [
            Message.system("You are a helpful assistant"),
            Message.user("Hello"),
        ]
        result = converter.to_prompt_string(messages)

        assert result == "Hello"
        assert "helpful assistant" not in result

    def test_to_prompt_string_ignores_assistant_messages(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test that assistant messages are not included in prompt string."""
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there!"),
            Message.user("Follow-up question"),
        ]
        result = converter.to_prompt_string(messages)

        assert result == "Hello\n\nFollow-up question"
        assert "Hi there" not in result

    def test_to_prompt_string_empty_list(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test converting empty message list."""
        result = converter.to_prompt_string([])
        assert result == ""

    def test_to_prompt_string_no_user_messages(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test converting when there are no user messages."""
        messages = [
            Message.system("System prompt"),
            Message.assistant("Response"),
        ]
        result = converter.to_prompt_string(messages)

        assert result == ""

    # -------------------------------------------------------------------------
    # extract_system_prompt tests
    # -------------------------------------------------------------------------

    def test_extract_system_prompt_single(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test extracting a single system prompt."""
        messages = [Message.system("You are a biology expert")]
        result = converter.extract_system_prompt(messages)

        assert result == "You are a biology expert"

    def test_extract_system_prompt_multiple(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test extracting multiple system prompts."""
        messages = [
            Message.system("Be helpful"),
            Message.system("Be concise"),
        ]
        result = converter.extract_system_prompt(messages)

        assert result == "Be helpful\n\nBe concise"

    def test_extract_system_prompt_ignores_other_roles(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test that non-system messages are not extracted."""
        messages = [
            Message.system("System prompt"),
            Message.user("User message"),
            Message.assistant("Assistant response"),
        ]
        result = converter.extract_system_prompt(messages)

        assert result == "System prompt"
        assert "User message" not in result
        assert "Assistant response" not in result

    def test_extract_system_prompt_no_system_messages(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test extraction when no system messages present."""
        messages = [Message.user("Hello")]
        result = converter.extract_system_prompt(messages)

        assert result is None

    def test_extract_system_prompt_empty_list(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test extraction with empty message list."""
        result = converter.extract_system_prompt([])

        assert result is None

    # -------------------------------------------------------------------------
    # from_provider tests (with SDK not installed)
    # -------------------------------------------------------------------------

    def test_from_provider_empty_list(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test converting empty SDK message list."""
        result = converter.from_provider([])

        # With empty input, should return empty list
        assert result == []
