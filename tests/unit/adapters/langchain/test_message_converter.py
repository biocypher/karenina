"""Tests for LangChainMessageConverter.

Tests round-trip message conversion between unified Message and LangChain formats.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from karenina.adapters.langchain import LangChainMessageConverter
from karenina.ports import (
    Message,
    Role,
    ToolResultContent,
    ToolUseContent,
)


class TestLangChainMessageConverter:
    """Tests for LangChainMessageConverter."""

    @pytest.fixture
    def converter(self) -> LangChainMessageConverter:
        """Create a converter instance."""
        return LangChainMessageConverter()

    # -------------------------------------------------------------------------
    # to_provider tests
    # -------------------------------------------------------------------------

    def test_to_provider_system_message(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of system message to LangChain format."""
        messages = [Message.system("You are a helpful assistant")]
        result = converter.to_provider(messages)

        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are a helpful assistant"

    def test_to_provider_user_message(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of user message to LangChain format."""
        messages = [Message.user("Hello, world!")]
        result = converter.to_provider(messages)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello, world!"

    def test_to_provider_assistant_message_text_only(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of assistant message with text only."""
        messages = [Message.assistant("I am a helpful assistant")]
        result = converter.to_provider(messages)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "I am a helpful assistant"
        assert result[0].tool_calls == []

    def test_to_provider_assistant_message_with_tool_calls(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of assistant message with tool calls."""
        tool_call = ToolUseContent(
            id="call_123",
            name="search",
            input={"query": "test"},
        )
        messages = [Message.assistant("Let me search for that", tool_calls=[tool_call])]
        result = converter.to_provider(messages)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "Let me search for that"
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0]["id"] == "call_123"
        assert result[0].tool_calls[0]["name"] == "search"
        assert result[0].tool_calls[0]["args"] == {"query": "test"}

    def test_to_provider_tool_result_message(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of tool result message to LangChain format."""
        messages = [Message.tool_result(tool_use_id="call_123", content="Search results here")]
        result = converter.to_provider(messages)

        assert len(result) == 1
        assert isinstance(result[0], ToolMessage)
        assert result[0].content == "Search results here"
        assert result[0].tool_call_id == "call_123"

    def test_to_provider_multiple_messages(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of a conversation with multiple message types."""
        messages = [
            Message.system("You are helpful"),
            Message.user("Hi"),
            Message.assistant("Hello!"),
        ]
        result = converter.to_provider(messages)

        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)

    def test_to_provider_empty_list(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of empty message list."""
        result = converter.to_provider([])
        assert result == []

    # -------------------------------------------------------------------------
    # from_provider tests
    # -------------------------------------------------------------------------

    def test_from_provider_system_message(self, converter: LangChainMessageConverter) -> None:
        """Test conversion from LangChain SystemMessage."""
        lc_messages: list[BaseMessage] = [SystemMessage(content="You are helpful")]
        result = converter.from_provider(lc_messages)

        assert len(result) == 1
        assert result[0].role == Role.SYSTEM
        assert result[0].text == "You are helpful"

    def test_from_provider_human_message(self, converter: LangChainMessageConverter) -> None:
        """Test conversion from LangChain HumanMessage."""
        lc_messages: list[BaseMessage] = [HumanMessage(content="Hello")]
        result = converter.from_provider(lc_messages)

        assert len(result) == 1
        assert result[0].role == Role.USER
        assert result[0].text == "Hello"

    def test_from_provider_ai_message_text_only(self, converter: LangChainMessageConverter) -> None:
        """Test conversion from LangChain AIMessage with text only."""
        lc_messages: list[BaseMessage] = [AIMessage(content="I can help with that")]
        result = converter.from_provider(lc_messages)

        assert len(result) == 1
        assert result[0].role == Role.ASSISTANT
        assert result[0].text == "I can help with that"
        assert result[0].tool_calls == []

    def test_from_provider_ai_message_with_tool_calls_dict(self, converter: LangChainMessageConverter) -> None:
        """Test conversion from LangChain AIMessage with tool calls (dict format)."""
        lc_messages: list[BaseMessage] = [
            AIMessage(
                content="Searching...",
                tool_calls=[{"id": "call_456", "name": "search", "args": {"q": "test"}}],
            )
        ]
        result = converter.from_provider(lc_messages)

        assert len(result) == 1
        assert result[0].role == Role.ASSISTANT
        assert result[0].text == "Searching..."
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].id == "call_456"
        assert result[0].tool_calls[0].name == "search"
        assert result[0].tool_calls[0].input == {"q": "test"}

    def test_from_provider_tool_message(self, converter: LangChainMessageConverter) -> None:
        """Test conversion from LangChain ToolMessage."""
        lc_messages: list[BaseMessage] = [ToolMessage(content="Result data", tool_call_id="call_789")]
        result = converter.from_provider(lc_messages)

        assert len(result) == 1
        assert result[0].role == Role.TOOL
        assert len(result[0].content) == 1
        tool_result = result[0].content[0]
        assert isinstance(tool_result, ToolResultContent)
        assert tool_result.tool_use_id == "call_789"
        assert tool_result.content == "Result data"

    # -------------------------------------------------------------------------
    # Round-trip tests
    # -------------------------------------------------------------------------

    def test_roundtrip_system_message(self, converter: LangChainMessageConverter) -> None:
        """Test round-trip conversion preserves system message."""
        original = [Message.system("Test system prompt")]
        lc = converter.to_provider(original)
        roundtrip = converter.from_provider(lc)

        assert len(roundtrip) == 1
        assert roundtrip[0].role == original[0].role
        assert roundtrip[0].text == original[0].text

    def test_roundtrip_user_message(self, converter: LangChainMessageConverter) -> None:
        """Test round-trip conversion preserves user message."""
        original = [Message.user("User question")]
        lc = converter.to_provider(original)
        roundtrip = converter.from_provider(lc)

        assert len(roundtrip) == 1
        assert roundtrip[0].role == original[0].role
        assert roundtrip[0].text == original[0].text

    def test_roundtrip_assistant_with_tool_calls(self, converter: LangChainMessageConverter) -> None:
        """Test round-trip conversion preserves assistant message with tool calls."""
        tool_call = ToolUseContent(
            id="tc_001",
            name="calculator",
            input={"expression": "2+2"},
        )
        original = [Message.assistant("Computing...", tool_calls=[tool_call])]
        lc = converter.to_provider(original)
        roundtrip = converter.from_provider(lc)

        assert len(roundtrip) == 1
        assert roundtrip[0].role == Role.ASSISTANT
        assert roundtrip[0].text == "Computing..."
        assert len(roundtrip[0].tool_calls) == 1
        assert roundtrip[0].tool_calls[0].id == "tc_001"
        assert roundtrip[0].tool_calls[0].name == "calculator"
        assert roundtrip[0].tool_calls[0].input == {"expression": "2+2"}

    def test_roundtrip_full_conversation(self, converter: LangChainMessageConverter) -> None:
        """Test round-trip conversion of full conversation flow."""
        tool_call = ToolUseContent(id="tc_1", name="search", input={"q": "BCL2"})
        original = [
            Message.system("You are a biology expert"),
            Message.user("What is BCL2?"),
            Message.assistant("Let me search for that", tool_calls=[tool_call]),
            Message.tool_result(tool_use_id="tc_1", content="BCL2 is a gene..."),
            Message.assistant("BCL2 is a proto-oncogene..."),
        ]

        lc = converter.to_provider(original)
        roundtrip = converter.from_provider(lc)

        assert len(roundtrip) == 5
        assert roundtrip[0].role == Role.SYSTEM
        assert roundtrip[1].role == Role.USER
        assert roundtrip[2].role == Role.ASSISTANT
        assert roundtrip[3].role == Role.TOOL
        assert roundtrip[4].role == Role.ASSISTANT
