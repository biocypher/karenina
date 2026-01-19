"""Unit tests for Message types in the ports module.

Tests cover:
- Role enum values and string compatibility
- ContentType enum values and string compatibility
- TextContent, ToolUseContent, ToolResultContent, ThinkingContent dataclasses
- Message class factory methods and properties
"""

from dataclasses import FrozenInstanceError

import pytest

from karenina.ports import (
    Content,
    ContentType,
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)

# =============================================================================
# Role Enum Tests
# =============================================================================


@pytest.mark.unit
class TestRole:
    """Tests for the Role enum."""

    def test_role_has_four_values(self) -> None:
        """Test that Role enum has exactly four values."""
        assert len(Role) == 4

    def test_role_system_value(self) -> None:
        """Test SYSTEM role value."""
        assert Role.SYSTEM == "system"
        assert Role.SYSTEM.value == "system"

    def test_role_user_value(self) -> None:
        """Test USER role value."""
        assert Role.USER == "user"
        assert Role.USER.value == "user"

    def test_role_assistant_value(self) -> None:
        """Test ASSISTANT role value."""
        assert Role.ASSISTANT == "assistant"
        assert Role.ASSISTANT.value == "assistant"

    def test_role_tool_value(self) -> None:
        """Test TOOL role value."""
        assert Role.TOOL == "tool"
        assert Role.TOOL.value == "tool"

    def test_role_is_string_compatible(self) -> None:
        """Test that Role inherits from str for string operations."""
        assert isinstance(Role.SYSTEM, str)
        assert Role.SYSTEM.upper() == "SYSTEM"
        # Direct string comparison works (not str() representation)
        assert Role.SYSTEM == "system"


# =============================================================================
# ContentType Enum Tests
# =============================================================================


@pytest.mark.unit
class TestContentType:
    """Tests for the ContentType enum."""

    def test_content_type_has_four_values(self) -> None:
        """Test that ContentType enum has exactly four values."""
        assert len(ContentType) == 4

    def test_content_type_text_value(self) -> None:
        """Test TEXT content type value."""
        assert ContentType.TEXT == "text"
        assert ContentType.TEXT.value == "text"

    def test_content_type_tool_use_value(self) -> None:
        """Test TOOL_USE content type value."""
        assert ContentType.TOOL_USE == "tool_use"
        assert ContentType.TOOL_USE.value == "tool_use"

    def test_content_type_tool_result_value(self) -> None:
        """Test TOOL_RESULT content type value."""
        assert ContentType.TOOL_RESULT == "tool_result"
        assert ContentType.TOOL_RESULT.value == "tool_result"

    def test_content_type_thinking_value(self) -> None:
        """Test THINKING content type value."""
        assert ContentType.THINKING == "thinking"
        assert ContentType.THINKING.value == "thinking"

    def test_content_type_is_string_compatible(self) -> None:
        """Test that ContentType inherits from str for string operations."""
        assert isinstance(ContentType.TEXT, str)
        assert ContentType.TEXT.upper() == "TEXT"


# =============================================================================
# TextContent Tests
# =============================================================================


@pytest.mark.unit
class TestTextContent:
    """Tests for the TextContent dataclass."""

    def test_text_content_creation_with_text(self) -> None:
        """Test TextContent creation with text field."""
        content = TextContent(text="Hello, world!")
        assert content.text == "Hello, world!"
        assert content.type == ContentType.TEXT

    def test_text_content_type_defaults_to_text(self) -> None:
        """Test that type defaults to ContentType.TEXT."""
        content = TextContent("Some text")
        assert content.type == ContentType.TEXT

    def test_text_content_is_frozen(self) -> None:
        """Test that TextContent is immutable (frozen)."""
        content = TextContent("Hello")
        with pytest.raises(FrozenInstanceError):
            content.text = "World"

    def test_text_content_empty_string(self) -> None:
        """Test TextContent with empty string."""
        content = TextContent("")
        assert content.text == ""


# =============================================================================
# ToolUseContent Tests
# =============================================================================


@pytest.mark.unit
class TestToolUseContent:
    """Tests for the ToolUseContent dataclass."""

    def test_tool_use_content_creation(self) -> None:
        """Test ToolUseContent creation with all required fields."""
        content = ToolUseContent(
            id="tool_123",
            name="search",
            input={"query": "test query"},
        )
        assert content.id == "tool_123"
        assert content.name == "search"
        assert content.input == {"query": "test query"}
        assert content.type == ContentType.TOOL_USE

    def test_tool_use_content_type_defaults_to_tool_use(self) -> None:
        """Test that type defaults to ContentType.TOOL_USE."""
        content = ToolUseContent(id="1", name="test", input={})
        assert content.type == ContentType.TOOL_USE

    def test_tool_use_content_is_frozen(self) -> None:
        """Test that ToolUseContent is immutable (frozen)."""
        content = ToolUseContent(id="1", name="test", input={})
        with pytest.raises(FrozenInstanceError):
            content.name = "modified"

    def test_tool_use_content_with_complex_input(self) -> None:
        """Test ToolUseContent with nested input dictionary."""
        complex_input = {
            "query": "search term",
            "options": {"limit": 10, "offset": 0},
            "filters": ["type:article", "date:2024"],
        }
        content = ToolUseContent(id="123", name="search", input=complex_input)
        assert content.input == complex_input


# =============================================================================
# ToolResultContent Tests
# =============================================================================


@pytest.mark.unit
class TestToolResultContent:
    """Tests for the ToolResultContent dataclass."""

    def test_tool_result_content_creation(self) -> None:
        """Test ToolResultContent creation with required fields."""
        content = ToolResultContent(
            tool_use_id="tool_123",
            content="Result data",
        )
        assert content.tool_use_id == "tool_123"
        assert content.content == "Result data"
        assert content.is_error is False
        assert content.type == ContentType.TOOL_RESULT

    def test_tool_result_content_is_error_true(self) -> None:
        """Test ToolResultContent with is_error=True."""
        content = ToolResultContent(
            tool_use_id="tool_123",
            content="Error: something went wrong",
            is_error=True,
        )
        assert content.is_error is True

    def test_tool_result_content_is_error_false(self) -> None:
        """Test ToolResultContent with explicit is_error=False."""
        content = ToolResultContent(
            tool_use_id="tool_123",
            content="Success",
            is_error=False,
        )
        assert content.is_error is False

    def test_tool_result_content_is_frozen(self) -> None:
        """Test that ToolResultContent is immutable (frozen)."""
        content = ToolResultContent(tool_use_id="1", content="result")
        with pytest.raises(FrozenInstanceError):
            content.is_error = True


# =============================================================================
# ThinkingContent Tests
# =============================================================================


@pytest.mark.unit
class TestThinkingContent:
    """Tests for the ThinkingContent dataclass."""

    def test_thinking_content_creation(self) -> None:
        """Test ThinkingContent creation with thinking field."""
        content = ThinkingContent(thinking="Let me think about this...")
        assert content.thinking == "Let me think about this..."
        assert content.signature is None
        assert content.type == ContentType.THINKING

    def test_thinking_content_with_signature(self) -> None:
        """Test ThinkingContent with signature."""
        content = ThinkingContent(
            thinking="My reasoning...",
            signature="abc123def456",
        )
        assert content.signature == "abc123def456"

    def test_thinking_content_without_signature(self) -> None:
        """Test ThinkingContent without signature (defaults to None)."""
        content = ThinkingContent(thinking="Thinking...")
        assert content.signature is None

    def test_thinking_content_is_frozen(self) -> None:
        """Test that ThinkingContent is immutable (frozen)."""
        content = ThinkingContent(thinking="Original")
        with pytest.raises(FrozenInstanceError):
            content.thinking = "Modified"


# =============================================================================
# Content Type Alias Tests
# =============================================================================


@pytest.mark.unit
class TestContentTypeAlias:
    """Tests for the Content type alias."""

    def test_text_content_is_content(self) -> None:
        """Test that TextContent is a valid Content type."""
        content: Content = TextContent("Hello")
        assert isinstance(content, TextContent)

    def test_tool_use_content_is_content(self) -> None:
        """Test that ToolUseContent is a valid Content type."""
        content: Content = ToolUseContent(id="1", name="test", input={})
        assert isinstance(content, ToolUseContent)

    def test_tool_result_content_is_content(self) -> None:
        """Test that ToolResultContent is a valid Content type."""
        content: Content = ToolResultContent(tool_use_id="1", content="result")
        assert isinstance(content, ToolResultContent)

    def test_thinking_content_is_content(self) -> None:
        """Test that ThinkingContent is a valid Content type."""
        content: Content = ThinkingContent(thinking="...")
        assert isinstance(content, ThinkingContent)


# =============================================================================
# Message Factory Method Tests
# =============================================================================


@pytest.mark.unit
class TestMessageFactoryMethods:
    """Tests for Message factory classmethods."""

    def test_message_system_creates_system_role(self) -> None:
        """Test Message.system() creates message with SYSTEM role."""
        msg = Message.system("You are a helpful assistant.")
        assert msg.role == Role.SYSTEM

    def test_message_system_creates_text_content(self) -> None:
        """Test Message.system() creates message with TextContent."""
        msg = Message.system("System prompt")
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "System prompt"

    def test_message_user_creates_user_role(self) -> None:
        """Test Message.user() creates message with USER role."""
        msg = Message.user("Hello!")
        assert msg.role == Role.USER

    def test_message_user_creates_text_content(self) -> None:
        """Test Message.user() creates message with TextContent."""
        msg = Message.user("User message")
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "User message"

    def test_message_assistant_creates_assistant_role(self) -> None:
        """Test Message.assistant() creates message with ASSISTANT role."""
        msg = Message.assistant("Hello, I'm here to help!")
        assert msg.role == Role.ASSISTANT

    def test_message_assistant_with_text_only(self) -> None:
        """Test Message.assistant() with text only."""
        msg = Message.assistant("Response text")
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "Response text"

    def test_message_assistant_with_tool_calls_only(self) -> None:
        """Test Message.assistant() with tool calls only (no text)."""
        tool_call = ToolUseContent(id="1", name="search", input={"q": "test"})
        msg = Message.assistant(tool_calls=[tool_call])

        assert msg.role == Role.ASSISTANT
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ToolUseContent)

    def test_message_assistant_with_text_and_tool_calls(self) -> None:
        """Test Message.assistant() with both text and tool calls."""
        tool_call = ToolUseContent(id="1", name="search", input={"q": "test"})
        msg = Message.assistant("Let me search for that.", tool_calls=[tool_call])

        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextContent)
        assert isinstance(msg.content[1], ToolUseContent)

    def test_message_assistant_empty_text_with_no_tool_calls(self) -> None:
        """Test Message.assistant() with empty text and no tool calls."""
        msg = Message.assistant("")
        assert msg.role == Role.ASSISTANT
        assert len(msg.content) == 0

    def test_message_tool_result_creates_tool_role(self) -> None:
        """Test Message.tool_result() creates message with TOOL role."""
        msg = Message.tool_result(tool_use_id="1", content="Result")
        assert msg.role == Role.TOOL

    def test_message_tool_result_creates_tool_result_content(self) -> None:
        """Test Message.tool_result() creates ToolResultContent."""
        msg = Message.tool_result(tool_use_id="tool_123", content="Success result")
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ToolResultContent)
        assert msg.content[0].tool_use_id == "tool_123"
        assert msg.content[0].content == "Success result"
        assert msg.content[0].is_error is False

    def test_message_tool_result_with_error(self) -> None:
        """Test Message.tool_result() with is_error=True."""
        msg = Message.tool_result(
            tool_use_id="tool_123",
            content="Error occurred",
            is_error=True,
        )
        assert msg.content[0].is_error is True


# =============================================================================
# Message Properties Tests
# =============================================================================


@pytest.mark.unit
class TestMessageProperties:
    """Tests for Message properties (text and tool_calls)."""

    def test_text_property_returns_text_content(self) -> None:
        """Test that text property returns TextContent text."""
        msg = Message.user("Hello, world!")
        assert msg.text == "Hello, world!"

    def test_text_property_joins_multiple_text_contents(self) -> None:
        """Test that text property joins multiple TextContent items."""
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent("First line"),
                TextContent("Second line"),
                TextContent("Third line"),
            ],
        )
        assert msg.text == "First line\nSecond line\nThird line"

    def test_text_property_ignores_non_text_content(self) -> None:
        """Test that text property ignores non-TextContent items."""
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent("Some text"),
                ToolUseContent(id="1", name="search", input={}),
                TextContent("More text"),
            ],
        )
        assert msg.text == "Some text\nMore text"

    def test_text_property_returns_empty_string_when_no_text(self) -> None:
        """Test that text property returns empty string when no TextContent."""
        tool_call = ToolUseContent(id="1", name="search", input={})
        msg = Message(role=Role.ASSISTANT, content=[tool_call])
        assert msg.text == ""

    def test_tool_calls_property_returns_tool_use_content(self) -> None:
        """Test that tool_calls property returns ToolUseContent items."""
        tool_call = ToolUseContent(id="1", name="search", input={"q": "test"})
        msg = Message.assistant("Using tool", tool_calls=[tool_call])

        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0] == tool_call

    def test_tool_calls_property_returns_multiple_tool_calls(self) -> None:
        """Test that tool_calls property returns all ToolUseContent items."""
        tool1 = ToolUseContent(id="1", name="search", input={"q": "test1"})
        tool2 = ToolUseContent(id="2", name="fetch", input={"url": "http://..."})
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent("Let me help"),
                tool1,
                tool2,
            ],
        )

        assert len(msg.tool_calls) == 2
        assert tool1 in msg.tool_calls
        assert tool2 in msg.tool_calls

    def test_tool_calls_property_filters_to_only_tool_use(self) -> None:
        """Test that tool_calls only returns ToolUseContent, not ToolResultContent."""
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent("Response"),
                ToolUseContent(id="1", name="search", input={}),
                ToolResultContent(tool_use_id="0", content="prev result"),
            ],
        )

        assert len(msg.tool_calls) == 1
        assert isinstance(msg.tool_calls[0], ToolUseContent)

    def test_tool_calls_property_returns_empty_list_when_none(self) -> None:
        """Test that tool_calls returns empty list when no ToolUseContent."""
        msg = Message.user("Hello")
        assert msg.tool_calls == []


# =============================================================================
# Message Direct Construction Tests
# =============================================================================


@pytest.mark.unit
class TestMessageDirectConstruction:
    """Tests for direct Message construction (without factory methods)."""

    def test_message_direct_construction(self) -> None:
        """Test Message can be constructed directly with role and content."""
        msg = Message(
            role=Role.USER,
            content=[TextContent("Direct construction")],
        )
        assert msg.role == Role.USER
        assert len(msg.content) == 1

    def test_message_with_mixed_content_types(self) -> None:
        """Test Message with multiple different content types."""
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent("Starting response"),
                ThinkingContent(thinking="Let me think..."),
                ToolUseContent(id="1", name="tool", input={}),
                TextContent("Continuing response"),
            ],
        )

        assert len(msg.content) == 4
        assert msg.text == "Starting response\nContinuing response"
        assert len(msg.tool_calls) == 1

    def test_message_with_empty_content_list(self) -> None:
        """Test Message with empty content list."""
        msg = Message(role=Role.ASSISTANT, content=[])
        assert len(msg.content) == 0
        assert msg.text == ""
        assert msg.tool_calls == []
