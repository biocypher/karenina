"""Tests for the claude_tool adapter trace module."""

from karenina.adapters.claude_tool.trace import (
    claude_tool_messages_to_raw_trace,
    claude_tool_messages_to_trace_messages,
)
from karenina.ports.messages import (
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolUseContent,
)


class TestClaudeToolMessagesToRawTrace:
    """Tests for claude_tool_messages_to_raw_trace."""

    def test_single_assistant(self):
        messages = [Message.assistant("Hello")]
        result = claude_tool_messages_to_raw_trace(messages)
        assert "--- AI Message ---" in result
        assert "Hello" in result

    def test_multi_turn(self):
        tc = ToolUseContent(id="abc", name="search", input={"q": "test"})
        messages = [
            Message.assistant("Searching.", tool_calls=[tc]),
            Message.tool_result("abc", "found it"),
            Message.assistant("The answer is 42."),
        ]
        result = claude_tool_messages_to_raw_trace(messages)
        assert "--- AI Message ---" in result
        assert "--- Tool Message" in result
        assert "answer is 42" in result


class TestClaudeToolMessagesToTraceMessages:
    """Tests for claude_tool_messages_to_trace_messages."""

    def test_single_assistant(self):
        messages = [Message.assistant("Hello")]
        result = claude_tool_messages_to_trace_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hello"
        assert result[0]["block_index"] == 0

    def test_user_messages_skipped(self):
        messages = [
            Message.user("What?"),
            Message.assistant("42."),
        ]
        result = claude_tool_messages_to_trace_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"

    def test_tool_calls_included(self):
        tc = ToolUseContent(id="tc_1", name="search", input={"q": "test"})
        messages = [Message.assistant("Searching.", tool_calls=[tc])]
        result = claude_tool_messages_to_trace_messages(messages)
        assert len(result) == 1
        assert result[0]["tool_calls"][0]["name"] == "search"
        assert result[0]["tool_calls"][0]["id"] == "tc_1"

    def test_tool_result(self):
        messages = [Message.tool_result("tc_1", "results", is_error=False)]
        result = claude_tool_messages_to_trace_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_result"]["tool_use_id"] == "tc_1"
        assert result[0]["tool_result"]["is_error"] is False

    def test_tool_result_error(self):
        messages = [Message.tool_result("tc_1", "failed", is_error=True)]
        result = claude_tool_messages_to_trace_messages(messages)
        assert result[0]["tool_result"]["is_error"] is True

    def test_thinking_blocks(self):
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ThinkingContent(thinking="reasoning...", signature="sig"),
                TextContent(text="Answer."),
            ],
        )
        result = claude_tool_messages_to_trace_messages([msg])
        assert len(result) == 1
        assert result[0]["thinking"]["thinking"] == "reasoning..."
        assert result[0]["thinking"]["signature"] == "sig"

    def test_block_index_increments(self):
        tc = ToolUseContent(id="abc", name="search", input={})
        messages = [
            Message.assistant("Search.", tool_calls=[tc]),
            Message.tool_result("abc", "result"),
            Message.assistant("Answer."),
        ]
        result = claude_tool_messages_to_trace_messages(messages)
        assert len(result) == 3
        assert result[0]["block_index"] == 0
        assert result[1]["block_index"] == 1
        assert result[2]["block_index"] == 2
