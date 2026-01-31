"""Tests for the unified trace formatting module."""

from karenina.benchmark.verification.utils.trace_formatting import (
    DEFAULT_TRACE_FORMAT,
    TraceFormatConfig,
    messages_to_raw_trace,
)
from karenina.ports.messages import (
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolUseContent,
)


class TestMessagesToRawTrace:
    """Tests for messages_to_raw_trace."""

    def test_empty_messages(self):
        result = messages_to_raw_trace([])
        assert result == ""

    def test_single_assistant_message(self):
        messages = [Message.assistant("The answer is 42.")]
        result = messages_to_raw_trace(messages)
        assert result == "--- AI Message ---\nThe answer is 42."

    def test_assistant_with_tool_calls(self):
        tc = ToolUseContent(id="abc123", name="search", input={"query": "test"})
        messages = [Message.assistant("Let me search.", tool_calls=[tc])]
        result = messages_to_raw_trace(messages)
        assert "--- AI Message ---" in result
        assert "Let me search." in result
        assert "Tool Calls:" in result
        assert "search (call_abc123)" in result

    def test_tool_result_message(self):
        messages = [Message.tool_result("call_abc", "Search results here")]
        result = messages_to_raw_trace(messages)
        assert "--- Tool Message (call_id: call_abc) ---" in result
        assert "Search results here" in result

    def test_tool_result_error(self):
        messages = [Message.tool_result("call_abc", "Connection failed", is_error=True)]
        result = messages_to_raw_trace(messages)
        assert "[ERROR] Connection failed" in result

    def test_multi_turn_conversation(self):
        tc = ToolUseContent(id="abc", name="search", input={"q": "test"})
        messages = [
            Message.assistant("Let me search.", tool_calls=[tc]),
            Message.tool_result("abc", "results found"),
            Message.assistant("Based on the results, the answer is 42."),
        ]
        result = messages_to_raw_trace(messages)
        parts = result.split("\n\n")
        assert len(parts) == 3
        assert parts[0].startswith("--- AI Message ---")
        assert parts[1].startswith("--- Tool Message")
        assert parts[2].startswith("--- AI Message ---")
        assert "answer is 42" in parts[2]

    def test_user_messages_excluded_by_default(self):
        messages = [
            Message.user("What is the answer?"),
            Message.assistant("42."),
        ]
        result = messages_to_raw_trace(messages)
        assert "Human Message" not in result
        assert "--- AI Message ---" in result

    def test_user_messages_included_with_config(self):
        config = TraceFormatConfig(include_user_messages=True)
        messages = [
            Message.user("What is the answer?"),
            Message.assistant("42."),
        ]
        result = messages_to_raw_trace(messages, config=config)
        assert "--- Human Message ---" in result
        assert "What is the answer?" in result

    def test_system_messages_excluded_by_default(self):
        messages = [
            Message.system("You are helpful."),
            Message.assistant("Hello!"),
        ]
        result = messages_to_raw_trace(messages)
        assert "System Message" not in result

    def test_system_messages_included_with_config(self):
        config = TraceFormatConfig(include_system_messages=True)
        messages = [
            Message.system("You are helpful."),
            Message.assistant("Hello!"),
        ]
        result = messages_to_raw_trace(messages, config=config)
        assert "--- System Message ---" in result

    def test_thinking_blocks(self):
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ThinkingContent(thinking="Let me think..."),
                TextContent(text="The answer."),
            ],
        )
        result = messages_to_raw_trace([msg])
        assert "--- Thinking ---" in result
        assert "Let me think..." in result
        assert "--- AI Message ---" in result

    def test_custom_headers(self):
        config = TraceFormatConfig(
            assistant_header="== Assistant ==",
            tool_header_template="== Tool ({call_id}) ==",
        )
        messages = [
            Message.assistant("Hello"),
            Message.tool_result("tc_1", "result"),
        ]
        result = messages_to_raw_trace(messages, config=config)
        assert "== Assistant ==" in result
        assert "== Tool (tc_1) ==" in result


class TestTraceFormatConfig:
    """Tests for TraceFormatConfig defaults."""

    def test_default_config_values(self):
        config = DEFAULT_TRACE_FORMAT
        assert config.assistant_header == "--- AI Message ---"
        assert config.include_user_messages is False
        assert config.include_system_messages is False
        assert config.include_tool_calls is True
        assert config.tool_call_format == "inline"
