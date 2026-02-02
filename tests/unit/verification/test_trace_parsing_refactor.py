"""Tests for the refactored trace_parsing module that accepts both str and list[Message]."""

from karenina.benchmark.verification.utils.trace_parsing import (
    extract_final_ai_message,
    prepare_evaluation_input,
)
from karenina.ports.messages import Message, Role, ToolUseContent


class TestExtractFinalAiMessageWithMessages:
    """Tests for extract_final_ai_message with list[Message] input."""

    def test_single_assistant_message(self):
        messages = [Message.assistant("The answer is 42.")]
        result, error = extract_final_ai_message(messages)
        assert result == "The answer is 42."
        assert error is None

    def test_multi_turn_with_tools(self):
        tc = ToolUseContent(id="abc", name="search", input={})
        messages = [
            Message.assistant("Searching.", tool_calls=[tc]),
            Message.tool_result("abc", "results"),
            Message.assistant("The answer is 42."),
        ]
        result, error = extract_final_ai_message(messages)
        assert result == "The answer is 42."
        assert error is None

    def test_empty_message_list(self):
        result, error = extract_final_ai_message([])
        assert result is None
        assert "Empty" in error

    def test_last_message_is_tool(self):
        messages = [
            Message.assistant("Searching."),
            Message.tool_result("abc", "results"),
        ]
        result, error = extract_final_ai_message(messages)
        assert result is None
        assert "not an AI message" in error

    def test_assistant_with_only_tool_calls(self):
        tc = ToolUseContent(id="abc", name="search", input={})
        messages = [Message(role=Role.ASSISTANT, content=[tc])]
        result, error = extract_final_ai_message(messages)
        assert result is None
        assert "no text content" in error


class TestExtractFinalAiMessageWithString:
    """Existing string-based tests still work."""

    def test_simple_trace(self):
        trace = "--- AI Message ---\nThe answer is 42."
        result, error = extract_final_ai_message(trace)
        assert result == "The answer is 42."
        assert error is None

    def test_plain_text(self):
        result, error = extract_final_ai_message("Just a plain answer")
        assert result == "Just a plain answer"
        assert error is None

    def test_empty_string(self):
        result, error = extract_final_ai_message("")
        assert result is None
        assert error is not None


class TestPrepareEvaluationInputWithMessages:
    """Tests for prepare_evaluation_input with list[Message] input."""

    def test_full_trace_mode(self):
        messages = [
            Message.user("Question"),
            Message.assistant("The answer is 42."),
        ]
        result, error = prepare_evaluation_input(messages, use_full_trace=True)
        assert error is None
        # Should return the full raw trace string
        assert "AI Message" in result
        assert "42" in result

    def test_extract_final_mode(self):
        messages = [
            Message.assistant("Searching."),
            Message.tool_result("abc", "results"),
            Message.assistant("The answer is 42."),
        ]
        result, error = prepare_evaluation_input(messages, use_full_trace=False)
        assert result == "The answer is 42."
        assert error is None

    def test_extract_final_mode_with_string(self):
        """String input still works."""
        trace = "--- AI Message ---\nI'll search.\n\n--- AI Message ---\nThe answer is 42."
        result, error = prepare_evaluation_input(trace, use_full_trace=False)
        assert result == "The answer is 42."
        assert error is None
