"""Tests for issue 170: log_trace(string) doubles AI message prefix.

When a plain string containing '--- AI Message ---' prefix is passed to
log_trace(), it gets wrapped as Message.assistant(text), which later
gets serialized with messages_to_raw_trace() adding another
'--- AI Message ---' prefix. This results in doubled prefixes in
raw_llm_response.
"""

import pytest

from karenina.benchmark.task_eval.task_eval import TaskEval
from karenina.ports.messages import Message


@pytest.mark.unit
class TestLogTraceStringPrefix:
    """Tests for issue 170: AI message prefix duplication in log_trace()."""

    def test_log_trace_strips_ai_message_prefix(self) -> None:
        """log_trace() should strip existing AI message prefix from string input.

        When a user passes a string that already contains the '--- AI Message ---'
        prefix, the method should strip it before wrapping it as a Message to
        avoid double-prefixing in the serialized output.
        """
        task = TaskEval(task_id="test")
        prefixed_text = "--- AI Message ---\nThe answer is 42"

        task.log_trace(prefixed_text)

        # The stored trace_messages should have the text without the prefix
        assert len(task.global_logs) == 1
        log_event = task.global_logs[0]
        assert log_event.trace_messages is not None
        assert len(log_event.trace_messages) == 1

        msg = log_event.trace_messages[0]
        assert msg.text == "The answer is 42"

    def test_log_trace_string_without_prefix_unchanged(self) -> None:
        """log_trace() should not alter strings without the AI message prefix."""
        task = TaskEval(task_id="test")
        plain_text = "The answer is 42"

        task.log_trace(plain_text)

        assert len(task.global_logs) == 1
        log_event = task.global_logs[0]
        assert log_event.trace_messages is not None
        msg = log_event.trace_messages[0]
        assert msg.text == "The answer is 42"

    def test_log_trace_message_list_not_affected(self) -> None:
        """log_trace() with a Message list should not be affected by prefix stripping."""
        task = TaskEval(task_id="test")
        messages = [Message.assistant("--- AI Message ---\nSome text")]

        task.log_trace(messages)

        # Message lists are passed through unchanged
        assert len(task.global_logs) == 1
        log_event = task.global_logs[0]
        assert log_event.trace_messages is not None
        msg = log_event.trace_messages[0]
        # The text should be unchanged since it was passed as a Message object
        assert msg.text == "--- AI Message ---\nSome text"
