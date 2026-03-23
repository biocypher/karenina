"""Tests for TaskEval issues.

Issue 170: log_trace(string) doubles AI message prefix.
Issue 025: task.log() text-only logs fail template parsing in TaskEval.
"""

import pytest

from karenina.benchmark.task_eval.helpers import merge_logs_and_traces
from karenina.benchmark.task_eval.models import LogEvent
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


@pytest.mark.unit
class TestIssue025LogTextOnlyRegression:
    """Regression tests for issue 025: text-only logs must produce Messages.

    merge_logs_and_traces() converts text-only LogEvents (those with .text
    but no .trace_messages) into Message.assistant() objects so they are
    available to the verification pipeline for template parsing. Without
    this conversion, text-only logs were silently dropped, causing parsing
    failures.
    """

    def test_text_only_log_produces_messages(self) -> None:
        """A LogEvent with only text (no trace_messages) produces a Message list.

        The "concatenate" strategy must convert text-only logs into assistant
        Messages so the pipeline can parse them.
        """
        logs = [LogEvent(level="info", text="The answer is 42")]

        response_text, messages = merge_logs_and_traces(logs, strategy="concatenate")

        assert messages is not None
        assert len(messages) == 1
        assert messages[0].role.value == "assistant"
        assert messages[0].text == "The answer is 42"

    def test_text_only_log_response_text_nonempty(self) -> None:
        """The response_text string is non-empty for text-only logs.

        merge_logs_and_traces() must return a non-empty string when text-only
        logs are present, since downstream stages rely on response_text being
        populated.
        """
        logs = [LogEvent(level="info", text="Some model output")]

        response_text, messages = merge_logs_and_traces(logs, strategy="concatenate")

        assert response_text != ""
        assert "Some model output" in response_text

    def test_text_only_and_trace_mixed(self) -> None:
        """Mix of text-only and trace LogEvents both contribute to output.

        When using "concatenate", both text-only logs (converted to Messages)
        and trace-bearing logs (with existing Message objects) should appear
        in the final message list.
        """
        trace_msg = Message.assistant("Trace response content")
        logs = [
            LogEvent(level="info", text="Text-only log entry"),
            LogEvent(level="info", text="", trace_messages=[trace_msg]),
        ]

        response_text, messages = merge_logs_and_traces(logs, strategy="concatenate")

        assert messages is not None
        assert len(messages) == 2
        # The text-only log is converted to an assistant Message (processed first)
        assert messages[0].text == "Text-only log entry"
        # The trace message is included directly (processed second)
        assert messages[1].text == "Trace response content"

    def test_traces_only_strategy_ignores_text_logs(self) -> None:
        """The "traces_only" strategy skips text-only LogEvents.

        Text-only logs must not appear in the output when the strategy is
        "traces_only", even if they have non-empty text.
        """
        trace_msg = Message.assistant("From trace")
        logs = [
            LogEvent(level="info", text="Should be ignored"),
            LogEvent(level="info", text="", trace_messages=[trace_msg]),
        ]

        response_text, messages = merge_logs_and_traces(logs, strategy="traces_only")

        assert messages is not None
        assert len(messages) == 1
        assert messages[0].text == "From trace"

    def test_empty_text_log_skipped(self) -> None:
        """A LogEvent with empty or whitespace-only text is skipped.

        Line 48 of helpers.py checks `log.text.strip()` before converting
        to a Message. Whitespace-only text should not produce a Message.
        """
        logs = [
            LogEvent(level="info", text=""),
            LogEvent(level="info", text="   "),
            LogEvent(level="info", text="\n\t"),
        ]

        response_text, messages = merge_logs_and_traces(logs, strategy="concatenate")

        assert response_text == ""
        assert messages is None
