"""Tests for public stored-trace formatting."""

import pytest

from karenina.benchmark import abstention_detection_instruction, format_trace_messages
from karenina.ports.messages import Message, ToolUseContent


@pytest.mark.unit
class TestFormatTraceMessages:
    """Check live and stored message inputs."""

    def test_formats_stored_tool_calls_and_results(self) -> None:
        messages = [
            Message.assistant("Let me check.").to_dict(),
            Message.assistant(
                "Searching",
                tool_calls=[ToolUseContent(id="call-1", name="search", input={"q": "BCL2"})],
            ).to_dict(),
            Message.tool_result("call-1", "BCL2 is the reported target").to_dict(),
        ]
        trace = format_trace_messages(messages)
        assert "Tool Calls:" in trace
        assert "search" in trace
        assert "BCL2 is the reported target" in trace

    def test_accepts_live_messages(self) -> None:
        assert format_trace_messages([Message.assistant("Final answer")]) == "--- AI Message ---\nFinal answer"

    def test_exposes_standard_abstention_instruction(self) -> None:
        instruction = abstention_detection_instruction()
        assert "Find a verbatim capability disclaimer" in instruction
        assert "Check whether the response still commits to an answer" in instruction
