"""Tests for scenario handover: TaggedMessage and transcript formatting."""

from __future__ import annotations

import pytest

from karenina.ports.messages import Message, ToolUseContent
from karenina.scenario.handover import TaggedMessage, format_transcript


@pytest.mark.unit
class TestTaggedMessage:
    def test_creation(self) -> None:
        msg = Message.user("hello")
        tm = TaggedMessage(message=msg, agent_id="primary")
        assert tm.message is msg
        assert tm.agent_id == "primary"


@pytest.mark.unit
class TestFormatTranscript:
    def test_simple_text_messages(self) -> None:
        tagged = [
            TaggedMessage(Message.user("What is BCL2?"), agent_id="__user__"),
            TaggedMessage(Message.assistant("BCL2 is a protein."), agent_id="primary"),
        ]
        result = format_transcript(tagged)
        assert "[__user__] What is BCL2?" in result
        assert "[primary:text] BCL2 is a protein." in result

    def test_tool_use_message(self) -> None:
        tool_call = ToolUseContent(id="t1", name="search", input={"q": "BCL2"})
        msg = Message.assistant("", tool_calls=[tool_call])
        tagged = [TaggedMessage(msg, agent_id="agent")]
        result = format_transcript(tagged)
        assert "[agent:tool_use]" in result
        assert "search" in result

    def test_tool_result_message(self) -> None:
        msg = Message.tool_result(tool_use_id="t1", content='{"found": true}')
        tagged = [TaggedMessage(msg, agent_id="agent")]
        result = format_transcript(tagged)
        assert "[agent:tool_result]" in result
        assert '{"found": true}' in result

    def test_thinking_blocks_excluded(self) -> None:
        from karenina.ports.messages import Role, TextContent, ThinkingContent

        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ThinkingContent(thinking="internal reasoning"),
                TextContent(text="visible answer"),
            ],
        )
        tagged = [TaggedMessage(msg, agent_id="agent")]
        result = format_transcript(tagged)
        assert "internal reasoning" not in result
        assert "[agent:text] visible answer" in result

    def test_empty_list_returns_empty_string(self) -> None:
        assert format_transcript([]) == ""

    def test_multiple_agents_labeled(self) -> None:
        tagged = [
            TaggedMessage(Message.user("Q1"), agent_id="__user__"),
            TaggedMessage(Message.assistant("A1"), agent_id="primary"),
            TaggedMessage(Message.user("Q2"), agent_id="__user__"),
            TaggedMessage(Message.assistant("A2"), agent_id="guardrail"),
        ]
        result = format_transcript(tagged)
        assert "[primary:text] A1" in result
        assert "[guardrail:text] A2" in result
