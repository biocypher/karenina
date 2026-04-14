"""Tests for scenario handover: TaggedMessage, transcript formatting, and apply_handover."""

from __future__ import annotations

from pathlib import Path

import pytest

from karenina.ports.messages import Message, ToolUseContent
from karenina.scenario.handover import TaggedMessage, apply_handover, format_transcript
from karenina.schemas.scenario.types import ScenarioEdge


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
        assert "[primary:assistant:text] BCL2 is a protein." in result

    def test_tool_use_message(self) -> None:
        tool_call = ToolUseContent(id="t1", name="search", input={"q": "BCL2"})
        msg = Message.assistant("", tool_calls=[tool_call])
        tagged = [TaggedMessage(msg, agent_id="agent")]
        result = format_transcript(tagged)
        assert "[agent:assistant:tool_use]" in result
        assert "search" in result

    def test_tool_result_message(self) -> None:
        msg = Message.tool_result(tool_use_id="t1", content='{"found": true}')
        tagged = [TaggedMessage(msg, agent_id="agent")]
        result = format_transcript(tagged)
        assert "[agent:tool:tool_result]" in result
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
        assert "[agent:assistant:text] visible answer" in result

    def test_system_message_tagged_with_role(self) -> None:
        tagged = [
            TaggedMessage(Message.system("You are a biomedical expert."), agent_id="primary"),
            TaggedMessage(Message.user("What is BCL2?"), agent_id="__user__"),
            TaggedMessage(Message.assistant("BCL2 is a protein."), agent_id="primary"),
        ]
        result = format_transcript(tagged)
        assert "[primary:system:text] You are a biomedical expert." in result
        assert "[__user__] What is BCL2?" in result
        assert "[primary:assistant:text] BCL2 is a protein." in result

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
        assert "[primary:assistant:text] A1" in result
        assert "[guardrail:assistant:text] A2" in result


def _make_state(**overrides):
    from karenina.schemas.scenario.state import ScenarioState

    defaults = {
        "turn": 0,
        "current_node": "a",
        "verify_result": None,
        "parsed": {},
        "node_visits": {},
        "history": [],
        "accumulated": {},
        "node_results": {},
    }
    defaults.update(overrides)
    return ScenarioState(**defaults)


@pytest.mark.unit
class TestApplyHandover:
    def test_transcript_prepend(self) -> None:
        tagged = [
            TaggedMessage(Message.user("Q1"), agent_id="__user__"),
            TaggedMessage(Message.assistant("A1"), agent_id="primary"),
        ]
        edge = ScenarioEdge(source="a", target="b", handover="transcript_prepend")
        question_text = "Review the conversation above."
        result_text, history = apply_handover(edge, tagged, _make_state(), question_text)
        assert "[__user__] Q1" in result_text
        assert "[primary:assistant:text] A1" in result_text
        assert result_text.index("[primary:assistant:text] A1") < result_text.index("Review the conversation above.")
        assert history == []

    def test_transcript_append(self) -> None:
        tagged = [
            TaggedMessage(Message.user("Q1"), agent_id="__user__"),
            TaggedMessage(Message.assistant("A1"), agent_id="primary"),
        ]
        edge = ScenarioEdge(source="a", target="b", handover="transcript_append")
        question_text = "Review the conversation below."
        result_text, history = apply_handover(edge, tagged, _make_state(), question_text)
        assert result_text.index("Review the conversation below.") < result_text.index("[primary:assistant:text] A1")
        assert history == []

    def test_callable_handover(self) -> None:
        tagged = [
            TaggedMessage(Message.user("Q1"), agent_id="__user__"),
            TaggedMessage(Message.assistant("A1"), agent_id="primary"),
        ]

        def my_handover(msgs, state):
            return [Message.system("custom context")]

        edge = ScenarioEdge(source="a", target="b", handover_callable=my_handover)
        result_text, history = apply_handover(edge, tagged, _make_state(), "original question")
        assert result_text == "original question"
        assert len(history) == 1
        assert history[0].text == "custom context"

    def test_no_handover_returns_none(self) -> None:
        edge = ScenarioEdge(source="a", target="b")
        result = apply_handover(edge, [], _make_state(), "Q")
        assert result is None

    def test_separator_between_transcript_and_question(self) -> None:
        tagged = [
            TaggedMessage(Message.user("Q1"), agent_id="__user__"),
        ]
        edge = ScenarioEdge(source="a", target="b", handover="transcript_prepend")
        result_text, _ = apply_handover(edge, tagged, _make_state(), "My question")
        assert "---" in result_text


@pytest.mark.unit
class TestTranscriptMaterialize:
    def test_materialize_writes_file_and_enriches_question(self, tmp_path: Path) -> None:
        tagged = [
            TaggedMessage(Message.user("What is BCL2?"), agent_id="__user__"),
            TaggedMessage(Message.assistant("BCL2 is a protein."), agent_id="primary"),
        ]
        edge = ScenarioEdge(source="a", target="b", handover="transcript_materialize")
        result_text, history = apply_handover(
            edge,
            tagged,
            _make_state(),
            "Evaluate sycophancy.",
            turn_dir=tmp_path,
        )
        # Original question text preserved after the preamble
        assert "Evaluate sycophancy." in result_text
        # Preamble with file path comes before the question
        assert "Trace file path:" in result_text
        assert "traces/" in result_text
        assert result_text.index("Trace file path:") < result_text.index("Evaluate sycophancy.")
        # MUST read instruction present
        assert "MUST read" in result_text
        # History is empty (same pattern as transcript_prepend)
        assert history == []
        # File actually exists
        trace_files = list((tmp_path / "traces").glob("*.txt"))
        assert len(trace_files) == 1
        # Trace file should NOT contain the question text (only transcript)
        trace_content = trace_files[0].read_text()
        assert "Evaluate sycophancy." not in trace_content

    def test_materialize_without_workspace_falls_back_to_tempdir(self) -> None:
        tagged = [
            TaggedMessage(Message.user("Q1"), agent_id="__user__"),
        ]
        edge = ScenarioEdge(source="a", target="b", handover="transcript_materialize")
        result_text, history = apply_handover(
            edge,
            tagged,
            _make_state(),
            "Evaluate.",
            turn_dir=None,
        )
        assert "MUST read" in result_text
        assert history == []
