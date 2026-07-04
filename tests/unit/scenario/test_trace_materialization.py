"""Tests for scenario trace materialization utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from karenina.ports.messages import Message
from karenina.scenario.trace_materialization import (
    format_turns_as_xml,
    group_entries_into_turns,
    materialize_trace,
    parse_transcript_entries,
    reformat_transcript_as_xml,
    serialize_conversation_history,
)


@pytest.mark.unit
class TestParseTranscriptEntries:
    def test_user_tag(self) -> None:
        text = "[__user__] What is BCL2?"
        entries = parse_transcript_entries(text)
        assert len(entries) == 1
        assert entries[0]["role"] == "user"
        assert entries[0]["agent_id"] is None
        assert entries[0]["content_type"] == "text"
        assert entries[0]["content"] == "What is BCL2?"

    def test_three_part_tag(self) -> None:
        text = "[primary:assistant:text] BCL2 is a protein."
        entries = parse_transcript_entries(text)
        assert len(entries) == 1
        assert entries[0]["role"] == "assistant"
        assert entries[0]["agent_id"] == "primary"
        assert entries[0]["content_type"] == "text"

    def test_system_tag(self) -> None:
        text = "[primary:system:text] You are a biomedical expert."
        entries = parse_transcript_entries(text)
        assert entries[0]["role"] == "system"

    def test_tool_use_tag(self) -> None:
        text = "[agent:assistant:tool_use] search(q='BCL2')"
        entries = parse_transcript_entries(text)
        assert entries[0]["content_type"] == "tool_use"

    def test_multiline_continuation(self) -> None:
        text = "[primary:assistant:text] Line 1\nLine 2\nLine 3"
        entries = parse_transcript_entries(text)
        assert len(entries) == 1
        assert "Line 2" in entries[0]["content"]
        assert "Line 3" in entries[0]["content"]

    def test_multiple_entries(self) -> None:
        text = "[__user__] Q1\n[primary:assistant:text] A1\n[__user__] Q2"
        entries = parse_transcript_entries(text)
        assert len(entries) == 3

    def test_bracketed_agent_id_suffix(self) -> None:
        """Agent display strings like ``name +[otp]`` must parse intact."""
        tag = "[openai_endpoint:qwen3.5-122b-a10b +[otp]:assistant:tool_use] search(q='x')"
        entries = parse_transcript_entries(tag)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["agent_id"] == "openai_endpoint:qwen3.5-122b-a10b +[otp]"
        assert entry["role"] == "assistant"
        assert entry["content_type"] == "tool_use"
        assert entry["content"] == "search(q='x')"

    def test_nested_brackets_in_agent_id(self) -> None:
        """Multiple bracket groups inside the agent_id still parse."""
        tag = "[foo[a]bar[b]:user:text] hi"
        entries = parse_transcript_entries(tag)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["agent_id"] == "foo[a]bar[b]"
        assert entry["role"] == "user"
        assert entry["content_type"] == "text"
        assert entry["content"] == "hi"

    def test_plain_agent_id_regression_guard(self) -> None:
        """Original bracket-free agent_id parsing must keep working."""
        entries = parse_transcript_entries("[agent:assistant:text] hello")
        assert len(entries) == 1
        entry = entries[0]
        assert entry["agent_id"] == "agent"
        assert entry["role"] == "assistant"
        assert entry["content_type"] == "text"
        assert entry["content"] == "hello"


@pytest.mark.unit
class TestGroupEntriesIntoTurns:
    def test_user_starts_new_turn(self) -> None:
        entries = [
            {"role": "user", "agent_id": None, "content_type": "text", "content": "Q1"},
            {"role": "assistant", "agent_id": "primary", "content_type": "text", "content": "A1"},
        ]
        turns = group_entries_into_turns(entries)
        assert len(turns) == 1
        assert turns[0]["user_content"] == "Q1"
        assert len(turns[0]["blocks"]) == 1

    def test_system_entry_becomes_system_prompt(self) -> None:
        entries = [
            {"role": "system", "agent_id": "primary", "content_type": "text", "content": "Be helpful."},
            {"role": "user", "agent_id": None, "content_type": "text", "content": "Q1"},
            {"role": "assistant", "agent_id": "primary", "content_type": "text", "content": "A1"},
        ]
        turns = group_entries_into_turns(entries)
        assert len(turns) == 1
        assert turns[0]["system_prompt"] == "Be helpful."
        assert turns[0]["system_agent"] == "primary"

    def test_multiple_turns(self) -> None:
        entries = [
            {"role": "user", "agent_id": None, "content_type": "text", "content": "Q1"},
            {"role": "assistant", "agent_id": "a", "content_type": "text", "content": "A1"},
            {"role": "user", "agent_id": None, "content_type": "text", "content": "Q2"},
            {"role": "assistant", "agent_id": "b", "content_type": "text", "content": "A2"},
        ]
        turns = group_entries_into_turns(entries)
        assert len(turns) == 2
        assert turns[0]["agent_id"] == "a"
        assert turns[1]["agent_id"] == "b"

    def test_assistant_only_entries_produce_single_turn(self) -> None:
        """QA traces lack a user tag; callers still expect one turn."""
        entries = [
            {"role": "assistant", "agent_id": "primary", "content_type": "text", "content": "reasoning"},
            {"role": "assistant", "agent_id": "primary", "content_type": "tool_use", "content": "search(q='x')"},
            {"role": "tool", "agent_id": "primary", "content_type": "tool_result", "content": "hit"},
            {"role": "assistant", "agent_id": "primary", "content_type": "text", "content": "done"},
        ]
        turns = group_entries_into_turns(entries)
        assert len(turns) == 1
        turn = turns[0]
        assert turn["user_content"] is None
        assert turn["agent_id"] == "primary"
        assert turn["system_prompt"] is None
        assert len(turn["blocks"]) == 4

    def test_assistant_only_with_leading_system(self) -> None:
        """Leading system entry still populates system_prompt on the synthesized turn."""
        entries = [
            {"role": "system", "agent_id": "primary", "content_type": "text", "content": "Be helpful."},
            {"role": "assistant", "agent_id": "primary", "content_type": "text", "content": "ok"},
        ]
        turns = group_entries_into_turns(entries)
        assert len(turns) == 1
        assert turns[0]["system_prompt"] == "Be helpful."
        assert turns[0]["system_agent"] == "primary"
        assert turns[0]["user_content"] is None
        assert len(turns[0]["blocks"]) == 1


@pytest.mark.unit
class TestFormatTurnsAsXml:
    def test_basic_turn_structure(self) -> None:
        turns = [
            {
                "system_prompt": None,
                "system_agent": None,
                "user_content": "What is BCL2?",
                "agent_id": "primary",
                "blocks": [
                    {"role": "assistant", "agent_id": "primary", "content_type": "text", "content": "A protein."},
                ],
            },
        ]
        xml = format_turns_as_xml(turns)
        assert '<turn number="1">' in xml
        assert "<user>" in xml
        assert "What is BCL2?" in xml
        assert '<assistant agent="primary">' in xml
        assert "A protein." in xml

    def test_system_prompt_element(self) -> None:
        turns = [
            {
                "system_prompt": "Be helpful.",
                "system_agent": "primary",
                "user_content": "Q1",
                "agent_id": "primary",
                "blocks": [],
            },
        ]
        xml = format_turns_as_xml(turns)
        assert '<system_prompt agent="primary">' in xml
        assert "Be helpful." in xml

    def test_tool_call_name_attribute(self) -> None:
        turns = [
            {
                "system_prompt": None,
                "system_agent": None,
                "user_content": "Search for BCL2",
                "agent_id": "agent",
                "blocks": [
                    {
                        "role": "assistant",
                        "agent_id": "agent",
                        "content_type": "tool_use",
                        "content": "search(q='BCL2')",
                    },
                    {"role": "tool", "agent_id": "agent", "content_type": "tool_result", "content": '{"found": true}'},
                ],
            },
        ]
        xml = format_turns_as_xml(turns)
        assert '<tool_call name="search">' in xml
        assert '<tool_result name="search">' in xml

    def test_offloading_large_blocks(self, tmp_path: Path) -> None:
        long_content = "x" * 3000
        turns = [
            {
                "system_prompt": None,
                "system_agent": None,
                "user_content": "Q",
                "agent_id": "a",
                "blocks": [
                    {"role": "assistant", "agent_id": "a", "content_type": "text", "content": long_content},
                ],
            },
        ]
        xml = format_turns_as_xml(turns, artifacts_dir=tmp_path, truncation_threshold=2000)
        assert 'offloaded="true"' in xml
        assert "3,000 chars" in xml
        offloaded_files = list(tmp_path.glob("*.txt"))
        assert len(offloaded_files) == 1
        assert offloaded_files[0].read_text() == long_content


@pytest.mark.unit
class TestSerializeConversationHistory:
    def test_basic_serialization(self) -> None:
        history = [
            Message.user("What is BCL2?"),
            Message.assistant("BCL2 is a protein."),
        ]
        result = serialize_conversation_history(history)
        assert "--- User Message ---" in result
        assert "What is BCL2?" in result
        assert "--- Assistant Message ---" in result
        assert "BCL2 is a protein." in result


@pytest.mark.unit
class TestReformatTranscriptAsXml:
    def test_plain_text_returned_unchanged(self) -> None:
        text = "Just a plain question."
        assert reformat_transcript_as_xml(text) == text

    def test_transcript_prepend_pattern_detected(self) -> None:
        transcript = "[__user__] Q1\n[primary:assistant:text] A1"
        separator = "\n\n---\n\n"
        question = "Evaluate the conversation."
        text = transcript + separator + question
        result = reformat_transcript_as_xml(text)
        assert "<turn" in result
        assert "Evaluate the conversation." in result

    def test_no_user_tags_returns_unchanged(self) -> None:
        text = "no tags here\n\n---\n\nquestion"
        assert reformat_transcript_as_xml(text) == text


@pytest.mark.unit
class TestMaterializeTrace:
    def test_writes_file_and_returns_path(self, tmp_path: Path) -> None:
        path = materialize_trace(
            question_text="Evaluate sycophancy.",
            conversation_history=None,
            trace_dir=tmp_path,
            question_id="q1",
        )
        assert path.exists()
        content = path.read_text()
        assert "# KARENINA CONVERSATION TRACE" in content
        assert "Evaluate sycophancy." in content

    def test_includes_conversation_history(self, tmp_path: Path) -> None:
        history = [Message.user("Q1"), Message.assistant("A1")]
        path = materialize_trace(
            question_text="Evaluate.",
            conversation_history=history,
            trace_dir=tmp_path,
            question_id="q2",
        )
        content = path.read_text()
        assert "CONVERSATION HISTORY" in content
        assert "Q1" in content

    def test_falls_back_to_tempdir_when_no_workspace(self) -> None:
        path = materialize_trace(
            question_text="Evaluate.",
            conversation_history=None,
            trace_dir=None,
            question_id="q3",
        )
        assert path.exists()
        # Clean up temp directory
        import shutil

        shutil.rmtree(path.parent, ignore_errors=True)

    def test_scenario_turn_in_filename(self, tmp_path: Path) -> None:
        path = materialize_trace(
            question_text="Evaluate.",
            conversation_history=None,
            trace_dir=tmp_path,
            question_id="q4",
            scenario_turn=2,
        )
        assert "_turn2_" in path.name

    def test_xml_reformatting_applied(self, tmp_path: Path) -> None:
        transcript = "[__user__] Q1\n[primary:assistant:text] A1"
        separator = "\n\n---\n\n"
        question = "Evaluate the conversation."
        path = materialize_trace(
            question_text=transcript + separator + question,
            conversation_history=None,
            trace_dir=tmp_path,
            question_id="q5",
        )
        content = path.read_text()
        assert "<turn" in content

    def test_env_threshold_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KARENINA_TRACE_TRUNCATION_THRESHOLD", "100")
        long_transcript = "[__user__] Q1\n[primary:assistant:text] " + "x" * 200
        separator = "\n\n---\n\n"
        path = materialize_trace(
            question_text=long_transcript + separator + "Evaluate.",
            conversation_history=None,
            trace_dir=tmp_path,
            question_id="q6",
        )
        artifacts_dir = path.parent / "artifacts"
        assert artifacts_dir.exists()
