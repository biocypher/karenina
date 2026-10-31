"""Tests for codex trace conversion: canonical raw_trace and trace_messages."""

from __future__ import annotations

from karenina.adapters.codex_sdk.trace import (
    codex_items_to_raw_trace,
    codex_items_to_trace_messages,
    extract_final_response,
)

from .conftest import make_agent_message, make_command_execution, make_item, make_reasoning


def _full_turn_items() -> list:
    return [
        make_reasoning(content=["I should create the file."]),
        make_command_execution(item_id="cmd_1", command="echo hi > hello.txt", output="", exit_code=0),
        make_command_execution(item_id="cmd_2", command="cat missing.txt", output="no such file", exit_code=1),
        make_agent_message("Created hello.txt.", phase="final_answer"),
    ]


class TestRawTrace:
    def test_canonical_delimiters_present(self) -> None:
        trace = codex_items_to_raw_trace(_full_turn_items())
        assert "--- Thinking ---\nI should create the file." in trace
        assert "--- AI Message ---" in trace
        assert "--- Tool Message (call_id: cmd_1) ---" in trace

    def test_shell_tool_call_block_format(self) -> None:
        trace = codex_items_to_raw_trace(_full_turn_items())
        assert "Tool Calls:" in trace
        assert "  shell (call_cmd_1)" in trace
        assert "   Call ID: cmd_1" in trace
        assert "   Args: {'command': 'echo hi > hello.txt'}" in trace

    def test_failed_command_marked_as_error(self) -> None:
        trace = codex_items_to_raw_trace(_full_turn_items())
        assert "--- Tool Message (call_id: cmd_2) ---\n[ERROR] no such file" in trace

    def test_final_text_in_ai_message_section(self) -> None:
        trace = codex_items_to_raw_trace(_full_turn_items())
        assert "--- AI Message ---\nCreated hello.txt." in trace

    def test_user_messages_excluded_by_default(self) -> None:
        items = [make_item("userMessage", id="u1", content=[]), make_agent_message("hi")]
        trace = codex_items_to_raw_trace(items)
        assert "Human Message" not in trace

    def test_empty_items_give_empty_trace(self) -> None:
        assert codex_items_to_raw_trace([]) == ""


class TestTraceMessages:
    def test_structure_and_block_indices(self) -> None:
        messages = codex_items_to_trace_messages(_full_turn_items())
        assert [m["block_index"] for m in messages] == list(range(len(messages)))
        roles = [m["role"] for m in messages]
        # thinking, shell use, shell result, shell use, shell result, final text
        assert roles == ["assistant", "assistant", "tool", "assistant", "tool", "assistant"]

    def test_tool_use_and_result_pair_share_id(self) -> None:
        messages = codex_items_to_trace_messages(_full_turn_items())
        use = next(m for m in messages if m.get("tool_calls"))
        result = next(m for m in messages if m.get("tool_result"))
        assert use["tool_calls"][0]["id"] == result["tool_result"]["tool_use_id"] == "cmd_1"
        assert use["tool_calls"][0]["name"] == "shell"

    def test_error_result_flagged(self) -> None:
        messages = codex_items_to_trace_messages(_full_turn_items())
        failing = [m for m in messages if m.get("tool_result", {}).get("tool_use_id") == "cmd_2"]
        assert failing[0]["tool_result"]["is_error"] is True

    def test_thinking_metadata_present(self) -> None:
        messages = codex_items_to_trace_messages(_full_turn_items())
        assert messages[0]["thinking"] == {"thinking": "I should create the file."}


class TestExtractFinalResponse:
    def test_last_agent_message_wins(self) -> None:
        items = [make_agent_message("first"), make_agent_message("second")]
        assert extract_final_response(items) == "second"

    def test_none_when_no_text(self) -> None:
        assert extract_final_response([make_command_execution()]) is None
