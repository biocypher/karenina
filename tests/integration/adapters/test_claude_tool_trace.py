"""Integration tests for claude_tool adapter trace collection.

Validates that the claude_tool adapter correctly captures tool result messages
in trace_messages and raw_trace when using MCP tools.

Uses fixtures captured from real API calls via:
    python scripts/capture_fixtures.py --scenario claude_tool_trace

Fixture: tests/fixtures/llm_responses/claude-haiku-4-5/claude_tool_trace/
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_DIR = (
    Path(__file__).parent.parent.parent / "fixtures" / "llm_responses" / "claude-haiku-4-5" / "claude_tool_trace"
)


def _load_trace_fixture() -> dict | None:
    """Load the first trace fixture from the fixture directory."""
    if not FIXTURE_DIR.exists():
        return None
    fixtures = list(FIXTURE_DIR.glob("trace_*.json"))
    if not fixtures:
        return None
    with fixtures[0].open() as f:
        return json.load(f)


# Skip all tests if fixture not available
_fixture = _load_trace_fixture()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_fixture is None, reason="claude_tool_trace fixture not captured"),
]


class TestClaudeToolTraceStructure:
    """Tests that the captured trace has the correct message structure."""

    def test_trace_messages_not_empty(self) -> None:
        """trace_messages must contain at least one message."""
        trace_msgs = _fixture["trace_messages"]
        assert len(trace_msgs) > 0, "trace_messages should not be empty"

    def test_trace_contains_assistant_messages(self) -> None:
        """Trace must contain assistant messages."""
        trace_msgs = _fixture["trace_messages"]
        roles = [m["role"] for m in trace_msgs]
        assert "assistant" in roles, "No assistant messages in trace"

    def test_trace_contains_tool_messages(self) -> None:
        """Trace must contain tool result messages (the bug we fixed)."""
        trace_msgs = _fixture["trace_messages"]
        roles = [m["role"] for m in trace_msgs]
        assert "tool" in roles, (
            "No tool result messages in trace_messages. "
            "This was the original bug: tool_runner yields only assistant messages."
        )

    def test_trace_message_roles_are_valid(self) -> None:
        """All messages must have a valid role."""
        valid_roles = {"assistant", "tool", "user", "system"}
        trace_msgs = _fixture["trace_messages"]
        for i, msg in enumerate(trace_msgs):
            assert msg["role"] in valid_roles, f"Message [{i}] has invalid role: {msg['role']}"

    def test_assistant_messages_have_content_or_tool_calls(self) -> None:
        """Assistant messages must have content or tool_calls (or both)."""
        trace_msgs = _fixture["trace_messages"]
        for i, msg in enumerate(trace_msgs):
            if msg["role"] != "assistant":
                continue
            has_content = bool(msg.get("content"))
            has_tool_calls = bool(msg.get("tool_calls"))
            assert has_content or has_tool_calls, f"Assistant message [{i}] has neither content nor tool_calls"

    def test_tool_messages_have_tool_result(self) -> None:
        """Tool messages must have a tool_result with tool_use_id and is_error."""
        trace_msgs = _fixture["trace_messages"]
        for i, msg in enumerate(trace_msgs):
            if msg["role"] != "tool":
                continue
            tr = msg.get("tool_result")
            assert tr is not None, f"Tool message [{i}] missing tool_result"
            assert "tool_use_id" in tr, f"Tool message [{i}] missing tool_use_id"
            assert "is_error" in tr, f"Tool message [{i}] missing is_error"


class TestClaudeToolTraceToolCallLinking:
    """Tests that tool_call → tool_result links are correct."""

    def test_every_tool_call_has_matching_result(self) -> None:
        """Every tool_call in an assistant message must have a matching tool_result."""
        trace_msgs = _fixture["trace_messages"]

        for j, msg in enumerate(trace_msgs):
            if msg["role"] != "assistant" or not msg.get("tool_calls"):
                continue
            for tc in msg["tool_calls"]:
                tc_id = tc["id"]
                tc_name = tc["name"]
                # Search for matching tool_result in subsequent messages
                found = any(m2.get("tool_result", {}).get("tool_use_id") == tc_id for m2 in trace_msgs[j + 1 :])
                assert found, (
                    f"tool_call '{tc_name}' (id={tc_id[:20]}...) at message [{j}] "
                    f"has no matching tool_result in subsequent messages"
                )

    def test_tool_results_follow_their_tool_calls(self) -> None:
        """Tool result messages must appear after their corresponding tool_call."""
        trace_msgs = _fixture["trace_messages"]

        # Build map: tool_use_id -> index of assistant message containing the tool_call
        tool_call_indices: dict[str, int] = {}
        for j, msg in enumerate(trace_msgs):
            for tc in msg.get("tool_calls", []):
                tool_call_indices[tc["id"]] = j

        # Check that each tool_result appears after its tool_call
        for k, msg in enumerate(trace_msgs):
            tr = msg.get("tool_result")
            if not tr:
                continue
            use_id = tr["tool_use_id"]
            call_idx = tool_call_indices.get(use_id)
            assert call_idx is not None, f"tool_result at [{k}] references unknown tool_use_id: {use_id[:20]}..."
            assert k > call_idx, f"tool_result at [{k}] appears before its tool_call at [{call_idx}]"

    def test_interleaved_assistant_tool_pattern(self) -> None:
        """Messages should follow assistant → tool → assistant → tool pattern."""
        trace_msgs = _fixture["trace_messages"]
        roles = [m["role"] for m in trace_msgs]

        # First message should be assistant
        assert roles[0] == "assistant", f"First message should be assistant, got {roles[0]}"

        # After an assistant with tool_calls, the next message(s) should be tool
        for j, msg in enumerate(trace_msgs):
            if msg["role"] == "assistant" and msg.get("tool_calls") and j + 1 < len(trace_msgs):
                assert trace_msgs[j + 1]["role"] == "tool", (
                    f"After assistant with tool_calls at [{j}], "
                    f"expected tool message at [{j + 1}], got {trace_msgs[j + 1]['role']}"
                )


class TestClaudeToolTraceToolCallContent:
    """Tests that tool_call content is properly structured."""

    def test_tool_calls_have_required_fields(self) -> None:
        """Each tool_call must have id, name, and input."""
        trace_msgs = _fixture["trace_messages"]
        for j, msg in enumerate(trace_msgs):
            for tc in msg.get("tool_calls", []):
                assert "id" in tc, f"tool_call in message [{j}] missing 'id'"
                assert "name" in tc, f"tool_call in message [{j}] missing 'name'"
                assert "input" in tc, f"tool_call in message [{j}] missing 'input'"

    def test_tool_result_content_is_string(self) -> None:
        """Tool result content should be a string."""
        trace_msgs = _fixture["trace_messages"]
        for i, msg in enumerate(trace_msgs):
            if msg["role"] != "tool":
                continue
            content = msg.get("content", "")
            assert isinstance(content, str), f"Tool message [{i}] content should be str, got {type(content)}"


class TestClaudeToolRawTrace:
    """Tests that raw_trace (string format) is correct."""

    def test_raw_trace_not_empty(self) -> None:
        """raw_trace must not be empty."""
        raw_trace = _fixture["raw_trace"]
        assert len(raw_trace) > 0, "raw_trace should not be empty"

    def test_raw_trace_has_ai_messages(self) -> None:
        """raw_trace must contain AI Message sections."""
        raw_trace = _fixture["raw_trace"]
        assert "--- AI Message ---" in raw_trace, "raw_trace missing '--- AI Message ---' sections"

    def test_raw_trace_has_tool_messages(self) -> None:
        """raw_trace must contain Tool Message sections (the bug we fixed)."""
        raw_trace = _fixture["raw_trace"]
        assert "--- Tool Message" in raw_trace, (
            "raw_trace missing '--- Tool Message' sections. "
            "This was the original bug: tool results not appearing in raw_trace."
        )

    def test_raw_trace_tool_message_count_matches(self) -> None:
        """Number of Tool Message sections in raw_trace should match tool messages in trace_messages."""
        raw_trace = _fixture["raw_trace"]
        trace_msgs = _fixture["trace_messages"]

        n_tool_sections = raw_trace.count("--- Tool Message")
        n_tool_messages = sum(1 for m in trace_msgs if m["role"] == "tool")

        assert n_tool_sections == n_tool_messages, (
            f"raw_trace has {n_tool_sections} Tool Message sections "
            f"but trace_messages has {n_tool_messages} tool messages"
        )


class TestClaudeToolTraceMetadata:
    """Tests that fixture metadata is present and valid."""

    def test_fixture_has_metadata(self) -> None:
        """Fixture must have metadata section."""
        assert "metadata" in _fixture

    def test_fixture_has_verify_result(self) -> None:
        """Fixture must have verify_result."""
        assert "verify_result" in _fixture

    def test_metadata_has_model(self) -> None:
        """Metadata must record the model used."""
        assert "model" in _fixture["metadata"]

    def test_metadata_has_question_id(self) -> None:
        """Metadata must record the question_id."""
        assert "question_id" in _fixture["metadata"]
