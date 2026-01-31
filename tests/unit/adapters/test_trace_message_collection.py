"""Cross-adapter trace message collection tests.

Verifies that all three adapter interfaces (claude_tool, claude_agent_sdk,
langchain) produce consistent trace_messages structure for the same
conversation patterns. The TypeScript frontend expects a uniform
TraceMessage[] shape regardless of which adapter produced the trace.

Each test scenario checks:
- Correct number of messages
- Correct roles (assistant, tool)
- Presence of tool_calls on assistant messages that invoke tools
- Presence of tool_result metadata on tool messages
- Correct tool_use_id linking between tool_calls and tool_results
- Correct content in tool result messages
"""

from __future__ import annotations

import pytest

from karenina.ports.messages import (
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolUseContent,
)

# ======================================================================
# Helpers
# ======================================================================


def _assert_trace_shape(
    trace: list[dict],
    expected_roles: list[str],
) -> None:
    """Assert the trace has the expected role sequence and block_index progression."""
    assert len(trace) == len(expected_roles), (
        f"Expected {len(expected_roles)} messages, got {len(trace)}: {[m['role'] for m in trace]}"
    )
    for i, (msg, expected_role) in enumerate(zip(trace, expected_roles, strict=False)):
        assert msg["role"] == expected_role, f"Message {i}: expected role '{expected_role}', got '{msg['role']}'"
        assert msg["block_index"] == i, f"Message {i}: expected block_index {i}, got {msg['block_index']}"


def _assert_tool_call_link(trace: list[dict], assistant_idx: int, tool_idx: int) -> None:
    """Assert a tool_call in an assistant message links to a tool result message."""
    assistant_msg = trace[assistant_idx]
    tool_msg = trace[tool_idx]

    assert "tool_calls" in assistant_msg, f"Message {assistant_idx} has no tool_calls"
    assert "tool_result" in tool_msg, f"Message {tool_idx} has no tool_result"

    # Find matching tool_call -> tool_result by id
    tool_call_ids = {tc["id"] for tc in assistant_msg["tool_calls"]}
    tool_result_id = tool_msg["tool_result"]["tool_use_id"]
    assert tool_result_id in tool_call_ids, (
        f"tool_result.tool_use_id '{tool_result_id}' not found in tool_calls ids: {tool_call_ids}"
    )


# ======================================================================
# Port Message (claude_tool adapter) tests
# ======================================================================


class TestClaudeToolTraceCollection:
    """Tests for trace message collection via the claude_tool adapter's trace module.

    These test the trace output shape that the adapter produces when tool
    result messages ARE properly captured (the bug fix).
    """

    def _to_trace(self, messages: list[Message]) -> list[dict]:
        from karenina.adapters.claude_tool.trace import claude_tool_messages_to_trace_messages

        return claude_tool_messages_to_trace_messages(messages)

    def test_simple_response_no_tools(self) -> None:
        """Single assistant response without tool use."""
        messages = [Message.assistant("The answer is 42.")]
        trace = self._to_trace(messages)
        _assert_trace_shape(trace, ["assistant"])
        assert trace[0]["content"] == "The answer is 42."
        assert "tool_calls" not in trace[0]

    def test_single_tool_call_round_trip(self) -> None:
        """Assistant calls a tool, gets result, then responds."""
        tc = ToolUseContent(id="tc_001", name="search", input={"query": "BCL2"})
        messages = [
            Message.assistant("Let me search.", tool_calls=[tc]),
            Message.tool_result("tc_001", '{"gene": "BCL2", "score": 0.95}'),
            Message.assistant("BCL2 has a score of 0.95."),
        ]
        trace = self._to_trace(messages)
        _assert_trace_shape(trace, ["assistant", "tool", "assistant"])
        _assert_tool_call_link(trace, 0, 1)
        assert trace[1]["content"] == '{"gene": "BCL2", "score": 0.95}'
        assert trace[1]["tool_result"]["is_error"] is False

    def test_multiple_tool_calls_single_turn(self) -> None:
        """Assistant calls multiple tools in one message."""
        tc1 = ToolUseContent(id="tc_A", name="search_gene", input={"gene": "KRAS"})
        tc2 = ToolUseContent(id="tc_B", name="search_disease", input={"disease": "cancer"})
        messages = [
            Message.assistant("Searching both.", tool_calls=[tc1, tc2]),
            Message.tool_result("tc_A", "KRAS data"),
            Message.tool_result("tc_B", "cancer data"),
            Message.assistant("Here are the results."),
        ]
        trace = self._to_trace(messages)
        _assert_trace_shape(trace, ["assistant", "tool", "tool", "assistant"])
        assert len(trace[0]["tool_calls"]) == 2
        assert trace[1]["tool_result"]["tool_use_id"] == "tc_A"
        assert trace[2]["tool_result"]["tool_use_id"] == "tc_B"

    def test_multi_turn_tool_use(self) -> None:
        """Multiple rounds of tool calls."""
        tc1 = ToolUseContent(id="tc_1", name="search", input={"q": "first"})
        tc2 = ToolUseContent(id="tc_2", name="analyze", input={"data": "..."})
        messages = [
            Message.assistant("Step 1.", tool_calls=[tc1]),
            Message.tool_result("tc_1", "first result"),
            Message.assistant("Step 2.", tool_calls=[tc2]),
            Message.tool_result("tc_2", "second result"),
            Message.assistant("Final answer."),
        ]
        trace = self._to_trace(messages)
        _assert_trace_shape(trace, ["assistant", "tool", "assistant", "tool", "assistant"])
        _assert_tool_call_link(trace, 0, 1)
        _assert_tool_call_link(trace, 2, 3)

    def test_tool_error_result(self) -> None:
        """Tool result with is_error=True."""
        tc = ToolUseContent(id="tc_err", name="failing_tool", input={})
        messages = [
            Message.assistant("Calling.", tool_calls=[tc]),
            Message.tool_result("tc_err", "Connection refused", is_error=True),
            Message.assistant("The tool failed."),
        ]
        trace = self._to_trace(messages)
        _assert_trace_shape(trace, ["assistant", "tool", "assistant"])
        assert trace[1]["tool_result"]["is_error"] is True
        assert trace[1]["content"] == "Connection refused"

    def test_thinking_blocks_preserved(self) -> None:
        """Thinking content is included in trace."""
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ThinkingContent(thinking="reasoning about the problem", signature="sig123"),
                TextContent(text="The answer."),
            ],
        )
        trace = self._to_trace([msg])
        assert len(trace) == 1
        assert trace[0]["thinking"]["thinking"] == "reasoning about the problem"
        assert trace[0]["thinking"]["signature"] == "sig123"

    def test_user_messages_excluded(self) -> None:
        """User messages are not included in trace output."""
        messages = [
            Message.user("What is BCL2?"),
            Message.assistant("BCL2 is a gene."),
        ]
        trace = self._to_trace(messages)
        _assert_trace_shape(trace, ["assistant"])


# ======================================================================
# ToolResultCollector tests
# ======================================================================


class TestToolResultCollector:
    """Tests for the ToolResultCollector used by the claude_tool agent adapter."""

    def test_record_and_drain(self) -> None:
        from karenina.adapters.claude_tool.tools import ToolResultCollector

        collector = ToolResultCollector()
        collector.record("search", "found it", is_error=False)
        collector.record("analyze", "error!", is_error=True)

        results = collector.drain()
        assert len(results) == 2
        assert results[0].tool_name == "search"
        assert results[0].content == "found it"
        assert results[0].is_error is False
        assert results[1].tool_name == "analyze"
        assert results[1].content == "error!"
        assert results[1].is_error is True

    def test_drain_clears_results(self) -> None:
        from karenina.adapters.claude_tool.tools import ToolResultCollector

        collector = ToolResultCollector()
        collector.record("tool1", "result1")
        collector.drain()
        assert collector.drain() == []

    def test_drain_empty_returns_empty_list(self) -> None:
        from karenina.adapters.claude_tool.tools import ToolResultCollector

        collector = ToolResultCollector()
        assert collector.drain() == []


# ======================================================================
# MCP tool wrapper collector integration tests
# ======================================================================


class TestMcpToolCollectorIntegration:
    """Tests that MCP tool wrappers correctly record results in the collector."""

    @pytest.mark.asyncio
    async def test_mcp_tool_records_success(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from karenina.adapters.claude_tool.tools import ToolResultCollector, create_mcp_tool_function

        collector = ToolResultCollector()

        mock_session = AsyncMock()
        mock_content = MagicMock()
        mock_content.text = "Search result data"
        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_session.call_tool.return_value = mock_result

        mock_tool = MagicMock()
        mock_tool.name = "search"

        tool_fn = create_mcp_tool_function(mock_session, mock_tool, "server1", collector=collector)
        result = await tool_fn(query="BCL2")

        assert result == "Search result data"
        records = collector.drain()
        assert len(records) == 1
        assert records[0].tool_name == "search"
        assert records[0].content == "Search result data"
        assert records[0].is_error is False

    @pytest.mark.asyncio
    async def test_mcp_tool_records_error(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from karenina.adapters.claude_tool.tools import ToolResultCollector, create_mcp_tool_function

        collector = ToolResultCollector()

        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = Exception("Connection refused")

        mock_tool = MagicMock()
        mock_tool.name = "search"

        tool_fn = create_mcp_tool_function(mock_session, mock_tool, "server1", collector=collector)
        result = await tool_fn(query="test")

        assert "Error calling MCP tool" in result
        records = collector.drain()
        assert len(records) == 1
        assert records[0].is_error is True

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_ordered(self) -> None:
        """Multiple tool calls record results in execution order."""
        from unittest.mock import AsyncMock, MagicMock

        from karenina.adapters.claude_tool.tools import ToolResultCollector, create_mcp_tool_function

        collector = ToolResultCollector()

        def make_tool_fn(name: str, response_text: str):
            mock_session = AsyncMock()
            mock_content = MagicMock()
            mock_content.text = response_text
            mock_result = MagicMock()
            mock_result.content = [mock_content]
            mock_session.call_tool.return_value = mock_result
            mock_tool = MagicMock()
            mock_tool.name = name
            return create_mcp_tool_function(mock_session, mock_tool, "server", collector=collector)

        fn1 = make_tool_fn("search_gene", "KRAS data")
        fn2 = make_tool_fn("search_disease", "cancer data")

        await fn1(gene="KRAS")
        await fn2(disease="cancer")

        records = collector.drain()
        assert len(records) == 2
        assert records[0].tool_name == "search_gene"
        assert records[0].content == "KRAS data"
        assert records[1].tool_name == "search_disease"
        assert records[1].content == "cancer data"


# ======================================================================
# LangChain adapter trace tests
# ======================================================================


class TestLangchainTraceCollection:
    """Tests for trace message collection via the langchain adapter."""

    def _to_trace(self, messages) -> list[dict]:
        from karenina.adapters.langchain.trace import langchain_messages_to_trace_messages

        return langchain_messages_to_trace_messages(messages)

    def test_simple_response_no_tools(self) -> None:
        from langchain_core.messages import AIMessage

        messages = [AIMessage(content="The answer is 42.")]
        trace = self._to_trace(messages)
        _assert_trace_shape(trace, ["assistant"])
        assert trace[0]["content"] == "The answer is 42."

    def test_single_tool_call_round_trip(self) -> None:
        from langchain_core.messages import AIMessage, ToolMessage

        messages = [
            AIMessage(
                content="Let me search.",
                tool_calls=[{"id": "tc_001", "name": "search", "args": {"query": "BCL2"}}],
            ),
            ToolMessage(content='{"gene": "BCL2"}', tool_call_id="tc_001"),
            AIMessage(content="BCL2 is a gene."),
        ]
        trace = self._to_trace(messages)
        _assert_trace_shape(trace, ["assistant", "tool", "assistant"])
        _assert_tool_call_link(trace, 0, 1)

    def test_multiple_tool_calls_single_turn(self) -> None:
        from langchain_core.messages import AIMessage, ToolMessage

        messages = [
            AIMessage(
                content="Searching both.",
                tool_calls=[
                    {"id": "tc_A", "name": "search_gene", "args": {"gene": "KRAS"}},
                    {"id": "tc_B", "name": "search_disease", "args": {"disease": "cancer"}},
                ],
            ),
            ToolMessage(content="KRAS data", tool_call_id="tc_A"),
            ToolMessage(content="cancer data", tool_call_id="tc_B"),
            AIMessage(content="Results."),
        ]
        trace = self._to_trace(messages)
        _assert_trace_shape(trace, ["assistant", "tool", "tool", "assistant"])
        assert len(trace[0]["tool_calls"]) == 2

    def test_multi_turn_tool_use(self) -> None:
        from langchain_core.messages import AIMessage, ToolMessage

        messages = [
            AIMessage(
                content="Step 1.",
                tool_calls=[{"id": "tc_1", "name": "search", "args": {}}],
            ),
            ToolMessage(content="first result", tool_call_id="tc_1"),
            AIMessage(
                content="Step 2.",
                tool_calls=[{"id": "tc_2", "name": "analyze", "args": {}}],
            ),
            ToolMessage(content="second result", tool_call_id="tc_2"),
            AIMessage(content="Final answer."),
        ]
        trace = self._to_trace(messages)
        _assert_trace_shape(trace, ["assistant", "tool", "assistant", "tool", "assistant"])
        _assert_tool_call_link(trace, 0, 1)
        _assert_tool_call_link(trace, 2, 3)


# ======================================================================
# Claude Agent SDK adapter trace tests
# ======================================================================


try:
    import claude_agent_sdk  # noqa: F401

    _has_claude_sdk = True
except ImportError:
    _has_claude_sdk = False


@pytest.mark.skipif(not _has_claude_sdk, reason="claude_agent_sdk not installed")
class TestClaudeSDKTraceCollection:
    """Tests for trace message collection via the claude_agent_sdk adapter."""

    def _to_trace(self, messages) -> list[dict]:
        from karenina.adapters.claude_agent_sdk.trace import sdk_messages_to_trace_messages

        return sdk_messages_to_trace_messages(messages)

    def test_simple_response_no_tools(self) -> None:
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock

        messages = [
            AssistantMessage(content=[TextBlock(text="The answer is 42.")]),
        ]
        trace = self._to_trace(messages)
        _assert_trace_shape(trace, ["assistant"])
        assert trace[0]["content"] == "The answer is 42."

    def test_single_tool_call_round_trip(self) -> None:
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock, ToolResultBlock, ToolUseBlock

        messages = [
            AssistantMessage(
                content=[
                    TextBlock(text="Let me search."),
                    ToolUseBlock(id="tc_001", name="search", input={"query": "BCL2"}),
                    ToolResultBlock(tool_use_id="tc_001", content='{"gene": "BCL2"}'),
                    TextBlock(text="BCL2 is a gene."),
                ]
            ),
        ]
        trace = self._to_trace(messages)
        # SDK puts tool_result inline within AssistantMessage.content,
        # so the converter splits them into separate trace messages
        _assert_trace_shape(trace, ["tool", "assistant"])
        # The SDK trace converter emits tool results before the next assistant block
        assert trace[0]["role"] == "tool"
        assert trace[0]["tool_result"]["tool_use_id"] == "tc_001"

    def test_multi_turn_with_tools(self) -> None:
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk.types import TextBlock, ToolResultBlock, ToolUseBlock

        messages = [
            AssistantMessage(
                content=[
                    TextBlock(text="Step 1."),
                    ToolUseBlock(id="tc_1", name="search", input={"q": "first"}),
                    ToolResultBlock(tool_use_id="tc_1", content="first result"),
                    TextBlock(text="Step 2."),
                    ToolUseBlock(id="tc_2", name="analyze", input={"data": "..."}),
                    ToolResultBlock(tool_use_id="tc_2", content="second result"),
                    TextBlock(text="Final answer."),
                ]
            ),
        ]
        trace = self._to_trace(messages)
        # Key invariant: tool results appear with correct tool_use_id
        tool_results = [m for m in trace if m["role"] == "tool"]
        assert len(tool_results) == 2
        assert tool_results[0]["tool_result"]["tool_use_id"] == "tc_1"
        assert tool_results[1]["tool_result"]["tool_use_id"] == "tc_2"


# ======================================================================
# Cross-adapter consistency tests
# ======================================================================


class TestCrossAdapterTraceConsistency:
    """Verify that the same conversation produces structurally equivalent
    traces across all adapters.

    These tests check the invariants that the frontend relies on:
    - tool_calls on assistant messages have id, name, input
    - tool messages have tool_result with tool_use_id, is_error
    - block_index is monotonically increasing
    """

    def test_tool_call_dict_shape_claude_tool(self) -> None:
        """Claude tool adapter tool_calls have required keys."""
        from karenina.adapters.claude_tool.trace import claude_tool_messages_to_trace_messages

        tc = ToolUseContent(id="tc_1", name="search", input={"q": "test"})
        messages = [Message.assistant("Searching.", tool_calls=[tc])]
        trace = claude_tool_messages_to_trace_messages(messages)

        tool_call = trace[0]["tool_calls"][0]
        assert set(tool_call.keys()) == {"id", "name", "input"}

    def test_tool_call_dict_shape_langchain(self) -> None:
        """LangChain adapter tool_calls have required keys."""
        from langchain_core.messages import AIMessage

        from karenina.adapters.langchain.trace import langchain_messages_to_trace_messages

        messages = [
            AIMessage(
                content="Searching.",
                tool_calls=[{"id": "tc_1", "name": "search", "args": {"q": "test"}}],
            ),
        ]
        trace = langchain_messages_to_trace_messages(messages)

        tool_call = trace[0]["tool_calls"][0]
        assert set(tool_call.keys()) == {"id", "name", "input"}

    def test_tool_result_dict_shape_claude_tool(self) -> None:
        """Claude tool adapter tool_result has required keys."""
        from karenina.adapters.claude_tool.trace import claude_tool_messages_to_trace_messages

        messages = [Message.tool_result("tc_1", "data", is_error=False)]
        trace = claude_tool_messages_to_trace_messages(messages)

        tool_result = trace[0]["tool_result"]
        assert set(tool_result.keys()) == {"tool_use_id", "is_error"}

    def test_tool_result_dict_shape_langchain(self) -> None:
        """LangChain adapter tool_result has required keys."""
        from langchain_core.messages import ToolMessage

        from karenina.adapters.langchain.trace import langchain_messages_to_trace_messages

        messages = [ToolMessage(content="data", tool_call_id="tc_1")]
        trace = langchain_messages_to_trace_messages(messages)

        tool_result = trace[0]["tool_result"]
        assert set(tool_result.keys()) == {"tool_use_id", "is_error"}

    def test_message_to_dict_roundtrip(self) -> None:
        """Message.to_dict() produces same shape as trace converters."""
        tc = ToolUseContent(id="tc_1", name="search", input={"q": "test"})
        assistant_msg = Message.assistant("Searching.", tool_calls=[tc])
        tool_msg = Message.tool_result("tc_1", "result data", is_error=False)

        assistant_dict = assistant_msg.to_dict()
        tool_dict = tool_msg.to_dict()

        # Assistant message shape
        assert assistant_dict["role"] == "assistant"
        assert "tool_calls" in assistant_dict
        assert assistant_dict["tool_calls"][0]["id"] == "tc_1"

        # Tool message shape
        assert tool_dict["role"] == "tool"
        assert "tool_result" in tool_dict
        assert tool_dict["tool_result"]["tool_use_id"] == "tc_1"
        assert tool_dict["tool_result"]["is_error"] is False
        assert tool_dict["content"] == "result data"
