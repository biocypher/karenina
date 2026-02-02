"""Tests for extract_agent_metrics_from_messages.

Validates that tool counting, iteration tracking, and failure detection
work correctly against the canonical Message-based metrics extraction.
These tests are adapter-agnostic — they exercise the function that all
adapters feed into via AgentResult.trace_messages.

Scenarios simulate the trace shapes produced by each adapter family
(claude_tool, claude_agent_sdk, langchain) to guard against regressions
when refactoring adapter internals.
"""

from __future__ import annotations

from karenina.benchmark.verification.utils.trace_agent_metrics import (
    extract_agent_metrics_from_messages,
)
from karenina.ports.messages import (
    Message,
    ToolUseContent,
)

# ======================================================================
# Helpers — build realistic trace Message lists
# ======================================================================


def _tool_call(name: str, tool_id: str, input_data: dict | None = None) -> ToolUseContent:
    return ToolUseContent(id=tool_id, name=name, input=input_data or {})


# ======================================================================
# Basic counting
# ======================================================================


class TestBasicCounting:
    """Core iteration and tool counting logic."""

    def test_empty_messages(self) -> None:
        metrics = extract_agent_metrics_from_messages([])
        assert metrics["iterations"] == 0
        assert metrics["tool_calls"] == 0
        assert metrics["tools_used"] == []
        assert metrics["tool_call_counts"] == {}
        assert metrics["suspect_failed_tool_calls"] == 0
        assert metrics["suspect_failed_tools"] == []

    def test_single_assistant_no_tools(self) -> None:
        """Simple LLM response with no tool use — 1 iteration, 0 tool calls."""
        msgs = [Message.assistant("The answer is 42.")]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["iterations"] == 1
        assert metrics["tool_calls"] == 0
        assert metrics["tools_used"] == []
        assert metrics["tool_call_counts"] == {}

    def test_single_tool_round_trip(self) -> None:
        """One assistant→tool→assistant cycle."""
        msgs = [
            Message.assistant(
                "Let me search.",
                tool_calls=[_tool_call("web_search", "tc1", {"query": "test"})],
            ),
            Message.tool_result("tc1", "Result: found it"),
            Message.assistant("Based on the search, the answer is X."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["iterations"] == 2
        assert metrics["tool_calls"] == 1
        assert metrics["tools_used"] == ["web_search"]
        assert metrics["tool_call_counts"] == {"web_search": 1}

    def test_multiple_tools_single_turn(self) -> None:
        """Assistant invokes two tools in one turn (parallel tool calls)."""
        msgs = [
            Message.assistant(
                "I'll search both.",
                tool_calls=[
                    _tool_call("web_search", "tc1", {"query": "alpha"}),
                    _tool_call("calculator", "tc2", {"expr": "2+2"}),
                ],
            ),
            Message.tool_result("tc1", "alpha result"),
            Message.tool_result("tc2", "4"),
            Message.assistant("Alpha result is X and 2+2=4."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["iterations"] == 2
        assert metrics["tool_calls"] == 2
        assert metrics["tools_used"] == ["calculator", "web_search"]  # sorted
        assert metrics["tool_call_counts"] == {"web_search": 1, "calculator": 1}

    def test_multi_turn_same_tool(self) -> None:
        """Same tool called across multiple turns."""
        msgs = [
            Message.assistant(
                "Search 1.",
                tool_calls=[_tool_call("web_search", "tc1")],
            ),
            Message.tool_result("tc1", "result 1"),
            Message.assistant(
                "Search 2.",
                tool_calls=[_tool_call("web_search", "tc2")],
            ),
            Message.tool_result("tc2", "result 2"),
            Message.assistant(
                "Search 3.",
                tool_calls=[_tool_call("web_search", "tc3")],
            ),
            Message.tool_result("tc3", "result 3"),
            Message.assistant("Done."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["iterations"] == 4
        assert metrics["tool_calls"] == 3
        assert metrics["tools_used"] == ["web_search"]
        assert metrics["tool_call_counts"] == {"web_search": 3}

    def test_many_distinct_tools(self) -> None:
        """Multiple distinct tools across turns — verifies sorted output."""
        msgs = [
            Message.assistant(
                "Step 1.",
                tool_calls=[_tool_call("z_tool", "tc1")],
            ),
            Message.tool_result("tc1", "ok"),
            Message.assistant(
                "Step 2.",
                tool_calls=[_tool_call("a_tool", "tc2")],
            ),
            Message.tool_result("tc2", "ok"),
            Message.assistant(
                "Step 3.",
                tool_calls=[_tool_call("m_tool", "tc3")],
            ),
            Message.tool_result("tc3", "ok"),
            Message.assistant("Final."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["tools_used"] == ["a_tool", "m_tool", "z_tool"]
        assert metrics["tool_call_counts"] == {"z_tool": 1, "a_tool": 1, "m_tool": 1}


# ======================================================================
# Failure detection
# ======================================================================


class TestFailureDetection:
    """Suspect-failed tool detection via is_error flag and content patterns."""

    def test_is_error_flag(self) -> None:
        """Tool result with is_error=True is flagged as suspect failure."""
        msgs = [
            Message.assistant(
                "Try tool.",
                tool_calls=[_tool_call("flaky_api", "tc1")],
            ),
            Message.tool_result("tc1", "Something went wrong", is_error=True),
            Message.assistant("That failed."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["flaky_api"]

    def test_error_pattern_in_content(self) -> None:
        """Tool result containing error-like text (no is_error flag)."""
        msgs = [
            Message.assistant(
                "Try tool.",
                tool_calls=[_tool_call("web_search", "tc1")],
            ),
            Message.tool_result("tc1", "Error: connection timeout after 30s"),
            Message.assistant("Search failed."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["web_search"]

    def test_http_error_pattern(self) -> None:
        """HTTP status codes trigger failure detection."""
        msgs = [
            Message.assistant(
                "Fetch data.",
                tool_calls=[_tool_call("http_client", "tc1")],
            ),
            Message.tool_result("tc1", "HTTP 500 Internal Server Error"),
            Message.assistant("Server error."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["http_client"]

    def test_no_false_positive_on_clean_result(self) -> None:
        """Normal tool output should not trigger failure detection."""
        msgs = [
            Message.assistant(
                "Search.",
                tool_calls=[_tool_call("web_search", "tc1")],
            ),
            Message.tool_result("tc1", "The population of France is 67 million."),
            Message.assistant("France has 67 million people."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["suspect_failed_tool_calls"] == 0
        assert metrics["suspect_failed_tools"] == []

    def test_mixed_success_and_failure(self) -> None:
        """One tool succeeds, another fails — only the failure is flagged."""
        msgs = [
            Message.assistant(
                "Two calls.",
                tool_calls=[
                    _tool_call("good_tool", "tc1"),
                    _tool_call("bad_tool", "tc2"),
                ],
            ),
            Message.tool_result("tc1", "All good"),
            Message.tool_result("tc2", "Traceback (most recent call last):\n  File ...", is_error=True),
            Message.assistant("One succeeded, one failed."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["tool_calls"] == 2
        assert metrics["suspect_failed_tool_calls"] == 1
        assert metrics["suspect_failed_tools"] == ["bad_tool"]

    def test_failure_name_resolved_not_id(self) -> None:
        """Suspect-failed tools should report the tool NAME, not the tool_use_id."""
        msgs = [
            Message.assistant(
                "Call.",
                tool_calls=[_tool_call("my_calculator", "toolu_abc123xyz")],
            ),
            Message.tool_result("toolu_abc123xyz", "Error: division by zero", is_error=True),
            Message.assistant("Oops."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["suspect_failed_tools"] == ["my_calculator"]
        # Must NOT contain the raw id
        assert "toolu_abc123xyz" not in str(metrics["suspect_failed_tools"])

    def test_failure_with_unknown_tool_id(self) -> None:
        """Tool result with an id that doesn't match any assistant tool_call."""
        msgs = [
            Message.assistant("Hmm."),
            # Orphan tool result — no matching assistant tool_call
            Message.tool_result("orphan_id", "Error: something broke", is_error=True),
            Message.assistant("Done."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["suspect_failed_tool_calls"] == 1
        # Can't resolve name, so list stays empty
        assert metrics["suspect_failed_tools"] == []


# ======================================================================
# Non-assistant/tool messages are ignored
# ======================================================================


class TestMessageFiltering:
    """System and user messages should not affect metrics."""

    def test_system_and_user_messages_ignored(self) -> None:
        """System/user messages don't count as iterations or tool calls."""
        msgs = [
            Message.system("You are a helpful assistant."),
            Message.user("What is 2+2?"),
            Message.assistant(
                "Let me calculate.",
                tool_calls=[_tool_call("calculator", "tc1", {"expr": "2+2"})],
            ),
            Message.tool_result("tc1", "4"),
            Message.assistant("The answer is 4."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["iterations"] == 2
        assert metrics["tool_calls"] == 1

    def test_interleaved_user_messages(self) -> None:
        """User messages mid-conversation (e.g., from summarization) don't affect counts."""
        msgs = [
            Message.assistant(
                "Searching.",
                tool_calls=[_tool_call("search", "tc1")],
            ),
            Message.tool_result("tc1", "result"),
            Message.user("[Conversation summary]"),  # summarization middleware
            Message.assistant(
                "Continuing.",
                tool_calls=[_tool_call("search", "tc2")],
            ),
            Message.tool_result("tc2", "more results"),
            Message.assistant("Done."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["iterations"] == 3
        assert metrics["tool_calls"] == 2


# ======================================================================
# Adapter-shaped traces — simulate what each adapter produces
# ======================================================================


class TestAdapterShapedTraces:
    """Simulate the trace_messages shape each adapter produces.

    These tests ensure extract_agent_metrics_from_messages works correctly
    with the specific message patterns emitted by each adapter family.
    Adapters filter out USER messages from trace_messages, so these
    traces only contain assistant and tool messages.
    """

    def test_claude_tool_shape(self) -> None:
        """claude_tool adapter: assistant→tool→assistant with explicit tool_calls."""
        # claude_tool builds messages via convert_from_anthropic_message +
        # manual Message.tool_result injection. No user/system messages.
        msgs = [
            Message.assistant(
                "I'll look that up.",
                tool_calls=[
                    _tool_call("mcp__bio_db__search_genes", "toolu_01A"),
                    _tool_call("mcp__bio_db__get_pathways", "toolu_01B"),
                ],
            ),
            Message.tool_result("toolu_01A", '{"genes": ["BRCA1", "TP53"]}'),
            Message.tool_result("toolu_01B", '{"pathways": ["apoptosis"]}'),
            Message.assistant(
                "Now let me verify.",
                tool_calls=[_tool_call("mcp__bio_db__search_genes", "toolu_02A")],
            ),
            Message.tool_result("toolu_02A", '{"genes": ["BRCA1"]}'),
            Message.assistant("BRCA1 is involved in apoptosis."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["iterations"] == 3
        assert metrics["tool_calls"] == 3
        assert metrics["tools_used"] == ["mcp__bio_db__get_pathways", "mcp__bio_db__search_genes"]
        assert metrics["tool_call_counts"] == {
            "mcp__bio_db__search_genes": 2,
            "mcp__bio_db__get_pathways": 1,
        }

    def test_langchain_shape(self) -> None:
        """langchain adapter: same shape after conversion — assistant and tool only."""
        # LangChain adapter converts via LangChainMessageConverter.from_provider()
        # then filters out USER messages. Result is assistant+tool only.
        msgs = [
            Message.assistant(
                "",  # LangChain sometimes has empty text with just tool_calls
                tool_calls=[_tool_call("tavily_search", "call_abc")],
            ),
            Message.tool_result("call_abc", "Search results: ..."),
            Message.assistant("Based on search results, the answer is Y."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["iterations"] == 2
        assert metrics["tool_calls"] == 1
        assert metrics["tools_used"] == ["tavily_search"]

    def test_claude_sdk_shape(self) -> None:
        """claude_agent_sdk adapter: messages from SDK converter."""
        # Claude SDK produces structured messages via ClaudeSDKMessageConverter
        msgs = [
            Message.assistant(
                "Let me use MCP tools.",
                tool_calls=[_tool_call("mcp_tool_search", "sdk_tc1")],
            ),
            Message.tool_result("sdk_tc1", "Found 3 results."),
            Message.assistant(
                "Let me get details.",
                tool_calls=[_tool_call("mcp_tool_details", "sdk_tc2")],
            ),
            Message.tool_result("sdk_tc2", "Detailed info here."),
            Message.assistant("The final answer is Z."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["iterations"] == 3
        assert metrics["tool_calls"] == 2
        assert metrics["tools_used"] == ["mcp_tool_details", "mcp_tool_search"]

    def test_manual_adapter_empty_trace(self) -> None:
        """manual adapter: returns empty trace_messages — metrics should be zeroed."""
        metrics = extract_agent_metrics_from_messages([])
        assert metrics["iterations"] == 0
        assert metrics["tool_calls"] == 0
        assert metrics["tools_used"] == []
        assert metrics["tool_call_counts"] == {}
        assert metrics["suspect_failed_tool_calls"] == 0
        assert metrics["suspect_failed_tools"] == []


# ======================================================================
# Schema completeness
# ======================================================================


class TestSchemaCompleteness:
    """Verify the returned dict contains all documented fields."""

    EXPECTED_KEYS = {
        "iterations",
        "tool_calls",
        "tools_used",
        "tool_call_counts",
        "suspect_failed_tool_calls",
        "suspect_failed_tools",
    }

    def test_all_keys_present_empty(self) -> None:
        metrics = extract_agent_metrics_from_messages([])
        assert set(metrics.keys()) == self.EXPECTED_KEYS

    def test_all_keys_present_with_data(self) -> None:
        msgs = [
            Message.assistant("Hi.", tool_calls=[_tool_call("t", "tc1")]),
            Message.tool_result("tc1", "ok"),
            Message.assistant("Done."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert set(metrics.keys()) == self.EXPECTED_KEYS

    def test_types_correct(self) -> None:
        msgs = [
            Message.assistant("Hi.", tool_calls=[_tool_call("t", "tc1")]),
            Message.tool_result("tc1", "Error", is_error=True),
            Message.assistant("Done."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert isinstance(metrics["iterations"], int)
        assert isinstance(metrics["tool_calls"], int)
        assert isinstance(metrics["tools_used"], list)
        assert isinstance(metrics["tool_call_counts"], dict)
        assert isinstance(metrics["suspect_failed_tool_calls"], int)
        assert isinstance(metrics["suspect_failed_tools"], list)


# ======================================================================
# Delegation from manual/message_utils
# ======================================================================


class TestMessageUtilsDelegation:
    """Verify that adapters.manual.message_utils.extract_agent_metrics
    delegates to the canonical function and produces identical results."""

    def test_delegation_produces_same_result(self) -> None:
        from karenina.adapters.manual.message_utils import (
            extract_agent_metrics as manual_extract,
        )

        msgs = [
            Message.assistant(
                "Search.",
                tool_calls=[_tool_call("web_search", "tc1")],
            ),
            Message.tool_result("tc1", "Error: timeout", is_error=True),
            Message.assistant(
                "Retry.",
                tool_calls=[_tool_call("web_search", "tc2")],
            ),
            Message.tool_result("tc2", "Success"),
            Message.assistant("Done."),
        ]
        canonical = extract_agent_metrics_from_messages(msgs)
        delegated = manual_extract(msgs)
        assert canonical == delegated
