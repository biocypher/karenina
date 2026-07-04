"""Tests confirming retained symbols survive dead-code removal.

Issue 097: extract_agent_metrics (legacy) and extract_middleware_metrics removed
from trace_agent_metrics.py, along with aliases in llm_invocation.py and
re-exports in __init__.py.

Issue 100: UsageTracker.track_agent_metrics() removed; set_agent_metrics()
remains the canonical path.

These tests verify that kept symbols still import and behave correctly after
the removals.
"""

from __future__ import annotations

import re

import pytest

from karenina.benchmark.verification.utils.trace_agent_metrics import (
    TOOL_FAILURE_PATTERNS,
    extract_agent_metrics_from_messages,
)
from karenina.benchmark.verification.utils.trace_usage_tracker import (
    UsageTracker,
)
from karenina.ports.messages import (
    Message,
    ToolUseContent,
)

# ======================================================================
# Issue 097: TOOL_FAILURE_PATTERNS still available
# ======================================================================


@pytest.mark.unit
class TestToolFailurePatternsRetained:
    """TOOL_FAILURE_PATTERNS constant must survive removal of legacy functions."""

    def test_is_tuple_of_compiled_patterns(self) -> None:
        assert isinstance(TOOL_FAILURE_PATTERNS, tuple)
        assert len(TOOL_FAILURE_PATTERNS) > 0
        for pat in TOOL_FAILURE_PATTERNS:
            assert isinstance(pat, re.Pattern)

    def test_matches_known_error_text(self) -> None:
        text = "Error: connection timeout"
        matched = any(p.search(text) for p in TOOL_FAILURE_PATTERNS)
        assert matched

    def test_no_match_on_clean_text(self) -> None:
        text = "The population of France is 67 million."
        matched = any(p.search(text) for p in TOOL_FAILURE_PATTERNS)
        assert not matched


# ======================================================================
# Issue 097: extract_agent_metrics_from_messages still works
# ======================================================================


@pytest.mark.unit
class TestExtractAgentMetricsFromMessagesRetained:
    """Canonical extraction function must survive removal of legacy functions."""

    def test_empty_messages(self) -> None:
        metrics = extract_agent_metrics_from_messages([])
        assert metrics["iterations"] == 0
        assert metrics["tool_calls"] == 0
        assert metrics["tools_used"] == []

    def test_basic_tool_round_trip(self) -> None:
        msgs = [
            Message.assistant(
                "Searching.",
                tool_calls=[ToolUseContent(id="tc1", name="search", input={})],
            ),
            Message.tool_result("tc1", "found it"),
            Message.assistant("Done."),
        ]
        metrics = extract_agent_metrics_from_messages(msgs)
        assert metrics["iterations"] == 2
        assert metrics["tool_calls"] == 1
        assert metrics["tools_used"] == ["search"]
        assert metrics["tool_call_counts"] == {"search": 1}


# ======================================================================
# Issue 097: legacy functions are gone
# ======================================================================


@pytest.mark.unit
class TestLegacyFunctionsRemoved:
    """Legacy symbols must not be importable after removal."""

    def test_extract_agent_metrics_removed_from_trace_module(self) -> None:
        from karenina.benchmark.verification.utils import trace_agent_metrics

        assert not hasattr(trace_agent_metrics, "extract_agent_metrics")

    def test_extract_middleware_metrics_removed_from_trace_module(self) -> None:
        from karenina.benchmark.verification.utils import trace_agent_metrics

        assert not hasattr(trace_agent_metrics, "extract_middleware_metrics")

    def test_aliases_removed_from_llm_invocation(self) -> None:
        from karenina.benchmark.verification.utils import llm_invocation

        assert not hasattr(llm_invocation, "_extract_agent_metrics")
        assert not hasattr(llm_invocation, "_extract_middleware_metrics")

    def test_removed_from_utils_init(self) -> None:
        from karenina.benchmark.verification import utils

        assert not hasattr(utils, "extract_middleware_metrics")


# ======================================================================
# Issue 100: UsageTracker.set_agent_metrics still works
# ======================================================================


@pytest.mark.unit
class TestUsageTrackerSetAgentMetrics:
    """set_agent_metrics() is the canonical path and must survive removal."""

    def test_set_and_get_agent_metrics(self) -> None:
        tracker = UsageTracker()
        metrics = {
            "iterations": 3,
            "tool_calls": 5,
            "tools_used": ["search", "calculator"],
        }
        tracker.set_agent_metrics(metrics)
        result = tracker.get_agent_metrics()
        assert result == metrics

    def test_get_agent_metrics_returns_none_initially(self) -> None:
        tracker = UsageTracker()
        assert tracker.get_agent_metrics() is None

    def test_set_overwrites_previous(self) -> None:
        tracker = UsageTracker()
        tracker.set_agent_metrics({"iterations": 1, "tool_calls": 0, "tools_used": []})
        tracker.set_agent_metrics({"iterations": 5, "tool_calls": 10, "tools_used": ["a"]})
        result = tracker.get_agent_metrics()
        assert result is not None
        assert result["iterations"] == 5


# ======================================================================
# Issue 100: track_agent_metrics is gone
# ======================================================================


@pytest.mark.unit
class TestTrackAgentMetricsRemoved:
    """track_agent_metrics() must not exist on UsageTracker after removal."""

    def test_method_removed(self) -> None:
        tracker = UsageTracker()
        assert not hasattr(tracker, "track_agent_metrics")
