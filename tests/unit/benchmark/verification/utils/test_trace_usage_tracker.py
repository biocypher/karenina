"""Tests for UsageTracker optional usage counts and the port usage converter."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.utils.trace_usage_tracker import (
    UsageTracker,
    usage_to_callback_metadata,
)
from karenina.ports import UsageMetadata


@pytest.mark.unit
class TestOptionalUsageCounts:
    """reasoning_tokens and web_search_queries reach the per-stage summary only when set."""

    def test_stage_summary_without_optional_counts_is_unchanged(self) -> None:
        tracker = UsageTracker()
        tracker.track_call("parsing", "m", {"m": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}})

        assert tracker.get_stage_summary("parsing") == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "model": "m",
            "input_token_details": {"audio": 0, "cache_read": 0},
            "output_token_details": {"audio": 0, "reasoning": 0},
        }
        assert tracker.get_total_summary()["total"] == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }

    def test_model_keyed_counts_are_summed_per_stage_and_in_total(self) -> None:
        tracker = UsageTracker()
        tracker.track_call(
            "agentic_rubric_evaluation",
            "judge",
            {"judge": {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140, "web_search_queries": 3}},
        )
        tracker.track_call(
            "agentic_rubric_evaluation",
            "judge",
            {"judge": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70, "reasoning_tokens": 12}},
        )
        tracker.track_call("parsing", "parser", {"parser": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}})

        stage = tracker.get_stage_summary("agentic_rubric_evaluation")
        assert stage is not None
        assert stage["input_tokens"] == 150
        assert stage["web_search_queries"] == 3
        assert stage["reasoning_tokens"] == 12

        parsing = tracker.get_stage_summary("parsing")
        assert parsing is not None
        assert "web_search_queries" not in parsing
        assert "reasoning_tokens" not in parsing

        total = tracker.get_total_summary()["total"]
        assert total["web_search_queries"] == 3
        assert total["reasoning_tokens"] == 12

    def test_flat_port_usage_dict_carries_counts(self) -> None:
        tracker = UsageTracker()
        usage = UsageMetadata(input_tokens=8, output_tokens=2, total_tokens=10, web_search_queries=1)
        tracker.track_call("agentic_parsing_investigation", "m", usage.to_dict())

        stage = tracker.get_stage_summary("agentic_parsing_investigation")
        assert stage is not None
        assert stage["web_search_queries"] == 1
        assert "reasoning_tokens" not in stage


@pytest.mark.unit
class TestUsageToCallbackMetadata:
    """usage_to_callback_metadata builds generate_answer's inner_usage dict."""

    def test_core_counts_only(self) -> None:
        usage = UsageMetadata(input_tokens=3, output_tokens=4, total_tokens=7)
        assert usage_to_callback_metadata("m", usage) == {
            "m": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }

    def test_optional_keys_present_when_set(self) -> None:
        usage = UsageMetadata(
            input_tokens=3,
            output_tokens=4,
            total_tokens=7,
            cost_usd=0.01,
            cache_read_tokens=2,
            cache_creation_tokens=1,
            reasoning_tokens=5,
            web_search_queries=6,
        )
        assert usage_to_callback_metadata("m", usage) == {
            "m": {
                "input_tokens": 3,
                "output_tokens": 4,
                "total_tokens": 7,
                "cost_usd": 0.01,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 1,
                "reasoning_tokens": 5,
                "web_search_queries": 6,
            },
        }
