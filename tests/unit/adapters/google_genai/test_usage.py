"""Tests for google_genai usage extraction and aggregation."""

from __future__ import annotations

import pytest

from karenina.adapters.google_genai.usage import aggregate_usage, extract_usage, per_call_usage
from karenina.ports import UsageMetadata
from tests.unit.adapters.google_genai._fakes import usage


@pytest.mark.unit
class TestExtractUsage:
    def test_full_mapping(self) -> None:
        result = extract_usage(
            usage(prompt=1000, candidates=200, thoughts=300, tool_prompt=50, cached=600, total=1550),
            model="gemini-4-argon",
            web_search_queries=4,
        )
        assert result.input_tokens == 1050  # prompt + tool-use prompt (cached already inside prompt)
        assert result.output_tokens == 500  # candidates + thoughts
        assert result.total_tokens == 1550
        assert result.cache_read_tokens == 600
        assert result.reasoning_tokens == 300
        assert result.web_search_queries == 4
        assert result.model == "gemini-4-argon"
        assert result.cost_usd is None

    def test_total_falls_back_to_the_sum(self) -> None:
        result = extract_usage(usage(prompt=10, candidates=5), model="m")
        assert result.total_tokens == 15
        assert result.cache_read_tokens is None
        assert result.reasoning_tokens is None
        assert result.web_search_queries is None

    def test_missing_metadata_gives_zeros(self) -> None:
        result = extract_usage(None, model="m", web_search_queries=0)
        assert (result.input_tokens, result.output_tokens, result.total_tokens) == (0, 0, 0)
        assert result.web_search_queries == 0


@pytest.mark.unit
class TestAggregateUsage:
    def test_sums_several_turns_with_cache_and_thinking(self) -> None:
        turns = [
            extract_usage(usage(prompt=100, candidates=10, thoughts=40, total=150), model="m", web_search_queries=3),
            extract_usage(usage(prompt=200, candidates=20, cached=90, total=220), model="m", web_search_queries=None),
            extract_usage(usage(prompt=300, candidates=30, thoughts=60, cached=250, total=390), model="m"),
        ]
        total = UsageMetadata(model="m")
        for turn in turns:
            total = aggregate_usage(total, turn)
        assert total.input_tokens == 600
        assert total.output_tokens == 160
        assert total.total_tokens == 760
        assert total.cache_read_tokens == 340
        assert total.reasoning_tokens == 100
        assert total.web_search_queries == 3
        assert total.model == "m"

    def test_none_only_when_no_call_reported(self) -> None:
        total = aggregate_usage(UsageMetadata(), UsageMetadata(input_tokens=1, total_tokens=1))
        assert total.cache_read_tokens is None
        assert total.reasoning_tokens is None
        assert total.web_search_queries is None
        assert total.cost_usd is None

    def test_real_zero_is_kept(self) -> None:
        total = aggregate_usage(UsageMetadata(web_search_queries=0), UsageMetadata())
        assert total.web_search_queries == 0


@pytest.mark.unit
def test_per_call_usage_keys() -> None:
    call = extract_usage(usage(prompt=10, candidates=5, thoughts=2, cached=4), model="m", web_search_queries=1)
    assert per_call_usage(call) == {
        "input_tokens": 10,
        "output_tokens": 7,
        "cache_read_input_tokens": 4,
        "reasoning_tokens": 2,
        "web_search_queries": 1,
    }
    assert per_call_usage(extract_usage(usage(), model="m")) == {"input_tokens": 10, "output_tokens": 5}
