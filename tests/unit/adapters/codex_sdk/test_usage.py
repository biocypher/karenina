"""Tests for codex usage extraction."""

from __future__ import annotations

from types import SimpleNamespace

from karenina.adapters.codex_sdk.usage import extract_codex_usage

from .conftest import make_usage


class TestExtractCodexUsage:
    def test_total_breakdown_mapped(self) -> None:
        usage = extract_codex_usage(make_usage(input_tokens=29306, output_tokens=125, cached=7), model="qwen")
        assert usage.input_tokens == 29306
        assert usage.output_tokens == 125
        assert usage.total_tokens == 29431
        assert usage.cache_read_tokens == 7
        assert usage.cache_creation_tokens is None
        assert usage.cost_usd is None
        assert usage.model == "qwen"

    def test_none_usage_returns_zeroed(self) -> None:
        usage = extract_codex_usage(None, model="qwen")
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.model == "qwen"

    def test_falls_back_to_last_breakdown(self) -> None:
        wrapper = SimpleNamespace(
            total=None,
            last=SimpleNamespace(
                cached_input_tokens=0,
                input_tokens=10,
                output_tokens=5,
                reasoning_output_tokens=0,
                total_tokens=15,
            ),
        )
        usage = extract_codex_usage(wrapper)
        assert usage.input_tokens == 10
        assert usage.total_tokens == 15

    def test_bare_breakdown_tolerated(self) -> None:
        breakdown = SimpleNamespace(
            cached_input_tokens=1,
            input_tokens=8,
            output_tokens=2,
            reasoning_output_tokens=0,
            total_tokens=10,
        )
        usage = extract_codex_usage(breakdown)
        assert usage.input_tokens == 8
        assert usage.output_tokens == 2

    def test_missing_total_tokens_computed(self) -> None:
        breakdown = SimpleNamespace(input_tokens=3, output_tokens=4)
        usage = extract_codex_usage(SimpleNamespace(total=breakdown, last=None))
        assert usage.total_tokens == 7
        assert usage.cache_read_tokens is None
