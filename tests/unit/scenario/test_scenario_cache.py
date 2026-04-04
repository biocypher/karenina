"""Tests for scenario-level answer caching integration."""

from __future__ import annotations

import inspect

import pytest

from karenina.scenario.manager import ScenarioManager, build_scenario_cache_key


@pytest.mark.unit
class TestBuildScenarioCacheKey:
    """Tests for build_scenario_cache_key determinism and uniqueness."""

    def test_different_scenario_ids_produce_different_keys(self):
        key_a = build_scenario_cache_key("alpha", "node1", "claude", ["hi"])
        key_b = build_scenario_cache_key("beta", "node1", "claude", ["hi"])
        assert key_a != key_b

    def test_different_node_ids_produce_different_keys(self):
        key_a = build_scenario_cache_key("s", "node1", "claude", ["hi"])
        key_b = build_scenario_cache_key("s", "node2", "claude", ["hi"])
        assert key_a != key_b

    def test_different_models_produce_different_keys(self):
        key_a = build_scenario_cache_key("s", "n", "claude", ["hi"])
        key_b = build_scenario_cache_key("s", "n", "gpt-4o", ["hi"])
        assert key_a != key_b

    def test_different_histories_produce_different_keys(self):
        key_a = build_scenario_cache_key("s", "n", "m", ["hello"])
        key_b = build_scenario_cache_key("s", "n", "m", ["goodbye"])
        assert key_a != key_b

    def test_same_inputs_produce_same_key(self):
        key_a = build_scenario_cache_key("s", "n", "m", ["hello", "world"])
        key_b = build_scenario_cache_key("s", "n", "m", ["hello", "world"])
        assert key_a == key_b

    def test_key_contains_identifiers(self):
        key = build_scenario_cache_key("my_scenario", "ask_node", "claude-3", ["msg"])
        assert "my_scenario" in key
        assert "ask_node" in key
        assert "claude-3" in key

    def test_empty_history_is_valid(self):
        key = build_scenario_cache_key("s", "n", "m", [])
        assert isinstance(key, str)
        assert len(key) > 0


@pytest.mark.unit
class TestScenarioManagerSignature:
    """Tests for ScenarioManager API surface after cache wiring."""

    def test_run_accepts_answer_cache_kwarg(self):
        sig = inspect.signature(ScenarioManager.run)
        assert "answer_cache" in sig.parameters

    def test_answer_cache_defaults_to_none(self):
        sig = inspect.signature(ScenarioManager.run)
        param = sig.parameters["answer_cache"]
        assert param.default is None

    def test_arun_removed(self):
        assert not hasattr(ScenarioManager, "arun")
