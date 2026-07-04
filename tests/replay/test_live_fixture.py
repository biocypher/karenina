"""Smoke tests for the live-model fixture."""

from __future__ import annotations

import pytest


@pytest.mark.live
def test_live_primary_model_fixture_shape(live_primary_model):
    """The fixture should produce a ModelConfig pointing at the vLLM endpoint."""
    assert live_primary_model.interface == "openai_endpoint"
    assert "codon-gpu-001" in (live_primary_model.endpoint_base_url or "")
    assert live_primary_model.endpoint_api_key is not None


@pytest.mark.live
def test_live_parsing_model_fixture_shape(live_parsing_model):
    assert live_parsing_model.interface == "openai_endpoint"
    assert "codon-gpu-001" in (live_parsing_model.endpoint_base_url or "")


def test_live_tests_skipped_when_env_var_unset(monkeypatch):
    """Without KARENINA_LIVE_TESTS=1, the gate helper reports False."""
    monkeypatch.delenv("KARENINA_LIVE_TESTS", raising=False)
    from tests.replay.conftest import _live_tests_enabled

    assert _live_tests_enabled() is False


def test_live_tests_enabled_when_env_var_set(monkeypatch):
    monkeypatch.setenv("KARENINA_LIVE_TESTS", "1")
    from tests.replay.conftest import _live_tests_enabled

    assert _live_tests_enabled() is True
