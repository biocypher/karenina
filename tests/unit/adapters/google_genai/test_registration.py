"""Tests for the google_genai registration and availability check."""

from __future__ import annotations

import sys

import pytest

from karenina.adapters.factory import get_agent, get_llm, get_parser
from karenina.adapters.google_genai import registration
from karenina.adapters.registry import AdapterRegistry
from tests.unit.adapters.google_genai._fakes import make_config


@pytest.mark.unit
class TestRegistration:
    def test_spec(self) -> None:
        spec = AdapterRegistry.get_spec("google_genai")
        assert spec is not None
        assert spec.agent_tier == "tool_loop"
        assert spec.fallback_interface is None
        assert spec.supports_mcp is True
        assert spec.supports_tools is True
        assert spec.requires_provider is False

    def test_factories(self) -> None:
        config = make_config()
        assert type(get_agent(config)).__name__ == "GoogleGenAIAgentAdapter"
        assert type(get_llm(config)).__name__ == "GoogleGenAILLMAdapter"
        assert type(get_parser(config)).__name__ == "GoogleGenAIParserAdapter"

    def test_available_when_installed(self) -> None:
        availability = registration._check_availability()
        assert availability.available is True

    def test_unavailable_without_the_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "google.genai", None)
        availability = registration._check_availability()
        assert availability.available is False
        assert "karenina[google-genai]" in availability.reason
        assert availability.fallback_interface is None

    def test_lazy_package_exports(self) -> None:
        import karenina.adapters.google_genai as package

        assert package.GoogleGenAIAgentAdapter.__name__ == "GoogleGenAIAgentAdapter"
        assert package.GoogleGenAISettings.__name__ == "GoogleGenAISettings"
        with pytest.raises(AttributeError):
            _ = package.NotAThing
