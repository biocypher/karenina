"""Tests for codex_sdk registration, availability, and ModelConfig validation."""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from karenina.adapters.codex_sdk.availability import check_codex_available
from karenina.adapters.registry import AdapterRegistry
from karenina.schemas.config import ModelConfig


class TestRegistration:
    def test_spec_registered(self) -> None:
        spec = AdapterRegistry.get_spec("codex_sdk")
        assert spec is not None
        assert spec.interface == "codex_sdk"
        assert spec.agent_factory is not None
        assert spec.llm_factory is None
        assert spec.parser_factory is None
        assert spec.fallback_interface == "langchain"
        assert spec.agent_tier == "deep_agent"
        assert spec.requires_provider is False
        # MCP is a warn-and-skip stub for now (see mcp.py), so the spec
        # must not advertise support.
        assert spec.supports_mcp is False
        assert spec.supports_tools is True

    def test_model_config_validates_interface(self) -> None:
        config = ModelConfig(
            id="codex",
            model_name="qwen3.5-122b-a10b",
            interface="codex_sdk",
            endpoint_base_url="http://example-endpoint:8000/v1",
        )
        assert config.interface == "codex_sdk"

    def test_model_provider_not_required(self) -> None:
        config = ModelConfig(id="codex", model_name="m", interface="codex_sdk")
        assert config.model_provider is None

    def test_agent_factory_returns_adapter(self) -> None:
        from karenina.adapters.codex_sdk.agent import CodexSDKAgentAdapter

        spec = AdapterRegistry.get_spec("codex_sdk")
        assert spec is not None and spec.agent_factory is not None
        adapter = spec.agent_factory(ModelConfig(id="codex", model_name="m", interface="codex_sdk"))
        assert isinstance(adapter, CodexSDKAgentAdapter)

    def test_runtime_capabilities_profile_registered(self) -> None:
        from karenina.adapters.agent_runtime import get_agent_runtime_capabilities

        read_write = ModelConfig(id="c", model_name="m", interface="codex_sdk")
        caps = get_agent_runtime_capabilities(read_write)
        assert caps.supports_file_tools is True
        assert caps.supports_code_execution is True
        assert caps.uses_sandboxed_execution is True

        read_only = ModelConfig(
            id="c",
            model_name="m",
            interface="codex_sdk",
            extra_kwargs={"agent_runtime": {"access_mode": "read_only"}},
        )
        assert get_agent_runtime_capabilities(read_only).supports_code_execution is False


class TestAvailability:
    def test_available_when_sdk_installed(self) -> None:
        pytest.importorskip("openai_codex", reason="openai-codex not installed")
        availability = check_codex_available()
        assert availability.available is True
        assert "binary" in availability.reason

    def test_unavailable_when_sdk_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "openai_codex":
                raise ImportError("No module named 'openai_codex'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        availability = check_codex_available()
        assert availability.available is False
        assert availability.fallback_interface == "langchain"
        assert "openai-codex" in availability.reason
