from __future__ import annotations

from karenina.adapters.registry import AdapterRegistry
from karenina.schemas.config import ModelConfig


def test_omp_registration_is_agent_only_deep_agent() -> None:
    spec = AdapterRegistry.get_spec("omp")

    assert spec is not None
    assert spec.agent_factory is not None
    assert spec.llm_factory is None
    assert spec.parser_factory is None
    assert spec.agent_tier == "deep_agent"
    assert spec.supports_mcp is True
    assert spec.supports_tools is True
    assert spec.requires_provider is True


def test_omp_runtime_capabilities_are_not_claimed_as_sandboxed() -> None:
    spec = AdapterRegistry.get_spec("omp")
    config = ModelConfig(
        id="glm",
        model_provider="zhipu-coding-plan",
        model_name="glm-5.3",
        interface="omp",
    )

    capabilities = spec.runtime_profile.capabilities(config)

    assert capabilities.supports_file_tools is True
    assert capabilities.supports_code_execution is True
    assert capabilities.uses_sandboxed_execution is False
