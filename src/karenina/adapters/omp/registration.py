"""Register the Oh My Pi ACP adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from karenina.adapters.agent_runtime import AgentRuntimeProfile, get_agent_runtime_access_mode
from karenina.adapters.registry import AdapterRegistry, AdapterSpec
from karenina.ports.capabilities import PortCapabilities

from .availability import check_omp_available

if TYPE_CHECKING:
    from karenina.ports import AgentPort
    from karenina.schemas.config import ModelConfig


def _create_agent(config: ModelConfig) -> AgentPort:
    from .agent import OmpAgentAdapter

    return OmpAgentAdapter(config)


def _omp_capabilities(model_config: ModelConfig) -> PortCapabilities:
    access_mode = get_agent_runtime_access_mode(model_config)
    return PortCapabilities(
        supports_system_prompt=True,
        supports_file_tools=True,
        supports_code_execution=access_mode == "read_write",
        uses_sandboxed_execution=False,
    )


AdapterRegistry.register(
    AdapterSpec(
        interface="omp",
        description="Oh My Pi coding agent over Agent Client Protocol v1",
        agent_factory=_create_agent,
        llm_factory=None,
        parser_factory=None,
        availability_checker=check_omp_available,
        fallback_interface="langchain",
        runtime_profile=AgentRuntimeProfile(capabilities=_omp_capabilities),
        supports_mcp=True,
        supports_tools=True,
        agent_tier="deep_agent",
        requires_provider=True,
    )
)
