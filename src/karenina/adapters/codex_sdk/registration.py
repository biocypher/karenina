"""Registration module for the Codex SDK adapter.

Registers the codex_sdk interface with the AdapterRegistry. The adapter is
agent-only: LLM and parser duties fall back to the langchain interface.
The runtime profile registers codex capabilities with the shared agent
runtime helpers so pipeline prompts report the correct feature set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from karenina.adapters.agent_runtime import AgentRuntimeProfile, get_agent_runtime_access_mode
from karenina.adapters.registry import AdapterAvailability, AdapterRegistry, AdapterSpec
from karenina.ports.capabilities import PortCapabilities

if TYPE_CHECKING:
    from karenina.ports import AgentPort
    from karenina.schemas.config import ModelConfig


def _check_availability() -> AdapterAvailability:
    """Check availability via the dedicated module."""
    from karenina.adapters.codex_sdk.availability import check_codex_available

    return check_codex_available()


def _create_agent(config: ModelConfig) -> AgentPort:
    """Factory function to create the Codex SDK agent adapter."""
    from karenina.adapters.codex_sdk.agent import CodexSDKAgentAdapter

    return CodexSDKAgentAdapter(config)


def _codex_capabilities(model_config: ModelConfig) -> PortCapabilities:
    """Capabilities implied by the codex runtime for this model config.

    Codex always executes inside its own OS-level sandbox (read_only or
    workspace_write), so uses_sandboxed_execution is unconditionally True.
    """
    access_mode = get_agent_runtime_access_mode(model_config)
    return PortCapabilities(
        supports_system_prompt=True,
        supports_file_tools=True,
        supports_code_execution=access_mode != "read_only",
        uses_sandboxed_execution=True,
    )


_codex_sdk_spec = AdapterSpec(
    interface="codex_sdk",
    description="OpenAI Codex SDK for natively agentic execution",
    agent_factory=_create_agent,
    llm_factory=None,
    parser_factory=None,
    availability_checker=_check_availability,
    fallback_interface="langchain",
    routes_to=None,
    runtime_profile=AgentRuntimeProfile(capabilities=_codex_capabilities),
    # MCP wiring is a warn-and-skip stub (see mcp.py). Flip this to True
    # when the mcp_servers config-table mapping lands and is validated.
    supports_mcp=False,
    supports_tools=True,
    agent_tier="deep_agent",
    requires_provider=False,
)

AdapterRegistry.register(_codex_sdk_spec)
