"""Registration module for LangChain Deep Agents adapters.

This module registers the langchain_deep_agents interface with the AdapterRegistry.
The Deep Agents adapter requires the deepagents package to be installed.

Adapter instructions for PARSING, RUBRIC, and DEEP JUDGMENT tasks are
registered via the prompts subpackage (imported at the bottom of this module).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from karenina.adapters.registry import AdapterAvailability, AdapterRegistry, AdapterSpec

if TYPE_CHECKING:
    from karenina.ports import AgentPort, LLMPort, ParserPort
    from karenina.schemas.config import ModelConfig


def _check_availability() -> AdapterAvailability:
    """Check availability via dedicated module."""
    from karenina.adapters.langchain_deep_agents.availability import check_deep_agents_available

    return check_deep_agents_available()


def _create_agent(config: ModelConfig) -> AgentPort:
    """Factory function to create Deep Agents agent adapter."""
    from karenina.adapters.langchain_deep_agents.agent import DeepAgentsAgentAdapter

    return DeepAgentsAgentAdapter(config)


def _create_llm(config: ModelConfig) -> LLMPort:
    """Factory function to create Deep Agents LLM adapter."""
    from karenina.adapters.langchain_deep_agents.llm import DeepAgentsLLMAdapter

    return DeepAgentsLLMAdapter(config)


def _create_parser(config: ModelConfig) -> ParserPort:
    """Factory function to create Deep Agents parser adapter."""
    from karenina.adapters.langchain_deep_agents.parser import DeepAgentsParserAdapter

    return DeepAgentsParserAdapter(config)


# Register the LangChain Deep Agents adapter
_deep_agents_spec = AdapterSpec(
    interface="langchain_deep_agents",
    description="LangChain Deep Agents for autonomous task execution",
    agent_factory=_create_agent,
    llm_factory=_create_llm,
    parser_factory=_create_parser,
    availability_checker=_check_availability,
    fallback_interface=None,  # No fallback: explicit install required
    routes_to=None,
    supports_mcp=True,
    supports_tools=True,
    agent_tier="deep_agent",
)

AdapterRegistry.register(_deep_agents_spec)

# Import prompt modules to trigger adapter instruction registration
import karenina.adapters.langchain_deep_agents.prompts.abstention  # noqa: E402, F401
import karenina.adapters.langchain_deep_agents.prompts.deep_judgment  # noqa: E402, F401
import karenina.adapters.langchain_deep_agents.prompts.parsing  # noqa: E402, F401
import karenina.adapters.langchain_deep_agents.prompts.rubric  # noqa: E402, F401
import karenina.adapters.langchain_deep_agents.prompts.sufficiency  # noqa: E402, F401
