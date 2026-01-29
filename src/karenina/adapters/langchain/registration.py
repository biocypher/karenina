"""Registration module for LangChain adapters.

This module registers the LangChain interface and related routing interfaces
(openrouter, openai_endpoint) with the AdapterRegistry.

Adapter instructions for PARSING, RUBRIC, and DEEP JUDGMENT tasks are
registered via the prompts subpackage (imported at the bottom of this module).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from karenina.adapters.registry import AdapterAvailability, AdapterRegistry, AdapterSpec

if TYPE_CHECKING:
    from karenina.ports import AgentPort, LLMPort, ParserPort
    from karenina.schemas.workflow.models import ModelConfig


def _check_langchain_availability() -> AdapterAvailability:
    """Check if LangChain is available."""
    try:
        import langchain_core  # noqa: F401

        return AdapterAvailability(
            available=True,
            reason="LangChain is installed and available",
        )
    except ImportError:
        return AdapterAvailability(
            available=False,
            reason="LangChain packages not installed. Install with: pip install langchain-core langchain-anthropic",
            fallback_interface=None,  # No fallback if LangChain unavailable
        )


def _create_langchain_agent(config: ModelConfig) -> AgentPort:
    """Factory function to create LangChain agent adapter."""
    from karenina.adapters.langchain.agent import LangChainAgentAdapter

    return LangChainAgentAdapter(config)


def _create_langchain_llm(config: ModelConfig) -> LLMPort:
    """Factory function to create LangChain LLM adapter."""
    from karenina.adapters.langchain.llm import LangChainLLMAdapter

    return LangChainLLMAdapter(config)


def _create_langchain_parser(config: ModelConfig) -> ParserPort:
    """Factory function to create LangChain parser adapter."""
    from karenina.adapters.langchain.parser import LangChainParserAdapter

    return LangChainParserAdapter(config)


def _format_langchain_model_string(config: ModelConfig) -> str:
    """Format model string for standard LangChain interface."""
    if config.model_provider:
        return f"{config.model_provider}/{config.model_name}"
    return config.model_name or "unknown"


def _format_openrouter_model_string(config: ModelConfig) -> str:
    """Format model string for OpenRouter interface.

    OpenRouter uses just the model name, not provider/model format.
    """
    return config.model_name or "unknown"


def _format_openai_endpoint_model_string(config: ModelConfig) -> str:
    """Format model string for OpenAI endpoint interface."""
    return f"endpoint/{config.model_name}" if config.model_name else "endpoint/unknown"


# Register the langchain adapter (primary interface)
_langchain_spec = AdapterSpec(
    interface="langchain",
    description="LangChain adapter for standard LLM providers",
    agent_factory=_create_langchain_agent,
    llm_factory=_create_langchain_llm,
    parser_factory=_create_langchain_parser,
    availability_checker=_check_langchain_availability,
    fallback_interface=None,
    model_string_formatter=_format_langchain_model_string,
    routes_to=None,
    supports_mcp=True,
    supports_tools=True,
)

AdapterRegistry.register(_langchain_spec)


# Register openrouter adapter (routes through langchain)
_openrouter_spec = AdapterSpec(
    interface="openrouter",
    description="OpenRouter interface (routes through LangChain)",
    agent_factory=_create_langchain_agent,
    llm_factory=_create_langchain_llm,
    parser_factory=_create_langchain_parser,
    availability_checker=_check_langchain_availability,
    fallback_interface=None,
    model_string_formatter=_format_openrouter_model_string,
    routes_to="langchain",  # Indicates this routes to langchain
    supports_mcp=True,
    supports_tools=True,
)

AdapterRegistry.register(_openrouter_spec)


# Register openai_endpoint adapter (routes through langchain)
_openai_endpoint_spec = AdapterSpec(
    interface="openai_endpoint",
    description="OpenAI-compatible endpoint interface (routes through LangChain)",
    agent_factory=_create_langchain_agent,
    llm_factory=_create_langchain_llm,
    parser_factory=_create_langchain_parser,
    availability_checker=_check_langchain_availability,
    fallback_interface=None,
    model_string_formatter=_format_openai_endpoint_model_string,
    routes_to="langchain",  # Indicates this routes to langchain
    supports_mcp=True,
    supports_tools=True,
)

AdapterRegistry.register(_openai_endpoint_spec)

# Import prompt modules to trigger adapter instruction registration
import karenina.adapters.langchain.prompts.deep_judgment  # noqa: E402, F401
import karenina.adapters.langchain.prompts.parsing  # noqa: E402, F401
import karenina.adapters.langchain.prompts.rubric  # noqa: E402, F401
