"""Self-registration for the claude_tool adapter.

This module registers the claude_tool adapter with the AdapterRegistry.
Import this module to make the adapter available for use.

The adapter is registered with:
- LLM factory: ClaudeToolLLMAdapter
- Agent factory: ClaudeToolAgentAdapter
- Parser factory: ClaudeToolParserAdapter

Adapter instructions for PARSING, RUBRIC, and DEEP JUDGMENT tasks are
registered via the prompts subpackage (imported at the bottom of this module).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from karenina.adapters.registry import AdapterAvailability, AdapterRegistry, AdapterSpec

if TYPE_CHECKING:
    from karenina.ports import AgentPort, LLMPort, ParserPort
    from karenina.schemas.config import ModelConfig

logger = logging.getLogger(__name__)


def _check_availability() -> AdapterAvailability:
    """Check if the claude_tool adapter is available.

    Checks that the required anthropic and mcp packages are installed.

    Returns:
        AdapterAvailability indicating whether the adapter can be used.
    """
    try:
        import anthropic  # noqa: F401

        # Check for mcp as well
        try:
            import mcp  # noqa: F401

            return AdapterAvailability(
                available=True,
                reason="anthropic and mcp packages are installed",
            )
        except ImportError:
            return AdapterAvailability(
                available=True,
                reason="anthropic package installed (mcp optional for MCP server support)",
            )

    except ImportError:
        return AdapterAvailability(
            available=False,
            reason="anthropic package not installed. Install with: pip install anthropic",
            fallback_interface="langchain",
        )


def _create_llm(config: ModelConfig) -> LLMPort:
    """Create a ClaudeToolLLMAdapter for the given config.

    Args:
        config: Model configuration.

    Returns:
        Initialized ClaudeToolLLMAdapter.
    """
    from karenina.adapters.claude_tool.llm import ClaudeToolLLMAdapter

    return ClaudeToolLLMAdapter(config)


def _create_agent(config: ModelConfig) -> AgentPort:
    """Create a ClaudeToolAgentAdapter for the given config.

    Args:
        config: Model configuration.

    Returns:
        Initialized ClaudeToolAgentAdapter.
    """
    from karenina.adapters.claude_tool.agent import ClaudeToolAgentAdapter

    return ClaudeToolAgentAdapter(config)


def _create_parser(config: ModelConfig) -> ParserPort:
    """Create a ClaudeToolParserAdapter for the given config.

    Args:
        config: Model configuration.

    Returns:
        Initialized ClaudeToolParserAdapter.
    """
    from karenina.adapters.claude_tool.parser import ClaudeToolParserAdapter

    return ClaudeToolParserAdapter(config)


# Create and register the adapter spec
_claude_tool_spec = AdapterSpec(
    interface="claude_tool",
    description="Anthropic Python SDK adapter with tool_runner for agents",
    agent_factory=_create_agent,
    llm_factory=_create_llm,
    parser_factory=_create_parser,
    availability_checker=_check_availability,
    fallback_interface="langchain",
    supports_mcp=True,
    supports_tools=True,
)

# Register with the AdapterRegistry
AdapterRegistry.register(_claude_tool_spec)

logger.debug("Registered claude_tool adapter with AdapterRegistry")

# Import prompt modules to trigger adapter instruction registration
import karenina.adapters.claude_tool.prompts.deep_judgment  # noqa: E402, F401
import karenina.adapters.claude_tool.prompts.parsing  # noqa: E402, F401
import karenina.adapters.claude_tool.prompts.rubric  # noqa: E402, F401
