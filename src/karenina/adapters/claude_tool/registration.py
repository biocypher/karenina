"""Self-registration for the claude_tool adapter.

This module registers the claude_tool adapter with the AdapterRegistry.
Import this module to make the adapter available for use.

The adapter is registered with:
- LLM factory: ClaudeToolLLMAdapter
- Agent factory: ClaudeToolAgentAdapter
- Parser factory: ClaudeToolParserAdapter

Also registers adapter instructions for PARSING that append extraction
directives to the system prompt (Claude Tool uses native structured output
via beta.messages.parse, so no format/schema sections are needed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from karenina.adapters.registry import AdapterAvailability, AdapterRegistry, AdapterSpec
from karenina.ports.adapter_instruction import AdapterInstructionRegistry

if TYPE_CHECKING:
    from karenina.ports import AgentPort, LLMPort, ParserPort
    from karenina.schemas.workflow.models import ModelConfig

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


def _format_model_string(config: ModelConfig) -> str:
    """Format the model string for display.

    Args:
        config: Model configuration.

    Returns:
        Formatted model string.
    """
    # For claude_tool, just use the model name directly
    return config.model_name or "unknown"


# Create and register the adapter spec
_claude_tool_spec = AdapterSpec(
    interface="claude_tool",
    description="Anthropic Python SDK adapter with tool_runner for agents",
    agent_factory=_create_agent,
    llm_factory=_create_llm,
    parser_factory=_create_parser,
    availability_checker=_check_availability,
    fallback_interface="langchain",
    model_string_formatter=_format_model_string,
    supports_mcp=True,
    supports_tools=True,
)

# Register with the AdapterRegistry
AdapterRegistry.register(_claude_tool_spec)

logger.debug("Registered claude_tool adapter with AdapterRegistry")


# =============================================================================
# Adapter instructions for PARSING
# =============================================================================


@dataclass
class _ClaudeToolParsingInstruction:
    """Append extraction directives for Claude Tool parsing.

    Claude Tool uses native structured output (beta.messages.parse), so no
    format/schema sections are needed. Only critical extraction rules are appended.
    """

    @property
    def system_addition(self) -> str:
        return (
            "Extract only what's actually stated - don't infer or add information not present.\n"
            "Use null for information not present (if field allows null)."
        )

    @property
    def user_addition(self) -> str:
        return ""


def _claude_tool_parsing_instruction_factory(**kwargs: object) -> _ClaudeToolParsingInstruction:  # noqa: ARG001
    return _ClaudeToolParsingInstruction()


AdapterInstructionRegistry.register("claude_tool", "parsing", _claude_tool_parsing_instruction_factory)
