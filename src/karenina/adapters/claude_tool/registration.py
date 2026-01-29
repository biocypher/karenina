"""Self-registration for the claude_tool adapter.

This module registers the claude_tool adapter with the AdapterRegistry.
Import this module to make the adapter available for use.

The adapter is registered with:
- LLM factory: ClaudeToolLLMAdapter
- Agent factory: ClaudeToolAgentAdapter
- Parser factory: ClaudeToolParserAdapter

Also registers adapter instructions for PARSING that replace the system
prompt with a minimal variant and strip format-related sections from the
user prompt (since Claude Tool uses native structured output).
"""

from __future__ import annotations

import logging
import re
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

# Minimal system prompt for Claude Tool (native structured output handles format)
_CLAUDE_TOOL_SYSTEM = (
    "You are an evaluator that extracts structured information from responses.\n\n"
    "Extract information according to the schema field descriptions. "
    "Each field description specifies what to extract.\n\n"
    "Critical rules:\n"
    "- Extract only what's actually stated - don't infer or add information not present\n"
    "- Use null for information not present (if field allows null)"
)


@dataclass
class _ClaudeToolParsingInstruction:
    """Replace system prompt and strip format sections for Claude Tool.

    Claude Tool uses native structured output (beta.messages.parse), so:
    - System prompt is replaced with a minimal 3-line variant
    - JSON schema, parsing notes, and format_instructions are stripped from user text
    """

    def apply(self, system_text: str, user_text: str) -> tuple[str, str]:  # noqa: ARG002
        # Replace system with minimal variant
        system_text = _CLAUDE_TOOL_SYSTEM

        # Strip JSON schema block from user text
        user_text = re.sub(
            r"\*\*JSON SCHEMA \(your response MUST conform to this\):\*\*\s*```json\s*.*?```",
            "",
            user_text,
            flags=re.DOTALL,
        )

        # Strip parsing notes
        user_text = re.sub(
            r"\*\*PARSING NOTES:\*\*.*?(?=\*\*YOUR JSON RESPONSE:\*\*|\Z)",
            "",
            user_text,
            flags=re.DOTALL,
        )

        # Strip format_instructions section
        user_text = re.sub(
            r"<format_instructions>.*?</format_instructions>",
            "",
            user_text,
            flags=re.DOTALL,
        )

        # Strip format instruction block appended by LangChain-style instruction
        user_text = re.sub(
            r"\n\nYou must respond with valid JSON that matches this schema:.*?Return ONLY the JSON object, no additional text\.",
            "",
            user_text,
            flags=re.DOTALL,
        )

        # Clean up trailing whitespace
        user_text = user_text.rstrip()

        return system_text, user_text


def _claude_tool_parsing_instruction_factory(**kwargs: object) -> _ClaudeToolParsingInstruction:  # noqa: ARG001
    return _ClaudeToolParsingInstruction()


AdapterInstructionRegistry.register("claude_tool", "parsing", _claude_tool_parsing_instruction_factory)
