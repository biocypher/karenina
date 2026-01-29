"""Registration module for Claude Agent SDK adapters.

This module registers the claude_agent_sdk interface with the AdapterRegistry.
The Claude Agent SDK requires the Claude Code CLI to be installed.

Also registers adapter instructions for PARSING that append a best-interpretation
directive to the system prompt (the SDK uses native structured output via
output_format, so no format/schema sections are needed).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from karenina.adapters.registry import AdapterAvailability, AdapterRegistry, AdapterSpec
from karenina.ports.adapter_instruction import AdapterInstructionRegistry

if TYPE_CHECKING:
    from karenina.ports import AgentPort, LLMPort, ParserPort
    from karenina.schemas.workflow.models import ModelConfig


def _check_availability() -> AdapterAvailability:
    """Check if Claude Code CLI is installed and available.

    The Claude Agent SDK requires the 'claude' CLI binary to be in PATH.
    """
    claude_path = shutil.which("claude")

    if claude_path is not None:
        return AdapterAvailability(
            available=True,
            reason=f"Claude CLI found at: {claude_path}",
        )
    else:
        return AdapterAvailability(
            available=False,
            reason=(
                "Claude Code CLI not found in PATH. "
                "Install from: https://claude.ai/code "
                "or run: npm install -g @anthropic-ai/claude-code"
            ),
            fallback_interface="langchain",
        )


def _create_agent(config: ModelConfig) -> AgentPort:
    """Factory function to create Claude SDK agent adapter."""
    from karenina.adapters.claude_agent_sdk.agent import ClaudeSDKAgentAdapter

    return ClaudeSDKAgentAdapter(config)


def _create_llm(config: ModelConfig) -> LLMPort:
    """Factory function to create Claude SDK LLM adapter."""
    from karenina.adapters.claude_agent_sdk.llm import ClaudeSDKLLMAdapter

    return ClaudeSDKLLMAdapter(config)


def _create_parser(config: ModelConfig) -> ParserPort:
    """Factory function to create Claude SDK parser adapter."""
    from karenina.adapters.claude_agent_sdk.parser import ClaudeSDKParserAdapter

    return ClaudeSDKParserAdapter(config)


def _format_model_string(config: ModelConfig) -> str:
    """Format model string for Claude SDK interface."""
    return f"claude_sdk/{config.model_name}" if config.model_name else "claude_sdk/unknown"


# Register the Claude Agent SDK adapter
_claude_sdk_spec = AdapterSpec(
    interface="claude_agent_sdk",
    description="Claude Agent SDK for native Anthropic integration",
    agent_factory=_create_agent,
    llm_factory=_create_llm,
    parser_factory=_create_parser,
    availability_checker=_check_availability,
    fallback_interface="langchain",  # Fall back to langchain if CLI unavailable
    model_string_formatter=_format_model_string,
    routes_to=None,
    supports_mcp=True,
    supports_tools=True,
)

AdapterRegistry.register(_claude_sdk_spec)


# =============================================================================
# Adapter instructions for PARSING
# =============================================================================


@dataclass
class _ClaudeSDKParsingInstruction:
    """Append best-interpretation directive for Claude Agent SDK parsing.

    The Claude Agent SDK uses native structured output via output_format,
    so no format/schema sections are needed. Only a best-interpretation
    directive is appended.
    """

    @property
    def system_addition(self) -> str:
        return "If uncertain, use your best interpretation based on the text."

    @property
    def user_addition(self) -> str:
        return ""


def _claude_sdk_parsing_instruction_factory(**kwargs: object) -> _ClaudeSDKParsingInstruction:  # noqa: ARG001
    return _ClaudeSDKParsingInstruction()


AdapterInstructionRegistry.register("claude_agent_sdk", "parsing", _claude_sdk_parsing_instruction_factory)
