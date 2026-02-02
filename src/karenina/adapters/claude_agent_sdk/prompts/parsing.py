"""Parsing instruction for Claude Agent SDK adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = "If uncertain, use your best interpretation based on the text."

USER_ADDITION = ""


@dataclass
class _ClaudeSDKParsingInstruction:
    """Append best-interpretation directive for Claude Agent SDK parsing.

    The Claude Agent SDK uses native structured output via output_format,
    so no format/schema sections are needed. Only a best-interpretation
    directive is appended.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _claude_sdk_parsing_instruction_factory(**kwargs: object) -> _ClaudeSDKParsingInstruction:  # noqa: ARG001
    return _ClaudeSDKParsingInstruction()


AdapterInstructionRegistry.register("claude_agent_sdk", "parsing", _claude_sdk_parsing_instruction_factory)
