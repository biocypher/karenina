"""Parsing instruction for Claude Tool adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = """\
Extract only what's actually stated - don't infer or add information not present.
Use null for information not present (if field allows null)."""

USER_ADDITION = ""


@dataclass
class _ClaudeToolParsingInstruction:
    """Append extraction directives for Claude Tool parsing.

    Claude Tool uses native structured output (beta.messages.parse), so no
    format/schema sections are needed. Only critical extraction rules are appended.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _claude_tool_parsing_instruction_factory(**kwargs: object) -> _ClaudeToolParsingInstruction:  # noqa: ARG001
    return _ClaudeToolParsingInstruction()


AdapterInstructionRegistry.register("claude_tool", "parsing", _claude_tool_parsing_instruction_factory)
