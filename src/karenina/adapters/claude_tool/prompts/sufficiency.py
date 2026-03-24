"""Sufficiency detection instruction for Claude Tool adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = (
    'Respond as a JSON object with "reasoning" (string) and "sufficient" '
    '(boolean) fields. The "reasoning" field MUST come first.'
)

USER_ADDITION = ""


@dataclass
class _ClaudeToolSufficiencyInstruction:
    """Minimal sufficiency detection directives for Claude Tool adapter.

    Native structured output handles format enforcement, so only a brief
    schema reminder is appended.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _claude_tool_sufficiency_factory(**kwargs: object) -> _ClaudeToolSufficiencyInstruction:  # noqa: ARG001
    """Factory producing Claude Tool sufficiency detection instructions."""
    return _ClaudeToolSufficiencyInstruction()


AdapterInstructionRegistry.register("claude_tool", "sufficiency_detection", _claude_tool_sufficiency_factory)
