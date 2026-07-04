"""Abstention detection instruction for Claude Tool adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = (
    'Respond as a JSON object with "reasoning" (string) and "abstention_detected" '
    '(boolean) fields. The "reasoning" field MUST come first.'
)

USER_ADDITION = ""


@dataclass
class _ClaudeToolAbstentionInstruction:
    """Minimal abstention detection directives for Claude Tool adapter.

    Native structured output handles format enforcement, so only a brief
    schema reminder is appended.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _claude_tool_abstention_factory(**kwargs: object) -> _ClaudeToolAbstentionInstruction:  # noqa: ARG001
    """Factory producing Claude Tool abstention detection instructions."""
    return _ClaudeToolAbstentionInstruction()


AdapterInstructionRegistry.register("claude_tool", "abstention_detection", _claude_tool_abstention_factory)
