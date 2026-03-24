"""Sufficiency detection instruction for Claude Agent SDK adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = (
    'Respond as a JSON object with "reasoning" (string) and "sufficient" '
    '(boolean) fields. The "reasoning" field MUST come first.'
)

USER_ADDITION = ""


@dataclass
class _ClaudeSDKSufficiencyInstruction:
    """Minimal sufficiency detection directives for Claude Agent SDK adapter.

    The Claude Agent SDK uses native structured output via output_format,
    so only a brief schema reminder is appended.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _claude_sdk_sufficiency_factory(**kwargs: object) -> _ClaudeSDKSufficiencyInstruction:  # noqa: ARG001
    """Factory producing Claude SDK sufficiency detection instructions."""
    return _ClaudeSDKSufficiencyInstruction()


AdapterInstructionRegistry.register("claude_agent_sdk", "sufficiency_detection", _claude_sdk_sufficiency_factory)
