"""Deep judgment instruction for Claude Agent SDK adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = "If uncertain, use your best interpretation based on the text."

USER_ADDITION = ""


@dataclass
class _ClaudeSDKDJInstruction:
    """Minimal deep judgment directives for Claude Agent SDK.

    The SDK uses native structured output, so no format/schema sections
    are needed. Only interpretation guidance is appended.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _claude_sdk_dj_factory(**kwargs: object) -> _ClaudeSDKDJInstruction:  # noqa: ARG001
    return _ClaudeSDKDJInstruction()


_DJ_STRUCTURED_TASKS = [
    "dj_rubric_excerpt_extraction",
    "dj_rubric_hallucination",
    "dj_rubric_score_extraction",
    "dj_template_excerpt_extraction",
    "dj_template_hallucination",
]
for _task in _DJ_STRUCTURED_TASKS:
    AdapterInstructionRegistry.register("claude_agent_sdk", _task, _claude_sdk_dj_factory)
