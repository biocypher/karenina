"""Deep judgment instruction for Claude Tool adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = "Extract only what's actually stated. Use null for missing information."

USER_ADDITION = ""


@dataclass
class _ClaudeToolDJInstruction:
    """Minimal deep judgment directives for Claude Tool.

    Claude Tool uses native structured output, so no format/schema sections
    are needed. Only extraction guidance is appended.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _claude_tool_dj_factory(**kwargs: object) -> _ClaudeToolDJInstruction:  # noqa: ARG001
    return _ClaudeToolDJInstruction()


_DJ_STRUCTURED_TASKS = [
    "dj_rubric_excerpt_extraction",
    "dj_rubric_hallucination",
    "dj_rubric_score_extraction",
    "dj_template_excerpt_extraction",
    "dj_template_hallucination",
]
for _task in _DJ_STRUCTURED_TASKS:
    AdapterInstructionRegistry.register("claude_tool", _task, _claude_tool_dj_factory)
