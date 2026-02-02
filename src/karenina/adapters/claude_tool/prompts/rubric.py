"""Rubric instruction for Claude Tool adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = "Evaluate based solely on the provided criteria. Be precise and consistent."

USER_ADDITION = ""


@dataclass
class _ClaudeToolRubricInstruction:
    """Minimal rubric evaluation directives for Claude Tool.

    Claude Tool uses native structured output, so no format/schema sections
    are needed. Only evaluation guidance is appended.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _claude_tool_rubric_factory(**kwargs: object) -> _ClaudeToolRubricInstruction:  # noqa: ARG001
    return _ClaudeToolRubricInstruction()


_RUBRIC_TASKS = [
    "rubric_llm_trait_batch",
    "rubric_llm_trait_single",
    "rubric_literal_trait_batch",
    "rubric_literal_trait_single",
    "rubric_metric_trait",
]
for _task in _RUBRIC_TASKS:
    AdapterInstructionRegistry.register("claude_tool", _task, _claude_tool_rubric_factory)
