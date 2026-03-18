"""Deep judgment instruction for LangChain Deep Agents adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = "If uncertain, use your best interpretation based on the text."

USER_ADDITION = ""


@dataclass
class _DeepAgentsDJInstruction:
    """Minimal deep judgment directives for Deep Agents adapter.

    The adapter uses LangChain's structured output, so no format/schema
    sections are needed. Only interpretation guidance is appended.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _deep_agents_dj_factory(**kwargs: object) -> _DeepAgentsDJInstruction:  # noqa: ARG001
    return _DeepAgentsDJInstruction()


_DJ_STRUCTURED_TASKS = [
    "dj_rubric_excerpt_extraction",
    "dj_rubric_hallucination",
    "dj_rubric_score_extraction",
    "dj_template_excerpt_extraction",
    "dj_template_hallucination",
]
for _task in _DJ_STRUCTURED_TASKS:
    AdapterInstructionRegistry.register("langchain_deep_agents", _task, _deep_agents_dj_factory)
