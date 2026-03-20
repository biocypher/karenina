"""Rubric instruction for LangChain Deep Agents adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = "If uncertain, use your best interpretation based on the response."

USER_ADDITION = ""


@dataclass
class _DeepAgentsRubricInstruction:
    """Minimal rubric evaluation directives for Deep Agents adapter.

    The adapter uses LangChain's structured output, so no format/schema
    sections are needed. Only interpretation guidance is appended.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _deep_agents_rubric_factory(**kwargs: object) -> _DeepAgentsRubricInstruction:  # noqa: ARG001
    return _DeepAgentsRubricInstruction()


_RUBRIC_TASKS = [
    "rubric_llm_trait_batch",
    "rubric_llm_trait_single",
    "rubric_literal_trait_batch",
    "rubric_literal_trait_single",
    "rubric_metric_trait",
]
for _task in _RUBRIC_TASKS:
    AdapterInstructionRegistry.register("langchain_deep_agents", _task, _deep_agents_rubric_factory)
