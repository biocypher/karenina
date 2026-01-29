"""Rubric instruction for Claude Agent SDK adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = "If uncertain, use your best interpretation based on the response."

USER_ADDITION = ""


@dataclass
class _ClaudeSDKRubricInstruction:
    """Minimal rubric evaluation directives for Claude Agent SDK.

    The SDK uses native structured output, so no format/schema sections
    are needed. Only interpretation guidance is appended.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _claude_sdk_rubric_factory(**kwargs: object) -> _ClaudeSDKRubricInstruction:  # noqa: ARG001
    return _ClaudeSDKRubricInstruction()


_RUBRIC_TASKS = [
    "rubric_llm_trait_batch",
    "rubric_llm_trait_single",
    "rubric_literal_trait_batch",
    "rubric_literal_trait_single",
    "rubric_metric_trait",
]
for _task in _RUBRIC_TASKS:
    AdapterInstructionRegistry.register("claude_agent_sdk", _task, _claude_sdk_rubric_factory)
