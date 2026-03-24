"""Abstention detection instruction for LangChain Deep Agents adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = """\
<output_format>
Respond with ONLY a JSON object with this exact structure (reasoning MUST come first):
{
    "reasoning": "Brief explanation of why this was classified as abstention or genuine attempt",
    "abstention_detected": true or false
}
</output_format>"""

USER_ADDITION = ""


@dataclass
class _DeepAgentsAbstentionInstruction:
    """Append JSON output format for Deep Agents abstention detection.

    The Deep Agents adapter delegates to LangChain-style LLM calls, so the
    full JSON format block must be included in the system prompt.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _deep_agents_abstention_factory(**kwargs: object) -> _DeepAgentsAbstentionInstruction:  # noqa: ARG001
    """Factory producing Deep Agents abstention detection instructions."""
    return _DeepAgentsAbstentionInstruction()


AdapterInstructionRegistry.register("langchain_deep_agents", "abstention_detection", _deep_agents_abstention_factory)
