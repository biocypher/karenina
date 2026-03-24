"""Sufficiency detection instruction for LangChain Deep Agents adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = """\
<output_format>
Respond with ONLY a JSON object with this exact structure (reasoning MUST come first):
{
    "reasoning": "For each field, explain whether information exists. End with overall determination.",
    "sufficient": true or false
}
</output_format>"""

USER_ADDITION = ""


@dataclass
class _DeepAgentsSufficiencyInstruction:
    """Append JSON output format for Deep Agents sufficiency detection.

    The Deep Agents adapter delegates to LangChain-style LLM calls, so the
    full JSON format block must be included in the system prompt.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _deep_agents_sufficiency_factory(**kwargs: object) -> _DeepAgentsSufficiencyInstruction:  # noqa: ARG001
    """Factory producing Deep Agents sufficiency detection instructions."""
    return _DeepAgentsSufficiencyInstruction()


AdapterInstructionRegistry.register("langchain_deep_agents", "sufficiency_detection", _deep_agents_sufficiency_factory)
