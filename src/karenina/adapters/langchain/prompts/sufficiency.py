"""Sufficiency detection instruction for LangChain adapter."""

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
class _LangChainSufficiencyInstruction:
    """Append JSON output format for LangChain sufficiency detection.

    LangChain does not have native structured output, so the full JSON
    format block must be included in the system prompt.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _langchain_sufficiency_factory(**kwargs: object) -> _LangChainSufficiencyInstruction:  # noqa: ARG001
    """Factory producing LangChain sufficiency detection instructions."""
    return _LangChainSufficiencyInstruction()


AdapterInstructionRegistry.register("langchain", "sufficiency_detection", _langchain_sufficiency_factory)
AdapterInstructionRegistry.register("openrouter", "sufficiency_detection", _langchain_sufficiency_factory)
AdapterInstructionRegistry.register("openai_endpoint", "sufficiency_detection", _langchain_sufficiency_factory)
