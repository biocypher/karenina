"""Abstention detection instruction for LangChain adapter."""

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
class _LangChainAbstentionInstruction:
    """Append JSON output format for LangChain abstention detection.

    LangChain does not have native structured output, so the full JSON
    format block must be included in the system prompt.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _langchain_abstention_factory(**kwargs: object) -> _LangChainAbstentionInstruction:  # noqa: ARG001
    """Factory producing LangChain abstention detection instructions."""
    return _LangChainAbstentionInstruction()


AdapterInstructionRegistry.register("langchain", "abstention_detection", _langchain_abstention_factory)
AdapterInstructionRegistry.register("openrouter", "abstention_detection", _langchain_abstention_factory)
AdapterInstructionRegistry.register("openai_endpoint", "abstention_detection", _langchain_abstention_factory)
