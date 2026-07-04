"""Parsing instruction for LangChain Deep Agents adapter."""

from __future__ import annotations

from dataclasses import dataclass

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = "If uncertain, use your best interpretation based on the text."

USER_ADDITION = ""


@dataclass
class _DeepAgentsParsingInstruction:
    """Append best-interpretation directive for Deep Agents parsing.

    The Deep Agents adapter uses LangChain's with_structured_output(),
    so no format/schema sections are needed. Only a best-interpretation
    directive is appended.
    """

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        return USER_ADDITION


def _deep_agents_parsing_factory(**kwargs: object) -> _DeepAgentsParsingInstruction:  # noqa: ARG001
    return _DeepAgentsParsingInstruction()


AdapterInstructionRegistry.register("langchain_deep_agents", "parsing", _deep_agents_parsing_factory)
