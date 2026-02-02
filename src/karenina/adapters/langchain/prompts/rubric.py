"""Rubric instruction for LangChain adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = """\
**RESPONSE FORMAT:**
You will receive a JSON Schema specifying the exact output structure. Your response MUST conform to this schema.
Return ONLY a JSON object - no explanations, no markdown, no surrounding text.

**WHAT NOT TO DO:**
- Do NOT wrap JSON in markdown code blocks (no ```)
- Do NOT add explanatory text before or after the JSON"""

USER_ADDITION = ""  # Built dynamically from fields


@dataclass
class _LangChainRubricFormatInstruction:
    """Append format instructions for LangChain rubric evaluation.

    LangChain does not have native structured output, so rubric prompts need
    explicit JSON schema and format sections appended by the adapter.
    """

    json_schema: dict[str, Any] | None = None
    example_json: str = ""
    output_format_hint: str = ""

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        parts: list[str] = []
        if self.json_schema is not None:
            schema_json = json.dumps(self.json_schema, indent=2)
            parts.append(f"**JSON SCHEMA (your response MUST conform to this):**\n```json\n{schema_json}\n```")
        if self.example_json:
            parts.append(
                f"**REQUIRED OUTPUT FORMAT:**\n"
                f"Return a JSON object matching the schema above.\n\n"
                f"Example:\n{self.example_json}"
            )
        elif self.output_format_hint:
            parts.append(f"**REQUIRED OUTPUT FORMAT:**\n{self.output_format_hint}")
        parts.append("**YOUR JSON RESPONSE:**")
        return "\n\n".join(parts)


def _langchain_rubric_format_factory(**kwargs: object) -> _LangChainRubricFormatInstruction:
    """Factory producing LangChain rubric format instructions."""
    return _LangChainRubricFormatInstruction(
        json_schema=kwargs.get("json_schema"),  # type: ignore[arg-type]
        example_json=kwargs.get("example_json", "") or "",  # type: ignore[arg-type]
        output_format_hint=kwargs.get("output_format_hint", "") or "",  # type: ignore[arg-type]
    )


_RUBRIC_TASKS = [
    "rubric_llm_trait_batch",
    "rubric_llm_trait_single",
    "rubric_literal_trait_batch",
    "rubric_literal_trait_single",
    "rubric_metric_trait",
]
_LANGCHAIN_INTERFACES = ["langchain", "openrouter", "openai_endpoint"]

for _task in _RUBRIC_TASKS:
    for _iface in _LANGCHAIN_INTERFACES:
        AdapterInstructionRegistry.register(_iface, _task, _langchain_rubric_format_factory)
