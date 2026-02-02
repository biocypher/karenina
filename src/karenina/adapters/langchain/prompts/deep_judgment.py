"""Deep judgment instruction for LangChain adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = """\
You will receive a JSON Schema specifying the exact output structure. Your response MUST conform to this schema.
Return ONLY a JSON object - no explanations, no markdown, no surrounding text.

**WHAT NOT TO DO:**
- Do NOT wrap JSON in markdown code blocks (no ```)
- Do NOT add explanatory text before or after the JSON"""

USER_ADDITION = ""  # Built dynamically from fields


@dataclass
class _LangChainDJFormatInstruction:
    """Append format instructions for LangChain deep judgment tasks.

    Deep judgment excerpt extraction, hallucination assessment, and score
    extraction all produce structured JSON output that needs explicit
    format instructions for LangChain adapters.
    """

    json_schema: dict[str, Any] | None = None
    parsing_notes: str = ""

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        parts: list[str] = []
        if self.json_schema is not None:
            schema_json = json.dumps(self.json_schema, indent=2)
            parts.append(f"**JSON SCHEMA (your response MUST conform to this):**\n```json\n{schema_json}\n```")
        if self.parsing_notes:
            parts.append(f"**PARSING NOTES:**\n{self.parsing_notes}")
        parts.append("**YOUR JSON RESPONSE:**")
        return "\n\n".join(parts)


def _langchain_dj_format_factory(**kwargs: object) -> _LangChainDJFormatInstruction:
    """Factory producing LangChain deep judgment format instructions."""
    return _LangChainDJFormatInstruction(
        json_schema=kwargs.get("json_schema"),  # type: ignore[arg-type]
        parsing_notes=kwargs.get("parsing_notes", "") or "",  # type: ignore[arg-type]
    )


_DJ_STRUCTURED_TASKS = [
    "dj_rubric_excerpt_extraction",
    "dj_rubric_hallucination",
    "dj_rubric_score_extraction",
    "dj_template_excerpt_extraction",
    "dj_template_hallucination",
]
_LANGCHAIN_INTERFACES = ["langchain", "openrouter", "openai_endpoint"]

for _task in _DJ_STRUCTURED_TASKS:
    for _iface in _LANGCHAIN_INTERFACES:
        AdapterInstructionRegistry.register(_iface, _task, _langchain_dj_format_factory)
