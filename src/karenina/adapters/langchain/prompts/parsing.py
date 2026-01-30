"""Parsing instruction for LangChain adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = """\
# Output Format

Return only the completed JSON object - no surrounding text, no markdown fences:

<format_instructions>
{format_instructions}
</format_instructions>"""

USER_ADDITION = """\
**JSON SCHEMA (your response MUST conform to this):**
```json
{schema_json}
```

**PARSING NOTES:**
- Extract values for each field based on its description in the schema
- If information for a field is not present, use null (if field allows null) or your best inference
- Return ONLY the JSON object - no surrounding text

**YOUR JSON RESPONSE:**"""


@dataclass
class _LangChainFormatInstruction:
    """Append format instructions for LangChain parsing.

    LangChain does not have native structured output, so both system and
    user prompts need explicit format instructions with the JSON schema.
    """

    json_schema: dict[str, Any] | None = None
    format_instructions: str = ""

    @property
    def system_addition(self) -> str:
        if not self.format_instructions:
            return ""
        return SYSTEM_ADDITION.format(format_instructions=self.format_instructions)

    @property
    def user_addition(self) -> str:
        if self.json_schema is None:
            return ""
        schema_json = json.dumps(self.json_schema, indent=2)
        return USER_ADDITION.format(schema_json=schema_json)


def _langchain_format_instruction_factory(**kwargs: object) -> _LangChainFormatInstruction:
    """Factory producing LangChain format instructions.

    Expects ``json_schema`` and optionally ``format_instructions`` in the
    instruction context dict.
    """
    return _LangChainFormatInstruction(
        json_schema=kwargs.get("json_schema"),  # type: ignore[arg-type]
        format_instructions=kwargs.get("format_instructions", "") or "",  # type: ignore[arg-type]
    )


AdapterInstructionRegistry.register("langchain", "parsing", _langchain_format_instruction_factory)
AdapterInstructionRegistry.register("openrouter", "parsing", _langchain_format_instruction_factory)
AdapterInstructionRegistry.register("openai_endpoint", "parsing", _langchain_format_instruction_factory)
