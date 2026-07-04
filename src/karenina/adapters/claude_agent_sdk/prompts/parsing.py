"""Parsing instruction for Claude Agent SDK adapter.

The Claude Agent SDK parser uses a direct API call (not the SDK subprocess)
for structured output extraction. The JSON schema and extraction rules are
included in the prompt to guide the model, complementing the API-level
schema enforcement (output_config on Anthropic, response_format on vLLM/sglang).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from karenina.ports.adapter_instruction import AdapterInstructionRegistry

SYSTEM_ADDITION = """\
# Extraction Rules

- Extract only what's actually stated; do not infer or add information not present.
- Use null for information not present (if the field allows null).
- Match field names and types exactly as defined in the JSON schema.
- If a field expects a string, return a single string, not a list.
- If a field expects a boolean, return true or false, not a string or number.
- If uncertain, use your best interpretation based on the text.

# Output Format

Return ONLY a valid JSON object. Do not wrap it in markdown fences, code blocks, or any surrounding text."""

USER_ADDITION = """\
**JSON SCHEMA (your response MUST conform to this):**
```json
{schema_json}
```

**CRITICAL:**
- Every field name must match the schema exactly.
- Every required field must be present.
- Return ONLY the raw JSON object, no explanation, no markdown fences."""


@dataclass
class _ClaudeSDKParsingInstruction:
    """Append extraction and format directives for Claude Agent SDK parsing.

    When targeting the Anthropic API, schema enforcement is handled by
    output_config (constrained decoding). When targeting custom endpoints
    (vLLM, sglang), enforcement uses the OpenAI response_format. In both
    cases, the prompt-level schema instructions serve as a reinforcing signal.
    """

    json_schema: dict[str, Any] | None = None

    @property
    def system_addition(self) -> str:
        return SYSTEM_ADDITION

    @property
    def user_addition(self) -> str:
        if self.json_schema is None:
            return ""
        schema_json = json.dumps(self.json_schema, indent=2)
        return USER_ADDITION.format(schema_json=schema_json)


def _claude_sdk_parsing_instruction_factory(**kwargs: object) -> _ClaudeSDKParsingInstruction:
    return _ClaudeSDKParsingInstruction(
        json_schema=kwargs.get("json_schema"),  # type: ignore[arg-type]
    )


AdapterInstructionRegistry.register("claude_agent_sdk", "parsing", _claude_sdk_parsing_instruction_factory)
