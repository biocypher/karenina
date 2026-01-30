"""Format instructions appended to user message for fallback structured output.

Variables:
    {schema_json} - JSON schema string (pre-formatted with indent=2)
"""

PROMPT = """

You must respond with valid JSON that matches this schema:
```json
{schema_json}
```
Return ONLY the JSON object, no additional text."""
