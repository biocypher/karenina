"""User prompt for LLM-based response parsing.

Variables:
    {response} - The raw LLM response text to parse
    {json_schema} - JSON schema string (pre-formatted with indent=2)
"""

PROMPT = """Parse the following response and extract structured information.

**RESPONSE TO PARSE:**
{response}

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**PARSING NOTES:**
- Extract values for each field based on its description in the schema
- If information for a field is not present, use null (if field allows null) or your best inference
- Return ONLY the JSON object - no surrounding text

**YOUR JSON RESPONSE:**"""
