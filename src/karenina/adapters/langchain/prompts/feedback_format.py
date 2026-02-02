"""Retry feedback prompt for JSON format errors.

Variables:
    {failed_response} - The response that failed to parse (truncated to 1000 chars by caller)
    {error} - Error message (truncated to 500 chars by caller)
    {schema_hint} - Optional schema hint (empty string or formatted schema)
"""

PROMPT = """Your previous response could not be parsed as valid JSON.

**CRITICAL**: You must output ONLY a valid JSON object. Do not include:
- Any reasoning, explanation, or thinking
- Any text before or after the JSON
- Any markdown formatting (no ``` blocks)
- Any comments

**Your previous response that failed to parse:**
{failed_response}

**Error message:**
{error}
{schema_hint}

Please respond with ONLY the JSON object, nothing else."""
