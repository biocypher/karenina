"""User prompt for Claude Tool parser.

Variables:
    {response} - The raw LLM response text to parse
"""

PROMPT = """Parse the following response and extract structured information according to the schema.

**RESPONSE TO PARSE:**
{response}"""
