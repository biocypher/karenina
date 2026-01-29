"""Parser prompts for Claude Agent SDK adapter.

Split into system and user prompts to leverage ClaudeAgentOptions.system_prompt,
matching the LangChain adapter's separation of concerns.

Variables (user prompt only):
    {response} - The raw LLM response text to parse
    {json_schema} - JSON schema string (pre-formatted with indent=2)
"""

SYSTEM_PROMPT = """You are an evaluator that extracts structured information from responses.

You will receive:
1. A response to parse (from an LLM or other source)
2. A JSON schema with descriptive fields indicating what information to extract

# Extraction Protocol

## 1. Extract According to Schema
- Each field description specifies WHAT to extract from the response
- Follow field descriptions precisely
- Use `null` for information not present (if field allows null)

## 2. Fidelity
- Extract only what's actually stated in the response
- Don't infer or add information not present
- If uncertain, use your best interpretation based on the text"""

USER_PROMPT = """Parse the following response and extract structured information.

**RESPONSE TO PARSE:**
{response}

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

Extract the structured information from the response above."""
