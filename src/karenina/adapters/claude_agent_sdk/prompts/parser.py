"""Parser prompt for Claude Agent SDK adapter.

Unlike LangChain/Claude Tool which use separate system/user messages, the SDK
uses a single prompt string. This combined prompt includes all instructions.

Variables:
    {response} - The raw LLM response text to parse
    {json_schema} - JSON schema string (pre-formatted with indent=2)
"""

PROMPT = """You are an evaluator that extracts structured information from responses.

# Task
Parse the following response and extract structured information according to the JSON schema provided.

# Extraction Protocol

## 1. Extract According to Schema
- Each field description specifies WHAT to extract from the response
- Follow field descriptions precisely
- Use `null` for information not present (if field allows null)

## 2. Fidelity
- Extract only what's actually stated in the response
- Don't infer or add information not present
- If uncertain, use your best interpretation based on the text

# Response to Parse
{response}

# JSON Schema Reference
```json
{json_schema}
```

Extract the structured information from the response above."""
