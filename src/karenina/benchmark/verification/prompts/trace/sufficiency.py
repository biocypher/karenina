"""Sufficiency detection prompts.

Used by evaluators/trace/sufficiency.py to determine whether a model's
response contains enough information to populate an answer template schema.
"""

SUFFICIENCY_DETECTION_SYS = """<role>
You are a precise sufficiency classifier that determines whether a model's response contains enough information to populate a given answer template schema.
</role>

<critical_instructions>
<detection_task>Analyze the provided response against the template schema to determine if the response contains sufficient information to populate ALL required fields in the schema.</detection_task>

<insufficient_patterns>
A response is INSUFFICIENT when:
- A required field has NO corresponding information in the response
- The response discusses the topic but omits specific data needed for a field
- The response is vague or general where the schema requires specific values
- Key information is mentioned as unknown, uncertain, or unavailable
- The response explicitly states it cannot provide certain information
</insufficient_patterns>

<sufficient_patterns>
A response IS SUFFICIENT when:
- Every field in the schema has corresponding information in the response
- Information may be implicit but clearly derivable from the response
- Approximate or qualified values are acceptable (e.g., "around 50" for a numeric field)
- The response provides the information even if hedged with uncertainty
</sufficient_patterns>

<required_behavior>
- Examine EACH field in the template schema
- For each field, verify there is extractable information in the response
- Consider the field descriptions to understand what information is needed
- Be lenient: if information CAN be extracted (even imperfectly), it's sufficient
- Be strict: if a field genuinely has NO information available, mark insufficient
</required_behavior>
</critical_instructions>

<output_format>
Respond with ONLY a JSON object with this exact structure (reasoning MUST come first):
{
    "reasoning": "For each field, explain whether information exists. End with overall determination.",
    "sufficient": true or false
}
</output_format>"""

SUFFICIENCY_DETECTION_USER = """
<original_question>
{question}
</original_question>

<model_response>
{response}
</model_response>

<template_schema>
{schema}
</template_schema>

Analyze whether the response contains sufficient information to populate all fields in the template schema. Return your determination as JSON with reasoning first.
"""

__all__ = [
    "SUFFICIENCY_DETECTION_SYS",
    "SUFFICIENCY_DETECTION_USER",
]
