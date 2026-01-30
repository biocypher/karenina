"""Retry feedback prompt for null values in required fields.

Variables:
    {field_list} - Comma-separated list of field names with null values
    {failed_response} - The response that failed validation
"""

PROMPT = """The previous response contained null values for required fields: [{field_list}].

Required fields cannot be null. Please provide actual values instead:
- If the information is not available in the source, provide an appropriate default value:
  * 0.0 for numeric fields (float/int)
  * Empty string "" for text fields
  * false for boolean fields
- If the field represents "unknown" or "not applicable", use a sensible placeholder
- **Never use null/None for required fields**

Previous response that failed:
{failed_response}

Please provide a corrected response with all required fields populated."""
