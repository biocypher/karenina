"""
User prompt templates for ADeLe question classification.

These templates use {placeholder} syntax for .format() substitution.
"""

USER_PROMPT_SINGLE_TRAIT_TEMPLATE = """Classify the following QUESTION for the ADeLe dimension: **{trait_name}**

**QUESTION TO CLASSIFY:**
{question_text}

**DIMENSION: {trait_name}**
{trait_description}

Classes (in order of increasing complexity): {class_names}

{class_details}

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**REQUIRED OUTPUT FORMAT:**
Return a JSON object with a "classification" key containing the exact class name.

Example format (class value is just an example):
{example_json}

**YOUR JSON RESPONSE:**"""

USER_PROMPT_BATCH_TEMPLATE = """Classify the following QUESTION for each ADeLe dimension:

**QUESTION TO CLASSIFY:**
{question_text}

**ADeLe DIMENSIONS TO CLASSIFY:**
{traits_description}

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**REQUIRED OUTPUT FORMAT:**
Return a JSON object with a "classifications" key mapping trait names to class names.
Use EXACT trait and class names as shown above.

Example format (class values are just examples):
{example_json}

**YOUR JSON RESPONSE:**"""
