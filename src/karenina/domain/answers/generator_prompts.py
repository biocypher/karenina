"""
Prompt constants for answer template generation.

This module contains the system prompts and user prompt templates used
by the structured answer generator. Chain building has been moved to
generator.py using the port/adapter pattern.
"""

# ============================================================================
# GROUND TRUTH EXTRACTION PROMPTS
# ============================================================================

GROUND_TRUTH_SYSTEM_PROMPT = """
You are an expert evaluation designer extracting ground-truth attributes from a question and its ideal answer. Build a Pydantic-friendly schema capturing what a judge model should read from candidate responses. Apply the following rules when specifying attributes:

- Prefer concise snake_case names that are stable and unambiguous.
- FORBIDDEN: Never use `str`, `List[str]`, or `Dict[str, str]` types. All text-based evaluations must be converted to boolean checks.
- Use `bool` whenever the judge needs to confirm whether a concept, entity, or pattern is present. This is the primary type for text-based evaluation.
- Use numeric types (int, float) only when measurable quantities are required.
- When the answer implies a categorical classification or grading scheme, use `Literal` types to enumerate all reasonable values in that domain.
- For lists of items (e.g., multiple drugs, genes, etc.), create separate boolean attributes for each expected item rather than using List[str].
- When the reference answer contains compound terms or phrases, treat them as single semantic units and create one boolean attribute for the complete concept, not separate attributes for individual words.
- Avoid redundant attributes; ensure each serves a unique decision-making purpose.
- Frame every attribute as something the judge can extract by reading the candidate response (e.g., `number_of_interacting_genes` to count genes mentioned, `mentions_control_group` to flag a concept).
- For each attribute, derive the `ground_truth` value from the reference answer that represents the expected correct response.
- Ensure the final response is valid JSON without trailing commentary.

Example JSON output:
{{
  "attributes": [
    {{
      "name": "count_of_items",
      "type": "int",
      "ground_truth": 3
    }},
    {{
      "name": "mentions_first_concept",
      "type": "bool",
      "ground_truth": true
    }},
    {{
      "name": "mentions_second_concept",
      "type": "bool",
      "ground_truth": false
    }},
    {{
      "name": "classification_level",
      "type": "Literal['high', 'medium', 'low']",
      "ground_truth": "high"
    }}
  ]
}}
""".strip()

GROUND_TRUTH_USER_PROMPT_TEMPLATE = """
You receive an evaluation sample consisting of a question and its reference answer:

Question:
{question}

Reference Answer:
{answer}

Identify the minimal set of structured attributes that a judge must extract from a candidate response to verify correctness. Construct a JSON object with a single key `attributes` containing a list of attribute definitions. Each definition must include `name`, `type`, and `ground_truth` fields, where `ground_truth` contains the expected correct value derived from the reference answer.

When selecting types, follow these guidelines:
- FORBIDDEN: Never use `str`, `List[str]`, or `Dict[str, str]` types.
- Use `bool` to capture presence or absence of concepts, entities, or patterns. This is the primary evaluation mechanism.
- When the reference answer suggests a categorical classification or scale, use `Literal` types with all reasonable values in that domain.
- For multiple items in the reference answer, create separate boolean attributes for each item instead of using lists.
- When the reference answer contains compound terms or phrases, treat them as single semantic units and create one boolean attribute for the complete concept, not separate attributes for individual words.

Return only valid JSON.
""".strip()


# ============================================================================
# FIELD DESCRIPTION PROMPTS
# ============================================================================

FIELD_DESCRIPTION_SYSTEM_PROMPT = """
You craft instructional text for judge models who must parse a candidate response to a question. For every attribute in the provided ground-truth specification, produce a short, direct description that explains exactly what the judge should read for in the response and how to answer. Highlight boolean expectations clearly.

Guidelines:
- Reference attribute names verbatim.
- Mention the expected type implicitly via phrasing (e.g., "Answer with true or false if ..." for booleans, "Provide the count of ..." for numeric fields).
- For boolean attributes checking concept presence, allow semantic equivalence and related terms rather than requiring exact string matches. Focus on whether the underlying concept is conveyed.
- When relevant, reference concrete response-focused examples such as "Number of interacting genes mentioned in the response" or "Does the response cite a control group?" to reinforce that extraction happens from the candidate answer.
- Stay concise (<= 2 sentences per attribute).
- Return only valid JSON.

Example mapping:
{{
  "field_descriptions": {{
    "count_of_items": "Provide an integer equal to the number of items mentioned in the response.",
    "mentions_first_concept": "Answer with true if the response refers to the first concept or semantically related terms; otherwise answer false.",
    "mentions_second_concept": "Answer with true if the response refers to the second concept or semantically related terms; otherwise answer false.",
    "classification_level": "Select the classification level mentioned in the response from the available options."
  }}
}}
""".strip()

FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE = """
Question:
{question}

Reference Answer:
{answer}

Ground-truth attribute specification:
{spec_json}

Produce JSON with key `field_descriptions` mapping attribute names to their instructional descriptions for judge prompts.

Ensure descriptions communicate type expectations, especially emphasizing when boolean values should be used to flag presence of concepts. For boolean concept checks, focus on semantic meaning rather than exact word matching. Make it explicit that the judge is reading the candidate response to this question.

Return only valid JSON.
""".strip()


__all__ = [
    "GROUND_TRUTH_SYSTEM_PROMPT",
    "GROUND_TRUTH_USER_PROMPT_TEMPLATE",
    "FIELD_DESCRIPTION_SYSTEM_PROMPT",
    "FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE",
]
