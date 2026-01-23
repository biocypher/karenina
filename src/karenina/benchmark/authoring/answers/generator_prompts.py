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
You are an expert evaluation designer extracting ground-truth attributes from a question and its ideal answer. Build a Pydantic-friendly schema capturing what a judge model should read from candidate responses.

CRITICAL: Use the MINIMUM number of attributes necessary. One attribute per independent piece of information.

Type selection rules (in order of preference):
1. For YES/NO questions: Use a single `bool` attribute.
2. For single-value factual answers (IDs, names, numbers): Use a single attribute with the appropriate type:
   - `Literal['value']` for exact text matches (IDs, codes, specific terms)
   - `int` or `float` for numeric answers
   - `bool` for presence/absence checks
3. For categorical answers with few options: Use `Literal['option1', 'option2', ...]`
4. For multiple independent items: Create ONE `bool` attribute per item.

FORBIDDEN:
- Never use `str`, `List[str]`, or `Dict[str, str]` types.
- Never create redundant attributes that check the same information twice.
- Never create a "mentions_X" boolean AND a Literal/value check for the same answer.

Examples of CORRECT minimal schemas:
- Q: "What is the MESH ID for X?" A: "D010300" → ONE attribute: `mesh_id: Literal['D010300']`
- Q: "Is X a symptom of Y?" A: "Yes" → ONE attribute: `is_symptom: bool`
- Q: "What genes are involved?" A: "BRCA1, TP53" → TWO attributes: `mentions_brca1: bool`, `mentions_tp53: bool`

Example JSON output:
{{
  "attributes": [
    {{
      "name": "mesh_id",
      "type": "Literal['D010300']",
      "ground_truth": "D010300"
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
