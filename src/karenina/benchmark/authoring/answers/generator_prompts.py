"""
Prompt constants for answer template generation.

This module contains the system prompts and user prompt templates used
by the structured answer generator. Chain building has been moved to
generator.py using the port/adapter pattern.
"""

# ============================================================================
# PLANNING PROMPTS (Phase 0)
# ============================================================================

PLANNING_SYSTEM_PROMPT = """
You are an expert evaluation designer. Given a question and its reference answer, reason about what structured fields a judge model should extract from candidate responses and how to verify them.

Think through:
1. What distinct pieces of information the reference answer contains.
2. For each piece, what type best captures it. STRONGLY prefer `bool` fields:
   - `bool` for presence/absence checks (PREFERRED for most factual claims)
   - `Literal['option1', 'option2']` for categorical values with few options
   - `int` or `float` for numeric values
   - `list[str]` with SetContainment for multiple expected items
   - `date` for date values
   - `str` with ExactMatch ONLY when answer_notes explicitly request string extraction
3. Whether the answer requires a single field or multiple independent fields.

IMPORTANT: String fields are fragile because they require exact normalization to match.
Boolean fields ("does the response identify X?") are more robust because the judge
answers true/false rather than extracting and normalizing text. Default to boolean
encoding unless answer_notes explicitly request otherwise.

Produce your reasoning as free text. Do not produce JSON or code.
""".strip()

PLANNING_USER_PROMPT_TEMPLATE = """
Question:
{question}

Reference Answer:
{answer}
{answer_notes_section}
Reason about the field design: what attributes should a judge extract, what types suit each, and whether any normalization or formatting concerns apply. Write your analysis as free text.
""".strip()


# ============================================================================
# GROUND TRUTH EXTRACTION PROMPTS
# ============================================================================

GROUND_TRUTH_SYSTEM_PROMPT = """
You are an expert evaluation designer extracting ground-truth attributes from a question and its ideal answer. Build a Pydantic-friendly schema capturing what a judge model should read from candidate responses.

CRITICAL: Use the MINIMUM number of attributes necessary. One attribute per independent piece of information.

BOOLEAN-FIRST PRINCIPLE:
String fields are fragile: they require exact normalization to match, and minor formatting
differences (capitalization, hyphens, abbreviations) cause false negatives. Boolean fields
are robust: the judge answers true/false about whether a concept is present, which is
reliable across judge models. ALWAYS prefer `bool` over `str` unless answer_notes
explicitly request string extraction.

Type selection rules (in order of preference):
1. `bool` for factual claims, concept presence, or identity checks (STRONGLY PREFERRED).
   Convert "What is X?" into "Does the response identify X as Y?" with a bool.
2. `Literal['option1', 'option2', ...]` for categorical answers with few fixed options.
3. `int` or `float` for numeric answers.
4. `list[str]` with SetContainment for multiple expected items (e.g., gene lists).
5. `date` for date values.
6. `str` ONLY when answer_notes explicitly request string extraction. Do not use `str`
   by default.

FORBIDDEN:
- Never use `Dict[str, str]` types.
- Never create redundant attributes that check the same information twice.
- Never create a "mentions_X" boolean AND a Literal/value check for the same answer.
- Never use `str` type unless answer_notes explicitly ask for it.

Examples of CORRECT minimal schemas:
- Q: "What is the MESH ID for X?" A: "D010300" → ONE attribute: `mesh_id: Literal['D010300']`
- Q: "Is X a symptom of Y?" A: "Yes" → ONE attribute: `is_symptom: bool`
- Q: "What genes are involved?" A: "BRCA1, TP53" → ONE attribute: `genes: list[str]` with ground_truth ["BRCA1", "TP53"]
- Q: "What is the primary target?" A: "BCL2" → ONE attribute: `identifies_bcl2_as_target: bool` with ground_truth True

Example JSON output:
{{
  "attributes": [
    {{
      "name": "identifies_correct_mesh_id",
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
{answer_notes_section}{plan_section}
Identify the minimal set of structured attributes that a judge must extract from a candidate response to verify correctness. Construct a JSON object with a single key `attributes` containing a list of attribute definitions. Each definition must include `name`, `type`, and `ground_truth` fields, where `ground_truth` contains the expected correct value derived from the reference answer.

When selecting types, follow these guidelines:
- FORBIDDEN: Never use `Dict[str, str]` types.
- FORBIDDEN: Never use `str` type unless answer_notes explicitly request string extraction.
- Use `bool` as the DEFAULT type for factual claims and concept identification. Convert factual answers into boolean checks (e.g., "BCL2" becomes "identifies_bcl2_as_target: bool = True").
- Use `Literal` types when the reference answer suggests a categorical classification with few fixed options.
- Use `list[str]` for multiple expected items that form a set.
- When the reference answer contains compound terms or phrases, treat them as single semantic units and create one boolean attribute for the complete concept.
- If answer_notes are provided and explicitly request string fields or specific extraction behavior, follow those instructions and override the boolean-first default.

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
- For string fields that may have formatting variations (capitalization, abbreviations, whitespace), provide an extraction_hint describing normalization that should be applied.
- Stay concise (<= 2 sentences per attribute for description, <= 1 sentence for extraction_hint).
- Return only valid JSON.

Example output:
{{
  "field_descriptions": [
    {{
      "name": "count_of_items",
      "description": "Provide an integer equal to the number of items mentioned in the response.",
      "extraction_hint": null
    }},
    {{
      "name": "drug_target",
      "description": "Extract the primary protein target mentioned in the response.",
      "extraction_hint": "Normalize to uppercase gene symbol format (e.g., BCL2 not Bcl-2)"
    }},
    {{
      "name": "mentions_first_concept",
      "description": "Answer with true if the response refers to the first concept or semantically related terms; otherwise answer false.",
      "extraction_hint": null
    }}
  ]
}}
""".strip()

FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE = """
Question:
{question}

Reference Answer:
{answer}
{answer_notes_section}
Ground-truth attribute specification:
{spec_json}

Produce JSON with key `field_descriptions` containing a list of objects. Each object must have `name` (matching an attribute name), `description` (instructional text for the judge), and `extraction_hint` (optional normalization guidance, or null if not needed).

Ensure descriptions communicate type expectations, especially emphasizing when boolean values should be used to flag presence of concepts. For boolean concept checks, focus on semantic meaning rather than exact word matching. Make it explicit that the judge is reading the candidate response to this question.

Return only valid JSON.
""".strip()


__all__ = [
    "GROUND_TRUTH_SYSTEM_PROMPT",
    "GROUND_TRUTH_USER_PROMPT_TEMPLATE",
    "FIELD_DESCRIPTION_SYSTEM_PROMPT",
    "FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE",
    "PLANNING_SYSTEM_PROMPT",
    "PLANNING_USER_PROMPT_TEMPLATE",
]
