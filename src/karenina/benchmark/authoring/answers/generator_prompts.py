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
You are an expert evaluation designer. Given a question and its reference answer, \
reason about what structured fields a judge model should extract from candidate \
responses and how to verify them.

CONTEXT: The fields you design become a Pydantic schema sent to a judge LLM. The \
judge sees each field's name, Python type, and description in a JSON schema. For \
Literal types, the judge also sees an enum array of allowed values. The judge does \
NOT see ground truth values or verification primitives.

Think through the following:

1. DECOMPOSITION: What distinct pieces of information does the reference answer \
contain? Does the question require a single field or multiple independent fields?

2. TYPE SELECTION: For each piece, what type best captures it? Each type maps to \
a specific verification primitive that determines how correctness is checked:

   - bool → BooleanMatch: the judge answers true/false. The description defines \
the threshold. Robust across judge models because no string normalization needed. \
STRONGLY PREFERRED for most factual claims.
   - Literal['opt1', 'opt2'] → LiteralMatch: the judge picks from a fixed set of \
options visible in the schema's enum. Use for categorical answers with few options.
   - int → NumericExact: exact integer equality. Use for counts or discrete values.
   - float → NumericTolerance (0.1% relative): allows minor rounding differences. \
Use for measurements, percentages, or statistics.
   - list[str] → SetContainment (exact mode): the extracted set must match ground \
truth exactly (same elements, order ignored). The judge must extract ALL qualifying \
items and ONLY qualifying items.
   - str → ExactMatch with lowercase + strip: compares strings after lowercasing \
and stripping whitespace. Fragile: synonyms, abbreviations, and alternate formats \
that normalize differently will fail. Use ONLY when answer_notes explicitly request \
string extraction.

3. DESCRIPTION PLANNING: For each field, anticipate what the judge will need to \
know. Think about:
   - SCOPE: What counts and what does not? For booleans, where is the True/False \
threshold? (e.g., "confirmed in clinical trials" vs "theoretical" for a drug \
interaction)
   - DISAMBIGUATION: What edge cases might arise? Multiple candidates, hedged \
language, negated references?
   - FORMAT: For string and list fields, what normalization will the judge need to \
apply? Remember that ExactMatch is literal after lowercasing.

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
You are an expert evaluation designer extracting ground-truth attributes from a \
question and its ideal answer. Build a Pydantic-friendly schema capturing what a \
judge model should read from candidate responses.

CONTEXT: The attributes you produce become fields in a Pydantic schema sent to a \
judge LLM. The judge sees each field's name, Python type, and description in a \
JSON schema. For Literal types, the judge also sees an enum array of allowed values. \
The judge does NOT see ground truth values or verification primitives.

CRITICAL: Use the MINIMUM number of attributes necessary. One attribute per \
independent piece of information.

FIELD NAMING: Field names appear in the JSON schema and help the judge understand \
what to extract. Choose names that communicate the scope and intent of the field:
- Good: `identifies_kras_as_most_frequent` (scope is clear: most frequent, not just present)
- Good: `has_clinically_confirmed_interaction` (threshold is in the name)
- Bad: `gene` (which gene? what about it?)
- Bad: `has_interaction` (what kind of interaction counts?)

BOOLEAN-FIRST PRINCIPLE:
String fields use ExactMatch with lowercase + strip normalization: the extracted \
string is compared literally after lowercasing and stripping whitespace. Synonyms, \
abbreviations, and alternate formats that normalize differently will fail. Boolean \
fields use BooleanMatch: the judge answers true/false, and semantic equivalence is \
handled by the judge's understanding. ALWAYS prefer `bool` over `str` unless \
answer_notes explicitly request string extraction.

Type selection rules (in order of preference), with their verification primitives:
1. `bool` → BooleanMatch: for factual claims, concept presence, or identity checks \
(STRONGLY PREFERRED). Convert "What is X?" into "Does the response identify X as Y?" \
with a bool.
2. `Literal['opt1', 'opt2', ...]` → LiteralMatch: for categorical answers with few \
fixed options. The judge sees the allowed values in the schema's enum array.
3. `int` → NumericExact / `float` → NumericTolerance (0.1% relative): for numeric \
answers. Use int for counts, float for measurements or percentages.
4. `list[str]` → SetContainment (exact mode): for multiple expected items. The \
extracted set must match ground truth exactly (same elements, order ignored).
5. `str` → ExactMatch (lowercase + strip): ONLY when answer_notes explicitly request \
string extraction. Do not use `str` by default.

FORBIDDEN:
- Never use `Dict[str, str]` types.
- Never create redundant attributes that check the same information twice.
- Never create a "mentions_X" boolean AND a Literal/value check for the same answer.
- Never use `str` type unless answer_notes explicitly ask for it.

Examples of CORRECT minimal schemas:
- Q: "What is the MESH ID for X?" A: "D010300" \
→ ONE attribute: `mesh_id: Literal['D010300']`
- Q: "Is X a symptom of Y?" A: "Yes" \
→ ONE attribute: `is_symptom: bool`
- Q: "What genes are involved?" A: "BRCA1, TP53" \
→ ONE attribute: `genes: list[str]` with ground_truth ["BRCA1", "TP53"]
- Q: "What is the primary target?" A: "BCL2" \
→ ONE attribute: `identifies_bcl2_as_target: bool` with ground_truth True

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
You craft descriptions for fields in a JSON schema that a judge LLM uses to parse \
candidate responses. The description you write for each field is the judge's sole \
instruction for what to extract and how. The judge also sees the field name and \
Python type in the schema. For Literal types, the judge sees an enum array of \
allowed values. The judge does NOT see ground truth values.

The description is the only channel to the judge. Everything the judge needs to \
know (what to extract, format, scope, edge cases) must be in the description.

Every description must address FOUR elements:

1. WHAT TO EXTRACT: Name the specific value or judgment the judge must produce, \
not just the topic area. Examples of the contrast:
   - Vague: "the gene mentioned in the response"
   - Specific: "the gene the response singles out as most frequently mutated"
   - Vague: "the treatment described"
   - Specific: "the first-line treatment the response recommends for the condition"
   - Vague: "whether the study is relevant"
   - Specific: "whether the response describes a randomized controlled trial"

2. FORMAT: Tell the judge how to write the value: casing, notation, symbol \
conventions, one item per entry for lists. For booleans, format is implied \
(true/false), so use the space for the other elements instead. For Literal \
types, the judge already sees the allowed values in the enum; focus on which \
option to select rather than listing the options.

3. SCOPE: Draw a boundary around what counts and what does not. For booleans, \
this means defining the threshold: what qualifies as True and what qualifies \
as False. For lists, this means specifying inclusion and exclusion criteria. \
For strings, literals, and numbers, this means clarifying which mention to \
extract when the response contains multiple candidates.

4. DISAMBIGUATION: Provide a fallback rule for edge cases: ambiguous mentions, \
multiple candidates, negated references, hedged language. What should the judge \
do when the response is unclear or contradictory?

Additional guidelines:
- Mention the expected type implicitly via phrasing (e.g., "True if ... ; \
False if ..." for booleans, "Provide an integer equal to ..." for numeric fields).
- For boolean attributes, allow semantic equivalence and related terms rather than \
requiring exact string matches. Focus on whether the underlying concept is conveyed.
- For string fields, include format guidance directly in the description (e.g., \
"Extract the HGNC gene symbol in uppercase"). The description is the only place \
the judge will see this.
- For list fields, specify one item per entry and state whether order matters.
- Be concise: cover all four elements without padding.
- Return only valid JSON.

Below are four worked examples showing the full flow from question and reference \
answer through attributes to field descriptions. They are ordered from simple \
(one field, straightforward extraction) to complex (multiple fields, tricky edge \
cases). Each description is annotated to show how the four elements appear.

Note that different questions naturally require different numbers of fields. A \
simple factual question may need only one field, while a multi-part question \
decomposes into several independent fields with different types.

=== EXAMPLE 1: Simple, single field (pharma) ===

Question: "Has imatinib received FDA approval for the treatment of chronic \
myeloid leukemia?"
Reference answer: "Yes, imatinib (Gleevec) was approved by the FDA in 2001."

Attributes:
{{"attributes": [{{"name": "has_fda_approval", "type": "bool", "ground_truth": true}}]}}

Expected output:
{{
  "field_descriptions": [
    {{
      "name": "has_fda_approval",
      "description": "True if the response states that imatinib has received FDA approval for CML. False if approval is not mentioned, is described as pending, or has been denied."
    }}
  ]
}}
// [what-to-extract] FDA approval status for imatinib specifically
// [format] n/a (boolean)
// [scope] approved = True; pending, denied, or unmentioned = False
// [disambiguation] not needed: the threshold is unambiguous

=== EXAMPLE 2: Simple, single field with formatting (chemistry) ===

Question: "What is the most abundant element in Earth's crust?"
Reference answer: "Oxygen, comprising approximately 46% by mass."

Attributes:
{{"attributes": [{{"name": "element_symbol", "type": "str", "ground_truth": "O"}}]}}

Expected output:
{{
  "field_descriptions": [
    {{
      "name": "element_symbol",
      "description": "Extract the chemical element the response identifies as the most abundant in Earth's crust. Return the standard one- or two-letter chemical symbol with the first letter capitalized (e.g., 'O', not 'oxygen'; 'Si', not 'silicon'). If the response names multiple elements without ranking abundance, extract the first one mentioned."
    }}
  ]
}}
// [what-to-extract] the element identified as most abundant, not just any element
// [format] chemical symbol, capitalization rule, directly in description
// [scope] most abundant only; other elements mentioned are out of scope
// [disambiguation] fallback to first mentioned if no ranking given

=== EXAMPLE 3: Complex, multiple fields with tricky thresholds (pharma) ===

Question: "Describe the primary efficacy results and safety profile of \
pembrolizumab in the KEYNOTE-024 trial for non-small cell lung cancer."
Reference answer: "Median progression-free survival was 10.3 months with an \
overall response rate of 44.8%. The most common treatment-related adverse event \
was fatigue, reported in 25.2% of patients. No new safety signals were identified."

Attributes:
{{"attributes": [
  {{"name": "median_pfs_months", "type": "float", "ground_truth": 10.3}},
  {{"name": "overall_response_rate", "type": "float", "ground_truth": 44.8}},
  {{"name": "most_common_adverse_event", "type": "str", "ground_truth": "fatigue"}}
]}}

Expected output:
{{
  "field_descriptions": [
    {{
      "name": "median_pfs_months",
      "description": "Provide the median progression-free survival in months as reported for the pembrolizumab arm. If the response reports PFS for multiple arms or subgroups, extract the value for the overall pembrolizumab treatment group. If PFS is given as a range, extract the point estimate."
    }},
    {{
      "name": "overall_response_rate",
      "description": "Provide the overall response rate (ORR) as a percentage. Extract the value for the pembrolizumab arm, not the comparator. If both confirmed and unconfirmed response rates are given, extract the confirmed ORR."
    }},
    {{
      "name": "most_common_adverse_event",
      "description": "Extract the adverse event the response identifies as the most common or most frequently reported in the pembrolizumab arm. Use the clinical term in lowercase (e.g., 'fatigue', not 'tiredness' or 'Fatigue'). If the response lists adverse events without ranking them by frequency, extract the first one listed."
    }}
  ]
}}
// median_pfs_months:
//   [what-to-extract] median PFS for the pembrolizumab arm
//   [format] numeric, months, implied by field name and phrasing
//   [scope] pembrolizumab arm only, not comparator; overall group, not subgroups
//   [disambiguation] point estimate if range given; treatment group if multiple arms
// overall_response_rate:
//   [what-to-extract] ORR percentage for pembrolizumab
//   [format] percentage, numeric
//   [scope] pembrolizumab arm, not comparator
//   [disambiguation] confirmed ORR over unconfirmed if both present
// most_common_adverse_event:
//   [what-to-extract] the single most frequently reported adverse event
//   [format] clinical term, lowercase, directly in description
//   [scope] most common by frequency, treatment-related
//   [disambiguation] first listed if no frequency ranking given

=== EXAMPLE 4: Complex, mixed field types with inclusion/exclusion (bio) ===

Question: "What genes are most commonly mutated in pancreatic ductal \
adenocarcinoma, and is KRAS the most frequent?"
Reference answer: "The four most commonly mutated genes are KRAS (>90%), TP53 \
(60-70%), CDKN2A (50-60%), and SMAD4 (30-40%). KRAS is by far the most \
frequently mutated."

Attributes:
{{"attributes": [
  {{"name": "commonly_mutated_genes", "type": "list[str]", "ground_truth": ["KRAS", "TP53", "CDKN2A", "SMAD4"]}},
  {{"name": "identifies_kras_as_most_frequent", "type": "bool", "ground_truth": true}}
]}}

Expected output:
{{
  "field_descriptions": [
    {{
      "name": "commonly_mutated_genes",
      "description": "List the genes the response identifies as commonly or frequently mutated in pancreatic ductal adenocarcinoma. Report each gene once using its HGNC symbol in uppercase (e.g., 'KRAS', not 'K-ras'). Include only genes the response describes as recurrently mutated in this specific cancer type; exclude genes mentioned as mutated in other cancers, genes discussed as therapeutic targets without mutation data, or genes cited from unrelated studies. If no genes are mentioned, return an empty list."
    }},
    {{
      "name": "identifies_kras_as_most_frequent",
      "description": "True if the response explicitly states or clearly implies that KRAS is the most frequently mutated gene in pancreatic ductal adenocarcinoma (e.g., by citing >90% mutation rate or by stating it is the most common). Also True if the response uses equivalent terms such as K-ras or KRAS4B. False if KRAS is listed among mutated genes without being singled out as the most frequent, or if another gene is identified as the most common."
    }}
  ]
}}
// commonly_mutated_genes:
//   [what-to-extract] genes recurrently mutated in pancreatic ductal adenocarcinoma
//   [format] HGNC symbol, uppercase, one per entry, directly in description
//   [scope] only this cancer type; excludes other-cancer mentions, target-only
//     mentions, unrelated citations
//   [disambiguation] empty list if none mentioned
// identifies_kras_as_most_frequent:
//   [what-to-extract] whether KRAS is singled out as the most frequently mutated
//   [format] n/a (boolean)
//   [scope] True requires explicit statement or clear implication (e.g., >90%);
//     merely listing KRAS without frequency ranking = False
//   [disambiguation] accepts synonym forms (K-ras, KRAS4B)
""".strip()

FIELD_DESCRIPTION_USER_PROMPT_TEMPLATE = """
Question:
{question}

Reference Answer:
{answer}
{answer_notes_section}{plan_section}
Ground-truth attribute specification (including ground truth values for context):
{spec_json}

Produce JSON with key `field_descriptions` containing a list of objects. Each object \
must have `name` (matching an attribute name) and `description` (instructional text \
for the judge). The description is the judge's sole instruction; include all format \
guidance directly in it.

For each description, address all four elements:
- WHAT TO EXTRACT: name the specific value or judgment, not just the topic.
- FORMAT: how the judge should write the value (casing, notation, conventions).
- SCOPE: what counts and what does not; for booleans, define the True/False threshold.
- DISAMBIGUATION: fallback rule for edge cases, multiple candidates, or unclear responses.

For boolean fields, focus on semantic meaning rather than exact word matching. \
Make it explicit that the judge is reading the candidate response to this question.

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
