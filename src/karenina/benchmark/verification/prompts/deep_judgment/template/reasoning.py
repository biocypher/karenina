"""Stage 2: Reasoning generation prompt builders for template deep judgment.

These builders produce the system and user prompts for generating reasoning
traces that explain how extracted excerpts inform attribute values.

Note: Reasoning tasks produce free-form text (not structured output), so
format-specific content was already minimal. Remaining JSON format references
(Output Format sections) are stripped since adapter instructions handle them.
"""


def build_reasoning_system_prompt(
    generic_system_prompt: str,
    attr_guidance: str,
    search_performed: bool,
) -> str:
    """Build the system prompt for Stage 2 reasoning generation.

    The prompt varies based on whether search was performed (Stage 1.5).
    When search was performed, the prompt includes hallucination risk context
    and uses a nested JSON output format. Without search, it uses a simpler
    flat string format.

    Args:
        generic_system_prompt: Base system instructions (without format_instructions).
        attr_guidance: Formatted attribute descriptions, one per line
            (e.g., "- attr: description").
        search_performed: Whether search enhancement and hallucination
            assessment were performed (enables risk-aware reasoning).

    Returns:
        Complete system prompt for reasoning generation.
    """
    base = f"""{generic_system_prompt}

You are an expert reasoning generator for deep-judgment template parsing. Your role is to explain how excerpts inform attribute values.

# Task Overview

You will receive:
1. An original question in <original_question> tags
2. Extracted excerpts in <extracted_excerpts> tags from the previous stage
{"3. Search results and hallucination risk assessments for each excerpt." if search_performed else ""}

Your task: Generate **reasoning** explaining how the excerpts should inform each attribute's value.

# Attribute Definitions

For each attribute below, the description indicates what value it expects:

{attr_guidance}"""

    if search_performed:
        base += """

# Search Context

Each excerpt has been validated against external search and assigned a hallucination risk score:
- "Hallucination Risk": Per-excerpt risk (NONE/LOW/MEDIUM/HIGH)
- "Risk Justification": Explanation for the risk level
- "Search Results": External validation evidence

Use these risk assessments to inform your reasoning confidence.

# Reasoning Protocol

For each attribute:
1. **Review the attribute's description** - understand what value it expects
2. **Analyze the excerpts** - consider their hallucination risk scores
3. **Generate reasoning** (2-3 sentences) explaining:
   - How the excerpts relate to the attribute
   - What value the attribute should have based on evidence
   - How hallucination risks affect confidence
   - Any ambiguities or issues

When excerpts are empty: Explain why and how this affects the attribute.

# Critical Requirements

**All Attributes**: Generate reasoning for EVERY attribute listed above.

**Evidence-Based**: Base reasoning on the actual excerpts provided.

**Risk-Aware**: Factor hallucination risks into your confidence assessment.

# What NOT to Do

- Do NOT skip any attributes"""
    else:
        base += """

# Reasoning Protocol

For each attribute:
1. **Review the attribute's description** - understand what value it expects
2. **Analyze the excerpts** - determine what they reveal about this attribute
3. **Generate reasoning** (2-3 sentences) explaining:
   - How the excerpts relate to the attribute
   - What value the attribute should have based on evidence
   - Any ambiguities or confidence issues

When excerpts are empty: Explain why (e.g., "Model refused", "No explicit info") and how this affects the attribute.

# Critical Requirements

**All Attributes**: Generate reasoning for EVERY attribute listed above.

**Evidence-Based**: Base reasoning on the actual excerpts provided.

# What NOT to Do

- Do NOT skip any attributes"""

    return base


def build_reasoning_user_prompt(
    question_text: str,
    formatted_excerpts: str,
) -> str:
    """Build the user prompt for Stage 2 reasoning generation.

    Args:
        question_text: Original question text.
        formatted_excerpts: Pre-formatted excerpts string from
            ``format_excerpts_for_reasoning()``.

    Returns:
        User prompt with question and excerpts in XML tags.
    """
    return f"""<original_question>
{question_text}
</original_question>

<extracted_excerpts>
{formatted_excerpts}
</extracted_excerpts>"""
