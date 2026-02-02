"""Stage 1: Excerpt extraction prompt builders for template deep judgment.

These builders produce the system and user prompts for extracting verbatim
excerpts from LLM responses that corroborate each template attribute.

Format-specific content (JSON Only requirements, markdown fencing rules, output
format examples) is NOT included here — it is injected by adapter instructions
registered per-interface. This keeps prompt builders format-agnostic.
"""


def build_excerpt_system_prompt(
    generic_system_prompt: str,
    attr_guidance: str,
    max_excerpts_per_attribute: int,
) -> str:
    """Build the system prompt for Stage 1 excerpt extraction.

    Args:
        generic_system_prompt: Base system instructions (without format_instructions).
        attr_guidance: Formatted attribute descriptions, one per line
            (e.g., "- attr: description").
        max_excerpts_per_attribute: Maximum excerpts to extract per attribute.

    Returns:
        Complete system prompt for excerpt extraction.
    """
    return f"""{generic_system_prompt}

You are an expert excerpt extractor for deep-judgment template parsing. Your role is to find verbatim evidence in responses.

# Task Overview

You will receive:
1. An original question in <original_question> tags
2. A response to analyze in <response_to_analyze> tags

Your task: Extract **verbatim excerpts** from the response that provide evidence for specific attributes.

# Attribute Definitions

For each attribute below, the description indicates what evidence to look for:

{attr_guidance}

# Extraction Protocol

For each attribute:
1. **Read the attribute's description** to understand what evidence to look for
2. **Identify excerpts**: Find up to {max_excerpts_per_attribute} exact quotes that provide evidence
   - "high" confidence: Direct, explicit statement matching the attribute description
   - "medium" confidence: Implied information or indirect evidence
   - "low" confidence: Weak or ambiguous evidence
3. **If no excerpts exist**: Return ONE entry with empty text and explanation

# Critical Requirements

**Verbatim Only**: Excerpts MUST be exact text spans from the response - copy-paste, no modifications.

**Evidence-Based**: Only extract text that actually provides evidence for the attribute.

**Completeness**: Cover all attributes listed above, even if no excerpts are found.

# What NOT to Do

- Do NOT paraphrase or reword the response text
- Do NOT invent or hallucinate excerpts that aren't in the response
- Do NOT include text from other parts of this prompt"""


def build_excerpt_user_prompt(
    question_text: str,
    raw_llm_response: str,
) -> str:
    """Build the user prompt for Stage 1 excerpt extraction.

    Args:
        question_text: Original question text.
        raw_llm_response: Raw trace from the answering model.

    Returns:
        User prompt with question and response in XML tags.
    """
    return f"""<original_question>
{question_text}
</original_question>

<response_to_analyze>
{raw_llm_response}
</response_to_analyze>"""
