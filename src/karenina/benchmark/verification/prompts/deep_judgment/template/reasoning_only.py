"""Reasoning-only prompt builders for template deep judgment.

These builders produce the system and user prompts for generating per-attribute
reasoning directly from the model response, without an excerpt extraction stage.
This is the reasoning-only path: the LLM reasons about attribute values by
reading the response text directly rather than working from pre-extracted excerpts.
"""


def build_reasoning_only_system_prompt(
    generic_system_prompt: str,
    attr_guidance: str,
) -> str:
    """Build the system prompt for reasoning-only deep judgment.

    Unlike the excerpt-based reasoning path, this prompt asks the LLM to reason
    directly from the response text. There is no reference to excerpts or any
    prior extraction stage.

    Args:
        generic_system_prompt: Base system instructions (without format_instructions).
        attr_guidance: Formatted attribute descriptions, one per line
            (e.g., "- attr: description").

    Returns:
        Complete system prompt for reasoning-only generation.
    """
    return f"""{generic_system_prompt}

You are an expert reasoning generator for deep-judgment template parsing. Your role is to explain how the response informs each attribute's value.

# Task Overview

You will receive:
1. An original question in <original_question> tags
2. The model's response in <response> tags

Your task: Generate **reasoning** explaining how the response should inform each attribute's value.

# Attribute Definitions

For each attribute below, the description indicates what value it expects:

{attr_guidance}

# Reasoning Protocol

For each attribute:
1. **Review the attribute's description** - understand what value it expects
2. **Read the response directly** - determine what the response reveals about this attribute
3. **Generate reasoning** (2-3 sentences) explaining:
   - How the response relates to the attribute
   - What value the attribute should have based on what the response says
   - Any ambiguities or confidence issues

When the response contains no relevant information for an attribute: Explain why (e.g., "Model refused", "No explicit info") and how this affects the attribute.

# Critical Requirements

**All Attributes**: Generate reasoning for EVERY attribute listed above.

**Response-Based**: Base reasoning on the actual response text provided.

# What NOT to Do

- Do NOT skip any attributes"""


def build_reasoning_only_user_prompt(
    question_text: str,
    raw_llm_response: str,
) -> str:
    """Build the user prompt for reasoning-only deep judgment.

    Args:
        question_text: Original question text.
        raw_llm_response: The raw response from the answering model.

    Returns:
        User prompt with question and response in XML tags.
    """
    return f"""<original_question>
{question_text}
</original_question>

<response>
{raw_llm_response}
</response>"""
