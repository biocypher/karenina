"""Stage 1.5: Hallucination assessment prompt builders for template deep judgment.

These builders produce the system and user prompts for assessing hallucination
risk of extracted excerpts against search results.

Format-specific content (JSON Only requirements, markdown fencing rules, output
format examples) is NOT included here — it is injected by adapter instructions
registered per-interface. This keeps prompt builders format-agnostic.
"""

from __future__ import annotations

from typing import Any


def build_assessment_system_prompt(
    generic_system_prompt: str,
) -> str:
    """Build the system prompt for Stage 1.5 hallucination assessment.

    Args:
        generic_system_prompt: Base system instructions (without format_instructions).

    Returns:
        Complete system prompt for hallucination risk assessment.
    """
    return f"""{generic_system_prompt}

You are an expert hallucination risk assessor. Your role is to evaluate whether extracted excerpts are grounded in external evidence.

# Task Overview

You will receive excerpts extracted from a response, along with search results used to validate each excerpt.

Your task: Assess the **hallucination risk** for each excerpt by comparing it against search evidence.

# Risk Level Definitions

- **none** (lowest risk): Search strongly supports the excerpt with multiple corroborating sources
- **low**: Search generally supports the excerpt with minor discrepancies or weak evidence
- **medium**: Search provides mixed evidence, contradictions, or very weak support
- **high** (highest risk): Search contradicts the excerpt or provides no supporting evidence

# Assessment Protocol

For each excerpt:
1. **Read the excerpt text** - understand what claim is being made
2. **Examine search results** - look for supporting or contradicting evidence
3. **Assign risk level** - based on how well search supports the excerpt
4. **Provide justification** - brief explanation of your assessment

# Critical Requirements

**Conservative Assessment**: Only assign "none" when evidence is very strong with multiple sources.

**Evidence-Based**: Base risk assessment solely on the search results provided.

**All Excerpts**: You MUST assess every excerpt provided, no exceptions.

# What NOT to Do

- Do NOT assign "none" unless search strongly corroborates the excerpt
- Do NOT skip any excerpts in your assessment"""


def build_assessment_user_prompt(
    excerpts_with_search: list[tuple[str, dict[str, Any]]],
    format_search_results_fn: Any,
) -> str:
    """Build the user prompt for Stage 1.5 hallucination assessment.

    This constructs the assessment prompt by iterating over excerpts that have
    search results, formatting each into an XML-tagged description block.

    Args:
        excerpts_with_search: List of (attr_name, excerpt_obj) tuples. Each
            excerpt_obj must have "_id", "text", "confidence", "similarity_score",
            and "search_results" keys.
        format_search_results_fn: Callable that formats search results for LLM
            display. Signature: (list[dict]) -> str. This is
            ``_format_search_results_for_llm`` from template_parsing_helpers.

    Returns:
        Complete user prompt for batch hallucination assessment.
    """
    excerpt_descriptions = []
    for attr_name, excerpt_obj in excerpts_with_search:
        excerpt_id = excerpt_obj["_id"]
        text = excerpt_obj.get("text", "")
        confidence = excerpt_obj.get("confidence", "unknown")
        similarity = excerpt_obj.get("similarity_score", 0.0)
        search_results_raw = excerpt_obj.get("search_results", [])

        # Format search results for LLM (list[dict] -> string)
        if isinstance(search_results_raw, list):
            search_results_formatted = format_search_results_fn(search_results_raw)
        else:
            # Backward compatibility: if stored as string, use as-is
            search_results_formatted = search_results_raw

        excerpt_descriptions.append(
            f'<excerpt id="{excerpt_id}" attribute="{attr_name}">\n'
            f"Text: {text}\n"
            f"Extraction Confidence: {confidence}\n"
            f"Similarity: {similarity:.3f}\n"
            f"Search Results:\n{search_results_formatted}\n"
            f"</excerpt>"
        )

    num_excerpts = len(excerpts_with_search)
    return f"""Assess hallucination risk for each of the {num_excerpts} excerpts below.

{chr(10).join(excerpt_descriptions)}

Provide an assessment for ALL {num_excerpts} excerpts. For each excerpt, include
the excerpt_id, attribute name, hallucination_risk level (none/low/medium/high),
and a brief justification based on search evidence."""
