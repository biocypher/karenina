"""Deep-judgment multi-stage parsing for excerpt extraction and reasoning.

This module implements the deep-judgment feature, which extends standard parsing
with a multi-stage approach:
1. Stage 1: Extract verbatim excerpts that corroborate each attribute
2. Stage 2: Generate reasoning traces explaining how excerpts support values
3. Stage 3: Parse final attribute values (standard parsing logic)

The feature gracefully handles missing excerpts (refusals, no corroborating evidence)
and validates excerpts using fuzzy matching to prevent hallucinations.
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...schemas.answer_class import BaseAnswer
from ..models import ModelConfig, VerificationConfig
from .fuzzy_match import fuzzy_match_excerpt
from .parser_utils import _extract_attribute_names_from_class, _invoke_llm_with_retry, _strip_markdown_fences

logger = logging.getLogger(__name__)


def deep_judgment_parse(
    raw_llm_response: str,
    RawAnswer: type[BaseAnswer],
    parsing_model: ModelConfig,  # noqa: ARG001 - Kept for interface consistency with standard parsing
    parsing_llm: Any,
    question_text: str,
    config: VerificationConfig,
    format_instructions: str,  # noqa: ARG001 - Kept for interface consistency with standard parsing
    combined_system_prompt: str,
) -> tuple[BaseAnswer, dict[str, list[dict[str, Any]]], dict[str, str], dict[str, Any]]:
    """Execute multi-stage deep-judgment parsing: excerpts → reasoning → parameters.

    This is a drop-in replacement for the standard parsing step. It performs three
    sequential stages using the parsing model to extract excerpts, generate reasoning,
    and finally parse attribute values.

    Args:
        raw_llm_response: Raw trace from answering model
        RawAnswer: Answer template class (BaseAnswer subclass)
        parsing_model: Parsing model configuration
        parsing_llm: Initialized parsing LLM instance
        question_text: Original question text for context
        config: Verification configuration with deep-judgment settings
        format_instructions: Parser format instructions from PydanticOutputParser
        combined_system_prompt: System prompt with format instructions

    Returns:
        Tuple of (parsed_answer, excerpts, reasoning, metadata):
        - parsed_answer: BaseAnswer instance (same as standard parsing)
        - excerpts: Dict mapping attribute names to lists of excerpt objects
          Structure: {"attr": [{"text": str, "confidence": "low|medium|high", "similarity_score": float}]}
          Empty list [] indicates no excerpts found (valid for refusals)
        - reasoning: Dict mapping attribute names to reasoning text
          Structure: {"attr": "reasoning explaining excerpt→value mapping"}
        - metadata: Dict with execution info
          Structure: {"stages_completed": [...], "model_calls": int, ...}

    Raises:
        ValueError: If excerpt JSON parsing fails after all retries
        Exception: If any stage fails (propagated from _invoke_llm_with_retry)

    Example:
        >>> parsed, excerpts, reasoning, meta = deep_judgment_parse(...)
        >>> excerpts["drug_target"]
        [{"text": "targets BCL-2", "confidence": "high", "similarity_score": 0.95}]
        >>> reasoning["drug_target"]
        "The excerpt clearly states BCL-2 as the target..."
    """
    # Extract attribute names from template (excludes 'id', 'correct', 'regex')
    attribute_names = _extract_attribute_names_from_class(RawAnswer)

    # Initialize tracking variables
    stages_completed = []
    model_calls = 0
    excerpt_retry_count = 0
    attributes_without_excerpts = []

    logger.info(f"Starting deep-judgment parsing for {len(attribute_names)} attributes: {', '.join(attribute_names)}")

    # ==========================================
    # STAGE 1: EXCERPT EXTRACTION WITH RETRY
    # ==========================================
    excerpts = {}
    max_retries = config.deep_judgment_excerpt_retry_attempts

    for attempt in range(max_retries + 1):
        # Build excerpt extraction prompt (EXPLICIT MISSING EXCERPTS HANDLING)
        excerpt_prompt = f"""<original_question>
{question_text}
</original_question>

<response_to_analyze>
{raw_llm_response}
</response_to_analyze>

<task>
Extract verbatim excerpts from the response that support each of these attributes: {", ".join(attribute_names)}

For each attribute, identify up to {config.deep_judgment_max_excerpts_per_attribute} exact quotes.
Provide a confidence level (low/medium/high) for each excerpt based on how strongly it supports the attribute.

IMPORTANT: If no corroborating excerpts exist for an attribute (e.g., refusals, no relevant information), return an empty list for that attribute.

Return JSON format:
{{
  "attribute_name": [
    {{"text": "exact quote from response", "confidence": "low|medium|high"}}
  ],
  "another_attribute": []  // Empty list when no excerpts found
}}
</task>"""

        # Add error feedback if this is a retry
        if attempt > 0:
            excerpt_prompt += "\n\n<error>Some excerpts from the previous attempt were not found verbatim in the response. Please provide EXACT quotes that appear in the text above, or return empty list [] if no valid excerpts exist.</error>"

        # Invoke parsing model with generic retry logic
        messages = [SystemMessage(content=combined_system_prompt), HumanMessage(content=excerpt_prompt)]
        raw_response = _invoke_llm_with_retry(parsing_llm, messages)
        model_calls += 1
        cleaned_response = _strip_markdown_fences(raw_response)

        try:
            parsed_excerpts = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse excerpt JSON (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                excerpt_retry_count += 1
                continue
            else:
                raise ValueError(f"Failed to parse excerpt JSON after {max_retries + 1} attempts: {e}") from e

        # Validate excerpts with fuzzy matching
        validation_failed = False
        validated_excerpts: dict[str, list[dict[str, Any]]] = {}

        for attr in attribute_names:
            excerpt_list = parsed_excerpts.get(attr, [])

            # MISSING EXCERPTS: Empty list is valid (no corroborating evidence)
            if not excerpt_list:
                validated_excerpts[attr] = []
                attributes_without_excerpts.append(attr)
                logger.info(f"No excerpts found for attribute '{attr}' (valid scenario: refusal/no evidence)")
                continue

            validated_list = []

            for excerpt_obj in excerpt_list:
                excerpt_text = excerpt_obj.get("text", "")
                confidence = excerpt_obj.get("confidence", "medium")

                # Validate confidence level
                if confidence not in ("low", "medium", "high"):
                    logger.warning(f"Invalid confidence '{confidence}' for '{attr}', defaulting to 'medium'")
                    confidence = "medium"

                # Fuzzy match against raw trace
                match_found, similarity = fuzzy_match_excerpt(excerpt_text, raw_llm_response)

                if not match_found and similarity < config.deep_judgment_fuzzy_match_threshold:
                    if attempt < max_retries:
                        # Retry with error feedback
                        validation_failed = True
                        excerpt_retry_count += 1
                        logger.warning(
                            f"Excerpt validation failed for '{attr}' (similarity: {similarity:.2f} < "
                            f"threshold: {config.deep_judgment_fuzzy_match_threshold})"
                        )
                        break
                    else:
                        # Max retries reached, skip this excerpt
                        logger.error(
                            f"Skipping invalid excerpt for '{attr}' after {max_retries + 1} attempts "
                            f"(similarity: {similarity:.2f})"
                        )
                        continue

                # Add validated excerpt
                validated_list.append({"text": excerpt_text, "confidence": confidence, "similarity_score": similarity})

            if validation_failed:
                break

            validated_excerpts[attr] = validated_list

            # Track attributes with no excerpts after validation
            if not validated_list and attr not in attributes_without_excerpts:
                attributes_without_excerpts.append(attr)

        if not validation_failed:
            excerpts = validated_excerpts
            break

    stages_completed.append("excerpts")
    total_excerpts = sum(len(v) for v in excerpts.values())
    logger.info(
        f"Stage 1 completed: Extracted {total_excerpts} excerpts across {len(excerpts)} attributes "
        f"({len(attributes_without_excerpts)} attributes without excerpts)"
    )

    # ==========================================
    # STAGE 2: REASONING GENERATION
    # (Works even with empty excerpt lists)
    # ==========================================
    reasoning_prompt = f"""<original_question>
{question_text}
</original_question>

<extracted_excerpts>
{json.dumps(excerpts, indent=2)}
</extracted_excerpts>

<task>
For each attribute, explain the reasoning for the attribute value.

When excerpts exist: Explain how the excerpts support the attribute value (2-3 sentences).
When excerpts are empty []: Explain why no excerpts were found (e.g., "The response contains a refusal" or "No corroborating evidence present").

Return JSON format:
{{
  "attribute_name": "reasoning text explaining excerpt → value mapping OR why no excerpts found"
}}
</task>"""

    messages = [SystemMessage(content=combined_system_prompt), HumanMessage(content=reasoning_prompt)]
    raw_response = _invoke_llm_with_retry(parsing_llm, messages)
    model_calls += 1
    cleaned_response = _strip_markdown_fences(raw_response)

    try:
        reasoning = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse reasoning JSON: {e}")
        reasoning = {}  # Gracefully handle failures

    stages_completed.append("reasoning")
    logger.info(f"Stage 2 completed: Generated reasoning for {len(reasoning)} attributes")

    # ==========================================
    # STAGE 3: PARAMETER EXTRACTION
    # ==========================================
    # Use standard parsing logic (same as existing single-stage parsing)
    parsing_prompt = f"""<original_question>
Your task is to parse an answer given to the question reported in this section. Use the question to contextualize the info from the schema fields below:

Original Question: {question_text}
</original_question>

<response_to_parse>
{raw_llm_response}
</response_to_parse>"""

    messages = [SystemMessage(content=combined_system_prompt), HumanMessage(content=parsing_prompt)]
    raw_response = _invoke_llm_with_retry(parsing_llm, messages)
    model_calls += 1
    cleaned_response = _strip_markdown_fences(raw_response)

    # Parse with PydanticOutputParser (standard logic)
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=RawAnswer)
    parsed_answer = parser.parse(cleaned_response)

    stages_completed.append("parameters")
    logger.info("Stage 3 completed: Parameter extraction finished")
    logger.info(f"Deep-judgment parsing completed successfully (total {model_calls} model calls)")

    # ==========================================
    # RETURN RESULTS
    # ==========================================
    metadata = {
        "stages_completed": stages_completed,
        "model_calls": model_calls,
        "excerpt_retry_count": excerpt_retry_count,
        "attributes_without_excerpts": attributes_without_excerpts,
    }

    return parsed_answer, excerpts, reasoning, metadata
