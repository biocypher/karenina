"""Deep-judgment multi-stage parsing for excerpt extraction and reasoning.

This module implements the deep-judgment feature, which extends standard parsing
with a multi-stage approach:

1. Stage 1: Extract verbatim excerpts that corroborate each attribute
   - Uses the answer template's JSON schema (with field descriptions) to guide excerpt selection
   - Field descriptions specify what evidence to look for in the response
   - Excerpts are validated using fuzzy matching to prevent hallucinations

2. Stage 2: Generate reasoning traces explaining how excerpts inform attribute values
   - Explains how each excerpt maps to the attribute based on its schema description
   - Determines what value the attribute should have based on the evidence

3. Stage 3: Parse final attribute values (standard parsing logic)
   - Uses PydanticOutputParser with the same schema to extract structured values

The feature gracefully handles missing excerpts (refusals, no corroborating evidence)
and provides detailed metadata about the parsing process.
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

    # Generate JSON schema with field descriptions from the template class
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=RawAnswer)
    json_schema = parser.get_format_instructions()

    for attempt in range(max_retries + 1):
        # Build excerpt extraction prompt with explicit field descriptions
        excerpt_prompt = f"""<original_question>
{question_text}
</original_question>

<response_to_analyze>
{raw_llm_response}
</response_to_analyze>

<answer_template_schema>
The response will be parsed into a structured answer using the following JSON schema:

{json_schema}

Each attribute in this schema has a description field that specifies what information should be extracted for that attribute.
</answer_template_schema>

<task>
Extract verbatim excerpts from the response that corroborate each attribute in the answer template schema above.

For each attribute ({", ".join(attribute_names)}):
1. **Read the attribute's description** in the schema to understand what evidence to look for
2. **Identify excerpts**: Find up to {config.deep_judgment_max_excerpts_per_attribute} exact quotes from the response that help determine how to populate this attribute based on its schema description
   - "high" confidence: Direct, explicit statement matching the attribute description
   - "medium" confidence: Implied information or indirect evidence
   - "low" confidence: Weak or ambiguous evidence
3. **If no excerpts exist**: Return ONE entry with empty text and explain why (e.g., model refused, no relevant information, implicit answer)

IMPORTANT: Excerpts should be verbatim text spans that provide evidence for filling the attribute according to its schema description. The excerpts will inform the next stage where we determine the actual attribute values.

Return JSON format:
{{
  "attribute_name": [
    {{"text": "exact quote from response", "confidence": "low|medium|high"}}
  ],
  "attribute_without_excerpts": [
    {{"text": "", "confidence": "none", "explanation": "Brief reason why no excerpts (e.g., 'Model refused to answer', 'No explicit statement found', 'Information was implicit')"}}
  ]
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
                explanation = excerpt_obj.get("explanation", "")

                # Handle missing excerpt with explanation
                if not excerpt_text and confidence == "none" and explanation:
                    # This is a missing excerpt with LLM-generated explanation
                    validated_list.append(
                        {"text": "", "confidence": "none", "similarity_score": 0.0, "explanation": explanation}
                    )
                    attributes_without_excerpts.append(attr)
                    logger.info(f"No excerpts for attribute '{attr}': {explanation}")
                    continue

                # Validate confidence level for normal excerpts
                if confidence not in ("low", "medium", "high", "none"):
                    logger.warning(f"Invalid confidence '{confidence}' for '{attr}', defaulting to 'medium'")
                    confidence = "medium"

                # Skip validation for empty excerpts (shouldn't happen after explanation check above)
                if not excerpt_text:
                    logger.warning(f"Empty excerpt text for '{attr}' without explanation, skipping")
                    continue

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

<answer_template_schema>
{json_schema}
</answer_template_schema>

<extracted_excerpts>
{json.dumps(excerpts, indent=2)}
</extracted_excerpts>

<task>
Generate reasoning that explains how the excerpts should inform each attribute's value.

IMPORTANT: Only generate reasoning for these specific attributes (excluding configuration fields like 'id', 'correct', 'regex'):
{", ".join(attribute_names)}

For each of the above attributes:
1. **Review the attribute's description** in the schema to understand what value it expects
2. **Analyze the excerpts** to determine what they tell us about this attribute
3. **Generate reasoning** (2-3 sentences) that explains:
   - How the excerpts map to the attribute based on its schema description
   - What value the attribute should have based on the evidence
   - Any ambiguities or confidence issues

When excerpts are empty: Explain why no excerpts were found and how this affects the attribute (e.g., "The response contains a refusal, so this attribute should be marked as not provided" or "No explicit evidence present, attribute may need inference from context").

Return JSON format with ONLY the attributes listed above:
{{
  "attribute_name": "reasoning text explaining how excerpts inform the attribute value based on its schema description"
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
