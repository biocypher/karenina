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

from ....schemas.domain import BaseAnswer
from ....schemas.shared import SearchResultItem
from ....schemas.workflow import ModelConfig, VerificationConfig
from ..tools.fuzzy_match import fuzzy_match_excerpt
from ..tools.search_tools import create_search_tool
from ..utils.parsing import (
    _extract_attribute_descriptions,
    _extract_attribute_names_from_class,
    _format_search_results_for_llm,
    _invoke_llm_with_retry,
    _strip_markdown_fences,
    format_excerpts_for_reasoning,
    format_reasoning_for_parsing,
)

logger = logging.getLogger(__name__)


def deep_judgment_parse(
    raw_llm_response: str,
    RawAnswer: type[BaseAnswer],
    parsing_model: ModelConfig,  # noqa: ARG001 - Kept for interface consistency with standard parsing
    parsing_llm: Any,
    question_text: str,
    config: VerificationConfig,
    format_instructions: str,
    combined_system_prompt: str,
    usage_tracker: Any | None = None,
    parsing_model_str: str | None = None,
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
        usage_tracker: Optional usage tracker to record token usage for each stage
        parsing_model_str: Model string identifier for usage tracking

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
    # PREPARE PROMPTS
    # ==========================================
    # Generate JSON schema with field descriptions from the template class
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=RawAnswer)
    json_schema = parser.get_format_instructions()

    # Extract generic system instructions (without format_instructions for stages 1&2)
    # The combined_system_prompt contains general instructions + format_instructions
    # For stages 1&2, we only need the general instructions
    generic_system_prompt = combined_system_prompt.split("<format_instructions>")[0].strip()

    # Build stage-specific system prompts
    # Extract attribute descriptions for guidance (without the full schema format)
    attribute_descriptions = _extract_attribute_descriptions(json_schema, attribute_names)
    attr_guidance = "\n".join([f"- {attr}: {desc}" for attr, desc in attribute_descriptions.items()])

    excerpt_system_prompt = f"""{generic_system_prompt}

<task_structure>
You will be provided with:
1. An original question in <original_question> tags
2. A response to analyze in <response_to_analyze> tags

Your task is to extract verbatim excerpts from the response that provide evidence for specific attributes.
</task_structure>

<attribute_guidance>
For each attribute below, the description indicates what evidence to look for in the response:

{attr_guidance}
</attribute_guidance>

<instructions>
Extract verbatim excerpts from the response for each attribute listed above.

For each attribute:
1. **Read the attribute's description** to understand what evidence to look for
2. **Identify excerpts**: Find up to {config.deep_judgment_max_excerpts_per_attribute} exact quotes from the response that provide evidence for this attribute
   - "high" confidence: Direct, explicit statement matching the attribute description
   - "medium" confidence: Implied information or indirect evidence
   - "low" confidence: Weak or ambiguous evidence
3. **If no excerpts exist**: Return ONE entry with empty text and explain why (e.g., model refused, no relevant information, implicit answer)

IMPORTANT: Excerpts should be verbatim text spans from the response. Do NOT try to generate or infer content - only extract what is explicitly present.

Return JSON format:
{{
  "attribute_name": [
    {{"text": "exact quote from response", "confidence": "low|medium|high"}}
  ],
  "attribute_without_excerpts": [
    {{"text": "", "confidence": "none", "explanation": "Brief reason why no excerpts (e.g., 'Model refused to answer', 'No explicit statement found', 'Information was implicit')"}}
  ]
}}
</instructions>"""

    # ==========================================
    # STAGE 1: EXCERPT EXTRACTION WITH RETRY
    # ==========================================
    excerpts = {}
    max_retries = config.deep_judgment_excerpt_retry_attempts

    for attempt in range(max_retries + 1):
        # Build excerpt extraction prompt
        excerpt_prompt = f"""<original_question>
{question_text}
</original_question>

<response_to_analyze>
{raw_llm_response}
</response_to_analyze>"""

        # Add error feedback if this is a retry
        if attempt > 0:
            excerpt_prompt += "\n\n<error>Some excerpts from the previous attempt were not found verbatim in the response. Please provide EXACT quotes that appear in the text above, or return empty list [] if no valid excerpts exist.</error>"

        # Invoke parsing model with excerpt-specific system prompt
        messages = [SystemMessage(content=excerpt_system_prompt), HumanMessage(content=excerpt_prompt)]
        raw_response, _, usage_metadata, _ = _invoke_llm_with_retry(parsing_llm, messages, is_agent=False)
        model_calls += 1
        # Track usage for excerpt extraction
        if usage_tracker and usage_metadata and parsing_model_str:
            usage_tracker.track_call("deep_judgment_excerpts", parsing_model_str, usage_metadata)
        cleaned_response = _strip_markdown_fences(raw_response)

        try:
            parsed_excerpts = {} if cleaned_response is None else json.loads(cleaned_response)
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
    # SEARCH ENHANCEMENT (Optional)
    # Validate excerpts against external search results
    # ==========================================
    logger.debug(
        f"Search enhancement config: enabled={config.deep_judgment_search_enabled}, "
        f"tool={config.deep_judgment_search_tool}"
    )
    if config.deep_judgment_search_enabled:
        logger.info("Search enhancement enabled - validating excerpts against external evidence")

        try:
            # Initialize search tool
            search_tool = create_search_tool(config.deep_judgment_search_tool)

            # Collect all excerpts to search (excluding empty/none excerpts)
            excerpts_to_search = []
            for attr, excerpt_list in excerpts.items():
                for excerpt_obj in excerpt_list:
                    excerpt_text = excerpt_obj.get("text", "")
                    confidence = excerpt_obj.get("confidence", "medium")
                    # Only search non-empty excerpts
                    if excerpt_text and confidence != "none":
                        excerpts_to_search.append((attr, excerpt_obj, excerpt_text))

            if excerpts_to_search:
                # Batch search all excerpts
                search_queries = [text for _, _, text in excerpts_to_search]
                logger.info(f"Performing search for {len(search_queries)} excerpts")

                search_results = search_tool(search_queries)

                # Add search results to each excerpt (convert SearchResultItem to dict for JSON storage)
                for (_attr, excerpt_obj, _), search_result in zip(excerpts_to_search, search_results, strict=False):
                    # Convert list[SearchResultItem] to list[dict] for JSON serialization
                    if isinstance(search_result, list):
                        # Check if it's a list of SearchResultItem objects or already dicts
                        if search_result and isinstance(search_result[0], SearchResultItem):
                            excerpt_obj["search_results"] = [
                                {"title": item.title, "content": item.content, "url": item.url}
                                for item in search_result
                            ]
                        elif search_result and isinstance(search_result[0], dict):
                            # Already in dict format
                            excerpt_obj["search_results"] = search_result
                        else:
                            # Empty list or unknown format
                            excerpt_obj["search_results"] = []
                    elif isinstance(search_result, str):
                        # Backward compatibility: string result
                        excerpt_obj["search_results"] = search_result
                    elif isinstance(search_result, SearchResultItem):
                        # Single SearchResultItem
                        excerpt_obj["search_results"] = [
                            {"title": search_result.title, "content": search_result.content, "url": search_result.url}
                        ]
                    else:
                        # Unknown format
                        excerpt_obj["search_results"] = []
                        logger.warning(f"Unexpected search result type: {type(search_result)}")

                logger.info(f"Search completed for {len(search_queries)} excerpts")
            else:
                logger.info("No excerpts to search (all empty or none confidence)")

        except Exception as e:
            # Log warning but don't fail the pipeline
            logger.warning(f"Search enhancement failed: {e}. Continuing without search results.")

    # ==========================================
    # STAGE 1.5: PER-EXCERPT HALLUCINATION ASSESSMENT
    # (Only when search was performed)
    # ==========================================
    search_performed = any(
        "search_results" in excerpt_obj for excerpt_list in excerpts.values() for excerpt_obj in excerpt_list
    )

    if search_performed:
        logger.info("Stage 1.5: Assessing hallucination risk for each excerpt")

        # Assign unique IDs to excerpts for matching
        excerpt_id_counter = 0
        excerpts_with_search = []

        for attr_name, excerpt_list in excerpts.items():
            for excerpt_obj in excerpt_list:
                if "search_results" in excerpt_obj:
                    excerpt_obj["_id"] = str(excerpt_id_counter)
                    excerpt_obj["_attribute"] = attr_name
                    excerpts_with_search.append((attr_name, excerpt_obj))
                    excerpt_id_counter += 1

        if excerpts_with_search:
            # Build batch assessment prompt
            assessment_system_prompt = f"""{generic_system_prompt}

You are assessing hallucination risk for extracted excerpts based on search validation.

For EACH excerpt below, evaluate the hallucination risk by comparing the excerpt against the search results:

- **none** (lowest risk): Search strongly supports the excerpt with multiple corroborating sources
- **low**: Search generally supports the excerpt with minor discrepancies or weak evidence
- **medium**: Search provides mixed evidence, contradictions, or very weak support
- **high** (highest risk): Search contradicts the excerpt or provides no supporting evidence whatsoever

Be conservative - only assign "none" when evidence is very strong."""

            # Format each excerpt for assessment
            excerpt_descriptions = []
            for attr_name, excerpt_obj in excerpts_with_search:
                excerpt_id = excerpt_obj["_id"]
                text = excerpt_obj.get("text", "")
                confidence = excerpt_obj.get("confidence", "unknown")
                similarity = excerpt_obj.get("similarity_score", 0.0)
                search_results_raw = excerpt_obj.get("search_results", [])

                # Format search results for LLM (list[dict] -> string)
                if isinstance(search_results_raw, list):
                    search_results_formatted = _format_search_results_for_llm(search_results_raw)
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

            assessment_prompt = f"""Assess hallucination risk for each of the {len(excerpts_with_search)} excerpts below.

{chr(10).join(excerpt_descriptions)}

Return JSON format with assessments for ALL excerpts:
{{
  "excerpt_assessments": [
    {{
      "excerpt_id": "0",
      "attribute": "attribute_name",
      "hallucination_risk": "none|low|medium|high",
      "justification": "Brief explanation of why this risk level was assigned based on search evidence"
    }},
    ...
  ]
}}"""

            # Invoke LLM for batch assessment
            messages = [SystemMessage(content=assessment_system_prompt), HumanMessage(content=assessment_prompt)]

            try:
                raw_response, _, usage_metadata, _ = _invoke_llm_with_retry(parsing_llm, messages, is_agent=False)
                model_calls += 1
                # Track usage for hallucination assessment
                if usage_tracker and usage_metadata and parsing_model_str:
                    usage_tracker.track_call(
                        "deep_judgment_hallucination_assessment", parsing_model_str, usage_metadata
                    )
                cleaned_response = _strip_markdown_fences(raw_response)
                assessment_data = {} if cleaned_response is None else json.loads(cleaned_response)

                # Match assessments back to excerpts
                for assessment in assessment_data.get("excerpt_assessments", []):
                    excerpt_id = assessment["excerpt_id"]
                    hallucination_risk = assessment.get("hallucination_risk", "high")
                    justification = assessment.get("justification", "")

                    # Find and update the excerpt
                    for excerpt_list in excerpts.values():
                        for excerpt_obj in excerpt_list:
                            if excerpt_obj.get("_id") == excerpt_id:
                                excerpt_obj["hallucination_risk"] = hallucination_risk
                                excerpt_obj["hallucination_justification"] = justification
                                break

                logger.info(
                    f"Stage 1.5 complete: Assessed {len(assessment_data.get('excerpt_assessments', []))} excerpts"
                )
                stages_completed.append("excerpt_hallucination_assessment")

            except (json.JSONDecodeError, Exception) as e:
                # Fail the entire deep-judgment process if hallucination assessment fails
                logger.error(f"Stage 1.5 hallucination assessment failed: {e}")
                raise ValueError(
                    f"Failed to assess per-excerpt hallucination risk: {e}. "
                    "Deep-judgment cannot continue without risk assessment."
                ) from e

            # Clean up temporary IDs
            for excerpt_list in excerpts.values():
                for excerpt_obj in excerpt_list:
                    excerpt_obj.pop("_id", None)
                    excerpt_obj.pop("_attribute", None)

    # ==========================================
    # STAGE 2: REASONING GENERATION
    # (Works even with empty excerpt lists)
    # ==========================================
    # Check if search was performed (any excerpt has search_results field)
    search_performed = any(
        "search_results" in excerpt_obj for excerpt_list in excerpts.values() for excerpt_obj in excerpt_list
    )

    # Build base reasoning prompt
    additional_task = "3. Search snipptes gathered starting from the excerpts." if search_performed else ""
    reasoning_system_prompt = f"""{generic_system_prompt}

<task_structure>
You will be provided with:
1. An original question in <original_question> tags
2. Extracted excerpts in <extracted_excerpts> tags from the previous stage
{additional_task}

Your task is to generate reasoning that explains how the excerpts should inform each attribute's value.
</task_structure>

<attribute_guidance>
For each attribute below, the description indicates what the attribute represents:

{attr_guidance}
</attribute_guidance>"""

    # Conditionally add search context and use nested format when search was performed
    if search_performed:
        reasoning_system_prompt += """

<search_context>
Each excerpt has been validated against external search results AND has been assigned a hallucination risk score.
The excerpt data now includes:
- "Hallucination Risk": Per-excerpt risk assessment (NONE/LOW/MEDIUM/HIGH)
- "Risk Justification": Explanation of why that risk level was assigned
- "Search Results": The actual search validation results

Use these per-excerpt risk assessments to inform your reasoning about each attribute.
</search_context>

<instructions>
Generate reasoning that explains how the excerpts should inform each attribute's value.

IMPORTANT: Only generate reasoning for the attributes listed above.

For each attribute:
1. **Review the attribute's description** to understand what value it expects
2. **Analyze the excerpts** considering their hallucination risk scores
3. **Generate reasoning** (2-3 sentences) that explains:
   - How the excerpts relate to the attribute based on its description
   - What value the attribute should have based on the evidence
   - How the per-excerpt hallucination risks affect confidence in this attribute
   - Any ambiguities or confidence issues

When excerpts are empty: Explain why no excerpts were found and how this affects the attribute.

Return JSON format with ONLY the attributes listed above:
{
  "attribute_name": {
    "reasoning": "reasoning text explaining how excerpts inform the attribute value"
  }
}
</instructions>"""
    else:
        # Use simple string format (backward compatible - no search context)
        reasoning_system_prompt += """

<instructions>
Generate reasoning that explains how the excerpts should inform each attribute's value.

IMPORTANT: Only generate reasoning for the attributes listed above.

For each attribute:
1. **Review the attribute's description** to understand what value it expects
2. **Analyze the excerpts** to determine what they tell us about this attribute
3. **Generate reasoning** (2-3 sentences) that explains:
   - How the excerpts relate to the attribute based on its description
   - What value the attribute should have based on the evidence
   - Any ambiguities or confidence issues

When excerpts are empty: Explain why no excerpts were found and how this affects the attribute (e.g., "The response contains a refusal, so this attribute should be marked as not provided" or "No explicit evidence present, attribute may need inference from context").

Return JSON format with ONLY the attributes listed above:
{
  "attribute_name": "reasoning text explaining how excerpts inform the attribute value"
}
</instructions>"""

    reasoning_prompt = f"""<original_question>
{question_text}
</original_question>

<extracted_excerpts>
{format_excerpts_for_reasoning(excerpts)}
</extracted_excerpts>"""

    messages = [SystemMessage(content=reasoning_system_prompt), HumanMessage(content=reasoning_prompt)]
    raw_response, _, usage_metadata, _ = _invoke_llm_with_retry(parsing_llm, messages, is_agent=False)
    model_calls += 1
    # Track usage for reasoning generation
    if usage_tracker and usage_metadata and parsing_model_str:
        usage_tracker.track_call("deep_judgment_reasoning", parsing_model_str, usage_metadata)
    cleaned_response = _strip_markdown_fences(raw_response)

    try:
        reasoning_raw = {} if cleaned_response is None else json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse reasoning JSON: {e}")
        reasoning_raw = {}  # Gracefully handle failures

    # Extract reasoning and hallucination risk (if present)
    reasoning = {}
    hallucination_risk = {}

    if search_performed:
        # Nested format: {"attr": {"reasoning": "...", "hallucination_risk": "none|low|medium|high"}}
        # NOTE: We now ignore the LLM's hallucination_risk from Stage 2
        # Instead, we calculate it from per-excerpt risks (Stage 1.5)
        for attr, value in reasoning_raw.items():
            if isinstance(value, dict):
                reasoning[attr] = value.get("reasoning", "")
            else:
                # Fallback: LLM returned string instead of nested format
                reasoning[attr] = str(value)
                logger.warning(f"Expected nested reasoning format for '{attr}' but got string. Using fallback.")

        # Calculate attribute-level hallucination risk as MAX of per-excerpt risks
        risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        for attr_name, excerpt_list in excerpts.items():
            excerpt_risks = []
            for excerpt_obj in excerpt_list:
                if "hallucination_risk" in excerpt_obj:
                    excerpt_risks.append(excerpt_obj["hallucination_risk"])

            if excerpt_risks:
                # Find the maximum risk level
                max_risk = max(excerpt_risks, key=lambda r: risk_order.get(r, 3))
                hallucination_risk[attr_name] = max_risk
            else:
                # No per-excerpt risks (shouldn't happen if Stage 1.5 ran)
                hallucination_risk[attr_name] = "high"

    else:
        # Simple format: {"attr": "reasoning text"}
        for attr, value in reasoning_raw.items():
            reasoning[attr] = str(value) if not isinstance(value, dict) else value.get("reasoning", "")

    stages_completed.append("reasoning")
    if search_performed:
        logger.info(
            f"Stage 2 completed: Generated reasoning with hallucination risk assessment for {len(reasoning)} attributes"
        )
    else:
        logger.info(f"Stage 2 completed: Generated reasoning for {len(reasoning)} attributes")

    # ==========================================
    # STAGE 3: PARAMETER EXTRACTION
    # ==========================================
    # Build system prompt with format_instructions for structured output
    parsing_system_prompt = f"""{generic_system_prompt}

<task_structure>
You will be provided with:
1. An original question in <original_question> tags
2. Reasoning traces in <reasoning_traces> tags that explain how excerpts from a response map to attribute values

Your task is to extract the final attribute values based on the reasoning traces, following the JSON schema format specified below.
</task_structure>

{format_instructions}"""

    parsing_prompt = f"""<original_question>
{question_text}
</original_question>

<reasoning_traces>
{format_reasoning_for_parsing(reasoning)}
</reasoning_traces>"""

    messages = [SystemMessage(content=parsing_system_prompt), HumanMessage(content=parsing_prompt)]
    raw_response, _, usage_metadata, _ = _invoke_llm_with_retry(parsing_llm, messages, is_agent=False)
    model_calls += 1
    # Track usage for parameter extraction
    if usage_tracker and usage_metadata and parsing_model_str:
        usage_tracker.track_call("deep_judgment_parameters", parsing_model_str, usage_metadata)
    cleaned_response = _strip_markdown_fences(raw_response)

    # Parse with PydanticOutputParser (standard logic)
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=RawAnswer)
    if cleaned_response is None:
        raise ValueError("Empty response from LLM for parameter extraction")
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

    # Add hallucination risk if search was performed
    if search_performed and hallucination_risk:
        metadata["hallucination_risk"] = hallucination_risk

    return parsed_answer, excerpts, reasoning, metadata
