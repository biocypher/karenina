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

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from karenina.utils.json_extraction import strip_markdown_fences as _strip_markdown_fences

from .....ports import LLMPort, ParserPort
from .....ports.capabilities import PortCapabilities
from .....schemas.config import ModelConfig
from .....schemas.entities import BaseAnswer
from .....schemas.shared import SearchResultItem
from .....schemas.verification import VerificationConfig
from ...prompts.assembler import PromptAssembler
from ...prompts.task_types import PromptTask

if TYPE_CHECKING:
    from .....schemas.verification.prompt_config import PromptConfig
from ...prompts.deep_judgment.template import (
    build_assessment_system_prompt,
    build_assessment_user_prompt,
    build_excerpt_system_prompt,
    build_excerpt_user_prompt,
    build_reasoning_system_prompt,
    build_reasoning_user_prompt,
)
from ...utils.search_provider import create_search_tool
from ...utils.template_parsing_helpers import (
    _extract_attribute_descriptions,
    _extract_attribute_names_from_class,
    _format_search_results_for_llm,
    format_excerpts_for_reasoning,
    format_reasoning_for_parsing,
)
from ...utils.trace_fuzzy_match import fuzzy_match_excerpt

logger = logging.getLogger(__name__)


def deep_judgment_parse(
    raw_llm_response: str,
    RawAnswer: type[BaseAnswer],
    parsing_model: ModelConfig,  # noqa: ARG001 - Kept for interface consistency with standard parsing
    parsing_llm: LLMPort,
    parser: ParserPort,
    question_text: str,
    config: VerificationConfig,
    format_instructions: str,  # noqa: ARG001 - No longer needed, ParserPort builds its own
    combined_system_prompt: str,  # noqa: ARG001 - No longer needed, ParserPort builds its own
    usage_tracker: Any | None = None,
    parsing_model_str: str | None = None,
    prompt_config: PromptConfig | None = None,
) -> tuple[BaseAnswer, dict[str, list[dict[str, Any]]], dict[str, str], dict[str, Any]]:
    """Execute multi-stage deep-judgment parsing: excerpts → reasoning → parameters.

    This is a drop-in replacement for the standard parsing step. It performs three
    sequential stages using the parsing model to extract excerpts, generate reasoning,
    and finally parse attribute values via ParserPort.

    Args:
        raw_llm_response: Raw trace from answering model
        RawAnswer: Answer template class (BaseAnswer subclass)
        parsing_model: Parsing model configuration (kept for interface consistency)
        parsing_llm: LLMPort adapter for Stages 1, 1.5, 2 (excerpt/reasoning extraction)
        parser: ParserPort adapter for Stage 3 (parameter extraction with retry)
        question_text: Original question text for context
        config: Verification configuration with deep-judgment settings
        format_instructions: Kept for interface consistency (ParserPort builds its own)
        combined_system_prompt: Kept for interface consistency (ParserPort builds its own)
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
        ParseError: If Stage 3 parsing fails after all adapter retries

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

    temp_parser = PydanticOutputParser(pydantic_object=RawAnswer)
    json_schema = temp_parser.get_format_instructions()

    # The combined_system_prompt is now format-agnostic (no format_instructions section).
    # Use it directly as the generic system prompt for stages 1&2.
    generic_system_prompt = combined_system_prompt

    # Build stage-specific system prompts
    # Extract attribute descriptions for guidance (without the full schema format)
    attribute_descriptions = _extract_attribute_descriptions(json_schema, attribute_names)
    attr_guidance = "\n".join([f"- {attr}: {desc}" for attr, desc in attribute_descriptions.items()])

    excerpt_system_prompt = build_excerpt_system_prompt(
        generic_system_prompt=generic_system_prompt,
        attr_guidance=attr_guidance,
        max_excerpts_per_attribute=config.deep_judgment_max_excerpts_per_attribute,
    )

    # ==========================================
    # STAGE 1: EXCERPT EXTRACTION WITH RETRY
    # ==========================================
    excerpts = {}
    max_retries = config.deep_judgment_excerpt_retry_attempts

    for attempt in range(max_retries + 1):
        # Build excerpt extraction prompt
        excerpt_prompt = build_excerpt_user_prompt(
            question_text=question_text,
            raw_llm_response=raw_llm_response,
        )

        # Add error feedback if this is a retry
        if attempt > 0:
            excerpt_prompt += "\n\n<error>Some excerpts from the previous attempt were not found verbatim in the response. Please provide EXACT quotes that appear in the text above, or return empty list [] if no valid excerpts exist.</error>"

        # Invoke parsing model with excerpt-specific system prompt
        excerpt_assembler = PromptAssembler(
            task=PromptTask.DJ_TEMPLATE_EXCERPT_EXTRACTION,
            interface=parsing_model.interface,
            capabilities=PortCapabilities(),
        )
        excerpt_messages = excerpt_assembler.assemble(
            system_text=excerpt_system_prompt,
            user_text=excerpt_prompt,
            user_instructions=prompt_config.get_for_task(PromptTask.DJ_TEMPLATE_EXCERPT_EXTRACTION.value)
            if prompt_config
            else None,
            instruction_context={
                "json_schema": None,  # Template DJ uses inline format in system prompt
                "parsing_notes": (
                    "- We validate excerpts using fuzzy matching against the original response\n"
                    "- Excerpts that don't match will be rejected and may trigger a retry"
                ),
            },
        )
        llm_response = parsing_llm.invoke(excerpt_messages)
        raw_response, usage_metadata = llm_response.content, llm_response.usage.to_dict()
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
            assessment_system_prompt = build_assessment_system_prompt(
                generic_system_prompt=generic_system_prompt,
            )

            assessment_prompt = build_assessment_user_prompt(
                excerpts_with_search=excerpts_with_search,
                format_search_results_fn=_format_search_results_for_llm,
            )

            # Invoke LLM for batch assessment
            assessment_assembler = PromptAssembler(
                task=PromptTask.DJ_TEMPLATE_HALLUCINATION,
                interface=parsing_model.interface,
                capabilities=PortCapabilities(),
            )
            assessment_messages = assessment_assembler.assemble(
                system_text=assessment_system_prompt,
                user_text=assessment_prompt,
                user_instructions=prompt_config.get_for_task(PromptTask.DJ_TEMPLATE_HALLUCINATION.value)
                if prompt_config
                else None,
                instruction_context={
                    "json_schema": None,  # Template DJ uses inline format in system prompt
                    "parsing_notes": (
                        '- The "hallucination_risk" field must be one of: "none", "low", "medium", "high"'
                    ),
                },
            )

            try:
                llm_response = parsing_llm.invoke(assessment_messages)
                raw_response, usage_metadata = llm_response.content, llm_response.usage.to_dict()
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

    reasoning_system_prompt = build_reasoning_system_prompt(
        generic_system_prompt=generic_system_prompt,
        attr_guidance=attr_guidance,
        search_performed=search_performed,
    )

    reasoning_prompt = build_reasoning_user_prompt(
        question_text=question_text,
        formatted_excerpts=format_excerpts_for_reasoning(excerpts),
    )

    reasoning_assembler = PromptAssembler(
        task=PromptTask.DJ_TEMPLATE_REASONING,
        interface=parsing_model.interface,
        capabilities=PortCapabilities(),
    )
    reasoning_messages = reasoning_assembler.assemble(
        system_text=reasoning_system_prompt,
        user_text=reasoning_prompt,
        user_instructions=prompt_config.get_for_task(PromptTask.DJ_TEMPLATE_REASONING.value) if prompt_config else None,
    )
    llm_response = parsing_llm.invoke(reasoning_messages)
    raw_response, usage_metadata = llm_response.content, llm_response.usage.to_dict()
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
    # STAGE 3: PARAMETER EXTRACTION (via ParserPort)
    # ==========================================
    # Format reasoning traces as text for the parser to extract from.
    # Include question context so parser has all information needed.
    reasoning_text = f"""Original Question: {question_text}

Reasoning Traces (explaining how excerpts inform each attribute value):

{format_reasoning_for_parsing(reasoning)}"""

    # Build messages via PromptAssembler for the parser
    stage3_assembler = PromptAssembler(
        task=PromptTask.PARSING,
        interface=parsing_model.interface,
        capabilities=parser.capabilities,
    )
    stage3_messages = stage3_assembler.assemble(
        system_text=combined_system_prompt,
        user_text=reasoning_text,
        user_instructions=prompt_config.get_for_task(PromptTask.PARSING.value) if prompt_config else None,
        instruction_context={"json_schema": RawAnswer.model_json_schema()},
    )

    # Use ParserPort for parsing - it handles LLM call, JSON parsing, and retry logic internally
    parse_port_result = parser.parse_to_pydantic(stage3_messages, RawAnswer)
    parsed_answer = parse_port_result.parsed
    model_calls += 1  # ParserPort makes at least one LLM call

    # Track parsing usage from deep judgment if tracker provided
    if usage_tracker is not None and parse_port_result.usage:
        usage_dict = parse_port_result.usage.to_dict()
        if usage_dict.get("input_tokens", 0) > 0 or usage_dict.get("output_tokens", 0) > 0:
            usage_tracker.track_call("parsing", parsing_model_str, usage_dict)

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
