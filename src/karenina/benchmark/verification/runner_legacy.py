"""LEGACY: Original monolithic verification implementation.

⚠️ DEPRECATION WARNING ⚠️

This module contains the original monolithic verification implementation that has been
replaced by a modern stage-based pipeline architecture. It is preserved ONLY for:
1. Regression testing to prove behavioral equivalence with the new implementation
2. Historical reference during the migration period

DO NOT USE THIS CODE FOR NEW FEATURES OR MODIFICATIONS.

The new stage-based implementation is located in:
- runner.py: Main entry point with run_single_model_verification()
- stage_orchestrator.py: Pipeline orchestration
- stages/: Individual verification stages

This file will be removed once all regression tests pass 100% and the migration is complete.

Migration Status: Week 2 Complete - 107/108 tests passing (99.1%)
Target Removal: End of Week 3 (after code cleanup and final validation)
"""

import logging
import time
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from ...answers.generator import inject_question_id_into_answer_class
from ...llm.interface import init_chat_model_unified
from ...schemas.rubric_class import Rubric
from ...utils.checkpoint_converter import generate_template_id
from ..models import ModelConfig, VerificationConfig, VerificationResult
from .abstention_checker import detect_abstention
from .deep_judgment import deep_judgment_parse
from .embedding_utils import perform_embedding_check
from .parser_utils import _strip_markdown_fences
from .rubric_evaluator import RubricEvaluator
from .validation import validate_answer_template

# Set up logger
logger = logging.getLogger(__name__)

# Import shared utility functions from verification_utils
from .verification_utils import (  # noqa: E402
    _construct_few_shot_prompt,
    _invoke_llm_with_retry,
    _is_valid_md5_hash,
    _should_expose_ground_truth,
    _split_parsed_response,
    _system_prompt_compose,
)

# ============================================================================
# LEGACY Monolithic Implementation
# ============================================================================


def run_single_model_verification(
    question_id: str,
    question_text: str,
    template_code: str,
    answering_model: ModelConfig,
    parsing_model: ModelConfig,
    run_name: str | None = None,
    job_id: str | None = None,
    answering_replicate: int | None = None,
    parsing_replicate: int | None = None,
    rubric: Rubric | None = None,
    keywords: list[str] | None = None,
    few_shot_examples: list[dict[str, str]] | None = None,
    few_shot_enabled: bool = False,
    abstention_enabled: bool = False,
    deep_judgment_enabled: bool = False,
    deep_judgment_max_excerpts_per_attribute: int = 3,
    deep_judgment_fuzzy_match_threshold: float = 0.80,
    deep_judgment_excerpt_retry_attempts: int = 2,
    deep_judgment_search_enabled: bool = False,
    deep_judgment_search_tool: str | Any = "tavily",
) -> VerificationResult:
    """
    LEGACY: Original monolithic implementation (pre-refactor).

    ⚠️ DEPRECATED: This function is kept for regression testing only.
    Use runner.run_single_model_verification() for new code.

    Kept for regression testing. Will be removed once stage-based
    implementation passes all regression tests.
    """
    from datetime import datetime

    start_time = time.time()
    timestamp = datetime.now().isoformat()

    # Compute template_id from template_code (composite key component)
    template_id = generate_template_id(template_code)

    # For OpenRouter interface, don't include provider in the model string
    if answering_model.interface == "openrouter":
        answering_model_str = answering_model.model_name
    else:
        answering_model_str = f"{answering_model.model_provider}/{answering_model.model_name}"

    if parsing_model.interface == "openrouter":
        parsing_model_str = parsing_model.model_name
    else:
        parsing_model_str = f"{parsing_model.model_provider}/{parsing_model.model_name}"

    # Extract MCP server names from answering model config
    answering_mcp_servers = list(answering_model.mcp_urls_dict.keys()) if answering_model.mcp_urls_dict else None

    # Debug logging for MCP configuration
    if answering_mcp_servers:
        logger.info(f"Answering model MCP servers: {answering_mcp_servers}")

    try:
        # Step 1: Validate the template
        is_valid, error_msg, RawAnswer = validate_answer_template(template_code)
        if not is_valid or RawAnswer is None:
            return VerificationResult(
                question_id=question_id,
                template_id=template_id,
                completed_without_errors=False,
                error=f"Template validation failed: {error_msg}",
                keywords=keywords,
                question_text=question_text,
                raw_llm_response="",
                parsed_gt_response=None,
                parsed_llm_response=None,
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
                evaluation_rubric=rubric.model_dump() if rubric else None,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
                answering_system_prompt=answering_model.system_prompt,
                parsing_system_prompt=parsing_model.system_prompt,
                run_name=run_name,
                job_id=job_id,
                answering_replicate=answering_replicate,
                parsing_replicate=parsing_replicate,
                # Embedding check metadata (defaults for error cases)
                embedding_check_performed=False,
                embedding_similarity_score=None,
                embedding_override_applied=False,
                embedding_model_used=None,
                # Regex validation metadata (defaults for error cases)
                regex_validations_performed=False,
                regex_validation_results=None,
                regex_validation_details=None,
                regex_overall_success=None,
                regex_extraction_results=None,
                # Recursion limit metadata (defaults for error cases)
                recursion_limit_reached=False,
                # Abstention detection metadata (defaults for error cases)
                abstention_check_performed=False,
                abstention_detected=None,
                abstention_override_applied=False,
                abstention_reasoning=None,
                # MCP server metadata
                answering_mcp_servers=answering_mcp_servers,
                # Deep-judgment metadata (defaults for error cases)
                deep_judgment_enabled=deep_judgment_enabled,
                deep_judgment_performed=False,
                extracted_excerpts=None,
                attribute_reasoning=None,
                deep_judgment_stages_completed=None,
                deep_judgment_model_calls=0,
                deep_judgment_excerpt_retry_count=0,
                attributes_without_excerpts=None,
                # Search-enhanced deep-judgment metadata (defaults for error cases)
                deep_judgment_search_enabled=deep_judgment_search_enabled,
                hallucination_risk_assessment=None,
            )

        # Step 1.5: Inject question ID into the Answer class
        Answer = inject_question_id_into_answer_class(RawAnswer, question_id)

        # Step 2: Initialize answering model
        # For manual interface, pass the question_id as question_hash
        # Note: question_id should be an MD5 hash from the question extraction process
        if answering_model.interface == "manual":
            # Validate that question_id is indeed an MD5 hash for manual interface
            if not _is_valid_md5_hash(question_id):
                raise ValueError(
                    f"Invalid question_id format for manual interface: '{question_id}'. "
                    "question_id must be a 32-character hexadecimal MD5 hash when using manual interface. "
                    "This hash is typically generated during question extraction from the question text."
                )

            answering_llm = init_chat_model_unified(
                model=answering_model.model_name,
                provider=answering_model.model_provider,
                temperature=answering_model.temperature,
                interface=answering_model.interface,
                question_hash=question_id,
                mcp_urls_dict=answering_model.mcp_urls_dict,
                mcp_tool_filter=answering_model.mcp_tool_filter,
            )
        else:
            answering_llm = init_chat_model_unified(
                model=answering_model.model_name,
                provider=answering_model.model_provider,
                temperature=answering_model.temperature,
                interface=answering_model.interface,
                mcp_urls_dict=answering_model.mcp_urls_dict,
                mcp_tool_filter=answering_model.mcp_tool_filter,
            )

        # Step 3: Get LLM response
        messages: list[BaseMessage] = []
        if answering_model.system_prompt:
            messages.append(SystemMessage(content=answering_model.system_prompt))

        # Construct prompt with optional few-shot examples
        constructed_prompt = _construct_few_shot_prompt(question_text, few_shot_examples, few_shot_enabled)
        messages.append(HumanMessage(content=constructed_prompt))

        # Track recursion limit status
        recursion_limit_reached = False

        try:
            # Use retry-wrapped invocation
            is_agent = answering_model.mcp_urls_dict is not None
            response, recursion_limit_reached = _invoke_llm_with_retry(answering_llm, messages, is_agent)

            # Process response based on type
            if is_agent:
                raw_llm_response = response
                # Add note if recursion limit was reached
                if recursion_limit_reached:
                    raw_llm_response += "\n\n[Note: Recursion limit reached - partial response shown]"
            else:
                raw_llm_response = response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            # Check if this is a recursion limit error that wasn't caught earlier
            if "GraphRecursionError" in str(type(e).__name__) or "recursion_limit" in str(e).lower():
                recursion_limit_reached = True
                raw_llm_response = f"[Note: Recursion limit reached before completion. Error: {e}]"
                # Continue processing with this partial response
            else:
                # Log detailed error information for debugging
                import traceback

                error_details = traceback.format_exc()
                logger.error(
                    f"LLM call failed for question {question_id}\n"
                    f"Exception type: {type(e).__name__}\n"
                    f"Exception message: {e}\n"
                    f"Full traceback:\n{error_details}"
                )

                # Create a more detailed error message
                error_msg = f"LLM call failed: {type(e).__name__}: {e}"

                return VerificationResult(
                    question_id=question_id,
                    template_id=template_id,
                    completed_without_errors=False,
                    error=error_msg,
                    keywords=keywords,
                    question_text=question_text,
                    raw_llm_response="",
                    parsed_gt_response=None,
                    parsed_llm_response=None,
                    answering_model=answering_model_str,
                    parsing_model=parsing_model_str,
                    evaluation_rubric=rubric.model_dump() if rubric else None,
                    execution_time=time.time() - start_time,
                    timestamp=timestamp,
                    answering_system_prompt=answering_model.system_prompt,
                    parsing_system_prompt=parsing_model.system_prompt,
                    run_name=run_name,
                    job_id=job_id,
                    answering_replicate=answering_replicate,
                    parsing_replicate=parsing_replicate,
                    # Embedding check metadata (defaults for error cases)
                    embedding_check_performed=False,
                    embedding_similarity_score=None,
                    embedding_override_applied=False,
                    embedding_model_used=None,
                    # Regex validation metadata (defaults for error cases)
                    regex_validations_performed=False,
                    regex_validation_results=None,
                    regex_validation_details=None,
                    regex_overall_success=None,
                    regex_extraction_results=None,
                    # Recursion limit metadata (defaults for error cases)
                    recursion_limit_reached=False,
                    # Abstention detection metadata (defaults for error cases)
                    abstention_check_performed=False,
                    abstention_detected=None,
                    abstention_override_applied=False,
                    abstention_reasoning=None,
                    # MCP server metadata
                    answering_mcp_servers=answering_mcp_servers,
                )

        # Step 4: Initialize parsing model and parse response
        parsing_llm = init_chat_model_unified(
            model=parsing_model.model_name,
            provider=parsing_model.model_provider,
            temperature=parsing_model.temperature,
            interface=parsing_model.interface,
        )

        # Create PydanticOutputParser
        try:
            parser: Any = PydanticOutputParser(pydantic_object=Answer)
        except Exception as e:
            return VerificationResult(
                question_id=question_id,
                template_id=template_id,
                completed_without_errors=False,
                error=f"Failed to create PydanticOutputParser: {e}",
                keywords=keywords,
                question_text=question_text,
                raw_llm_response=raw_llm_response,
                parsed_gt_response=None,
                parsed_llm_response=None,
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
                evaluation_rubric=rubric.model_dump() if rubric else None,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
                answering_system_prompt=answering_model.system_prompt,
                parsing_system_prompt=parsing_model.system_prompt,
                run_name=run_name,
                job_id=job_id,
                answering_replicate=answering_replicate,
                parsing_replicate=parsing_replicate,
                # Embedding check metadata (defaults for error cases)
                embedding_check_performed=False,
                embedding_similarity_score=None,
                embedding_override_applied=False,
                embedding_model_used=None,
                # Regex validation metadata (defaults for error cases)
                regex_validations_performed=False,
                regex_validation_results=None,
                regex_validation_details=None,
                regex_overall_success=None,
                regex_extraction_results=None,
                # Recursion limit metadata (defaults for error cases)
                recursion_limit_reached=False,
                # Abstention detection metadata (defaults for error cases)
                abstention_check_performed=False,
                abstention_detected=None,
                abstention_override_applied=False,
                abstention_reasoning=None,
                # MCP server metadata
                answering_mcp_servers=answering_mcp_servers,
                # Deep-judgment metadata (defaults for error cases)
                deep_judgment_enabled=deep_judgment_enabled,
                deep_judgment_performed=False,
                extracted_excerpts=None,
                attribute_reasoning=None,
                deep_judgment_stages_completed=None,
                deep_judgment_model_calls=0,
                deep_judgment_excerpt_retry_count=0,
                attributes_without_excerpts=None,
                # Search-enhanced deep-judgment metadata (defaults for error cases)
                deep_judgment_search_enabled=deep_judgment_search_enabled,
                hallucination_risk_assessment=None,
            )

        # Extract ground truth if enabled
        ground_truth = None
        if _should_expose_ground_truth():
            try:
                from .template_utils import create_test_instance_from_answer_class

                # Create test instance and extract ground truth
                _, ground_truth = create_test_instance_from_answer_class(RawAnswer)
            except Exception as e:
                # If we can't extract ground truth, continue without it
                # This ensures the feature is robust and doesn't break existing functionality
                print(f"Warning: Could not extract ground truth for question {question_id}: {e}")

        # Create parsing prompt with format instructions and optional ground truth
        format_instructions = parser.get_format_instructions()
        combined_system_prompt = _system_prompt_compose(parsing_model.system_prompt, format_instructions, ground_truth)

        # Construct the parsing prompt (user message) with question context
        parsing_prompt = f"""<original_question>
Your task is to parse an answer given to the question reported in this section. Use the question to contextualize the info from the schema fields below:

Original Question: {question_text}
</original_question>

<response_to_parse>
{raw_llm_response}
</response_to_parse>"""

        parsing_messages: list[BaseMessage] = []
        if combined_system_prompt:
            parsing_messages.append(SystemMessage(content=combined_system_prompt))
        parsing_messages.append(HumanMessage(content=parsing_prompt))

        # Initialize deep-judgment metadata variables
        deep_judgment_performed = False
        extracted_excerpts = None
        attribute_reasoning = None
        deep_judgment_stages_completed = None
        deep_judgment_model_calls = 0
        deep_judgment_excerpt_retry_count = 0
        attributes_without_excerpts = None
        hallucination_risk_assessment = None

        try:
            # Choose parsing strategy based on configuration
            if deep_judgment_enabled:
                # Create minimal config for deep-judgment
                dj_config = VerificationConfig(
                    answering_models=[],
                    parsing_models=[parsing_model],
                    parsing_only=True,
                    deep_judgment_enabled=True,
                    deep_judgment_max_excerpts_per_attribute=deep_judgment_max_excerpts_per_attribute,
                    deep_judgment_fuzzy_match_threshold=deep_judgment_fuzzy_match_threshold,
                    deep_judgment_excerpt_retry_attempts=deep_judgment_excerpt_retry_attempts,
                    deep_judgment_search_enabled=deep_judgment_search_enabled,
                    deep_judgment_search_tool=deep_judgment_search_tool,
                )

                # Deep-judgment multi-stage parsing
                parsed_answer, extracted_excerpts, attribute_reasoning, dj_metadata = deep_judgment_parse(
                    raw_llm_response=raw_llm_response,
                    RawAnswer=RawAnswer,
                    parsing_model=parsing_model,
                    parsing_llm=parsing_llm,
                    question_text=question_text,
                    config=dj_config,
                    format_instructions=format_instructions,
                    combined_system_prompt=combined_system_prompt,
                )
                deep_judgment_performed = True
                deep_judgment_stages_completed = dj_metadata.get("stages_completed", [])
                deep_judgment_model_calls = dj_metadata.get("model_calls", 0)
                deep_judgment_excerpt_retry_count = dj_metadata.get("excerpt_retry_count", 0)
                attributes_without_excerpts = dj_metadata.get("attributes_without_excerpts", None)
                hallucination_risk_assessment = dj_metadata.get("hallucination_risk", None)
            else:
                # Standard single-stage parsing (existing logic)
                parsing_response = parsing_llm.invoke(parsing_messages)
                raw_parsing_response = (
                    parsing_response.content if hasattr(parsing_response, "content") else str(parsing_response)
                )

                # Strip markdown fences and parse with PydanticOutputParser
                cleaned_response = _strip_markdown_fences(raw_parsing_response)
                parsed_answer = parser.parse(cleaned_response)
        except Exception as e:
            return VerificationResult(
                question_id=question_id,
                template_id=template_id,
                completed_without_errors=False,
                error=f"Parsing failed: {e}",
                keywords=keywords,
                question_text=question_text,
                raw_llm_response=raw_llm_response,
                parsed_gt_response=None,
                parsed_llm_response=None,
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
                evaluation_rubric=rubric.model_dump() if rubric else None,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
                answering_system_prompt=answering_model.system_prompt,
                parsing_system_prompt=parsing_model.system_prompt,
                run_name=run_name,
                job_id=job_id,
                answering_replicate=answering_replicate,
                parsing_replicate=parsing_replicate,
                # Embedding check metadata (defaults for error cases)
                embedding_check_performed=False,
                embedding_similarity_score=None,
                embedding_override_applied=False,
                embedding_model_used=None,
                # Regex validation metadata (defaults for error cases)
                regex_validations_performed=False,
                regex_validation_results=None,
                regex_validation_details=None,
                regex_overall_success=None,
                regex_extraction_results=None,
                # Recursion limit metadata (defaults for error cases)
                recursion_limit_reached=False,
                # Abstention detection metadata (defaults for error cases)
                abstention_check_performed=False,
                abstention_detected=None,
                abstention_override_applied=False,
                abstention_reasoning=None,
                # MCP server metadata
                answering_mcp_servers=answering_mcp_servers,
                # Deep-judgment metadata (defaults for error cases)
                deep_judgment_enabled=deep_judgment_enabled,
                deep_judgment_performed=deep_judgment_performed,
                extracted_excerpts=extracted_excerpts,
                attribute_reasoning=attribute_reasoning,
                deep_judgment_stages_completed=deep_judgment_stages_completed,
                deep_judgment_model_calls=deep_judgment_model_calls,
                deep_judgment_excerpt_retry_count=deep_judgment_excerpt_retry_count,
                attributes_without_excerpts=attributes_without_excerpts,
            )

        # Step 5: Run verification
        try:
            # Standard field verification
            field_verification_result = parsed_answer.verify()  # type: ignore[attr-defined]

            # Step 5.1: Run regex verification on the raw trace
            regex_verification_results = parsed_answer.verify_regex(raw_llm_response)

            # Extract regex results for display (what the regex actually matched)
            regex_extraction_results = {}
            if regex_verification_results["details"]:
                for field_name, details in regex_verification_results["details"].items():
                    regex_extraction_results[field_name] = details.get("matches_found", [])

            # Combine field and regex verification results
            verification_result = field_verification_result and regex_verification_results["success"]

            # Step 5.5: Embedding check fallback if verification failed
            embedding_check_performed = False
            embedding_similarity_score = None
            embedding_model_used = None
            embedding_override_applied = False

            if not field_verification_result:  # Only apply embedding check to field verification failures
                # Extract ground truth and LLM response for embedding check
                parsed_gt_response, parsed_llm_response = _split_parsed_response(parsed_answer)

                # Perform embedding check
                (should_override, similarity_score, model_name, check_performed) = perform_embedding_check(
                    parsed_gt_response, parsed_llm_response, parsing_model, question_text
                )

                embedding_check_performed = check_performed
                embedding_similarity_score = similarity_score
                embedding_model_used = model_name

                if should_override:
                    # If embedding check overrides field verification, recalculate overall result
                    field_verification_result = True
                    verification_result = True and regex_verification_results["success"]
                    embedding_override_applied = True

            # Step 5.6: Abstention detection (runs after all other verification)
            abstention_check_performed = False
            abstention_detected = None
            abstention_override_applied = False
            abstention_reasoning = None

            if abstention_enabled:
                # Detect if model refused to answer or abstained
                abstention_detected, abstention_check_performed, abstention_reasoning = detect_abstention(
                    raw_llm_response=raw_llm_response,
                    parsing_model=parsing_model,
                    question_text=question_text,
                )

                # If abstention is detected, override the verification result
                if abstention_detected and abstention_check_performed:
                    # Mark as failed since model didn't provide a real answer
                    verification_result = False
                    abstention_override_applied = True
                    logger.info(f"Abstention detected for question {question_id} - overriding result to False")

            # Step 5.7: Deep-judgment auto-fail (runs after abstention check)
            # If deep-judgment found missing excerpts and abstention was NOT detected, auto-fail
            if (
                deep_judgment_performed
                and attributes_without_excerpts
                and len(attributes_without_excerpts) > 0
                and not (abstention_detected and abstention_check_performed)
            ):
                # Short-circuit verification: success=True (test ran fine), verify_result=False (verification failed)
                verification_result = False
                field_verification_result = False
                logger.info(
                    f"Deep-judgment auto-fail for question {question_id}: "
                    f"{len(attributes_without_excerpts)} attributes without excerpts: {', '.join(attributes_without_excerpts)}"
                )

        except Exception as e:
            return VerificationResult(
                question_id=question_id,
                template_id=template_id,
                completed_without_errors=False,
                error=f"Verification failed: {e}",
                keywords=keywords,
                question_text=question_text,
                raw_llm_response=raw_llm_response,
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
                parsed_gt_response=_split_parsed_response(parsed_answer)[0],
                parsed_llm_response=_split_parsed_response(parsed_answer)[1],
                evaluation_rubric=rubric.model_dump() if rubric else None,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
                answering_system_prompt=answering_model.system_prompt,
                parsing_system_prompt=parsing_model.system_prompt,
                run_name=run_name,
                job_id=job_id,
                answering_replicate=answering_replicate,
                parsing_replicate=parsing_replicate,
                # Embedding check metadata (defaults for error cases)
                embedding_check_performed=False,
                embedding_similarity_score=None,
                embedding_override_applied=False,
                embedding_model_used=None,
                # Regex validation metadata (defaults for error cases)
                regex_validations_performed=False,
                regex_validation_results=None,
                regex_validation_details=None,
                regex_overall_success=None,
                regex_extraction_results=None,
                # Recursion limit metadata (defaults for error cases)
                recursion_limit_reached=False,
                # Abstention detection metadata (defaults for error cases)
                abstention_check_performed=False,
                abstention_detected=None,
                abstention_override_applied=False,
                abstention_reasoning=None,
                # MCP server metadata
                answering_mcp_servers=answering_mcp_servers,
                # Deep-judgment metadata (defaults for error cases)
                deep_judgment_enabled=deep_judgment_enabled,
                deep_judgment_performed=False,
                extracted_excerpts=None,
                attribute_reasoning=None,
                deep_judgment_stages_completed=None,
                deep_judgment_model_calls=0,
                deep_judgment_excerpt_retry_count=0,
                attributes_without_excerpts=None,
                # Search-enhanced deep-judgment metadata (defaults for error cases)
                deep_judgment_search_enabled=deep_judgment_search_enabled,
                hallucination_risk_assessment=None,
            )

        # Step 6: Run rubric evaluation (optional)
        rubric_result = None
        metric_confusion_lists = None
        metric_results = None
        if rubric and (rubric.traits or rubric.manual_traits or rubric.metric_traits):
            try:
                # Use parsing model for rubric evaluation
                evaluator = RubricEvaluator(parsing_model)
                rubric_result = evaluator.evaluate_rubric(
                    question=question_text, answer=raw_llm_response, rubric=rubric
                )

                # Evaluate metric traits separately (returns confusion lists and computed metrics)
                if rubric.metric_traits:
                    print(
                        f"🔍 Runner: Evaluating {len(rubric.metric_traits)} metric trait(s) for question {question_id}"
                    )
                    for trait in rubric.metric_traits:
                        print(f"  - Trait: {trait.name} (mode: {trait.evaluation_mode}, metrics: {trait.metrics})")

                    metric_confusion_lists, metric_results = evaluator.evaluate_metric_traits(
                        question=question_text, answer=raw_llm_response, metric_traits=rubric.metric_traits
                    )

                    print(
                        f"  ✅ Metric evaluation complete. Results: {list(metric_results.keys()) if metric_results else 'None'}"
                    )
                    if metric_results:
                        for trait_name, metrics in metric_results.items():
                            print(f"     {trait_name}: {metrics}")
            except (ValueError, RuntimeError) as e:
                # Handle specific rubric evaluator errors
                print(f"Warning: Rubric evaluator initialization/configuration failed for question {question_id}: {e}")
                rubric_result = None
            except Exception as e:
                # Don't fail the entire verification if rubric evaluation fails
                print(f"Warning: Rubric evaluation failed for question {question_id}: {e}")
                rubric_result = None

        return VerificationResult(
            question_id=question_id,
            template_id=template_id,
            completed_without_errors=True,
            verify_result=verification_result,
            verify_rubric=rubric_result,
            metric_trait_confusion_lists=metric_confusion_lists,
            metric_trait_metrics=metric_results,
            evaluation_rubric=rubric.model_dump() if rubric else None,
            keywords=keywords,
            question_text=question_text,
            raw_llm_response=raw_llm_response,
            answering_model=answering_model_str,
            parsing_model=parsing_model_str,
            parsed_gt_response=_split_parsed_response(parsed_answer)[0],
            parsed_llm_response=_split_parsed_response(parsed_answer)[1],
            execution_time=time.time() - start_time,
            timestamp=timestamp,
            answering_system_prompt=answering_model.system_prompt,
            parsing_system_prompt=parsing_model.system_prompt,
            run_name=run_name,
            job_id=job_id,
            answering_replicate=answering_replicate,
            parsing_replicate=parsing_replicate,
            # Embedding check metadata
            embedding_check_performed=embedding_check_performed,
            embedding_similarity_score=embedding_similarity_score,
            embedding_override_applied=embedding_override_applied,
            embedding_model_used=embedding_model_used,
            # Regex validation metadata
            regex_validations_performed=bool(regex_verification_results["results"]),
            regex_validation_results=regex_verification_results["results"],
            regex_validation_details=regex_verification_results["details"],
            regex_overall_success=regex_verification_results["success"],
            regex_extraction_results=regex_extraction_results,
            # Recursion limit metadata
            recursion_limit_reached=recursion_limit_reached,
            # Abstention detection metadata
            abstention_check_performed=abstention_check_performed,
            abstention_detected=abstention_detected,
            abstention_override_applied=abstention_override_applied,
            abstention_reasoning=abstention_reasoning,
            # MCP server metadata
            answering_mcp_servers=answering_mcp_servers,
            # Deep-judgment metadata
            deep_judgment_enabled=deep_judgment_enabled,
            deep_judgment_performed=deep_judgment_performed,
            extracted_excerpts=extracted_excerpts,
            attribute_reasoning=attribute_reasoning,
            deep_judgment_stages_completed=deep_judgment_stages_completed,
            deep_judgment_model_calls=deep_judgment_model_calls,
            deep_judgment_excerpt_retry_count=deep_judgment_excerpt_retry_count,
            attributes_without_excerpts=attributes_without_excerpts,
            # Search-enhanced deep-judgment metadata
            deep_judgment_search_enabled=deep_judgment_search_enabled,
            hallucination_risk_assessment=hallucination_risk_assessment,
        )

    except Exception as e:
        return VerificationResult(
            question_id=question_id,
            template_id=template_id,
            completed_without_errors=False,
            error=f"Unexpected error: {e}",
            keywords=keywords,
            question_text=question_text,
            raw_llm_response="",
            parsed_gt_response=None,
            parsed_llm_response=None,
            answering_model=answering_model_str,
            parsing_model=parsing_model_str,
            evaluation_rubric=rubric.model_dump() if rubric else None,
            execution_time=time.time() - start_time,
            timestamp=timestamp,
            answering_system_prompt=answering_model.system_prompt,
            parsing_system_prompt=parsing_model.system_prompt,
            run_name=run_name,
            job_id=job_id,
            answering_replicate=answering_replicate,
            parsing_replicate=parsing_replicate,
            # Embedding check metadata (defaults for error cases)
            embedding_check_performed=False,
            embedding_similarity_score=None,
            embedding_override_applied=False,
            embedding_model_used=None,
            # Regex validation metadata (defaults for error cases)
            regex_validations_performed=False,
            regex_validation_results=None,
            regex_validation_details=None,
            regex_overall_success=None,
            regex_extraction_results=None,
            # Recursion limit metadata (defaults for error cases)
            recursion_limit_reached=False,
            # Abstention detection metadata (defaults for error cases)
            abstention_check_performed=False,
            abstention_detected=None,
            abstention_override_applied=False,
            abstention_reasoning=None,
            # MCP server metadata
            answering_mcp_servers=answering_mcp_servers,
            # Deep-judgment metadata (defaults for error cases)
            deep_judgment_enabled=deep_judgment_enabled,
            deep_judgment_performed=False,
            extracted_excerpts=None,
            attribute_reasoning=None,
            deep_judgment_stages_completed=None,
            deep_judgment_model_calls=0,
            deep_judgment_excerpt_retry_count=0,
            attributes_without_excerpts=None,
            # Search-enhanced deep-judgment metadata (defaults for error cases)
            deep_judgment_search_enabled=deep_judgment_search_enabled,
            hallucination_risk_assessment=None,
        )
