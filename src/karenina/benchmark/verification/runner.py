"""Single model verification runner."""

import logging
import os
import re
import time
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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


# Define retryable exceptions
def is_retryable_error(exception: Exception) -> bool:
    """Check if an exception is retryable (transient error)."""
    exception_str = str(exception).lower()
    exception_type = type(exception).__name__

    # Connection-related errors
    if any(
        keyword in exception_str
        for keyword in [
            "connection",
            "timeout",
            "timed out",
            "rate limit",
            "429",
            "503",
            "502",
            "500",
            "network",
            "temporary failure",
        ]
    ):
        return True

    # Common retryable exception types
    retryable_types = [
        "ConnectionError",
        "TimeoutError",
        "HTTPError",
        "ReadTimeout",
        "ConnectTimeout",
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
    ]

    return exception_type in retryable_types


def _invoke_llm_with_retry(
    llm: Any, messages: list[BaseMessage], is_agent: bool, timeout: int = 120
) -> tuple[Any, bool]:
    """
    Invoke LLM with automatic retry logic for transient errors.

    Args:
        llm: The LLM or agent to invoke
        messages: List of messages to send to the LLM
        is_agent: Whether the LLM is a LangGraph agent
        timeout: Timeout in seconds for agent invocation

    Returns:
        Tuple of (response, recursion_limit_reached)

    Raises:
        Exception: If all retry attempts are exhausted
    """

    def _log_retry(retry_state: Any) -> None:
        """Log retry attempt with error details."""
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(f"Retrying LLM call (attempt {retry_state.attempt_number}/3) after error: {exc}")

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),  # Try 3 times
        wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff: 2s, 4s, 8s
        reraise=True,
        before_sleep=_log_retry,
    )
    def _invoke_with_retry_inner() -> tuple[Any, bool]:
        recursion_limit_reached = False

        try:
            if is_agent:
                # LangGraph agents with MCP tools need async invocation
                import asyncio

                async def invoke_agent_async() -> Any:
                    try:
                        return await llm.ainvoke({"messages": messages})
                    except Exception as e:
                        # Check if this is a GraphRecursionError
                        if "GraphRecursionError" in str(type(e).__name__) or "recursion_limit" in str(e).lower():
                            nonlocal recursion_limit_reached
                            recursion_limit_reached = True
                            # Try to extract partial state from the agent
                            try:
                                agent_state = llm.get_state({"messages": messages})
                                return agent_state
                            except Exception:
                                # If we can't get state, return the messages we have so far
                                return {"messages": messages}
                        else:
                            # Check if this is a retryable error
                            if is_retryable_error(e):
                                logger.info(f"Detected retryable error: {type(e).__name__}: {e}")
                                raise  # Re-raise to trigger retry
                            else:
                                raise e

                # Run the async invocation in the event loop
                try:
                    asyncio.get_running_loop()
                    # We're in an async context, use ThreadPoolExecutor
                    import concurrent.futures

                    def run_in_thread() -> Any:
                        return asyncio.run(invoke_agent_async())

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        response = future.result(timeout=timeout)

                except RuntimeError:
                    # No event loop running, safe to use asyncio.run
                    response = asyncio.run(invoke_agent_async())

                from ...llm.mcp_utils import harmonize_agent_response

                return harmonize_agent_response(response), recursion_limit_reached
            else:
                # Regular LLMs expect the messages list directly
                response = llm.invoke(messages)
                return response, recursion_limit_reached

        except Exception as e:
            # Check if this is a retryable error
            if is_retryable_error(e):
                logger.info(f"Detected retryable error: {type(e).__name__}: {e}")
                raise  # Re-raise to trigger retry
            else:
                # Non-retryable error, don't retry
                raise

    return _invoke_with_retry_inner()


def _should_expose_ground_truth() -> bool:
    """
    Check if ground truth should be exposed to the parser model.

    Reads from the KARENINA_EXPOSE_GROUND_TRUTH environment variable.
    Defaults to False for backward compatibility.

    Returns:
        True if ground truth should be exposed, False otherwise
    """
    return os.getenv("KARENINA_EXPOSE_GROUND_TRUTH", "false").lower() in ("true", "1", "yes", "on")


def _split_parsed_response(parsed_answer: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Split parsed answer into ground truth and LLM response components.

    Args:
        parsed_answer: The parsed answer object from LLM

    Returns:
        Tuple of (parsed_gt_response, parsed_llm_response)
        - parsed_gt_response: The 'correct' field content (ground truth)
        - parsed_llm_response: All other fields except 'id' and 'correct'
    """
    if not parsed_answer or not hasattr(parsed_answer, "model_dump"):
        return None, None

    parsed_dict = parsed_answer.model_dump()

    # Extract ground truth from 'correct' field
    parsed_gt_response = parsed_dict.get("correct")

    # Create LLM response by excluding 'id', 'correct', and 'regex' (configuration fields)
    parsed_llm_response = {k: v for k, v in parsed_dict.items() if k not in ("id", "correct", "regex")}

    return parsed_gt_response, parsed_llm_response


def _is_valid_md5_hash(hash_string: str) -> bool:
    """
    Validate that a string is a proper MD5 hash format.

    Args:
        hash_string: String to validate

    Returns:
        True if valid MD5 hash format, False otherwise
    """
    if not isinstance(hash_string, str):
        return False

    # MD5 hash is exactly 32 hexadecimal characters
    md5_pattern = re.compile(r"^[a-fA-F0-9]{32}$")
    return bool(md5_pattern.match(hash_string))


def _construct_few_shot_prompt(
    question_text: str, few_shot_examples: list[dict[str, str]] | None, few_shot_enabled: bool
) -> str:
    """
    Construct a prompt with few-shot examples if enabled.

    Args:
        question_text: The main question to ask
        few_shot_examples: List of question-answer pairs for few-shot prompting
        few_shot_enabled: Whether few-shot prompting is enabled

    Returns:
        The constructed prompt with optional few-shot examples
    """
    if not few_shot_enabled or not few_shot_examples:
        return question_text

    # Build the prompt with examples
    prompt_parts = []

    for example in few_shot_examples:
        if "question" in example and "answer" in example:
            prompt_parts.append(f"Question: {example['question']}")
            prompt_parts.append(f"Answer: {example['answer']}")
            prompt_parts.append("")  # Empty line for separation

    # Add the actual question
    prompt_parts.append(f"Question: {question_text}")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


def _system_prompt_compose(
    system_prompt: str | None, format_instructions: str, ground_truth: dict[str, Any] | None = None
) -> str:
    """
    Compose a system prompt with format instructions and optional ground truth information.

    Args:
        system_prompt: The system prompt to compose
        format_instructions: The format instructions to compose
        ground_truth: Optional ground truth information to include for parsing assistance

    Returns:
        The composed system prompt
    """
    prompt_parts = [
        f"<general_instructions>\n{system_prompt if system_prompt else ''}\n</general_instructions>",
        f"<format_instructions>\n{format_instructions}\n</format_instructions>",
    ]

    # Add ground truth instructions if provided
    if ground_truth is not None:
        import json

        ground_truth_str = json.dumps(ground_truth, indent=2, default=str)

        ground_truth_section = f"""<ground_truth_reference>
The following ground truth information is provided as reference to help with semantic matching and disambiguation.
Use this information carefully - do not blindly copy it, but it may help resolve ambiguities when the trace
and template are semantically close but differ in exact wording. IF AND ONLY IF the answer is very close to the ground truth,
use the ground truth as final answer.

Ground Truth:
{ground_truth_str}
</ground_truth_reference>"""

        prompt_parts.append(ground_truth_section)

    return "\n\n".join(prompt_parts) + "\n"


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
) -> VerificationResult:
    """
    Run verification for a single question with specific answering and parsing models.

    Args:
        question_id: Unique identifier for the question. For manual interface, this MUST be
                    a 32-character hexadecimal MD5 hash (generated during question extraction).
        question_text: The question to ask the LLM
        template_code: Python code defining the Answer class
        answering_model: Configuration for the answering model
        parsing_model: Configuration for the parsing model
        run_name: Optional run name for tracking
        job_id: Optional job ID for tracking
        answering_replicate: Optional replicate number for answering model
        parsing_replicate: Optional replicate number for parsing model
        rubric: Optional rubric for qualitative evaluation
        few_shot_examples: Optional list of question-answer pairs for few-shot prompting
        few_shot_enabled: Whether to use few-shot prompting (disabled by default)

    Returns:
        VerificationResult with all details and optional rubric scores

    Raises:
        ValueError: If question_id is not a valid MD5 hash when using manual interface
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
                success=False,
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
                    success=False,
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
                success=False,
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
                success=False,
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

        except Exception as e:
            return VerificationResult(
                question_id=question_id,
                template_id=template_id,
                success=False,
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
            )

        # Step 6: Run rubric evaluation (optional)
        rubric_result = None
        if rubric and (rubric.traits or rubric.manual_traits):
            try:
                # Use parsing model for rubric evaluation
                evaluator = RubricEvaluator(parsing_model)
                rubric_result = evaluator.evaluate_rubric(
                    question=question_text, answer=raw_llm_response, rubric=rubric
                )
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
            success=True,
            verify_result=verification_result,
            verify_rubric=rubric_result,
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
        )

    except Exception as e:
        return VerificationResult(
            question_id=question_id,
            template_id=template_id,
            success=False,
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
        )
