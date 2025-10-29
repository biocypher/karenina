"""Single model verification runner."""

import logging
import os
import re
from typing import Any

from langchain_core.messages import BaseMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ...schemas.rubric_class import Rubric
from ...utils.checkpoint_converter import generate_template_id
from ..models import ModelConfig, VerificationResult

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
    deep_judgment_search_enabled: bool = False,
    deep_judgment_search_tool: str | Any = "tavily",
) -> VerificationResult:
    """
    Run verification for a single question with specific answering and parsing models.

    This function uses a stage-based pipeline architecture for modularity and testability.
    Each verification step (validation, generation, parsing, verification, etc.) is
    implemented as a discrete stage that can be independently tested and configured.

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
        keywords: Optional keywords associated with the question
        few_shot_examples: Optional list of question-answer pairs for few-shot prompting
        few_shot_enabled: Whether to use few-shot prompting (disabled by default)
        abstention_enabled: Whether to enable abstention detection
        deep_judgment_enabled: Whether to enable deep-judgment parsing
        deep_judgment_max_excerpts_per_attribute: Max excerpts per attribute (deep-judgment)
        deep_judgment_fuzzy_match_threshold: Similarity threshold for excerpts (deep-judgment)
        deep_judgment_excerpt_retry_attempts: Retry attempts for excerpt validation (deep-judgment)
        deep_judgment_search_enabled: Whether to enable search enhancement (deep-judgment)
        deep_judgment_search_tool: Search tool name or callable (deep-judgment)

    Returns:
        VerificationResult with all details and optional rubric scores

    Raises:
        ValueError: If question_id is not a valid MD5 hash when using manual interface
        RuntimeError: If stage orchestration fails critically
    """
    from .stage import VerificationContext
    from .stage_orchestrator import StageOrchestrator

    # Compute template_id from template_code (composite key component)
    template_id = generate_template_id(template_code)

    # Initialize verification context with all parameters
    context = VerificationContext(
        # Identity & Metadata
        question_id=question_id,
        template_id=template_id,
        question_text=question_text,
        template_code=template_code,
        # Configuration
        answering_model=answering_model,
        parsing_model=parsing_model,
        rubric=rubric,
        keywords=keywords,
        # Run Metadata
        run_name=run_name,
        job_id=job_id,
        answering_replicate=answering_replicate,
        parsing_replicate=parsing_replicate,
        # Feature Flags
        few_shot_enabled=few_shot_enabled,
        abstention_enabled=abstention_enabled,
        deep_judgment_enabled=deep_judgment_enabled,
        # Deep-Judgment Configuration
        deep_judgment_max_excerpts_per_attribute=deep_judgment_max_excerpts_per_attribute,
        deep_judgment_fuzzy_match_threshold=deep_judgment_fuzzy_match_threshold,
        deep_judgment_excerpt_retry_attempts=deep_judgment_excerpt_retry_attempts,
        deep_judgment_search_enabled=deep_judgment_search_enabled,
        deep_judgment_search_tool=deep_judgment_search_tool,
        # Few-Shot Configuration
        few_shot_examples=few_shot_examples,
    )

    # Compute model strings for result (needed even if validation fails)
    # For OpenRouter interface, don't include provider in the model string
    if answering_model.interface == "openrouter":
        answering_model_str = answering_model.model_name
    else:
        answering_model_str = f"{answering_model.model_provider}/{answering_model.model_name}"

    if parsing_model.interface == "openrouter":
        parsing_model_str = parsing_model.model_name
    else:
        parsing_model_str = f"{parsing_model.model_provider}/{parsing_model.model_name}"

    # Store model strings in context for early access (e.g., in error cases)
    context.set_artifact("answering_model_str", answering_model_str)
    context.set_artifact("parsing_model_str", parsing_model_str)

    # Build stage orchestrator from configuration
    orchestrator = StageOrchestrator.from_config(
        answering_model=answering_model,
        parsing_model=parsing_model,
        rubric=rubric,
        abstention_enabled=abstention_enabled,
        deep_judgment_enabled=deep_judgment_enabled,
    )

    # Execute verification pipeline
    result = orchestrator.execute(context)

    return result
