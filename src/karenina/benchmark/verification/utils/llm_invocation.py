"""LLM invocation utilities for verification operations.

Shared utilities used by verification stages, runner, and evaluators.
These functions handle common tasks like LLM invocation with retries,
response parsing, and validation.

Functions provided:
- Error handling: is_retryable_error (imported from error_helpers)
- LLM invocation: _invoke_llm_with_retry
- Agent metrics: _extract_agent_metrics (from trace_agent_metrics)
- Response parsing: _split_parsed_response
- Prompt construction: _construct_few_shot_prompt
- Retry logic: _retry_parse_with_null_feedback, _extract_null_fields_from_error

Note: Template-specific parsing and prompt construction is handled by
TemplateEvaluator (evaluators/template_evaluator.py) which encapsulates
all template evaluation logic following the evaluator pattern.
"""

import json
import logging
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.callbacks.usage import UsageMetadataCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from karenina.adapters.langchain.agent import extract_partial_agent_state

from .error_helpers import is_retryable_error
from .trace_agent_metrics import (
    extract_agent_metrics,
    extract_middleware_metrics,
)

# Set up logger
logger = logging.getLogger(__name__)

# Re-export agent metrics functions with underscore prefix for backward compatibility
# These are the internal names used throughout the codebase
_extract_agent_metrics = extract_agent_metrics
_extract_middleware_metrics = extract_middleware_metrics

# ============================================================================
# Error Handling and Retry Logic
# ============================================================================

# is_retryable_error is imported from error_helpers


def _invoke_llm_with_retry(
    llm: Any, messages: list[BaseMessage], is_agent: bool, timeout: int = 120
) -> tuple[Any, bool, dict[str, Any], dict[str, Any] | None]:
    """
    Invoke LLM with automatic retry logic for transient errors.

    For agents (LangChain 1.1+): Middleware handles retries via ModelRetryMiddleware
    and ToolRetryMiddleware. This function only handles state recovery on limit errors.

    For regular LLMs: Uses tenacity for exponential backoff retry logic.

    Args:
        llm: The LLM or agent to invoke
        messages: List of messages to send to the LLM
        is_agent: Whether the LLM is a LangGraph agent (has middleware)
        timeout: Timeout in seconds for agent invocation

    Returns:
        Tuple of (response, limit_reached, usage_metadata, agent_metrics)
        - response: The LLM/agent response
        - limit_reached: Whether agent hit a limit (recursion, model calls, or tool calls)
        - usage_metadata: Token usage metadata from LangChain callback
        - agent_metrics: Agent execution metrics (iterations, tool_calls, tools_used, etc.) or None for non-agents

    Raises:
        Exception: If all retry attempts are exhausted (non-agent) or unrecoverable error (agent)
    """
    if is_agent:
        # Agent path: Middleware handles retries, we only handle state recovery
        return _invoke_agent_with_middleware(llm, messages, timeout)
    else:
        # Non-agent path: Use tenacity for retry logic
        return _invoke_llm_with_tenacity_retry(llm, messages)


def _invoke_agent_with_middleware(
    llm: Any, messages: list[BaseMessage], timeout: int = 120
) -> tuple[Any, bool, dict[str, Any], dict[str, Any] | None]:
    """
    Invoke a LangGraph agent with middleware.

    Middleware (ModelRetryMiddleware, ToolRetryMiddleware) handles retries internally.
    This function focuses on:
    1. Capturing usage metadata
    2. Handling limit exceptions (model call, tool call, recursion)
    3. Extracting partial state when limits are reached

    Args:
        llm: The LangGraph agent to invoke
        messages: List of messages to send
        timeout: Timeout in seconds

    Returns:
        Tuple of (response, limit_reached, usage_metadata, agent_metrics)
    """
    import asyncio

    async def invoke_agent_async() -> tuple[Any, dict[str, Any], bool]:
        """
        Invoke agent and return (response, usage_metadata, limit_reached).
        """
        cb_manager = None
        cb: UsageMetadataCallbackHandler | None = None
        local_usage_metadata: dict[str, Any] = {}
        limit_reached = False

        try:
            # Wrap async invoke with usage metadata callback
            cb_manager = get_usage_metadata_callback()
            cb = cb_manager.__enter__()

            # Use ainvoke with thread_id for checkpointer to track state
            # This enables partial state recovery when limits are hit
            config = {"configurable": {"thread_id": "default"}}
            response = await llm.ainvoke({"messages": messages}, config=config)

            # Capture usage metadata on success
            local_usage_metadata = dict(cb.usage_metadata) if cb and cb.usage_metadata else {}
            cb_manager.__exit__(None, None, None)
            return response, local_usage_metadata, limit_reached

        except Exception as e:
            # CRITICAL: Capture usage metadata BEFORE handling exception
            # This ensures we track tokens even when limits are hit
            if cb and cb.usage_metadata:
                local_usage_metadata = dict(cb.usage_metadata)
                # Close the callback context
                from contextlib import suppress

                if cb_manager:
                    with suppress(Exception):
                        cb_manager.__exit__(type(e), e, e.__traceback__)

            # Check if this is a limit-related error (recursion, model calls, tool calls)
            error_type = type(e).__name__
            error_str = str(e).lower()

            is_limit_error = (
                "GraphRecursionError" in error_type
                or "recursion_limit" in error_str
                or "ModelCallLimitExceeded" in error_type
                or "ToolCallLimitExceeded" in error_type
                or "model_call_limit" in error_str
                or "tool_call_limit" in error_str
                or "limit" in error_str
                and ("exceeded" in error_str or "reached" in error_str)
            )

            if is_limit_error:
                limit_reached = True
                logger.info(f"Agent hit limit: {error_type}")

                # Try to extract partial state
                partial_state = extract_partial_agent_state(llm, messages, e, config)
                return partial_state, local_usage_metadata, limit_reached
            else:
                # Non-limit error - let it propagate
                # Note: Middleware should have already retried transient errors
                raise e

    # Run the async invocation using the shared portal if available,
    # otherwise fall back to asyncio.run()
    from ..batch_runner import get_async_portal

    portal = get_async_portal()

    if portal is not None:
        # Use the shared BlockingPortal for proper event loop management
        # This prevents "Event loop is closed" errors and connection pool degradation
        response, usage_metadata, limit_reached = portal.call(invoke_agent_async)
    else:
        # No portal available - use asyncio.run() (may cause event loop issues in threads)
        try:
            asyncio.get_running_loop()
            # We're in an async context, use ThreadPoolExecutor
            import concurrent.futures

            def run_in_thread() -> Any:
                return asyncio.run(invoke_agent_async())

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                result = future.result(timeout=timeout)
                response, usage_metadata, limit_reached = result

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            response, usage_metadata, limit_reached = asyncio.run(invoke_agent_async())

    # Extract agent metrics before harmonization
    agent_metrics = _extract_agent_metrics(response)

    from ...infrastructure.llm.mcp_utils import harmonize_agent_response

    # Extract original question from input messages for summary detection
    # Skip SystemMessages to find the actual user question (HumanMessage)
    original_question = None
    if messages:
        from langchain_core.messages import SystemMessage

        for msg in messages:
            # Skip system messages
            if isinstance(msg, SystemMessage):
                continue
            # Found first non-system message (should be HumanMessage with question)
            if hasattr(msg, "content"):
                original_question = str(msg.content)
                break

    return harmonize_agent_response(response, original_question), limit_reached, usage_metadata, agent_metrics


def _invoke_llm_with_tenacity_retry(
    llm: Any, messages: list[BaseMessage]
) -> tuple[Any, bool, dict[str, Any], dict[str, Any] | None]:
    """
    Invoke a regular LLM (non-agent) with tenacity retry logic.

    Args:
        llm: The LLM to invoke
        messages: List of messages to send

    Returns:
        Tuple of (response, False, usage_metadata, None)
        - limit_reached is always False for non-agents
        - agent_metrics is always None for non-agents
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
    def _invoke_with_retry() -> tuple[Any, bool, dict[str, Any], dict[str, Any] | None]:
        try:
            # Wrap invoke with usage metadata callback
            with get_usage_metadata_callback() as cb:
                response = llm.invoke(messages)
            usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}
            # Extract content from AIMessage for consistency with agent path
            raw_response = response.content if hasattr(response, "content") else str(response)
            # Non-agents don't have agent metrics or limit reached
            return raw_response, False, usage_metadata, None

        except Exception as e:
            # Check if this is a retryable error
            if is_retryable_error(e):
                logger.info(f"Detected retryable error: {type(e).__name__}: {e}")
                raise  # Re-raise to trigger retry
            else:
                # Non-retryable error, don't retry
                raise

    return _invoke_with_retry()


# ============================================================================
# Response Parsing and Validation
# ============================================================================


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


# ============================================================================
# Prompt Construction
# ============================================================================


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


# ============================================================================
# Null-Value Retry Logic
# ============================================================================


def _extract_null_fields_from_error(error_str: str, failed_json: str | None = None) -> list[str]:
    """
    Extract field names that had null values from parsing error.

    Tries two approaches:
    1. Parse the JSON from error message to find null fields
    2. Parse field names from Pydantic validation error text

    Args:
        error_str: Error message string
        failed_json: Optional JSON string that failed to parse

    Returns:
        List of field names that had null/None values
    """
    null_fields = []

    # Approach 1: Try to extract JSON and find null fields
    if failed_json:
        try:
            data = json.loads(failed_json)
            null_fields = [k for k, v in data.items() if v is None]
            if null_fields:
                return null_fields
        except json.JSONDecodeError:
            pass

    # Approach 2: Parse Pydantic validation error for None/null mentions
    # Pattern: field name followed by line mentioning input_value=None or input_type=NoneType
    lines = error_str.split("\n")

    for i, line in enumerate(lines):
        # Check if this line mentions None/null as input
        if "input_value=None" in line or "input_type=NoneType" in line:
            # Field name is usually 1-2 lines before the error message
            for j in range(i - 1, max(i - 3, -1), -1):
                potential_field = lines[j].strip()
                # Field names are single words without spaces, excluding common error keywords
                if (
                    potential_field
                    and " " not in potential_field
                    and potential_field not in ["Answer", "Input", "For", "Got:", "validation", "error"]
                ):
                    null_fields.append(potential_field)
                    break

    return list(set(null_fields))  # Remove duplicates


def _retry_parse_with_null_feedback(
    parsing_llm: Any,
    parser: Any,  # PydanticOutputParser
    original_messages: list[BaseMessage],
    failed_response: str,
    error: Exception,
    usage_tracker: Any | None = None,
    model_str: str | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """
    Retry parsing with feedback about null values in required fields.

    When parsing fails due to null values, this function:
    1. Extracts which fields had null values
    2. Sends feedback to LLM asking for actual values instead of nulls
    3. Retries parsing once

    Args:
        parsing_llm: The LLM to use for retry
        parser: PydanticOutputParser instance
        original_messages: Original messages that produced failed_response
        failed_response: The response that failed to parse
        error: The validation error from first parse attempt
        usage_tracker: Optional usage tracker
        model_str: Optional model string for tracking

    Returns:
        Tuple of (parsed_answer, usage_metadata)
        parsed_answer is None if retry also fails
    """
    from .json_helpers import strip_markdown_fences as _strip_markdown_fences

    # Try to extract JSON from error message
    failed_json = None
    error_str = str(error)
    if "from completion" in error_str:
        # Format: "Failed to parse X from completion {...}. Got: ..."
        try:
            json_start = error_str.index("{")
            json_end = error_str.index("}.", json_start) + 1
            failed_json = error_str[json_start:json_end]
        except (ValueError, IndexError):
            pass

    # Extract null fields
    null_fields = _extract_null_fields_from_error(error_str, failed_json)

    if not null_fields:
        # Not a null-related error, can't help
        logger.debug("Parsing error is not null-related, skipping retry")
        return None, {}

    logger.info(f"Detected null values in required fields: {null_fields}. Retrying with feedback...")

    # Build feedback message
    field_list = ", ".join(null_fields)
    feedback_prompt = f"""The previous response contained null values for required fields: [{field_list}].

Required fields cannot be null. Please provide actual values instead:
- If the information is not available in the source, provide an appropriate default value:
  * 0.0 for numeric fields (float/int)
  * Empty string "" for text fields
  * false for boolean fields
- If the field represents "unknown" or "not applicable", use a sensible placeholder
- **Never use null/None for required fields**

Previous response that failed:
{failed_response}

Please provide a corrected response with all required fields populated."""

    # Create retry messages
    retry_messages = list(original_messages)  # Copy original messages
    retry_messages.append(HumanMessage(content=feedback_prompt))

    # Invoke LLM with retry
    try:
        raw_response, _, usage_metadata, _ = _invoke_llm_with_retry(parsing_llm, retry_messages, is_agent=False)

        # Track usage if tracker provided
        if usage_tracker and usage_metadata and model_str:
            usage_tracker.track_call("parsing_null_retry", model_str, usage_metadata)

        # Try parsing again
        cleaned = _strip_markdown_fences(raw_response)
        parsed = parser.parse(cleaned)

        logger.info(f"✓ Successfully parsed after null-value retry. Fixed fields: {field_list}")
        return parsed, usage_metadata

    except Exception as e:
        logger.warning(f"✗ Retry parsing failed after null-value feedback: {e}")
        return None, {}
