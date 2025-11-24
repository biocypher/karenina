"""Verification utility functions.

Shared utilities used by verification stages and runner.
These functions handle common tasks like LLM retries, prompt composition,
response parsing, and validation.
"""

import json
import logging
import os
import re
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.callbacks.usage import UsageMetadataCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Set up logger
logger = logging.getLogger(__name__)

# ============================================================================
# Tool Call Failure Detection Patterns
# ============================================================================

# Regex patterns to detect suspected tool failures in tool message content
# Compiled with case-insensitive flag for better matching
TOOL_FAILURE_PATTERNS = [
    # Error indicators
    re.compile(r"\berror\b", re.IGNORECASE),
    re.compile(r"\bfailed\b", re.IGNORECASE),
    re.compile(r"\bexception\b", re.IGNORECASE),
    re.compile(r"\btraceback\b", re.IGNORECASE),
    re.compile(r"\bstack\s+trace\b", re.IGNORECASE),
    # HTTP errors
    re.compile(r"\b404\b", re.IGNORECASE),
    re.compile(r"\b500\b", re.IGNORECASE),
    re.compile(r"\b502\b", re.IGNORECASE),
    re.compile(r"\b503\b", re.IGNORECASE),
    re.compile(r"\btimeout\b", re.IGNORECASE),
    # API failures
    re.compile(r"\binvalid\b", re.IGNORECASE),
    re.compile(r"\bunauthorized\b", re.IGNORECASE),
    re.compile(r"\bforbidden\b", re.IGNORECASE),
    re.compile(r"\bnot\s+found\b", re.IGNORECASE),
    re.compile(r"\bcannot\b", re.IGNORECASE),
    re.compile(r"\bunable\s+to\b", re.IGNORECASE),
]


# ============================================================================
# Error Handling and Retry Logic
# ============================================================================


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


def _extract_agent_metrics(response: Any) -> dict[str, Any] | None:
    """
    Extract agent execution metrics from LangGraph agent response.

    This function analyzes agent messages to track:
    - Iterations (AI message cycles)
    - Tool calls (successful tool invocations)
    - Tools used (unique tool names)
    - Suspected failed tool calls (tools with error-like output patterns)

    Args:
        response: Agent response object from LangGraph (dict with "messages" key)

    Returns:
        Dict with agent metrics:
        - iterations: Number of AI message cycles
        - tool_calls: Total tool invocations
        - tools_used: Sorted list of unique tool names
        - suspect_failed_tool_calls: Count of tool calls with error-like patterns
        - suspect_failed_tools: Sorted list of tools with suspected failures
        Returns None if extraction fails
    """
    if not response or not isinstance(response, dict):
        return None

    messages = response.get("messages", [])
    if not messages:
        return None

    # Count iterations (AI message cycles)
    iterations = 0
    tool_calls = 0
    tools_used = set()
    suspect_failed_tool_calls = 0
    suspect_failed_tools = set()

    for msg in messages:
        # Check message type
        msg_type = getattr(msg, "__class__", None)
        if msg_type:
            type_name = msg_type.__name__

            # Count AI messages as iterations
            if type_name == "AIMessage":
                iterations += 1

            # Count tool messages and extract tool names
            elif type_name == "ToolMessage":
                tool_calls += 1
                # Extract tool name from ToolMessage
                tool_name = getattr(msg, "name", None)
                if tool_name:
                    tools_used.add(tool_name)

                # Check for suspected failures in tool output
                is_suspect_failure = False

                # Check content field for error patterns
                content = getattr(msg, "content", None)
                if content and isinstance(content, str):
                    # Test against all failure patterns
                    for pattern in TOOL_FAILURE_PATTERNS:
                        if pattern.search(content):
                            is_suspect_failure = True
                            break

                # Check status field if available (some tool messages have status)
                if not is_suspect_failure:
                    status = getattr(msg, "status", None)
                    if status and isinstance(status, str) and status.lower() in ["error", "failed", "failure"]:
                        is_suspect_failure = True

                # Track suspected failure
                if is_suspect_failure:
                    suspect_failed_tool_calls += 1
                    if tool_name:
                        suspect_failed_tools.add(tool_name)

    return {
        "iterations": iterations,
        "tool_calls": tool_calls,
        "tools_used": sorted(tools_used),  # Sort for deterministic output
        "suspect_failed_tool_calls": suspect_failed_tool_calls,
        "suspect_failed_tools": sorted(suspect_failed_tools),  # Sort for deterministic output
    }


def _invoke_llm_with_retry(
    llm: Any, messages: list[BaseMessage], is_agent: bool, timeout: int = 120
) -> tuple[Any, bool, dict[str, Any], dict[str, Any] | None]:
    """
    Invoke LLM with automatic retry logic for transient errors.

    Args:
        llm: The LLM or agent to invoke
        messages: List of messages to send to the LLM
        is_agent: Whether the LLM is a LangGraph agent
        timeout: Timeout in seconds for agent invocation

    Returns:
        Tuple of (response, recursion_limit_reached, usage_metadata, agent_metrics)
        - response: The LLM/agent response
        - recursion_limit_reached: Whether agent hit recursion limit
        - usage_metadata: Token usage metadata from LangChain callback
        - agent_metrics: Agent execution metrics (iterations, tool_calls, tools_used) or None for non-agents

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
    def _invoke_with_retry_inner() -> tuple[Any, bool, dict[str, Any], dict[str, Any] | None]:
        recursion_limit_reached = False
        usage_metadata: dict[str, Any] = {}
        agent_metrics = None

        try:
            if is_agent:
                # LangGraph agents with MCP tools need async invocation
                import asyncio

                async def invoke_agent_async() -> tuple[Any, dict[str, Any], bool]:
                    """
                    Invoke agent and return (response, usage_metadata, recursion_limit_reached).
                    This avoids using 'nonlocal' which can cause issues with nested async functions.
                    """
                    cb_manager = None
                    cb: UsageMetadataCallbackHandler | None = None
                    accumulated_messages = list(messages)  # Start with input messages
                    last_known_state = None
                    local_usage_metadata: dict[str, Any] = {}
                    local_recursion_limit_reached = False

                    try:
                        # Wrap async invoke with usage metadata callback
                        cb_manager = get_usage_metadata_callback()
                        cb = cb_manager.__enter__()

                        # Use streaming to capture messages as they're generated
                        # This ensures we preserve the full trace even if recursion limit is hit
                        try:
                            async for chunk in llm.astream({"messages": messages}):
                                # Each chunk is a state update from the agent
                                # Accumulate the latest state to have full trace even on error
                                last_known_state = chunk
                                # Extract messages from chunk - handle both flat and nested structures
                                if isinstance(chunk, dict):
                                    # Nested structure from astream: {'agent': {'messages': [...]}}
                                    if (
                                        "agent" in chunk
                                        and isinstance(chunk["agent"], dict)
                                        and "messages" in chunk["agent"]
                                    ):
                                        accumulated_messages = chunk["agent"]["messages"]
                                    # Flat structure: {'messages': [...]}
                                    elif "messages" in chunk:
                                        accumulated_messages = chunk["messages"]
                        except StopAsyncIteration:
                            pass  # Stream completed normally

                        # Use final state if we have it
                        response = (
                            last_known_state if last_known_state is not None else {"messages": accumulated_messages}
                        )

                        # Capture usage metadata on success
                        local_usage_metadata = dict(cb.usage_metadata) if cb and cb.usage_metadata else {}
                        cb_manager.__exit__(None, None, None)
                        return response, local_usage_metadata, local_recursion_limit_reached

                    except Exception as e:
                        # CRITICAL: Capture usage metadata BEFORE handling exception
                        # This ensures we track tokens even when recursion limit is hit
                        if cb and cb.usage_metadata:
                            local_usage_metadata = dict(cb.usage_metadata)
                            # Close the callback context
                            from contextlib import suppress

                            if cb_manager:
                                with suppress(Exception):
                                    cb_manager.__exit__(type(e), e, e.__traceback__)

                        # Check if this is a GraphRecursionError
                        if "GraphRecursionError" in str(type(e).__name__) or "recursion_limit" in str(e).lower():
                            local_recursion_limit_reached = True

                            # CRITICAL: Return accumulated messages from streaming
                            # This preserves the full trace up to the point where recursion limit was hit
                            if accumulated_messages and len(accumulated_messages) > len(messages):
                                logger.info(
                                    f"Recursion limit hit. Returning accumulated trace with "
                                    f"{len(accumulated_messages)} messages (started with {len(messages)})"
                                )
                                return (
                                    {"messages": accumulated_messages},
                                    local_usage_metadata,
                                    local_recursion_limit_reached,
                                )

                            # Fallback methods if streaming didn't capture messages
                            # Method 1: Check if exception contains state information
                            if hasattr(e, "state") and e.state is not None:
                                logger.info("Extracted partial state from GraphRecursionError.state")
                                return e.state, local_usage_metadata, local_recursion_limit_reached

                            # Method 2: Try to get current graph state if checkpointer exists
                            if hasattr(llm, "checkpointer") and llm.checkpointer is not None:
                                try:
                                    if hasattr(llm, "get_state"):
                                        config = {"configurable": {"thread_id": "default"}}
                                        state = llm.get_state(config)
                                        if state and hasattr(state, "values") and "messages" in state.values:
                                            logger.info("Extracted partial state from graph checkpointer")
                                            return (
                                                {"messages": state.values["messages"]},
                                                local_usage_metadata,
                                                local_recursion_limit_reached,
                                            )
                                except Exception as state_error:
                                    logger.debug(f"Could not extract state from checkpointer: {state_error}")

                            # Method 3: Check if exception has accumulated messages attribute
                            if hasattr(e, "messages"):
                                logger.info("Extracted messages from exception.messages attribute")
                                return {"messages": e.messages}, local_usage_metadata, local_recursion_limit_reached

                            # FALLBACK: Return input messages with warning
                            logger.warning(
                                "Could not extract partial agent state after recursion limit. "
                                "Returning input messages only. Accumulated trace may be lost."
                            )
                            return {"messages": messages}, local_usage_metadata, local_recursion_limit_reached
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
                        result = future.result(timeout=timeout)
                        response, usage_metadata, recursion_limit_reached = result

                except RuntimeError:
                    # No event loop running, safe to use asyncio.run
                    response, usage_metadata, recursion_limit_reached = asyncio.run(invoke_agent_async())

                # Extract agent metrics before harmonization
                agent_metrics = _extract_agent_metrics(response)

                from ...infrastructure.llm.mcp_utils import harmonize_agent_response

                return harmonize_agent_response(response), recursion_limit_reached, usage_metadata, agent_metrics
            else:
                # Regular LLMs expect the messages list directly
                # Wrap invoke with usage metadata callback
                with get_usage_metadata_callback() as cb:
                    response = llm.invoke(messages)
                usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}
                # Extract content from AIMessage for consistency with agent path
                raw_response = response.content if hasattr(response, "content") else str(response)
                # Non-agents don't have agent metrics
                return raw_response, recursion_limit_reached, usage_metadata, None

        except Exception as e:
            # Check if this is a retryable error
            if is_retryable_error(e):
                logger.info(f"Detected retryable error: {type(e).__name__}: {e}")
                raise  # Re-raise to trigger retry
            else:
                # Non-retryable error, don't retry
                raise

    return _invoke_with_retry_inner()


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


# ============================================================================
# Configuration and Feature Flags
# ============================================================================


def _should_expose_ground_truth() -> bool:
    """
    Check if ground truth should be exposed to the parser model.

    Reads from the KARENINA_EXPOSE_GROUND_TRUTH environment variable.
    Defaults to False for backward compatibility.

    Returns:
        True if ground truth should be exposed, False otherwise
    """
    return os.getenv("KARENINA_EXPOSE_GROUND_TRUTH", "false").lower() in ("true", "1", "yes", "on")


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
    from .utils.parsing import _strip_markdown_fences

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
