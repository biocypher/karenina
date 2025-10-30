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
from langchain_core.messages import BaseMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Set up logger
logger = logging.getLogger(__name__)


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

    Args:
        response: Agent response object from LangGraph (dict with "messages" key)

    Returns:
        Dict with agent metrics (iterations, tool_calls, tools_used) or None if extraction fails
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

    return {
        "iterations": iterations,
        "tool_calls": tool_calls,
        "tools_used": sorted(tools_used),  # Sort for deterministic output
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

                async def invoke_agent_async() -> Any:
                    try:
                        # Wrap async invoke with usage metadata callback
                        with get_usage_metadata_callback() as cb:
                            response = await llm.ainvoke({"messages": messages})
                        # Capture usage metadata
                        nonlocal usage_metadata
                        usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}
                        return response
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
                # Non-agents don't have agent metrics
                return response, recursion_limit_reached, usage_metadata, None

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
