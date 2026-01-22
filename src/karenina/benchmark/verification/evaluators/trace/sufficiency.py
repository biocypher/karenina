"""Sufficiency detection for identifying when traces lack information to populate templates."""

import json
import logging
from functools import partial
from typing import Any

from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .....adapters import get_llm
from .....ports import LLMResponse, Message
from .....schemas.workflow import ModelConfig
from .....utils.errors import is_retryable_error
from .....utils.retry import log_retry
from ...utils.llm_judge_helpers import extract_judge_result, fallback_json_parse
from ...utils.prompts import SUFFICIENCY_DETECTION_SYS, SUFFICIENCY_DETECTION_USER

# Set up logger
logger = logging.getLogger(__name__)


class SufficiencyResult(BaseModel):
    """Result of sufficiency detection by LLM judge."""

    reasoning: str = Field(
        description="For each field, explain whether information exists. End with overall determination."
    )
    sufficient: bool = Field(description="True if response has info for all fields, False if information missing")


def detect_sufficiency(
    raw_llm_response: str,
    parsing_model: ModelConfig,
    question_text: str,
    template_schema: dict[str, Any],
) -> tuple[bool, bool, str | None, dict[str, Any]]:
    """
    Detect if the response contains sufficient information to populate the template schema.

    This function uses an LLM to analyze the response against the template schema and
    determine if all required fields can be populated. Uses retry logic for transient
    errors (connection issues, rate limits, etc.).

    Args:
        raw_llm_response: The raw response text from the answering model
        parsing_model: Configuration for the model to use for sufficiency detection
        question_text: The original question that was asked
        template_schema: The JSON schema of the answer template to populate

    Returns:
        Tuple of (sufficient, check_performed, reasoning, usage_metadata):
        - sufficient: True if response has info for all fields, False if information missing
        - check_performed: True if check completed successfully, False if check failed
        - reasoning: The LLM's explanation for its determination (None if check failed)
        - usage_metadata: Token usage metadata from the LLM invocation

    Examples:
        >>> config = ModelConfig(id="parser", model_provider="openai", ...)
        >>> schema = {"properties": {"answer": {"type": "string"}}}
        >>> sufficient, performed, reasoning, metadata = detect_sufficiency(
        ...     "The answer is 42", config, "What is X?", schema
        ... )
        >>> print(sufficient, performed)
        True, True
    """

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),  # Try 3 times
        wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff: 2s, 4s, 8s
        reraise=True,
        before_sleep=partial(log_retry, context="sufficiency detection"),
    )
    def _detect_with_retry() -> tuple[bool, bool, str | None, dict[str, Any]]:
        """Inner function with retry logic."""
        usage_metadata: dict[str, Any] = {}

        try:
            # Create config copy with temperature=0 for consistent detection
            # Note: LangChain adapter respects temperature; Claude SDK adapter ignores it
            detection_config = parsing_model.model_copy(update={"temperature": 0.0})

            # Get LLM via adapter factory
            llm = get_llm(detection_config)

            # Configure for structured output
            structured_llm = llm.with_structured_output(SufficiencyResult)

            # Convert schema to string for prompt
            schema_str = json.dumps(template_schema, indent=2)

            # Build messages using unified Message class
            user_prompt = SUFFICIENCY_DETECTION_USER.format(
                question=question_text,
                response=raw_llm_response,
                schema=schema_str,
            )
            messages = [
                Message.system(SUFFICIENCY_DETECTION_SYS),
                Message.user(user_prompt),
            ]

            # Invoke with structured output
            response: LLMResponse = structured_llm.invoke(messages)
            usage_metadata = response.usage.to_dict()

            # Extract result from structured output or fall back to manual parsing
            result = extract_judge_result(response, SufficiencyResult, "sufficient")
            if result is not None:
                logger.debug(f"Sufficiency check: {result.sufficient} - Reasoning: {result.reasoning}")
                return result.sufficient, True, result.reasoning, usage_metadata

            # Fallback: manual JSON parsing from content
            return fallback_json_parse(
                response.content, usage_metadata, "sufficient", True, "Sufficiency check (fallback)"
            )

        except json.JSONDecodeError as e:
            # JSON parsing failed - log and treat as check failure
            logger.warning(f"Failed to parse sufficiency detection response as JSON: {e}")
            return True, False, None, usage_metadata  # Default to sufficient on failure

        except Exception as e:
            # Check if this is a retryable error
            if is_retryable_error(e):
                logger.info(f"Detected retryable error in sufficiency check: {type(e).__name__}: {e}")
                raise  # Re-raise to trigger retry
            else:
                # Non-retryable error - log and treat as check failure
                logger.warning(f"Sufficiency detection failed with non-retryable error: {e}")
                return True, False, None, usage_metadata  # Default to sufficient on failure

    try:
        return _detect_with_retry()
    except Exception as e:
        # All retries exhausted or unhandled error
        logger.error(f"Sufficiency detection failed after all retries: {e}")
        return True, False, None, {}  # Default to sufficient on failure
