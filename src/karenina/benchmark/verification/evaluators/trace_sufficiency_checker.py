"""Sufficiency detection for identifying when traces lack information to populate templates."""

import json
import logging
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ....infrastructure.llm.interface import init_chat_model_unified
from ....schemas.workflow import ModelConfig
from ..utils.error_helpers import is_retryable_error
from ..utils.json_helpers import strip_markdown_fences as _strip_markdown_fences
from ..utils.prompts import SUFFICIENCY_DETECTION_SYS, SUFFICIENCY_DETECTION_USER

# Set up logger
logger = logging.getLogger(__name__)


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
        - usage_metadata: Token usage metadata from LangChain callback

    Examples:
        >>> config = ModelConfig(id="parser", model_provider="openai", ...)
        >>> schema = {"properties": {"answer": {"type": "string"}}}
        >>> sufficient, performed, reasoning, metadata = detect_sufficiency(
        ...     "The answer is 42", config, "What is X?", schema
        ... )
        >>> print(sufficient, performed)
        True, True
    """

    def _log_retry(retry_state: Any) -> None:
        """Log retry attempt with error details."""
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(f"Retrying sufficiency detection (attempt {retry_state.attempt_number}/3) after error: {exc}")

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),  # Try 3 times
        wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff: 2s, 4s, 8s
        reraise=True,
        before_sleep=_log_retry,
    )
    def _detect_with_retry() -> tuple[bool, bool, str | None, dict[str, Any]]:
        """Inner function with retry logic."""
        usage_metadata: dict[str, Any] = {}

        try:
            # Initialize the parsing model for sufficiency detection
            # Note: model_name is guaranteed non-None by ModelConfig validator
            assert parsing_model.model_name is not None, "model_name must not be None"

            # Build kwargs for model initialization
            model_kwargs: dict[str, Any] = {
                "model": parsing_model.model_name,
                "provider": parsing_model.model_provider,
                "temperature": 0.0,  # Use temperature 0 for consistent detection
                "interface": parsing_model.interface,
            }

            # Add endpoint configuration if using openai_endpoint interface
            if parsing_model.endpoint_base_url:
                model_kwargs["endpoint_base_url"] = parsing_model.endpoint_base_url
            if parsing_model.endpoint_api_key:
                model_kwargs["endpoint_api_key"] = parsing_model.endpoint_api_key

            # Add any extra kwargs if provided (e.g., vendor-specific API keys)
            if parsing_model.extra_kwargs:
                model_kwargs.update(parsing_model.extra_kwargs)

            llm = init_chat_model_unified(**model_kwargs)

            # Convert schema to string for prompt
            schema_str = json.dumps(template_schema, indent=2)

            # Construct the prompt
            user_prompt = SUFFICIENCY_DETECTION_USER.format(
                question=question_text,
                response=raw_llm_response,
                schema=schema_str,
            )

            messages = [
                SystemMessage(content=SUFFICIENCY_DETECTION_SYS),
                HumanMessage(content=user_prompt),
            ]

            # Invoke the LLM with usage metadata callback
            with get_usage_metadata_callback() as cb:
                response = llm.invoke(messages)
            usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}

            raw_response = response.content if hasattr(response, "content") else str(response)

            # Parse the JSON response
            cleaned_response = _strip_markdown_fences(raw_response)
            # raw_response is always a str, so cleaned_response will be str
            result = json.loads(cleaned_response)  # type: ignore[arg-type]

            # Extract the sufficiency determination
            sufficient = result.get("sufficient", True)  # Default to sufficient if missing

            # Log the reasoning for debugging
            reasoning = result.get("reasoning", "No reasoning provided")
            logger.debug(f"Sufficiency check result: {sufficient} - Reasoning: {reasoning}")

            return sufficient, True, reasoning, usage_metadata

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
