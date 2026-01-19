"""LangChain LLM adapter implementing the LLMPort interface.

This module provides the LangChainLLMAdapter class that wraps existing
LangChain infrastructure (init_chat_model) behind the unified LLMPort interface.

Retry Logic:
    This adapter includes tenacity-based retry for transient errors (connection
    errors, timeouts, rate limits, 5xx errors). Retry is applied to both
    ainvoke() and with_structured_output() calls with exponential backoff.

Structured Output Fallback:
    When the underlying model doesn't support with_structured_output(), the
    adapter falls back to manual JSON parsing with format instructions.

Module Organization:
    - Constants & Configuration: Retry decorator, error detection
    - Helper Functions: JSON extraction, logging
    - Main Adapter Class: LangChainLLMAdapter
    - Protocol Verification: Static compliance check
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from karenina.ports import LLMPort, LLMResponse, Message, UsageMetadata

from .messages import LangChainMessageConverter

logger = logging.getLogger(__name__)

# =============================================================================
# Constants & Configuration
# =============================================================================


def _extract_json_from_response(text: str) -> str:
    """Extract JSON from a response that may be wrapped in markdown code blocks.

    Args:
        text: Raw response text that may contain JSON.

    Returns:
        Extracted JSON string.

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    # Try direct parsing first
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return text

    # Try to extract from markdown code blocks
    import re

    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    matches: list[str] = re.findall(code_block_pattern, text)
    if matches:
        for match in matches:
            extracted = match.strip()
            if extracted.startswith("{") or extracted.startswith("["):
                return extracted

    # Last resort: find JSON-like content
    json_pattern = r"\{[\s\S]*\}"
    json_match = re.search(json_pattern, text)
    if json_match:
        return json_match.group()

    raise ValueError(f"Could not extract JSON from response: {text[:200]}...")


def _is_retryable_error(exception: BaseException) -> bool:
    """Check if an exception is retryable (transient error).

    Args:
        exception: The exception to check

    Returns:
        True if the error is transient and should be retried, False otherwise
    """
    exception_str = str(exception).lower()
    exception_type = type(exception).__name__

    # Connection-related errors (check error message content)
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
            "overloaded",
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
        "OverloadedError",
        "InternalServerError",
    ]

    return exception_type in retryable_types


def _log_retry(retry_state: Any) -> None:
    """Log retry attempt with error details."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(f"Retrying LLM call (attempt {retry_state.attempt_number}/3) after error: {exc}")


# Reusable retry decorator for transient errors (connection, timeout, rate limit, 5xx)
_TRANSIENT_RETRY = retry(
    retry=retry_if_exception(_is_retryable_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    before_sleep=_log_retry,
)


# =============================================================================
# Main Adapter Class
# =============================================================================

if TYPE_CHECKING:
    from karenina.schemas.workflow.models import ModelConfig


class LangChainLLMAdapter:
    """LLM adapter using LangChain's init_chat_model.

    This adapter implements the LLMPort Protocol and wraps the existing
    karenina infrastructure for LangChain-based LLM invocation.

    The adapter handles:
    - Message conversion between unified Message and LangChain formats
    - Usage metadata extraction from LangChain responses
    - Sync/async invocation with proper event loop handling
    - Structured output via with_structured_output()

    Example:
        >>> from karenina.schemas.workflow.models import ModelConfig
        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic"
        ... )
        >>> adapter = LangChainLLMAdapter(config)
        >>> response = await adapter.ainvoke([Message.user("Hello!")])
        >>> print(response.content)
        'Hello! How can I help you today?'

        >>> # With structured output
        >>> class Answer(BaseModel):
        ...     value: str
        >>> structured = adapter.with_structured_output(Answer)
        >>> response = await structured.ainvoke([Message.user("What is 2+2?")])
    """

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        _structured_schema: type[BaseModel] | None = None,
        _structured_model: Any = None,
        _base_model: Any = None,
        _max_retries: int = 0,
    ) -> None:
        """Initialize the LangChain LLM adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
            _structured_schema: Internal - schema for structured output mode.
            _structured_model: Internal - model configured with with_structured_output().
            _base_model: Internal - original model for fallback parsing.
            _max_retries: Internal - max validation retries with error feedback.
        """
        self._config = model_config
        self._converter = LangChainMessageConverter()
        self._structured_schema = _structured_schema
        self._structured_model = _structured_model
        self._base_model = _base_model
        self._max_retries = _max_retries

        if _base_model is not None:
            self._model = _structured_model if _structured_model else _base_model
        else:
            self._model = self._initialize_model()

    def _initialize_model(self) -> Any:
        """Initialize the underlying LangChain model.

        Returns:
            Initialized LangChain chat model.
        """
        from karenina.infrastructure.llm.interface import init_chat_model_unified

        # Build kwargs for model initialization
        kwargs: dict[str, Any] = {}

        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature

        if self._config.extra_kwargs:
            kwargs.update(self._config.extra_kwargs)

        # Initialize model via existing infrastructure
        model = init_chat_model_unified(
            model=self._config.model_name or "",
            provider=self._config.model_provider,
            interface=self._config.interface,
            endpoint_base_url=self._config.endpoint_base_url,
            endpoint_api_key=self._config.endpoint_api_key,
            max_context_tokens=self._config.max_context_tokens,
            # Note: We don't pass MCP config here - simple LLM invocation
            # For agents with MCP, use LangChainAgentAdapter instead
            **kwargs,
        )

        return model

    def _extract_usage(self, response: Any) -> UsageMetadata:
        """Extract usage metadata from LangChain response.

        Args:
            response: LangChain AIMessage or similar response object.

        Returns:
            UsageMetadata with token counts and cost info.
        """
        # LangChain stores usage in response_metadata or usage_metadata attribute
        usage_data: dict[str, Any] = {}

        # Try response_metadata first (newer LangChain versions)
        if hasattr(response, "response_metadata"):
            metadata = response.response_metadata or {}
            # Anthropic/OpenAI style: token_usage or usage
            usage_data = metadata.get("token_usage") or metadata.get("usage") or {}

        # Fall back to usage_metadata attribute
        if not usage_data and hasattr(response, "usage_metadata"):
            usage_data = response.usage_metadata or {}

        input_tokens = usage_data.get("input_tokens") or usage_data.get("prompt_tokens") or 0
        output_tokens = usage_data.get("output_tokens") or usage_data.get("completion_tokens") or 0
        total_tokens = usage_data.get("total_tokens") or (input_tokens + output_tokens)

        # Extract cache tokens if available (Anthropic)
        cache_read = usage_data.get("cache_read_input_tokens") or usage_data.get("cache_read_tokens")
        cache_creation = usage_data.get("cache_creation_input_tokens") or usage_data.get("cache_creation_tokens")

        return UsageMetadata(
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            total_tokens=int(total_tokens),
            cache_read_tokens=int(cache_read) if cache_read else None,
            cache_creation_tokens=int(cache_creation) if cache_creation else None,
            model=self._config.model_name,
        )

    async def ainvoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM asynchronously with automatic retry for transient errors.

        Uses tenacity for exponential backoff retry logic on transient errors
        (connection errors, timeouts, rate limits, 5xx errors).

        For structured output mode, tries native with_structured_output() first.
        If that fails, automatically falls back to manual JSON parsing.

        Args:
            messages: List of unified Message objects.

        Returns:
            LLMResponse with content, usage metadata, and raw response.
            When using structured output (via with_structured_output()), the
            raw field contains the parsed Pydantic model instance.

        Raises:
            PortError: If the invocation fails after all retries.
            ValidationError: If structured output parsing fails.
        """
        # For structured output mode, try native first then fallback
        if self._structured_schema is not None:
            return await self._ainvoke_structured(messages)

        # Regular text mode
        return await self._ainvoke_text(messages)

    async def _ainvoke_text(self, messages: list[Message]) -> LLMResponse:
        """Invoke LLM for regular text output."""
        lc_messages = self._converter.to_provider(messages)
        response = await self._invoke_model_with_retry(self._model, lc_messages)

        content = str(response.content) if hasattr(response, "content") else str(response)
        usage = self._extract_usage(response)

        return LLMResponse(content=content, usage=usage, raw=response)

    async def _invoke_model_with_retry(self, model: Any, lc_messages: list[Any]) -> Any:
        """Invoke a LangChain model with automatic retry for transient errors.

        This is a helper that applies the standard transient retry policy
        to any model invocation.

        Args:
            model: The LangChain model to invoke.
            lc_messages: LangChain-formatted messages.

        Returns:
            The model's response.

        Raises:
            Exception: After all retries are exhausted.
        """

        @_TRANSIENT_RETRY
        async def _invoke() -> Any:
            return await model.ainvoke(lc_messages)

        return await _invoke()

    async def _ainvoke_structured(self, messages: list[Message]) -> LLMResponse:
        """Invoke LLM for structured output, with automatic fallback and retry.

        Tries native with_structured_output() first. If that fails (model doesn't
        support it properly), falls back to manual JSON parsing.

        When max_retries > 0, validation errors trigger retry with error feedback
        appended to the messages, giving the LLM a chance to fix issues.
        """
        if self._structured_schema is None:
            raise RuntimeError("_ainvoke_structured called without structured schema")

        last_error: str | None = None

        for attempt in range(self._max_retries + 1):
            # Add error feedback on retry attempts
            effective_messages = (
                self._add_retry_feedback(messages, last_error) if attempt > 0 and last_error else messages
            )

            try:
                return await self._try_structured_invocation(effective_messages)
            except (ValidationError, ValueError) as e:
                last_error = str(e)
                if attempt == self._max_retries:
                    raise ValueError(
                        f"Structured output failed after {self._max_retries + 1} attempts. Last error: {last_error}"
                    ) from None
                logger.info(
                    f"Structured output validation failed (attempt {attempt + 1}/{self._max_retries + 1}): {e}. "
                    "Retrying with error feedback."
                )

        raise ValueError("Unexpected error in retry logic")

    async def _try_structured_invocation(self, messages: list[Message]) -> LLMResponse:
        """Try native structured output, falling back to manual parsing if needed.

        Args:
            messages: Messages to send to the LLM.

        Returns:
            LLMResponse with parsed structured output.

        Raises:
            ValidationError: If parsing fails.
            Exception: For transient errors (will be retried by caller).
        """
        # Try native structured output first (if available)
        if self._structured_model is not None:
            try:
                return await self._ainvoke_native_structured(messages)
            except Exception as e:
                if _is_retryable_error(e):
                    raise  # Let transient errors propagate
                logger.warning(
                    f"Native structured output failed for {self._config.model_name}: {e}. "
                    "Falling back to manual JSON parsing."
                )

        # Fallback to manual JSON parsing
        return await self._ainvoke_with_fallback_parsing(messages)

    def _add_retry_feedback(self, messages: list[Message], error: str) -> list[Message]:
        """Add retry feedback to messages after a validation failure.

        Appends a user message with the validation error, giving the LLM
        context to fix the issue on the next attempt.

        Args:
            messages: Original messages.
            error: The validation error message.

        Returns:
            New message list with error feedback appended.
        """
        feedback = Message.user(
            f"PREVIOUS ATTEMPT FAILED with error: {error}\nPlease fix the validation issues and try again."
        )
        return list(messages) + [feedback]

    async def _ainvoke_native_structured(self, messages: list[Message]) -> LLMResponse:
        """Invoke using native with_structured_output()."""
        lc_messages = self._converter.to_provider(messages)
        response = await self._invoke_model_with_retry(self._structured_model, lc_messages)

        # Native structured output - response should be a Pydantic model
        if isinstance(response, BaseModel):
            return LLMResponse(
                content=str(response),
                usage=self._extract_usage(response),
                raw=response,
            )

        # Unexpected response type - raise to trigger fallback
        raise TypeError(f"Native structured output returned unexpected type: {type(response).__name__}")

    async def _ainvoke_with_fallback_parsing(self, messages: list[Message]) -> LLMResponse:
        """Invoke LLM with manual JSON parsing for structured output.

        Used when native with_structured_output() fails or is unavailable.
        Adds format instructions to the prompt and parses the JSON response manually.

        Args:
            messages: List of unified Message objects.

        Returns:
            LLMResponse with the parsed Pydantic model in raw.

        Raises:
            ValidationError: If the response cannot be parsed into the schema.
        """
        if self._structured_schema is None:
            raise RuntimeError("Fallback parsing requires a structured schema")

        # Augment messages with format instructions
        augmented_messages = self._augment_with_format_instructions(messages)

        # Use base model for fallback (not the structured model)
        model_to_use = self._base_model if self._base_model is not None else self._model
        lc_messages = self._converter.to_provider(augmented_messages)
        response = await self._invoke_model_with_retry(model_to_use, lc_messages)

        # Extract and parse response
        text_content = str(response.content) if hasattr(response, "content") else str(response)
        parsed_model = self._parse_json_response(text_content)

        return LLMResponse(
            content=text_content,
            usage=self._extract_usage(response),
            raw=parsed_model,
        )

    def _augment_with_format_instructions(self, messages: list[Message]) -> list[Message]:
        """Append JSON schema format instructions to the last user message.

        Args:
            messages: Original messages.

        Returns:
            New message list with format instructions appended to the last user message.
        """
        from karenina.ports.messages import TextContent

        if self._structured_schema is None:
            return messages

        schema_json = json.dumps(self._structured_schema.model_json_schema(), indent=2)
        format_instruction = (
            f"\n\nYou must respond with valid JSON that matches this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Return ONLY the JSON object, no additional text."
        )

        augmented = list(messages)
        for i in range(len(augmented) - 1, -1, -1):
            msg = augmented[i]
            if msg.role.value == "user":
                new_content = [
                    TextContent(text=c.text + format_instruction) if isinstance(c, TextContent) else c
                    for c in msg.content
                ]
                augmented[i] = Message(role=msg.role, content=new_content)
                break

        return augmented

    def _parse_json_response(self, text_content: str) -> BaseModel:
        """Parse JSON from LLM response into the structured schema.

        Args:
            text_content: Raw response text that may contain JSON.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ValidationError: If parsing or validation fails.
        """
        if self._structured_schema is None:
            raise RuntimeError("Cannot parse without structured schema")

        try:
            json_str = _extract_json_from_response(text_content)
            return self._structured_schema.model_validate_json(json_str)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse structured output: {e}\nResponse: {text_content[:500]}")
            raise

    def invoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM synchronously.

        This is a convenience wrapper around ainvoke() for sync code.
        Uses the shared async portal if available, otherwise falls back
        to asyncio.run() with proper event loop handling.

        Args:
            messages: List of unified Message objects.

        Returns:
            LLMResponse with content, usage metadata, and raw response.

        Raises:
            PortError: If the invocation fails.
        """
        from karenina.benchmark.verification.batch_runner import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            # Use the shared BlockingPortal for proper event loop management
            return portal.call(self.ainvoke, messages)

        # No portal available - check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - use ThreadPoolExecutor to avoid
            # nested event loop issues

            def run_in_thread() -> LLMResponse:
                return asyncio.run(self.ainvoke(messages))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=300)  # 5 minute timeout

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.ainvoke(messages))

    def with_structured_output(self, schema: type[BaseModel], *, max_retries: int | None = None) -> LangChainLLMAdapter:
        """Return a new adapter configured for structured output.

        The returned adapter tries native with_structured_output() first at invoke time.
        If that fails, it automatically falls back to manual JSON parsing.

        When max_retries > 0, validation errors trigger retry with error feedback
        appended to the messages, giving the LLM a chance to fix issues.

        Args:
            schema: A Pydantic model class defining the output structure.
            max_retries: Maximum number of retry attempts on validation failure.
                Each retry includes error feedback to help the LLM fix issues.
                Default is 3 retries.

        Returns:
            A new LangChainLLMAdapter instance configured for structured output.
            The returned adapter guarantees that response.raw will be an instance
            of the provided schema.

        Example:
            >>> class Answer(BaseModel):
            ...     value: str
            ...     confidence: float
            >>> structured = adapter.with_structured_output(Answer, max_retries=2)
            >>> response = await structured.ainvoke(messages)
            >>> assert isinstance(response.raw, Answer)
        """
        # Try to create a structured model for native support
        structured_model = None
        if hasattr(self._model, "with_structured_output"):
            try:
                structured_model = self._model.with_structured_output(schema)
            except Exception as e:
                # Creation failed - will use fallback at invoke time
                logger.debug(
                    f"Could not create structured model for {self._config.model_name}: {e}. "
                    "Will use fallback parsing at invoke time."
                )

        return LangChainLLMAdapter(
            model_config=self._config,
            _structured_schema=schema,
            _structured_model=structured_model,
            _base_model=self._model,  # Keep base model for fallback
            _max_retries=max_retries if max_retries is not None else 3,
        )


# =============================================================================
# Protocol Verification
# =============================================================================


def _verify_protocol_compliance() -> None:
    """Verify LangChainLLMAdapter implements LLMPort protocol."""
    # This is a static check - if this fails, it means the adapter
    # doesn't properly implement the protocol
    adapter_instance: LLMPort = None  # type: ignore[assignment]

    # The following would fail mypy if the protocol wasn't properly implemented
    # We don't actually run this, it's just for static analysis
    _ = adapter_instance  # Suppress unused warning
