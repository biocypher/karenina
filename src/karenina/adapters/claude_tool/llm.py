"""Claude Tool LLM adapter implementing the LLMPort interface.

This module provides the ClaudeToolLLMAdapter class that uses the Anthropic Python SDK
directly (client.messages.create) for simple LLM invocations without agent loops.

Key features:
- Uses Anthropic's native Python SDK for efficient API calls
- Supports structured output via client.beta.messages.parse() with Pydantic
- Implements prompt caching for efficiency
- Derives SDK max_retries from RetryPolicy for consistent retry budgets
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel

from karenina.adapters._parallel_base import with_llm_semaphore
from karenina.ports import AdapterUnavailableError, LLMPort, LLMResponse, Message, ParseError
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.llm import StreamingLLMResponse
from karenina.schemas.config import ModelConfig
from karenina.utils.messages import append_error_feedback
from karenina.utils.retry_policy import RetryPolicy

from .messages import build_system_with_cache, convert_to_anthropic, extract_system_prompt
from .usage import extract_usage_from_response

logger = logging.getLogger(__name__)


class ClaudeToolLLMAdapter:
    """LLM adapter using Anthropic Python SDK for simple invocations.

    This adapter implements the LLMPort Protocol using client.messages.create()
    for stateless LLM calls without agent loops.

    The adapter handles:
    - Message conversion from unified Message to Anthropic SDK format
    - Usage metadata extraction from SDK responses
    - Structured output via client.beta.messages.parse() with Pydantic
    - Prompt caching for efficiency

    Note: Transient error retries are handled by the Anthropic SDK. The max_retries
    parameter is derived from the model's RetryPolicy so retry budgets stay consistent
    across all adapters.

    Example:
        >>> from karenina.schemas.config import ModelConfig
        >>> config = ModelConfig(
        ...     id="claude-haiku",
        ...     model_name="claude-haiku-4-5",
        ...     model_provider="anthropic",
        ...     interface="claude_tool"
        ... )
        >>> adapter = ClaudeToolLLMAdapter(config)
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
        _max_retries: int = 0,
    ) -> None:
        """Initialize the Claude Tool LLM adapter.

        Args:
            model_config: Configuration specifying model and parameters.
            _structured_schema: Internal - schema for structured output mode.
            _max_retries: Internal - max validation retries with error feedback.
        """
        self._config = model_config
        self._structured_schema = _structured_schema
        self._max_retries = _max_retries

        # Initialize clients lazily to avoid import issues
        self._client: Any = None
        self._async_client: Any = None

    def _get_client(self) -> Any:
        """Get or create the sync Anthropic client."""
        if self._client is None:
            from anthropic import Anthropic

            # Build kwargs for Anthropic client (api_key, base_url from config)
            kwargs: dict[str, Any] = {}
            if self._config.anthropic_api_key:
                kwargs["api_key"] = self._config.anthropic_api_key.get_secret_value()
            if self._config.anthropic_base_url:
                kwargs["base_url"] = self._config.anthropic_base_url
            if self._config.request_timeout is not None:
                kwargs["timeout"] = self._config.request_timeout

            # Derive SDK max_retries from RetryPolicy so retry budgets are consistent
            retry_policy = self._config.retry_policy or RetryPolicy()
            kwargs["max_retries"] = retry_policy.derive_sdk_max_retries()

            self._client = Anthropic(**kwargs)
        return self._client

    def _get_async_client(self) -> Any:
        """Get or create the async Anthropic client."""
        if self._async_client is None:
            from anthropic import AsyncAnthropic

            # Build kwargs for Anthropic client (api_key, base_url from config)
            kwargs: dict[str, Any] = {}
            if self._config.anthropic_api_key:
                kwargs["api_key"] = self._config.anthropic_api_key.get_secret_value()
            if self._config.anthropic_base_url:
                kwargs["base_url"] = self._config.anthropic_base_url
            if self._config.request_timeout is not None:
                kwargs["timeout"] = self._config.request_timeout

            # Derive SDK max_retries from RetryPolicy so retry budgets are consistent
            retry_policy = self._config.retry_policy or RetryPolicy()
            kwargs["max_retries"] = retry_policy.derive_sdk_max_retries()

            self._async_client = AsyncAnthropic(**kwargs)
        return self._async_client

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare what prompt features this adapter supports.

        Returns:
            PortCapabilities with system prompt, structured output, and streaming support.
        """
        return PortCapabilities(
            supports_system_prompt=True,
            supports_structured_output=True,
            supports_streaming=True,
        )

    async def ainvoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM asynchronously.

        Args:
            messages: List of unified Message objects.

        Returns:
            LLMResponse with content, usage metadata, and raw response.
            When using structured output (via with_structured_output()), the
            raw field contains the parsed Pydantic model instance.

        Raises:
            PortError: If the invocation fails.
        """
        if self._structured_schema is not None:
            return await self._ainvoke_structured(messages)
        return await self._ainvoke_text(messages)

    async def _ainvoke_text(self, messages: list[Message]) -> LLMResponse:
        """Invoke LLM for regular text output."""
        client = self._get_async_client()
        kwargs = self._build_invoke_kwargs(messages)
        response = await client.messages.create(**kwargs)

        # Extract text content
        content = ""
        if response.content:
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            content = "\n".join(text_blocks)

        usage = extract_usage_from_response(response, model=self._config.model_name)

        return LLMResponse(content=content, usage=usage, raw=response)

    async def _ainvoke_structured(self, messages: list[Message]) -> LLMResponse:
        """Invoke LLM for structured output using beta.messages.parse().

        Uses Anthropic's native structured output feature with Pydantic models.
        """
        if self._structured_schema is None:
            raise RuntimeError("_ainvoke_structured called without structured schema")

        last_error: str | None = None

        for attempt in range(self._max_retries + 1):
            effective_messages = append_error_feedback(messages, last_error) if attempt > 0 and last_error else messages

            try:
                return await self._try_structured_invocation(effective_messages)
            except Exception as e:
                last_error = str(e)
                if attempt == self._max_retries:
                    raise ParseError(
                        f"Structured output failed after {self._max_retries + 1} attempts. Last error: {last_error}"
                    ) from None
                logger.info(
                    f"Structured output failed (attempt {attempt + 1}/{self._max_retries + 1}): {e}. "
                    "Retrying with error feedback."
                )

        raise ParseError("Unexpected error in structured output retry logic")

    async def _try_structured_invocation(self, messages: list[Message]) -> LLMResponse:
        """Try structured output invocation."""
        client = self._get_async_client()

        # Convert messages
        anthropic_messages = convert_to_anthropic(messages)
        system_prompt = extract_system_prompt(messages)

        # Build kwargs for beta.messages.parse
        if not self._config.model_name:
            raise AdapterUnavailableError("model_name is required in ModelConfig", reason="missing_model_name")

        kwargs: dict[str, Any] = {
            "model": self._config.model_name,
            "max_tokens": self._config.max_tokens,
            "messages": anthropic_messages,
        }

        # Add system with caching if present
        if system_prompt:
            cached_system = build_system_with_cache(system_prompt)
            if cached_system:
                kwargs["system"] = cached_system

        # Add temperature if specified
        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature

        # Use beta.messages.parse for structured output
        response = await client.beta.messages.parse(
            output_format=self._structured_schema,
            **kwargs,
        )

        # At this point, _structured_schema is guaranteed to be set (checked by caller)
        schema = self._structured_schema
        assert schema is not None  # For type checker

        # Extract parsed output - beta.messages.parse returns this as parsed_output
        parsed_output = getattr(response, "parsed_output", None)

        if parsed_output is None:
            raise ParseError(f"beta.messages.parse did not return parsed_output for {schema.__name__}")

        if not isinstance(parsed_output, schema):
            raise ParseError(f"Parsed output is {type(parsed_output).__name__}, expected {schema.__name__}")

        usage = extract_usage_from_response(response, model=self._config.model_name)
        # Serialize to JSON so callers can json.loads(response.content)
        content = parsed_output.model_dump_json()
        return LLMResponse(
            content=content,
            usage=usage,
            raw=parsed_output,
        )

    @with_llm_semaphore
    def invoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM synchronously.

        This is a convenience wrapper around ainvoke() for sync code.

        Args:
            messages: List of unified Message objects.

        Returns:
            LLMResponse with content, usage metadata, and raw response.
        """
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            return portal.call(self.ainvoke, messages)

        try:
            asyncio.get_running_loop()

            def run_in_thread() -> LLMResponse:
                return asyncio.run(self.ainvoke(messages))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=300)

        except RuntimeError:
            return asyncio.run(self.ainvoke(messages))

    def _build_invoke_kwargs(self, messages: list[Message]) -> dict[str, Any]:
        """Build kwargs for messages.create / messages.stream.

        Shared between _ainvoke_text and astream to avoid duplication.
        """
        anthropic_messages = convert_to_anthropic(messages)
        system_prompt = extract_system_prompt(messages)

        if not self._config.model_name:
            raise AdapterUnavailableError("model_name is required in ModelConfig", reason="missing_model_name")

        kwargs: dict[str, Any] = {
            "model": self._config.model_name,
            "max_tokens": self._config.max_tokens,
            "messages": anthropic_messages,
        }

        if system_prompt:
            cached_system = build_system_with_cache(system_prompt)
            if cached_system:
                kwargs["system"] = cached_system

        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature

        return kwargs

    @asynccontextmanager
    async def astream(self, messages: list[Message]) -> AsyncIterator[StreamingLLMResponse]:  # noqa: ANN201
        """Stream LLM response, accumulating tokens as they arrive.

        Uses Anthropic SDK's ``client.messages.stream()`` for native streaming.
        On timeout, the StreamingLLMResponse contains whatever content
        was received before the interruption.

        Args:
            messages: List of unified Message objects.

        Yields:
            StreamingLLMResponse that can be iterated for text chunks.
        """
        client = self._get_async_client()
        kwargs = self._build_invoke_kwargs(messages)
        response = StreamingLLMResponse()

        async with client.messages.stream(**kwargs) as stream:
            response._set_chunk_source(stream.text_stream)
            try:
                yield response
            finally:
                # Best-effort usage extraction from the stream's final message
                try:
                    final_message = await stream.get_final_message()
                    response.usage = extract_usage_from_response(final_message, model=self._config.model_name)
                except Exception:
                    logger.warning("Could not extract usage from stream", exc_info=True)
        response.is_complete = True

    async def _astream_with_timeout(self, messages: list[Message], timeout: float | None) -> LLMResponse:
        """Stream with wall-clock timeout, returning accumulated content.

        Args:
            messages: List of unified Message objects.
            timeout: Wall-clock timeout in seconds. None means no timeout.

        Returns:
            LLMResponse with accumulated content.

        Raises:
            StreamingTimeoutError: If the stream exceeds the wall-clock timeout.
        """
        async with self.astream(messages) as sr:
            try:
                async with asyncio.timeout(timeout):
                    async for _chunk in sr:
                        pass
            except TimeoutError:
                from karenina.exceptions import StreamingTimeoutError

                raise StreamingTimeoutError(
                    f"Streaming timed out after {timeout}s",
                    partial_content=sr.accumulated_content,
                ) from None

        return LLMResponse(
            content=sr.accumulated_content,
            usage=sr.usage,
            raw=None,
        )

    @with_llm_semaphore
    def stream_invoke(self, messages: list[Message], timeout: float | None = None) -> LLMResponse:
        """Stream with wall-clock timeout, returning accumulated content synchronously.

        Uses streaming internally so that partial content can be captured on
        timeout. Returns the same LLMResponse type as invoke().

        Args:
            messages: List of unified Message objects.
            timeout: Wall-clock timeout in seconds. None means no timeout.

        Returns:
            LLMResponse with accumulated content. ``is_partial`` is True if
            the stream was interrupted by timeout.
        """
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            return portal.call(self._astream_with_timeout, messages, timeout)

        try:
            asyncio.get_running_loop()

            def run_in_thread() -> LLMResponse:
                return asyncio.run(self._astream_with_timeout(messages, timeout))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=(timeout or 300) + 30)

        except RuntimeError:
            return asyncio.run(self._astream_with_timeout(messages, timeout))

    def with_structured_output(
        self, schema: type[BaseModel], *, max_retries: int | None = None
    ) -> ClaudeToolLLMAdapter:
        """Return a new adapter configured for structured output.

        Uses Anthropic's beta.messages.parse() for native structured output
        with Pydantic models.

        Args:
            schema: A Pydantic model class defining the output structure.
            max_retries: Maximum retry attempts on validation failure.
                Default is 3 retries.

        Returns:
            A new ClaudeToolLLMAdapter configured for structured output.
        """
        return ClaudeToolLLMAdapter(
            model_config=self._config,
            _structured_schema=schema,
            _max_retries=max_retries if max_retries is not None else 3,
        )

    async def aclose(self) -> None:
        """Close underlying HTTP client resources.

        This method should be called when the adapter is no longer needed
        to properly close httpx connection pools and prevent resource leaks.
        Safe to call multiple times.
        """
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None
        if self._client is not None:
            self._client.close()
            self._client = None


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify ClaudeToolLLMAdapter implements LLMPort protocol."""
    adapter_instance: LLMPort = None  # type: ignore[assignment]
    _ = adapter_instance
