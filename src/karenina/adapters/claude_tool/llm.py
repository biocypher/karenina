"""Claude Tool LLM adapter implementing the LLMPort interface.

This module provides the ClaudeToolLLMAdapter class that uses the Anthropic Python SDK
directly (client.messages.create) for simple LLM invocations without agent loops.

Key features:
- Uses Anthropic's native Python SDK for efficient API calls
- Supports structured output via client.beta.messages.parse() with Pydantic
- Implements prompt caching for efficiency

Retry Logic:
    API calls route through RetryExecutor with per-category retry budgets
    for transient errors (connection errors, timeouts, rate limits, 5xx
    errors). The SDK clients run with max_retries=0 so RetryExecutor is
    the sole retry layer and retry telemetry via track_retries() is
    accurate. The structured-output validation retry loop (error feedback
    to the model) stays outside the transient executor so validation
    failures do not consume transient budgets.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from pydantic import BaseModel

from karenina.adapters._parallel_base import run_coro_in_thread, with_llm_semaphore
from karenina.adapters._timeouts import compute_sync_wrapper_timeout
from karenina.ports import AdapterUnavailableError, LLMPort, LLMResponse, Message, ParseError
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.llm import StreamingLLMResponse
from karenina.schemas.config import ModelConfig
from karenina.utils.errors import ErrorRegistry
from karenina.utils.messages import append_error_feedback
from karenina.utils.retry_policy import RetryExecutor, RetryPolicy

from .messages import build_system_with_cache, convert_to_anthropic, extract_system_prompt
from .usage import extract_usage, extract_usage_from_response, merge_stream_usage

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

    Note: Transient error retries are handled by RetryExecutor at the
    adapter layer (per-category budgets from the model's RetryPolicy).
    The SDK clients are constructed with max_retries=0.

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

        retry_policy = model_config.retry_policy or RetryPolicy()
        self._retry_executor = RetryExecutor(retry_policy, ErrorRegistry())

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

            # Suppress SDK-level retries. RetryExecutor is the sole retry
            # layer (design decision D1). Known cost: the SDK's honoring of
            # server retry-after headers is dropped. Deliberate deferred
            # follow-up for first-party Anthropic endpoints.
            kwargs["max_retries"] = 0

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

            # Suppress SDK-level retries. RetryExecutor is the sole retry
            # layer (design decision D1, see _get_client for the deferred
            # retry-after follow-up).
            kwargs["max_retries"] = 0

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
        """Invoke LLM for regular text output with automatic transient retry."""
        client = self._get_async_client()
        kwargs = self._build_invoke_kwargs(messages)
        response = await self._retry_executor.aexecute_with_timeout(
            self._acall_with_timeout,
            client.messages.create,
            timeout=self._config.request_timeout,
            **kwargs,
        )

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

        # Use beta.messages.parse for structured output. Transient errors
        # are retried here by RetryExecutor. The validation retry loop in
        # _ainvoke_structured stays outside so validation failures do not
        # consume transient budgets.
        response = await self._retry_executor.aexecute_with_timeout(
            self._acall_with_timeout,
            client.beta.messages.parse,
            timeout=self._config.request_timeout,
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

    async def _acall_with_timeout(
        self,
        call: Any,
        *,
        timeout: float | None = None,
        **call_kwargs: Any,
    ) -> Any:
        """Run an async SDK call under a wall-clock timeout as a guardrail.

        The SDK client already carries an httpx-level timeout from
        ``request_timeout``, but that does not catch every stall, so this
        karenina-layer ``asyncio.wait_for`` enforces a hard per-attempt
        wall-clock budget (mirrors the langchain adapter's helper).

        A fired timeout raises a stock ``asyncio.TimeoutError``, which
        ``ErrorRegistry`` classifies as ``TIMEOUT`` via the built-in MRO
        check, so the timeout retry budget applies inside
        ``RetryExecutor.aexecute_with_timeout``. Note that
        ``RetryPolicy.timeout_escalation`` only extends this guard on
        TIMEOUT retries. The SDK client's own timeout stays pinned at
        ``request_timeout`` from construction, so the SDK may cut the
        request before an escalated guard fires.

        Args:
            call: The bound async SDK method to invoke (for example
                ``client.messages.create``).
            timeout: Optional per-call wall-clock timeout in seconds.
                When None, falls back to ``self._config.request_timeout``.
                When that is also None, the call is made without a wrapper.
            **call_kwargs: Keyword arguments forwarded to ``call``.

        Returns:
            The raw SDK response object.

        Raises:
            asyncio.TimeoutError: If the call exceeds the effective timeout.
        """
        if timeout is None:
            timeout = self._config.request_timeout
        if timeout is None:
            return await call(**call_kwargs)
        return await asyncio.wait_for(call(**call_kwargs), timeout=timeout)

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
            # Fresh thread with the caller's context propagated, so
            # track_retries telemetry survives the dispatch.
            thread_timeout = compute_sync_wrapper_timeout(
                self._config.request_timeout,
                retry_policy=self._config.retry_policy,
            )
            return run_coro_in_thread(self.ainvoke, messages, timeout=thread_timeout)

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

    async def _aenter_stream(self, kwargs: dict[str, Any], *, timeout: float | None = None) -> tuple[Any, Any]:
        """Open a fresh SDK stream context (one establishment attempt).

        Called by RetryExecutor so transient connection-setup failures
        retry with per-category budgets. Each attempt creates a brand-new
        stream manager. The established stream itself is never retried
        mid-flight.

        Args:
            kwargs: Keyword arguments for ``client.messages.stream``.
            timeout: Optional wall-clock bound on establishment, escalated
                by RetryExecutor on TIMEOUT retries.

        Returns:
            Tuple of (manager, stream) where manager is the SDK context
            manager (its ``__aexit__`` must be awaited by the caller) and
            stream is the entered AsyncMessageStream.
        """
        client = self._get_async_client()
        manager = client.messages.stream(**kwargs)
        try:
            if timeout is None:
                stream = await manager.__aenter__()
            else:
                stream = await asyncio.wait_for(manager.__aenter__(), timeout=timeout)
        except BaseException:
            # Best-effort close of the half-opened manager (for example
            # when wait_for cancels __aenter__ mid-handshake) so no
            # unclosed-stream debris is left behind.
            with contextlib.suppress(Exception):
                await manager.__aexit__(None, None, None)
            raise
        return manager, stream

    def astream(self, messages: list[Message]) -> AbstractAsyncContextManager[StreamingLLMResponse]:
        """Stream LLM response, accumulating tokens as they arrive.

        Uses Anthropic SDK's ``client.messages.stream()`` for native streaming.
        On timeout, the StreamingLLMResponse contains whatever content
        was received before the interruption.

        Stream ESTABLISHMENT (entering the SDK stream context) routes
        through RetryExecutor, so transient connection-setup failures
        retry with per-category budgets even though the SDK client runs
        with max_retries=0 (design decision D1). The established stream
        is never retried mid-flight.

        Usage is captured inline as streaming events arrive:
        ``message_start`` carries input tokens (and cache fields),
        ``message_delta`` carries cumulative output tokens. A mid-stream
        timeout therefore still leaves whatever usage arrived before the
        interruption. ``get_final_message()`` remains the success-path
        enrichment and overrides the inline snapshot with final values.

        Args:
            messages: List of unified Message objects.

        Returns:
            Async context manager yielding a StreamingLLMResponse that
            can be iterated for text chunks.
        """
        return self._astream_impl(messages, establishment_retry=True)

    @asynccontextmanager
    async def _astream_impl(
        self,
        messages: list[Message],
        *,
        establishment_retry: bool,
    ) -> AsyncIterator[StreamingLLMResponse]:  # noqa: ANN202
        """Shared astream body with switchable establishment retries.

        The public astream keeps establishment retries on. The
        stream_invoke path (via _astream_with_timeout) turns them off
        because its outer RetryExecutor already retries the whole
        attempt, and nesting the two would amplify worst-case connection
        attempts multiplicatively.
        """
        kwargs = self._build_invoke_kwargs(messages)
        response = StreamingLLMResponse()

        if establishment_retry:
            manager, stream = await self._retry_executor.aexecute_with_timeout(
                self._aenter_stream,
                kwargs,
                timeout=self._config.request_timeout,
            )
        else:
            manager, stream = await self._aenter_stream(kwargs, timeout=self._config.request_timeout)

        async def _chunk_generator() -> AsyncIterator[str]:  # noqa: ANN202
            """Yield text deltas while capturing usage from stream events inline."""
            async for event in stream:
                event_type = getattr(event, "type", None)
                if event_type == "text":
                    yield event.text
                elif event_type == "message_start":
                    message_usage = getattr(getattr(event, "message", None), "usage", None)
                    if message_usage is not None:
                        response.usage = merge_stream_usage(response.usage, message_usage, self._config.model_name)
                elif event_type == "message_delta":
                    delta_usage = getattr(event, "usage", None)
                    if delta_usage is not None:
                        response.usage = merge_stream_usage(response.usage, delta_usage, self._config.model_name)

        try:
            response._set_chunk_source(_chunk_generator())
            try:
                yield response
            finally:
                # Success-path enrichment: the final message carries the
                # authoritative usage. On failure the inline snapshot
                # captured from stream events is kept.
                try:
                    final_message = await stream.get_final_message()
                    final_usage = getattr(final_message, "usage", None)
                    if final_usage is not None:
                        response.usage = extract_usage(final_usage, model=self._config.model_name)
                except Exception:
                    logger.warning(
                        "Could not extract usage from stream final message, keeping inline streamed usage",
                        exc_info=True,
                    )
        finally:
            await manager.__aexit__(None, None, None)
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
        # Establishment retries are off: the caller (stream_invoke) wraps
        # this whole attempt in RetryExecutor, so a single retry layer
        # covers connection-setup failures.
        async with self._astream_impl(messages, establishment_retry=False) as sr:
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
        timeout. Returns the same LLMResponse type as invoke(). Retries via
        RetryExecutor on transient errors (including StreamingTimeoutError
        with zero content, classified as RATE_LIMIT for queue congestion).
        The inner stream-establishment retry used by the bare astream path
        is disabled on this path, so this outer executor is the single
        retry layer and connection attempts are not amplified
        multiplicatively.

        Args:
            messages: List of unified Message objects.
            timeout: Wall-clock timeout in seconds. None means no timeout.

        Returns:
            LLMResponse with accumulated content.

        Raises:
            StreamingTimeoutError: If retries are exhausted and the stream
                still exceeds the wall-clock timeout.
        """
        result: LLMResponse = self._retry_executor.execute_with_timeout(
            self._stream_invoke_once, messages, timeout=timeout
        )
        return result

    def _stream_invoke_once(self, messages: list[Message], timeout: float | None = None) -> LLMResponse:
        """Single stream_invoke attempt (no retry). Called by RetryExecutor."""
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            return portal.call(self._astream_with_timeout, messages, timeout)

        try:
            asyncio.get_running_loop()
            return run_coro_in_thread(
                self._astream_with_timeout,
                messages,
                timeout,
                timeout=(timeout or 300) + 30,
            )

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
