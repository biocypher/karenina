"""LangChain Deep Agents LLM adapter implementing the LLMPort interface.

This module provides the DeepAgentsLLMAdapter class that implements LLMPort
using LangChain's init_chat_model for simple single-turn LLM calls.

For single-turn calls, the adapter uses the LangChain model directly (not
create_deep_agent), since no agent loop or tool calling is needed.

Retry Logic:
    ainvoke() routes through RetryExecutor with per-category retry budgets
    for transient errors (connection errors, timeouts, rate limits, 5xx
    errors). The underlying SDK clients run with max_retries=0 (see
    initialization.create_chat_model), so RetryExecutor is the sole retry
    layer and retry telemetry via track_retries() is accurate.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from karenina.adapters._parallel_base import run_coro_in_thread, with_llm_semaphore
from karenina.adapters._timeouts import DEEP_AGENTS_SYNC_WRAPPER_FLOOR, compute_sync_wrapper_timeout
from karenina.adapters.langchain.usage import extract_usage_from_chunk
from karenina.benchmark.verification.async_lifecycle import get_global_llm_limiter
from karenina.ports import LLMResponse, Message
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.llm import StreamingLLMResponse
from karenina.ports.usage import UsageMetadata
from karenina.utils.errors import ErrorRegistry
from karenina.utils.retry_policy import RetryExecutor, RetryPolicy

from .initialization import create_chat_model
from .messages import DeepAgentsMessageConverter

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig

logger = logging.getLogger(__name__)


class DeepAgentsLLMAdapter:
    """LLM adapter using LangChain's init_chat_model for single-turn calls.

    This adapter implements the LLMPort Protocol for simple LLM invocation
    without agent loops. Uses the LangChain model directly for efficiency.

    Example:
        >>> config = ModelConfig(
        ...     id="test",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic",
        ...     interface="langchain_deep_agents",
        ... )
        >>> adapter = DeepAgentsLLMAdapter(config)
        >>> response = await adapter.ainvoke([Message.user("Hello!")])
        >>> print(response.content)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        _structured_schema: type[BaseModel] | None = None,
    ) -> None:
        """Initialize the Deep Agents LLM adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
            _structured_schema: Internal; schema for structured output mode.
        """
        self._config = model_config
        self._converter = DeepAgentsMessageConverter()
        self._structured_schema = _structured_schema

        retry_policy = model_config.retry_policy or RetryPolicy()
        self._retry_executor = RetryExecutor(retry_policy, ErrorRegistry())

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare adapter capabilities.

        Returns:
            PortCapabilities with system_prompt, structured_output, and streaming support.
        """
        return PortCapabilities(
            supports_system_prompt=True,
            supports_structured_output=True,
            supports_streaming=True,
        )

    async def ainvoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM asynchronously with automatic retry for transient errors.

        Converts karenina Messages to LangChain format, invokes the model
        through RetryExecutor (per-category retry budgets for connection,
        timeout, rate limit, and server errors), and converts the response
        back.

        Args:
            messages: List of messages forming the conversation.

        Returns:
            LLMResponse containing the generated content and usage metadata.

        Raises:
            The last transient exception if all retries are exhausted, or
            immediately for permanent errors.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        # Build LangChain message list
        lc_messages: list[Any] = []
        for msg in messages:
            text = msg.text or ""
            if msg.role.value == "system":
                lc_messages.append(SystemMessage(content=text))
            elif msg.role.value == "user":
                lc_messages.append(HumanMessage(content=text))
            elif msg.role.value == "assistant":
                lc_messages.append(AIMessage(content=text))

        # Create the model once, outside the retried callable, so that
        # retries repeat the API call rather than model construction.
        chat_model = create_chat_model(self._config)

        # Apply structured output if configured
        if self._structured_schema is not None:
            chat_model = chat_model.with_structured_output(self._structured_schema)

        # Invoke through RetryExecutor with a wall-clock guard per attempt.
        # GlobalLLMLimiter gating happens per attempt inside
        # _ainvoke_with_timeout, so retry backoff never holds a permit.
        # Sync invoke() reaches the wire only through this method, so it
        # is covered without a sync acquire.
        response = await self._retry_executor.aexecute_with_timeout(
            self._ainvoke_with_timeout,
            chat_model,
            lc_messages,
            timeout=self._config.request_timeout,
        )

        # Extract content
        if self._structured_schema is not None:
            # Structured output returns a Pydantic model or dict
            if isinstance(response, BaseModel):
                content = response.model_dump_json()
            elif isinstance(response, dict):
                import json

                content = json.dumps(response)
            else:
                content = str(response)
        elif isinstance(response, AIMessage):
            content = response.content if isinstance(response.content, str) else str(response.content)
        else:
            content = str(response)

        # Extract usage
        usage = UsageMetadata(model=self._config.model_name)
        if isinstance(response, AIMessage):
            usage_meta = getattr(response, "usage_metadata", None)
            if usage_meta and isinstance(usage_meta, dict):
                usage = UsageMetadata(
                    input_tokens=usage_meta.get("input_tokens", 0),
                    output_tokens=usage_meta.get("output_tokens", 0),
                    total_tokens=usage_meta.get("input_tokens", 0) + usage_meta.get("output_tokens", 0),
                    model=self._config.model_name,
                )

        return LLMResponse(content=content, usage=usage, raw=response)

    async def _ainvoke_with_timeout(
        self,
        model: Any,
        lc_messages: list[Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        """Call ``model.ainvoke`` under a wall-clock timeout as a guardrail.

        LangChain's ``ainvoke`` can stall internally without ever raising,
        and the httpx ``request_timeout`` does not catch every such case,
        so this karenina-layer ``asyncio.wait_for`` enforces a hard
        per-attempt wall-clock budget. Mirrors the langchain adapter.

        A fired timeout raises a stock ``asyncio.TimeoutError``, which
        ``ErrorRegistry`` classifies as ``TIMEOUT`` via the built-in MRO
        check. The configured timeout retry budget then applies inside
        ``RetryExecutor.aexecute_with_timeout``. Note that
        ``RetryPolicy.timeout_escalation`` only extends this guard on
        TIMEOUT retries. The underlying SDK client's own timeout stays
        pinned at ``request_timeout`` from construction, so the SDK may
        cut the request before an escalated guard fires.

        When ``timeout`` is None, falls back to ``self._config.request_timeout``.
        When that is also None, the call is made without any wrapper.

        Each attempt holds one GlobalLLMLimiter permit for the wire call
        only (uniform per-attempt policy), so retry backoff sleeps never
        hold a permit. The permit wait itself is NOT bounded by the
        attempt timeout: a saturated limiter delays the attempt rather
        than timing it out, matching the legacy semaphore semantics.

        Args:
            model: The LangChain model exposing ``ainvoke``.
            lc_messages: Provider-formatted messages to send.
            timeout: Optional per-call wall-clock timeout in seconds.
                When None, falls back to ``self._config.request_timeout``.

        Returns:
            The raw model response object.

        Raises:
            asyncio.TimeoutError: If the call exceeds the effective timeout.
        """
        async with get_global_llm_limiter().borrow():
            if timeout is None:
                timeout = self._config.request_timeout
            if timeout is None:
                return await model.ainvoke(lc_messages)
            return await asyncio.wait_for(model.ainvoke(lc_messages), timeout=timeout)

    @with_llm_semaphore
    def invoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM synchronously.

        Args:
            messages: List of messages forming the conversation.

        Returns:
            LLMResponse containing the generated content and usage metadata.
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
                floor=DEEP_AGENTS_SYNC_WRAPPER_FLOOR,
                retry_policy=self._config.retry_policy,
            )
            return run_coro_in_thread(self.ainvoke, messages, timeout=thread_timeout)

        except RuntimeError:
            return asyncio.run(self.ainvoke(messages))

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        max_retries: int | None = None,
    ) -> DeepAgentsLLMAdapter:
        """Return a new adapter configured for structured output.

        Args:
            schema: A Pydantic model class defining the output structure.
            max_retries: Not supported by this adapter. A warning is emitted
                if a non-None value is provided. See
                karenina.ports.llm.LLMPort.with_structured_output for the
                cross-adapter contract (LangChain and Claude Tool respect
                max_retries, this adapter and claude_agent_sdk warn and
                ignore).

        Returns:
            A new DeepAgentsLLMAdapter configured with the schema.
        """
        if max_retries is not None:
            logger.warning(
                "max_retries=%d ignored by langchain_deep_agents adapter. "
                "Validation-feedback retries are not implemented here. "
                "Transient errors are retried via RetryExecutor regardless.",
                max_retries,
            )
        return DeepAgentsLLMAdapter(
            self._config,
            _structured_schema=schema,
        )

    def astream(self, messages: list[Message]) -> AbstractAsyncContextManager[StreamingLLMResponse]:
        """Stream LLM response using LangChain's cross-provider astream.

        Stream ESTABLISHMENT (opening the LangChain astream and fetching
        the first chunk) routes through RetryExecutor, so transient
        connection-setup failures (including langchain-openai's cross
        event loop pooled-connection errors) retry with per-category
        budgets. Each retry recreates the stream from scratch. Once
        established, the stream is never retried mid-flight.

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
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages: list[Any] = []
        for msg in messages:
            text = msg.text or ""
            if msg.role.value == "system":
                lc_messages.append(SystemMessage(content=text))
            elif msg.role.value == "user":
                lc_messages.append(HumanMessage(content=text))
            elif msg.role.value == "assistant":
                lc_messages.append(AIMessage(content=text))

        chat_model = create_chat_model(self._config)
        response = StreamingLLMResponse()

        async def _establish(*, timeout: float | None = None) -> tuple[AsyncIterator[Any], Any | None]:
            """One stream-establishment attempt: open astream, await the first chunk.

            Each attempt holds one GlobalLLMLimiter permit for the
            establishment only, released before the stream is consumed:
            max_concurrent_requests caps concurrent request setups, not
            concurrent open streams. Borrowing per attempt (instead of
            around the retry loop) releases the permit during backoff.
            """
            async with get_global_llm_limiter().borrow():
                iterator = chat_model.astream(lc_messages).__aiter__()
                try:
                    if timeout is None:
                        first = await iterator.__anext__()
                    else:
                        first = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
                except StopAsyncIteration:
                    return iterator, None
                except BaseException:
                    # Close the half-opened stream before surfacing the failure
                    # (or cancellation) so no "async generator was never closed"
                    # debris is left behind.
                    aclose = getattr(iterator, "aclose", None)
                    if aclose is not None:
                        with contextlib.suppress(Exception):
                            await aclose()
                    raise
                return iterator, first

        if establishment_retry:
            iterator, first_chunk = await self._retry_executor.aexecute_with_timeout(
                _establish,
                timeout=self._config.request_timeout,
            )
        else:
            iterator, first_chunk = await _establish(timeout=self._config.request_timeout)

        def _handle_chunk(chunk: Any) -> str:
            """Extract text and capture usage metadata from a chunk."""
            text = chunk.content if isinstance(chunk.content, str) else ""
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                response.usage = extract_usage_from_chunk(chunk, model_name=self._config.model_name)
            return text

        async def _chunk_generator() -> AsyncIterator[str]:  # noqa: ANN202
            """Yield text chunks from LangChain's astream, extracting usage from final chunk."""
            if first_chunk is not None:
                text = _handle_chunk(first_chunk)
                if text:
                    yield text
            async for chunk in iterator:
                text = _handle_chunk(chunk)
                if text:
                    yield text

        response._set_chunk_source(_chunk_generator())
        yield response
        response.is_complete = True

    async def _astream_with_timeout(self, messages: list[Message], timeout: float | None) -> LLMResponse:
        """Stream with wall-clock timeout, returning accumulated content.

        The timeout window covers stream ESTABLISHMENT as well as the
        drain: the stream is opened inside asyncio.timeout, so an
        establishment stall surfaces as StreamingTimeoutError (with empty
        partial content) within about ``timeout`` seconds even when
        request_timeout is unset. Establishment retries are disabled here
        because the caller (stream_invoke) already wraps each attempt in
        RetryExecutor.

        Args:
            messages: List of unified Message objects.
            timeout: Wall-clock timeout in seconds. None means no timeout.

        Returns:
            LLMResponse with accumulated content.

        Raises:
            StreamingTimeoutError: If establishment plus drain exceed the
                wall-clock timeout.
        """
        from karenina.exceptions import StreamingTimeoutError

        stream_cm = self._astream_impl(messages, establishment_retry=False)
        sr: StreamingLLMResponse | None = None
        try:
            try:
                async with asyncio.timeout(timeout):
                    sr = await stream_cm.__aenter__()
                    async for _chunk in sr:
                        pass
            except TimeoutError:
                raise StreamingTimeoutError(
                    f"Streaming timed out after {timeout}s",
                    partial_content=sr.accumulated_content if sr is not None else "",
                ) from None
        finally:
            if sr is not None:
                await stream_cm.__aexit__(None, None, None)

        return LLMResponse(
            content=sr.accumulated_content,
            usage=sr.usage,
            raw=None,
        )

    @with_llm_semaphore
    def stream_invoke(self, messages: list[Message], timeout: float | None = None) -> LLMResponse:
        """Stream with wall-clock timeout synchronously.

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

    async def aclose(self) -> None:
        """Close underlying resources.

        No resources to clean up: the LangChain model is created fresh
        per ainvoke() call. Provided for interface consistency with other
        adapters.
        """
