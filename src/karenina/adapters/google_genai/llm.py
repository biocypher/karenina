"""google_genai LLM adapter implementing the LLMPort interface.

Single Gemini calls through the shared ``calls.generate`` (one code path for
the wire call across all three ports), with optional structured output and
streaming. Built-in server-side tools apply when configured: Gemini searches
inside the call.

Structured output uses ``response_json_schema`` (not a forced function
call, which skips search on some routes). The content is validated with the
schema and retried with error feedback. If Gemini rejects the JSON Schema
form, the adapter falls back once to ``response_schema=<Pydantic class>``,
which lets the SDK convert the schema.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from karenina.adapters._parallel_base import run_coro_in_thread, with_llm_semaphore
from karenina.adapters._timeouts import SYNC_WRAPPER_GRACE_SECONDS, compute_sync_wrapper_timeout
from karenina.benchmark.verification.async_lifecycle import get_global_llm_limiter
from karenina.ports import LLMPort, LLMResponse, Message, ParseError, PortError, UsageMetadata
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.llm import StreamingLLMResponse
from karenina.schemas.config import ModelConfig
from karenina.utils.messages import append_error_feedback
from karenina.utils.retry_policy import RetryExecutor, RetryPolicy

from .calls import (
    CallOutcome,
    StreamAccumulator,
    build_generate_config,
    ensure_complete,
    generate,
    raise_for_llm_finish,
    translated_sdk_errors,
    with_request_timeout,
)
from .client import GenAIClientHolder
from .errors import GeminiClientError, build_error_registry
from .grounding import resolve_source_urls
from .messages import extract_system_instruction, to_contents
from .settings import BUILTIN_GOOGLE_SEARCH, GoogleGenAISettings
from .tools import builtin_tools
from .usage import aggregate_usage, extract_usage

logger = logging.getLogger(__name__)

_DEFAULT_STRUCTURED_RETRIES = 3


class GoogleGenAILLMAdapter:
    """LLM adapter for single Gemini calls.

    Transient errors are retried by ``RetryExecutor`` (the SDK's own HTTP
    retries are off). Each attempt holds one ``GlobalLLMLimiter`` permit.

    Example:
        >>> config = ModelConfig(id="flash", model_name="gemini-3.8-flash", interface="google_genai")
        >>> llm = GoogleGenAILLMAdapter(config)
        >>> response = await llm.ainvoke([Message.user("Hello!")])
        >>> structured = llm.with_structured_output(Answer)
        >>> parsed = (await structured.ainvoke([Message.user("2+2?")])).raw
    """

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        _structured_schema: type[BaseModel] | None = None,
        _max_retries: int = 0,
        _settings: GoogleGenAISettings | None = None,
        _client: GenAIClientHolder | None = None,
        _use_builtin_tools: bool = True,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_config: Configuration with ``interface="google_genai"``.
            _structured_schema: Internal: schema for structured output mode.
            _max_retries: Internal: validation retries with error feedback.
            _settings: Internal: settings shared with a parent adapter.
            _client: Internal: client holder shared with a parent adapter.
            _use_builtin_tools: Internal: False keeps server-side tools off
                (used by the parser, which only extracts from given text).

        Raises:
            ValueError: If the configuration is invalid.
        """
        self._config = model_config
        self._settings = _settings or GoogleGenAISettings.from_model_config(model_config)
        self._client = _client or GenAIClientHolder(self._settings)
        self._structured_schema = _structured_schema
        self._max_retries = _max_retries
        self._use_builtin_tools = _use_builtin_tools
        self._schema_mode: Literal["json_schema", "pydantic"] = "json_schema"
        self._retry_executor = RetryExecutor(model_config.retry_policy or RetryPolicy(), build_error_registry())

    @property
    def settings(self) -> GoogleGenAISettings:
        """The validated adapter settings."""
        return self._settings

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare adapter capabilities (system prompt, structured output, streaming)."""
        return PortCapabilities(
            supports_system_prompt=True,
            supports_structured_output=True,
            supports_streaming=True,
        )

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _builtins(self) -> tuple[str, ...]:
        """Built-in tools for this adapter's calls."""
        return self._settings.builtin_tools if self._use_builtin_tools else ()

    def _request(
        self,
        messages: list[Message],
        *,
        response_json_schema: dict[str, Any] | None = None,
        response_schema: Any | None = None,
    ) -> tuple[list[Any], Any]:
        """Build the contents and GenerateContentConfig of one call."""
        contents = to_contents(messages)
        if not contents:
            raise PortError("The google_genai LLM adapter needs at least one non-system message")
        config = build_generate_config(
            self._settings,
            system_instruction=extract_system_instruction(messages) or self._settings.system_prompt,
            tools=builtin_tools(self._builtins()),
            response_json_schema=response_json_schema,
            response_schema=response_schema,
        )
        return contents, config

    def _usage(self, usage_metadata: Any, query_count: int) -> UsageMetadata:
        """Convert one call's usage, counting searches when search was available."""
        search_available = BUILTIN_GOOGLE_SEARCH in self._builtins()
        return extract_usage(
            usage_metadata,
            model=self._settings.model_name,
            web_search_queries=query_count if search_available or query_count else None,
        )

    async def _generate(self, messages: list[Message], **schema_kwargs: Any) -> CallOutcome:
        """Make one call through ``calls.generate``."""
        contents, config = self._request(messages, **schema_kwargs)
        outcome = await generate(
            self._client,
            model=self._settings.model_name,
            contents=contents,
            config=config,
            stream=self._settings.stream,
            request_timeout=self._settings.request_timeout,
            retry_executor=self._retry_executor,
        )
        if self._settings.resolve_grounding_urls:
            await resolve_source_urls(outcome.grounding)
        return outcome

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    async def ainvoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the model asynchronously.

        Args:
            messages: The conversation messages.

        Returns:
            LLMResponse with the answer text (thoughts excluded), the call's
            usage and the raw SDK response (the last chunk when streamed).
            In structured mode the content is the validated model as JSON
            and ``raw`` is the model instance.

        Raises:
            PortError: If the call fails or Gemini stops without an answer.
            ParseError: If structured output stays invalid after all retries.
        """
        if self._structured_schema is not None:
            return await self._ainvoke_structured(messages, self._structured_schema)
        outcome = await self._generate(messages)
        raise_for_llm_finish(outcome)
        return LLMResponse(
            content=outcome.text,
            usage=self._usage(outcome.usage_metadata, outcome.search_query_count),
            raw=outcome.response,
        )

    async def _ainvoke_structured(self, messages: list[Message], schema: type[BaseModel]) -> LLMResponse:
        """Structured call with validation retries and error feedback.

        Only validation failures are retried here. Transport errors are
        retried per call inside ``calls.generate`` and API errors propagate.
        Usage is summed over every attempt.
        """
        total = UsageMetadata(model=self._settings.model_name)
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            effective = append_error_feedback(messages, last_error) if attempt > 0 and last_error else messages
            outcome = await self._generate_structured(effective, schema)
            total = aggregate_usage(total, self._usage(outcome.usage_metadata, outcome.search_query_count))
            raise_for_llm_finish(outcome)
            text = outcome.text
            try:
                parsed = _validate_json(schema, text)
            except (ValidationError, ValueError) as e:
                last_error = str(e)
                if attempt == self._max_retries:
                    raise ParseError(
                        f"Structured output for {schema.__name__} failed after {attempt + 1} attempts. "
                        f"Last error: {last_error}",
                        raw_response=text,
                    ) from None
                logger.info(
                    "Structured output for %s invalid (attempt %d/%d): %s. Retrying with error feedback",
                    schema.__name__,
                    attempt + 1,
                    self._max_retries + 1,
                    e,
                )
                continue
            return LLMResponse(content=parsed.model_dump_json(), usage=total, raw=parsed)
        raise ParseError("Unexpected end of the structured output retry loop")

    async def _generate_structured(self, messages: list[Message], schema: type[BaseModel]) -> CallOutcome:
        """One structured call, falling back to the SDK's schema conversion once."""
        if self._schema_mode == "json_schema":
            try:
                return await self._generate(messages, response_json_schema=schema.model_json_schema())
            except GeminiClientError as e:
                if e.code != 400:
                    raise
                logger.warning(
                    "Gemini rejected the JSON Schema of %s (%s). Retrying with the SDK's schema conversion",
                    schema.__name__,
                    e,
                )
                self._schema_mode = "pydantic"
        return await self._generate(messages, response_schema=schema)

    @with_llm_semaphore
    def invoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the model synchronously (wrapper around ``ainvoke``).

        Args:
            messages: The conversation messages.

        Returns:
            LLMResponse as from ``ainvoke``.
        """
        from karenina.benchmark.verification.async_lifecycle import get_async_portal

        portal = get_async_portal()
        if portal is not None:
            return portal.call(self.ainvoke, messages)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(messages))
        sequences = self._max_retries + 2 if self._structured_schema is not None else 1
        thread_timeout = compute_sync_wrapper_timeout(
            self._settings.request_timeout,
            retry_policy=self._config.retry_policy,
            internal_call_sequences=sequences,
        )
        return run_coro_in_thread(self.ainvoke, messages, timeout=thread_timeout)

    def with_structured_output(
        self, schema: type[BaseModel], *, max_retries: int | None = None
    ) -> GoogleGenAILLMAdapter:
        """Return a new adapter configured for structured output.

        The new adapter shares this adapter's settings and clients.

        Args:
            schema: Pydantic model class of the output.
            max_retries: Validation retries with error feedback (default 3).

        Returns:
            A new GoogleGenAILLMAdapter in structured mode.
        """
        return GoogleGenAILLMAdapter(
            self._config,
            _structured_schema=schema,
            _max_retries=max_retries if max_retries is not None else _DEFAULT_STRUCTURED_RETRIES,
            _settings=self._settings,
            _client=self._client,
            _use_builtin_tools=self._use_builtin_tools,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def astream(self, messages: list[Message]) -> AbstractAsyncContextManager[StreamingLLMResponse]:
        """Stream the answer text (thought parts are skipped).

        Stream establishment goes through ``RetryExecutor`` under one limiter
        permit. The established stream is never retried mid-flight. A stream
        that ends without a finish reason, or with STOP but no text, raises
        ``GeminiIncompleteResponseError`` from the iteration.

        Args:
            messages: The conversation messages.

        Returns:
            Async context manager yielding a StreamingLLMResponse.
        """
        return self._astream_impl(messages, establishment_retry=True)

    async def _open_stream(self, contents: list[Any], config: Any, *, timeout: float | None) -> Any:
        """Establish one stream under a limiter permit (one attempt)."""
        async with get_global_llm_limiter().borrow():
            with translated_sdk_errors():
                client = self._client.get()
                opening = client.aio.models.generate_content_stream(
                    model=self._settings.model_name, contents=contents, config=with_request_timeout(config, timeout)
                )
                return await (opening if timeout is None else asyncio.wait_for(opening, timeout=timeout))

    @asynccontextmanager
    async def _astream_impl(
        self, messages: list[Message], *, establishment_retry: bool
    ) -> AsyncIterator[StreamingLLMResponse]:
        """Shared astream body with switchable establishment retries.

        ``stream_invoke`` turns establishment retries off because its own
        RetryExecutor already retries the whole attempt.
        """
        contents, config = self._request(messages)
        if establishment_retry:
            stream = await self._retry_executor.aexecute_with_timeout(
                self._open_stream, contents, config, timeout=self._settings.request_timeout
            )
        else:
            stream = await self._open_stream(contents, config, timeout=self._settings.request_timeout)

        response = StreamingLLMResponse()
        accumulator = StreamAccumulator()

        async def _chunks() -> AsyncIterator[str]:
            with translated_sdk_errors():
                async for chunk in stream:
                    new_parts = accumulator.add(chunk)
                    if accumulator.usage_metadata is not None:
                        query_count = accumulator.grounding.query_count if accumulator.grounding else 0
                        response.usage = self._usage(accumulator.usage_metadata, query_count)
                    for part in new_parts:
                        text = getattr(part, "text", None)
                        if text and not getattr(part, "thought", None):
                            yield text
            outcome = accumulator.outcome()
            ensure_complete(outcome, stream=True)
            raise_for_llm_finish(outcome)

        response._set_chunk_source(_chunks())
        try:
            yield response
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
        response.is_complete = True

    async def _astream_with_timeout(self, messages: list[Message], timeout: float | None) -> LLMResponse:
        """Stream under a wall-clock timeout covering establishment and reading.

        Returns:
            LLMResponse. When the timeout fires after some text arrived,
            ``is_partial`` and ``usage_unavailable`` are both True.

        Raises:
            StreamingTimeoutError: If the timeout fires before any text
                arrived (classified TIMEOUT, so stream_invoke retries it).
        """
        from karenina.exceptions import StreamingTimeoutError

        scope = asyncio.timeout(timeout)
        streamed: StreamingLLMResponse | None = None
        try:
            async with scope, self._astream_impl(messages, establishment_retry=False) as sr:
                streamed = sr
                async for _chunk in sr:
                    pass
        except TimeoutError:
            if not scope.expired():
                raise
            partial = streamed.accumulated_content if streamed is not None else ""
            if not partial:
                raise StreamingTimeoutError(f"Streaming timed out after {timeout}s with no content") from None
            logger.warning("Streaming timeout after %ss: kept %d chars of partial response", timeout, len(partial))
            assert streamed is not None
            return LLMResponse(content=partial, usage=streamed.usage, raw=None, is_partial=True, usage_unavailable=True)

        assert streamed is not None
        return LLMResponse(content=streamed.accumulated_content, usage=streamed.usage, raw=None)

    @with_llm_semaphore
    def stream_invoke(self, messages: list[Message], timeout: float | None = None) -> LLMResponse:
        """Stream under a wall-clock timeout and return the (possibly partial) answer.

        Retries transient failures through RetryExecutor, including a
        timeout that fired before any text arrived.

        Args:
            messages: The conversation messages.
            timeout: Wall-clock timeout in seconds. None means no timeout.

        Returns:
            LLMResponse. ``is_partial=True`` when the timeout cut the stream.
        """
        result: LLMResponse = self._retry_executor.execute_with_timeout(
            self._stream_invoke_once, messages, timeout=timeout
        )
        return result

    def _stream_invoke_once(self, messages: list[Message], timeout: float | None = None) -> LLMResponse:
        """Single stream_invoke attempt (no retry). Called by RetryExecutor."""
        from karenina.benchmark.verification.async_lifecycle import get_async_portal

        portal = get_async_portal()
        if portal is not None:
            return portal.call(self._astream_with_timeout, messages, timeout)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._astream_with_timeout(messages, timeout))
        thread_timeout = (timeout if timeout is not None else 300.0) + SYNC_WRAPPER_GRACE_SECONDS
        return run_coro_in_thread(self._astream_with_timeout, messages, timeout, timeout=thread_timeout)

    async def aclose(self) -> None:
        """Close the google-genai clients (shared with derived adapters). Safe to call twice."""
        await self._client.aclose()


def _validate_json(schema: type[BaseModel], text: str) -> BaseModel:
    """Validate the model's JSON text against the schema.

    A Markdown code fence around the JSON is tolerated.

    Raises:
        ValueError: If the text is empty.
        ValidationError: If the JSON does not match the schema.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("The model returned no JSON text")
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0].strip()
    try:
        return schema.model_validate_json(stripped)
    except ValidationError:
        raise
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e


def _verify_protocol_compliance() -> None:
    """Verify GoogleGenAILLMAdapter implements the LLMPort protocol."""
    adapter_instance: LLMPort = GoogleGenAILLMAdapter.__new__(GoogleGenAILLMAdapter)
    _ = adapter_instance
