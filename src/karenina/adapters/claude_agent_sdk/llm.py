"""Claude Agent SDK LLM adapter implementing the LLMPort interface.

This module provides the ClaudeSDKLLMAdapter class that implements the LLMPort
interface using the Claude Agent SDK's query() function for simple LLM calls.

IMPORTANT: Uses query() NOT ClaudeSDKClient since no hooks/tools are needed
for simple LLM invocation. For agent loops with MCP/hooks, use ClaudeSDKAgentAdapter.

Key differences from LangChain:
- Claude SDK uses string prompts, not message arrays
- System prompts go in ClaudeAgentOptions.system_prompt
- Structured output returned in ResultMessage.structured_output (already dict, not JSON)
- SDK handles retries autonomously via max_turns (no manual retry logic needed)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from karenina.adapters._parallel_base import run_coro_in_thread, with_llm_semaphore
from karenina.adapters._timeouts import compute_sync_wrapper_timeout
from karenina.benchmark.verification.async_lifecycle import gate_stream_establishment, get_global_llm_limiter
from karenina.ports import LLMPort, LLMResponse, Message, ParseError
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.llm import StreamingLLMResponse
from karenina.utils.errors import ErrorRegistry
from karenina.utils.retry_policy import RetryExecutor, RetryPolicy

from .auth import subscription_auth_env
from .messages import ClaudeSDKMessageConverter
from .usage import extract_sdk_usage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

    from karenina.schemas.config import ModelConfig

_T = TypeVar("_T")


def _run_in_fresh_loop(
    coro_func: Callable[..., Coroutine[Any, Any, _T]],
    *args: Any,
    timeout: float = 300,
) -> _T:
    """Run an async function in a dedicated thread with a fresh event loop.

    The Claude Agent SDK (query(), ClaudeSDKClient) uses anyio cancel scopes
    internally. When run via BlockingPortal.call(), those cancel scopes conflict
    with the portal's own cancel scope hierarchy, producing:

        RuntimeError: Attempted to exit a cancel scope that isn't the
        current task's current cancel scope

    This helper always spawns a new thread with asyncio.run(), giving the SDK
    a completely isolated event loop and task context. This avoids the cancel
    scope mismatch regardless of whether a BlockingPortal is active.

    Other adapters (claude_tool, langchain) do not need this because their
    underlying async calls (httpx, LangChain ainvoke) do not create anyio
    cancel scopes.

    The caller's contextvars are propagated into the fresh thread (via
    run_coro_in_thread), so context-bound state such as the track_retries
    telemetry tracker survives the dispatch.

    Args:
        coro_func: Async function to call.
        *args: Positional arguments forwarded to coro_func.
        timeout: Thread-level timeout in seconds (default: 300).

    Returns:
        The return value of the coroutine.

    Raises:
        concurrent.futures.TimeoutError: If the thread does not finish
            within the timeout.
        Exception: Any exception raised by the coroutine is re-raised.
    """
    return run_coro_in_thread(coro_func, *args, timeout=timeout)


class ClaudeSDKLLMAdapter:
    """LLM adapter using Claude Agent SDK's query() function.

    This adapter implements the LLMPort Protocol for simple LLM invocation
    without agent loops or tool calling. Uses the SDK's query() function
    which is simpler than ClaudeSDKClient.

    The adapter handles:
    - Message conversion from unified Message to prompt string
    - System prompt extraction to ClaudeAgentOptions.system_prompt
    - Usage metadata extraction from ResultMessage
    - Sync/async invocation with proper event loop handling
    - Structured output via with_structured_output()

    Note: The SDK handles retries autonomously via max_turns, so no manual
    retry logic is needed in this adapter.

    Example:
        >>> from karenina.schemas.config import ModelConfig
        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic",
        ...     interface="claude_agent_sdk"
        ... )
        >>> adapter = ClaudeSDKLLMAdapter(config)
        >>> response = await adapter.ainvoke([Message.user("Hello!")])
        >>> print(response.content)
        'Hello! How can I help you today?'

        >>> # With structured output
        >>> class Answer(BaseModel):
        ...     value: str
        >>> structured = adapter.with_structured_output(Answer)
        >>> response = await structured.ainvoke([Message.user("What is 2+2?")])
    """

    # Default max_turns for structured output (minimum needed for SDK validation)
    DEFAULT_STRUCTURED_MAX_TURNS = 2

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        _structured_schema: type[BaseModel] | None = None,
        _max_turns: int | None = None,
    ) -> None:
        """Initialize the Claude SDK LLM adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
            _structured_schema: Internal - schema for structured output mode.
            _max_turns: Internal - max turns for SDK (handles retries autonomously).
        """
        self._config = model_config
        self._converter = ClaudeSDKMessageConverter()
        self._structured_schema = _structured_schema
        self._max_turns = _max_turns if _max_turns is not None else self.DEFAULT_STRUCTURED_MAX_TURNS

        retry_policy = model_config.retry_policy or RetryPolicy()
        self._retry_executor = RetryExecutor(retry_policy, ErrorRegistry())

    def _build_options(self, system_prompt: str | None) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions for the query.

        Args:
            system_prompt: System prompt extracted from messages.

        Returns:
            Configured ClaudeAgentOptions.
        """
        from claude_agent_sdk import ClaudeAgentOptions

        options_kwargs: dict[str, Any] = {
            # Use bypassPermissions for batch/automated calls
            "permission_mode": "bypassPermissions",
            # No tools needed for simple LLM invocation
            "allowed_tools": [],
        }

        # Add system prompt if provided
        if system_prompt:
            options_kwargs["system_prompt"] = system_prompt

        # Add model config system prompt as fallback
        if not system_prompt and self._config.system_prompt:
            options_kwargs["system_prompt"] = self._config.system_prompt

        # Add model specification if provided
        if self._config.model_name:
            options_kwargs["model"] = self._config.model_name

        # Add structured output if configured
        if self._structured_schema is not None:
            json_schema = self._structured_schema.model_json_schema()
            options_kwargs["output_format"] = {
                "type": "json_schema",
                "schema": json_schema,
            }
            # SDK uses max_turns for autonomous retry on structured output
            options_kwargs["max_turns"] = self._max_turns

        # Build env dict for Anthropic settings (api_key, base_url)
        # The Claude Agent SDK reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL from env
        env_vars: dict[str, str] = {}
        if self._config.anthropic_api_key:
            env_vars["ANTHROPIC_API_KEY"] = self._config.anthropic_api_key.get_secret_value()
        else:
            env_vars.update(subscription_auth_env())
        if self._config.anthropic_base_url:
            env_vars["ANTHROPIC_BASE_URL"] = self._config.anthropic_base_url

        if env_vars:
            options_kwargs["env"] = env_vars

        return ClaudeAgentOptions(**options_kwargs)

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

        When ModelConfig.request_timeout is set, the whole query() drain is
        wrapped in asyncio.wait_for so the call has a hard wall-clock bound.
        The guard bounds only the wall clock seen by the caller: the SDK
        subprocess behind query() may linger until garbage collection after
        the timeout fires. A fired timeout raises a stock asyncio.TimeoutError
        (the builtin TimeoutError), which the ErrorRegistry classifies as
        TIMEOUT, matching what the langchain adapter surfaces.

        Args:
            messages: List of unified Message objects.

        Returns:
            LLMResponse with content, usage metadata, and raw response.

        Raises:
            PortError: If the invocation fails.
            TimeoutError: If the call exceeds ModelConfig.request_timeout.
        """
        from claude_agent_sdk import ResultMessage, query

        # Convert unified messages to SDK format
        prompt_string = self._converter.to_prompt_string(messages)
        system_prompt = self._converter.extract_system_prompt(messages)

        # Build options
        options = self._build_options(system_prompt)

        async def _drain_query() -> ResultMessage | None:
            collected: ResultMessage | None = None
            async for message in query(prompt=prompt_string, options=options):
                if isinstance(message, ResultMessage):
                    collected = message
            return collected

        # Execute query and collect result. One GlobalLLMLimiter permit
        # held for the whole query() drain. This path has no retry loop,
        # so the borrow is already per-attempt (uniform policy). Sync
        # invoke() reaches the wire only through this method, so it is
        # covered without a sync acquire.
        async with get_global_llm_limiter().borrow():
            if self._config.request_timeout is not None:
                result = await asyncio.wait_for(_drain_query(), timeout=self._config.request_timeout)
            else:
                result = await _drain_query()

        if result is None:
            from karenina.ports.errors import AgentResponseError

            raise AgentResponseError("No ResultMessage received from SDK")

        # Handle structured vs text output
        if self._structured_schema is not None:
            return self._process_structured_result(result)
        else:
            return self._process_text_result(result)

    def _process_text_result(self, result: ResultMessage) -> LLMResponse:
        """Process a text (non-structured) result from the SDK."""
        content = result.result or ""
        usage = extract_sdk_usage(result, model=self._config.model_name)
        return LLMResponse(content=content, usage=usage, raw=result)

    def _process_structured_result(self, result: ResultMessage) -> LLMResponse:
        """Process a structured output result from the SDK."""
        if not result.structured_output:
            raise ParseError("No structured output received from SDK")

        # Assert schema exists (this method only called when _structured_schema is set)
        assert self._structured_schema is not None

        parsed_model = self._structured_schema.model_validate(result.structured_output)
        content = parsed_model.model_dump_json()
        usage = extract_sdk_usage(result, model=self._config.model_name)

        return LLMResponse(content=content, usage=usage, raw=parsed_model)

    @with_llm_semaphore
    def invoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM synchronously.

        This is a convenience wrapper around ainvoke() for sync code.
        Always uses a dedicated thread with asyncio.run() to give the
        Claude Agent SDK a fresh event loop, avoiding cancel scope
        conflicts with BlockingPortal.

        Args:
            messages: List of unified Message objects.

        Returns:
            LLMResponse with content, usage metadata, and raw response.

        Raises:
            PortError: If the invocation fails.
        """
        thread_timeout = compute_sync_wrapper_timeout(
            self._config.request_timeout,
            retry_policy=self._config.retry_policy,
        )
        return _run_in_fresh_loop(self.ainvoke, messages, timeout=thread_timeout)

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        max_turns: int | None = None,
        max_retries: int | None = None,
    ) -> ClaudeSDKLLMAdapter:
        """Return a new adapter configured for structured output.

        The returned adapter uses SDK's output_format option to constrain
        the LLM's output to match the provided JSON schema.

        The SDK handles retries autonomously via max_turns, so no manual retry
        logic is needed.

        Args:
            schema: A Pydantic model class defining the output structure.
            max_turns: Maximum turns for SDK (handles retries autonomously).
                Default is 2 (minimum needed for structured output).
            max_retries: Not supported by this adapter. A warning is emitted
                if a non-None value is provided. The SDK handles retries
                autonomously via max_turns. See
                karenina.ports.llm.LLMPort.with_structured_output for the
                cross-adapter contract (LangChain and Claude Tool respect
                max_retries, this adapter and Deep Agents warn and ignore).

        Returns:
            A new ClaudeSDKLLMAdapter instance configured for structured output.
            The returned adapter guarantees that response.raw will be an instance
            of the provided schema.

        Example:
            >>> class Answer(BaseModel):
            ...     value: str
            ...     confidence: float
            >>> structured = adapter.with_structured_output(Answer)
            >>> response = await structured.ainvoke(messages)
            >>> assert isinstance(response.raw, Answer)
        """
        if max_retries is not None:
            logger.warning(
                "max_retries=%d ignored by claude_agent_sdk adapter; "
                "retry behavior is managed internally by the SDK via max_turns",
                max_retries,
            )
        return ClaudeSDKLLMAdapter(
            model_config=self._config,
            _structured_schema=schema,
            _max_turns=max_turns,
        )

    @asynccontextmanager
    async def astream(self, messages: list[Message]) -> AsyncIterator[StreamingLLMResponse]:  # noqa: ANN201
        """Stream LLM response via Claude Agent SDK.

        Uses query() with include_partial_messages=True to receive
        AssistantMessage objects as the response is built. Text deltas
        are computed by tracking previously seen content.

        Usage capture: partial AssistantMessages do NOT carry usage in the
        SDK, the authoritative usage arrives only on the final
        ResultMessage. With include_partial_messages=True the SDK also
        exposes raw StreamEvent payloads, whose ``message_start`` and
        ``message_delta`` events DO carry usage, so this method captures
        those inline as they arrive. A mid-stream interruption therefore
        still leaves whatever usage was streamed before the cut, and the
        ResultMessage overrides the inline snapshot on the success path.

        Args:
            messages: List of unified Message objects.

        Yields:
            StreamingLLMResponse that can be iterated for text chunks.
        """
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, query

        try:
            from claude_agent_sdk import StreamEvent  # type: ignore[attr-defined]
        except ImportError:
            StreamEvent = None

        prompt_string = self._converter.to_prompt_string(messages)
        system_prompt = self._converter.extract_system_prompt(messages)

        options = self._build_options(system_prompt)
        options.include_partial_messages = True

        response = StreamingLLMResponse()

        def _capture_stream_event_usage(usage_dict: dict[str, Any]) -> None:
            """Overlay a raw stream-event usage dict onto the inline snapshot."""
            from karenina.ports.usage import UsageMetadata

            current = response.usage
            input_tokens = usage_dict.get("input_tokens")
            output_tokens = usage_dict.get("output_tokens")
            cache_read = usage_dict.get("cache_read_input_tokens")
            cache_creation = usage_dict.get("cache_creation_input_tokens")
            merged_input = input_tokens if input_tokens is not None else current.input_tokens
            merged_output = output_tokens if output_tokens is not None else current.output_tokens
            response.usage = UsageMetadata(
                input_tokens=merged_input,
                output_tokens=merged_output,
                total_tokens=(merged_input or 0) + (merged_output or 0),
                cache_read_tokens=cache_read if cache_read is not None else current.cache_read_tokens,
                cache_creation_tokens=cache_creation if cache_creation is not None else current.cache_creation_tokens,
                model=self._config.model_name,
            )

        async def _chunk_generator() -> AsyncIterator[str]:  # noqa: ANN202
            """Yield text deltas from SDK partial AssistantMessages."""
            emitted_length = 0
            # The SDK query() starts lazily on first iteration, so the
            # first message fetch is the request establishment: it holds
            # one GlobalLLMLimiter permit, released before the first chunk
            # is yielded (the cap bounds concurrent stream setups, not
            # concurrent open streams).
            async for message in gate_stream_establishment(query(prompt=prompt_string, options=options)):
                if isinstance(message, AssistantMessage):
                    # Extract full text from content blocks
                    full_text = "".join(
                        block.text for block in getattr(message, "content", []) if isinstance(block, TextBlock)
                    )
                    # Yield only the new portion
                    if len(full_text) > emitted_length:
                        delta = full_text[emitted_length:]
                        emitted_length = len(full_text)
                        yield delta
                elif isinstance(message, ResultMessage):
                    # Final message carries the authoritative usage data
                    response.usage = extract_sdk_usage(message, model=self._config.model_name)
                elif StreamEvent is not None and isinstance(message, StreamEvent):
                    event = getattr(message, "event", None) or {}
                    event_type = event.get("type")
                    if event_type == "message_start":
                        _capture_stream_event_usage((event.get("message") or {}).get("usage") or {})
                    elif event_type == "message_delta":
                        _capture_stream_event_usage(event.get("usage") or {})

        response._set_chunk_source(_chunk_generator())
        yield response
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
        """Stream with wall-clock timeout synchronously.

        Always uses a dedicated thread with asyncio.run() to give the
        Claude Agent SDK a fresh event loop, avoiding cancel scope
        conflicts with BlockingPortal. Retries via RetryExecutor on
        transient errors (including StreamingTimeoutError with zero
        content, classified as RATE_LIMIT for queue congestion).

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
        return _run_in_fresh_loop(
            self._astream_with_timeout,
            messages,
            timeout,
            timeout=(timeout or 300) + 30,
        )

    async def aclose(self) -> None:
        """Close underlying resources.

        The Claude SDK adapter uses query() which doesn't hold persistent
        connections, so this is a no-op. Provided for interface consistency
        with other adapters that do require cleanup.
        """
        pass


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify ClaudeSDKLLMAdapter implements LLMPort protocol."""
    # This is a static check - if this fails, it means the adapter
    # doesn't properly implement the protocol
    adapter_instance: LLMPort = None  # type: ignore[assignment]

    # The following would fail mypy if the protocol wasn't properly implemented
    # We don't actually run this, it's just for static analysis
    _ = adapter_instance  # Suppress unused warning
