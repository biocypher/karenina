"""LangChain Deep Agents LLM adapter implementing the LLMPort interface.

This module provides the DeepAgentsLLMAdapter class that implements LLMPort
using LangChain's init_chat_model for simple single-turn LLM calls.

For single-turn calls, the adapter uses the LangChain model directly (not
create_deep_agent), since no agent loop or tool calling is needed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from karenina.adapters._parallel_base import with_llm_semaphore
from karenina.adapters.langchain.usage import extract_usage_from_chunk
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
        """Invoke the LLM asynchronously.

        Converts karenina Messages to LangChain format, invokes the model,
        and converts the response back.

        Args:
            messages: List of messages forming the conversation.

        Returns:
            LLMResponse containing the generated content and usage metadata.
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

        # Create model
        chat_model = create_chat_model(self._config)

        # Apply structured output if configured
        if self._structured_schema is not None:
            chat_model = chat_model.with_structured_output(self._structured_schema)

        # Invoke
        response = await chat_model.ainvoke(lc_messages)

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

            def run_in_thread() -> LLMResponse:
                return asyncio.run(self.ainvoke(messages))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=600)

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
                if a non-None value is provided.

        Returns:
            A new DeepAgentsLLMAdapter configured with the schema.
        """
        if max_retries is not None:
            logger.warning(
                "max_retries=%d ignored by langchain_deep_agents adapter; "
                "retry behavior is managed internally by LangChain",
                max_retries,
            )
        return DeepAgentsLLMAdapter(
            self._config,
            _structured_schema=schema,
        )

    @asynccontextmanager
    async def astream(self, messages: list[Message]) -> AsyncIterator[StreamingLLMResponse]:  # noqa: ANN201
        """Stream LLM response using LangChain's cross-provider astream.

        Args:
            messages: List of unified Message objects.

        Yields:
            StreamingLLMResponse that can be iterated for text chunks.
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

        async def _chunk_generator() -> AsyncIterator[str]:  # noqa: ANN202
            """Yield text chunks from LangChain's astream, extracting usage from final chunk."""
            async for chunk in chat_model.astream(lc_messages):
                text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    yield text
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    response.usage = extract_usage_from_chunk(chunk, model_name=self._config.model_name)

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

        Uses streaming internally so that partial content can be captured on
        timeout. Returns the same LLMResponse type as invoke(). Retries via
        RetryExecutor on transient errors (including StreamingTimeoutError
        with zero content, classified as RATE_LIMIT for queue congestion).

        Args:
            messages: List of unified Message objects.
            timeout: Wall-clock timeout in seconds. None means no timeout.

        Returns:
            LLMResponse with accumulated content.

        Raises:
            StreamingTimeoutError: If retries are exhausted and the stream
                still exceeds the wall-clock timeout.
        """
        result: LLMResponse = self._retry_executor.execute(self._stream_invoke_once, messages, timeout)
        return result

    def _stream_invoke_once(self, messages: list[Message], timeout: float | None = None) -> LLMResponse:
        """Single stream_invoke attempt (no retry). Called by RetryExecutor."""
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

    async def aclose(self) -> None:
        """Close underlying resources.

        No resources to clean up: the LangChain model is created fresh
        per ainvoke() call. Provided for interface consistency with other
        adapters.
        """
