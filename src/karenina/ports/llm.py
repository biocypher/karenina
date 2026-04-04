"""LLM Port interface for simple LLM invocation.

This module defines the LLMPort Protocol for stateless LLM calls without
agent loops. Use this for simple text generation and structured output tasks.
For multi-turn agent execution with tools/MCP, use AgentPort instead.
"""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from karenina.ports.capabilities import PortCapabilities
from karenina.ports.messages import Message
from karenina.ports.usage import UsageMetadata


@dataclass
class LLMResponse:
    """Response from an LLM invocation.

    Attributes:
        content: The text content of the response.
        usage: Token usage and cost metadata.
        raw: Provider-specific raw response object for advanced use cases.
        is_partial: Whether the response was truncated (e.g., due to streaming timeout).
        usage_unavailable: Whether usage metadata could not be captured (e.g., streaming
            timeout interrupted the final chunk that carries token counts). When True,
            usage fields are zero but do not reflect actual consumption.

    Example:
        >>> response = LLMResponse(
        ...     content="Hello, world!",
        ...     usage=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
        ...     raw={"id": "msg_123", "model": "claude-sonnet-4-20250514"}
        ... )
        >>> response.content
        'Hello, world!'
    """

    content: str
    usage: UsageMetadata
    raw: Any = None
    is_partial: bool = False
    usage_unavailable: bool = False


class StreamingLLMResponse:
    """Accumulating response from a streaming LLM invocation.

    Async-iterable: yields str chunks. Accumulates content internally.
    On timeout or error, ``accumulated_content`` has whatever arrived.

    Example:
        >>> async with llm.astream(messages) as sr:
        ...     async for chunk in sr:
        ...         print(chunk, end="")
        ...     print(f"\\nTotal: {len(sr.accumulated_content)} chars")
    """

    def __init__(self) -> None:
        self.accumulated_content: str = ""
        self.usage: UsageMetadata = UsageMetadata()
        self.is_complete: bool = False
        self._chunks: AsyncIterator[str] | None = None

    def _set_chunk_source(self, chunks: AsyncIterator[str]) -> None:
        """Set the async iterator that produces text chunks."""
        self._chunks = chunks

    def __aiter__(self) -> "StreamingLLMResponse":
        return self

    async def __anext__(self) -> str:
        if self._chunks is None:
            raise StopAsyncIteration
        chunk = await self._chunks.__anext__()
        self.accumulated_content += chunk
        return chunk


@runtime_checkable
class LLMPort(Protocol):
    """Protocol for simple LLM invocation.

    This interface defines stateless LLM calls without agent loops.
    For multi-turn agent execution with tools/MCP, use AgentPort instead.

    Implementations must provide:
    - ainvoke(): Async invocation (primary API)
    - invoke(): Sync wrapper for convenience
    - with_structured_output(): Return a new LLMPort configured for structured output

    Example:
        >>> llm = get_llm(model_config)
        >>> response = await llm.ainvoke([Message.user("Hello!")])
        >>> print(response.content)
        'Hello! How can I help you today?'

        >>> # With structured output
        >>> class Answer(BaseModel):
        ...     value: str
        >>> structured_llm = llm.with_structured_output(Answer)
        >>> response = await structured_llm.ainvoke([Message.user("What is 2+2?")])
    """

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare what prompt features this LLM adapter supports.

        Returns:
            PortCapabilities with adapter-specific feature flags.
            Defaults to PortCapabilities() (system prompts supported,
            structured output not supported).
        """
        return PortCapabilities()

    async def ainvoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM asynchronously.

        Args:
            messages: List of messages forming the conversation.

        Returns:
            LLMResponse containing the generated content and usage metadata.

        Raises:
            PortError: If the invocation fails.
        """
        ...

    def invoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM synchronously.

        This is a convenience wrapper around ainvoke() for sync code.

        Args:
            messages: List of messages forming the conversation.

        Returns:
            LLMResponse containing the generated content and usage metadata.

        Raises:
            PortError: If the invocation fails.
        """
        ...

    def with_structured_output(self, schema: type[BaseModel], *, max_retries: int | None = None) -> "LLMPort":
        """Return a new LLMPort configured for structured output.

        The returned LLMPort will use the provided schema to constrain
        the LLM's output format. The schema is converted to JSON schema
        and passed to the LLM provider's structured output feature.

        Args:
            schema: A Pydantic model class defining the output structure.
            max_retries: Maximum retry attempts on validation failure.
                Not all adapters support this parameter. LangChain and Claude
                Tool adapters respect it; Claude SDK and Deep Agents adapters
                ignore it (with a warning). Check adapter documentation for
                details.

        Returns:
            A new LLMPort instance configured for structured output.

        Example:
            >>> class Answer(BaseModel):
            ...     value: str
            ...     confidence: float
            >>> structured_llm = llm.with_structured_output(Answer)
        """
        ...

    def astream(self, messages: list[Message]) -> AbstractAsyncContextManager[StreamingLLMResponse]:
        """Open a streaming LLM connection.

        Yields a StreamingLLMResponse that produces text chunks and accumulates
        content. Check capabilities.supports_streaming before calling.

        Args:
            messages: List of messages forming the conversation.

        Returns:
            Async context manager yielding StreamingLLMResponse.

        Raises:
            NotImplementedError: If the adapter does not support streaming.
        """
        raise NotImplementedError

    def stream_invoke(self, messages: list[Message], timeout: float | None = None) -> LLMResponse:
        """Invoke the LLM with streaming and optional wall-clock timeout.

        Sync wrapper that streams tokens and captures partial output on timeout.
        Check capabilities.supports_streaming before calling.

        Args:
            messages: List of messages forming the conversation.
            timeout: Wall-clock timeout in seconds. If exceeded, returns partial
                content with is_partial=True.

        Returns:
            LLMResponse with content (possibly partial) and usage metadata.

        Raises:
            NotImplementedError: If the adapter does not support streaming.
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """Close underlying resources.

        Implementations should release any held resources (HTTP connections,
        file handles, MCP sessions). Safe to call multiple times. The default
        is a no-op for adapters with no resources to clean up.
        """
        ...
