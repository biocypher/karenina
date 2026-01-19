"""LLM Port interface for simple LLM invocation.

This module defines the LLMPort Protocol for stateless LLM calls without
agent loops. Use this for simple text generation and structured output tasks.
For multi-turn agent execution with tools/MCP, use AgentPort instead.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from karenina.ports.messages import Message
from karenina.ports.usage import UsageMetadata


@dataclass
class LLMResponse:
    """Response from an LLM invocation.

    Attributes:
        content: The text content of the response.
        usage: Token usage and cost metadata.
        raw: Provider-specific raw response object for advanced use cases.

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

    def with_structured_output(self, schema: type[BaseModel]) -> "LLMPort":
        """Return a new LLMPort configured for structured output.

        The returned LLMPort will use the provided schema to constrain
        the LLM's output format. The schema is converted to JSON schema
        and passed to the LLM provider's structured output feature.

        Args:
            schema: A Pydantic model class defining the output structure.

        Returns:
            A new LLMPort instance configured for structured output.

        Example:
            >>> class Answer(BaseModel):
            ...     value: str
            ...     confidence: float
            >>> structured_llm = llm.with_structured_output(Answer)
        """
        ...
