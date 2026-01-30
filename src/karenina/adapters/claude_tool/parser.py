"""Claude Tool Parser adapter implementing the ParserPort interface.

This module provides the ClaudeToolParserAdapter class that uses the Anthropic Python SDK's
beta.messages.parse() for structured output parsing via the LLM adapter's with_structured_output().

Key features:
- Uses Anthropic's native structured output (beta.messages.parse)
- Supports Pydantic models for type-safe extraction
- Simple delegation to LLM adapter's structured output capability
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import TypeVar

from pydantic import BaseModel

from karenina.ports import Message, ParseError, ParserPort
from karenina.ports.capabilities import PortCapabilities
from karenina.schemas.workflow.models import ModelConfig

from .llm import ClaudeToolLLMAdapter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ClaudeToolParserAdapter:
    """Parser adapter using Anthropic SDK's beta.messages.parse().

    This adapter implements the ParserPort Protocol and provides LLM-based
    structured output parsing. It invokes Claude to interpret natural language
    responses and extract structured data according to a Pydantic schema.

    The adapter delegates to ClaudeToolLLMAdapter.with_structured_output() which
    uses Anthropic's native structured output API (beta.messages.parse). The API
    constrains model output to match the Pydantic schema exactly.

    Example:
        >>> from karenina.schemas.workflow.models import ModelConfig
        >>> from pydantic import BaseModel, Field

        >>> class Answer(BaseModel):
        ...     gene_name: str = Field(description="The gene mentioned")
        ...     is_oncogene: bool = Field(description="Whether it's an oncogene")

        >>> config = ModelConfig(
        ...     id="claude-haiku",
        ...     model_name="claude-haiku-4-5",
        ...     model_provider="anthropic",
        ...     interface="claude_tool"
        ... )
        >>> parser = ClaudeToolParserAdapter(config)
        >>> trace = "Based on my analysis, BCL2 is an anti-apoptotic gene..."
        >>> answer = await parser.aparse_to_pydantic(trace, Answer)
        >>> print(answer.gene_name)
        'BCL2'
    """

    def __init__(self, model_config: ModelConfig, *, max_retries: int = 2) -> None:
        """Initialize the Claude Tool parser adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
            max_retries: Maximum validation retry attempts (passed to structured output).
        """
        self._config = model_config
        self._llm_adapter = ClaudeToolLLMAdapter(model_config)
        self._max_retries = max_retries

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare what prompt features this parser adapter supports.

        Claude Tool adapter uses Anthropic SDK which supports both
        system prompts and native structured output.

        Returns:
            PortCapabilities with supports_system_prompt=True, supports_structured_output=True.
        """
        return PortCapabilities(supports_system_prompt=True, supports_structured_output=True)

    async def aparse_to_pydantic(self, messages: list[Message], schema: type[T]) -> T:
        """Parse using pre-assembled prompt messages into a structured Pydantic model.

        Uses Anthropic's native structured output API (beta.messages.parse) which
        constrains model output to match the Pydantic schema exactly.

        Args:
            messages: Pre-assembled prompt messages (system + user).
            schema: A Pydantic model class defining the expected structure.

        Returns:
            An instance of the schema type with extracted values.

        Raises:
            ParseError: If the LLM fails to extract valid structured data.
        """
        try:
            structured_adapter = self._llm_adapter.with_structured_output(schema, max_retries=self._max_retries)
            llm_response = await structured_adapter.ainvoke(messages)

            # The structured output API returns the parsed model in .raw
            if isinstance(llm_response.raw, schema):
                return llm_response.raw

            # Fallback: shouldn't happen with native structured output
            raise ParseError(
                f"Structured output did not return {schema.__name__} instance, got {type(llm_response.raw)}"
            )

        except Exception as e:
            if isinstance(e, ParseError):
                raise
            raise ParseError(f"Failed to parse response into {schema.__name__}: {e}") from e

    def parse_to_pydantic(self, messages: list[Message], schema: type[T]) -> T:
        """Parse using pre-assembled prompt messages (sync).

        Args:
            messages: Pre-assembled prompt messages (system + user).
            schema: A Pydantic model class defining the expected structure.

        Returns:
            An instance of the schema type with extracted values.

        Raises:
            ParseError: If parsing fails.
        """
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            return portal.call(self.aparse_to_pydantic, messages, schema)

        try:
            asyncio.get_running_loop()

            def run_in_thread() -> T:
                return asyncio.run(self.aparse_to_pydantic(messages, schema))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=300)

        except RuntimeError:
            return asyncio.run(self.aparse_to_pydantic(messages, schema))

    async def aclose(self) -> None:
        """Close underlying HTTP client resources.

        Delegates to the internal LLM adapter's aclose() method.
        Safe to call multiple times.
        """
        await self._llm_adapter.aclose()


# Protocol verification
def _verify_protocol_compliance() -> None:
    """Verify ClaudeToolParserAdapter implements ParserPort protocol."""
    adapter_instance: ParserPort = None  # type: ignore[assignment]
    _ = adapter_instance
