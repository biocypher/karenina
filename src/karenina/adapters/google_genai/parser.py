"""google_genai parser adapter implementing the ParserPort interface.

A thin wrapper over the LLM adapter's structured output, mirroring
``ClaudeToolParserAdapter``. Parsing extracts structured data from text the
caller provides, so Gemini's server-side tools (Google Search, URL reading)
stay off for parser calls even when the ModelConfig enables them for the
agent: an extraction step must not search and change its input.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TypeVar

from pydantic import BaseModel

from karenina.adapters._parallel_base import run_coro_in_thread
from karenina.adapters._timeouts import compute_sync_wrapper_timeout
from karenina.ports import Message, ParseError, ParsePortResult, ParserPort, PortError
from karenina.ports.capabilities import PortCapabilities
from karenina.schemas.config import ModelConfig

from .errors import build_error_registry
from .llm import GoogleGenAILLMAdapter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GoogleGenAIParserAdapter:
    """Parser adapter using Gemini structured output.

    Example:
        >>> parser = GoogleGenAIParserAdapter(config)
        >>> result = await parser.aparse_to_pydantic(messages, Answer)
        >>> result.parsed, result.usage
    """

    def __init__(self, model_config: ModelConfig, *, max_retries: int = 2) -> None:
        """Initialize the parser.

        Args:
            model_config: Configuration with ``interface="google_genai"``.
            max_retries: Validation retries with error feedback.

        Raises:
            ValueError: If the configuration is invalid.
        """
        self._config = model_config
        self._llm_adapter = GoogleGenAILLMAdapter(model_config, _use_builtin_tools=False)
        self._max_retries = max_retries
        self._error_registry = build_error_registry()

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare capabilities (system prompt and native structured output)."""
        return PortCapabilities(supports_system_prompt=True, supports_structured_output=True)

    async def aparse_to_pydantic(self, messages: list[Message], schema: type[T]) -> ParsePortResult[T]:
        """Parse pre-assembled prompt messages into the schema.

        Args:
            messages: Pre-assembled prompt messages (system + user).
            schema: Pydantic model class to extract.

        Returns:
            ParsePortResult with the validated model and the usage summed
            over the validation retries.

        Raises:
            ParseError: If validation fails after all retries.
            PortError: API, auth and finish-reason errors pass through.
        """
        structured = self._llm_adapter.with_structured_output(schema, max_retries=self._max_retries)
        try:
            response = await structured.ainvoke(messages)
        except PortError:
            raise
        except Exception as e:
            # Transient failures that outlived their retries keep their type,
            # so the pipeline still classifies them as retryable.
            if self._error_registry.classify(e).is_retryable():
                raise
            raise ParseError(f"Failed to parse response into {schema.__name__}: {e}") from e

        if not isinstance(response.raw, schema):
            raise ParseError(
                f"Structured output did not return a {schema.__name__} instance, got {type(response.raw).__name__}"
            )
        return ParsePortResult(parsed=response.raw, usage=response.usage)

    def parse_to_pydantic(self, messages: list[Message], schema: type[T]) -> ParsePortResult[T]:
        """Parse synchronously (wrapper around ``aparse_to_pydantic``).

        Args:
            messages: Pre-assembled prompt messages (system + user).
            schema: Pydantic model class to extract.

        Returns:
            ParsePortResult as from ``aparse_to_pydantic``.
        """
        from karenina.benchmark.verification.async_lifecycle import get_async_portal

        portal = get_async_portal()
        if portal is not None:
            return portal.call(self.aparse_to_pydantic, messages, schema)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aparse_to_pydantic(messages, schema))
        thread_timeout = compute_sync_wrapper_timeout(
            self._llm_adapter.settings.request_timeout,
            retry_policy=self._config.retry_policy,
            internal_call_sequences=self._max_retries + 2,
        )
        return run_coro_in_thread(self.aparse_to_pydantic, messages, schema, timeout=thread_timeout)

    async def aclose(self) -> None:
        """Close the inner LLM adapter. Safe to call twice."""
        await self._llm_adapter.aclose()


def _verify_protocol_compliance() -> None:
    """Verify GoogleGenAIParserAdapter implements the ParserPort protocol."""
    adapter_instance: ParserPort = GoogleGenAIParserAdapter.__new__(GoogleGenAIParserAdapter)
    _ = adapter_instance
