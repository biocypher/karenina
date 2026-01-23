"""Claude Tool LLM adapter implementing the LLMPort interface.

This module provides the ClaudeToolLLMAdapter class that uses the Anthropic Python SDK
directly (client.messages.create) for simple LLM invocations without agent loops.

Key features:
- Uses Anthropic's native Python SDK for efficient API calls
- Supports structured output via client.beta.messages.parse() with Pydantic
- Implements prompt caching for efficiency
- Uses SDK's built-in retry logic for transient errors
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from karenina.ports import AdapterUnavailableError, LLMPort, LLMResponse, Message, ParseError
from karenina.schemas.workflow.models import ModelConfig
from karenina.utils.messages import append_error_feedback

from .messages import build_system_with_cache, convert_to_anthropic, extract_system_prompt
from .usage import extract_usage_from_response

# Load environment variables from .env file (for ANTHROPIC_API_KEY)
load_dotenv()

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

    Note: Transient error retries are handled by the Anthropic SDK (default: 2 retries
    with exponential backoff for connection errors, timeouts, rate limits, and 5xx errors).

    Example:
        >>> from karenina.schemas.workflow.models import ModelConfig
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

            # SDK handles retries automatically (default: 2 retries with exponential backoff)
            self._client = Anthropic()
        return self._client

    def _get_async_client(self) -> Any:
        """Get or create the async Anthropic client."""
        if self._async_client is None:
            from anthropic import AsyncAnthropic

            # SDK handles retries automatically (default: 2 retries with exponential backoff)
            self._async_client = AsyncAnthropic()
        return self._async_client

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

        # Convert messages
        anthropic_messages = convert_to_anthropic(messages)
        system_prompt = extract_system_prompt(messages)

        # Build kwargs
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
        return LLMResponse(
            content=str(parsed_output),
            usage=usage,
            raw=parsed_output,
        )

    def invoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM synchronously.

        This is a convenience wrapper around ainvoke() for sync code.

        Args:
            messages: List of unified Message objects.

        Returns:
            LLMResponse with content, usage metadata, and raw response.
        """
        from karenina.benchmark.verification.batch_runner import get_async_portal

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


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify ClaudeToolLLMAdapter implements LLMPort protocol."""
    adapter_instance: LLMPort = None  # type: ignore[assignment]
    _ = adapter_instance
