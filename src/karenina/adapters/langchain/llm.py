"""LangChain LLM adapter implementing the LLMPort interface.

This module provides the LangChainLLMAdapter class that wraps existing
LangChain infrastructure (init_chat_model) behind the unified LLMPort interface.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from karenina.ports import LLMPort, LLMResponse, Message, UsageMetadata

from .messages import LangChainMessageConverter

if TYPE_CHECKING:
    from karenina.schemas.workflow.models import ModelConfig


class LangChainLLMAdapter:
    """LLM adapter using LangChain's init_chat_model.

    This adapter implements the LLMPort Protocol and wraps the existing
    karenina infrastructure for LangChain-based LLM invocation.

    The adapter handles:
    - Message conversion between unified Message and LangChain formats
    - Usage metadata extraction from LangChain responses
    - Sync/async invocation with proper event loop handling
    - Structured output via with_structured_output()

    Example:
        >>> from karenina.schemas.workflow.models import ModelConfig
        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic"
        ... )
        >>> adapter = LangChainLLMAdapter(config)
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
        _underlying_model: Any = None,
    ) -> None:
        """Initialize the LangChain LLM adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
            _structured_schema: Internal - schema for structured output mode.
            _underlying_model: Internal - pre-initialized model (used by with_structured_output).
        """
        self._config = model_config
        self._converter = LangChainMessageConverter()
        self._structured_schema = _structured_schema

        if _underlying_model is not None:
            self._model = _underlying_model
        else:
            self._model = self._initialize_model()

    def _initialize_model(self) -> Any:
        """Initialize the underlying LangChain model.

        Returns:
            Initialized LangChain chat model.
        """
        from karenina.infrastructure.llm.interface import init_chat_model_unified

        # Build kwargs for model initialization
        kwargs: dict[str, Any] = {}

        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature

        if self._config.extra_kwargs:
            kwargs.update(self._config.extra_kwargs)

        # Initialize model via existing infrastructure
        model = init_chat_model_unified(
            model=self._config.model_name or "",
            provider=self._config.model_provider,
            interface=self._config.interface,
            endpoint_base_url=self._config.endpoint_base_url,
            endpoint_api_key=self._config.endpoint_api_key,
            max_context_tokens=self._config.max_context_tokens,
            # Note: We don't pass MCP config here - simple LLM invocation
            # For agents with MCP, use LangChainAgentAdapter instead
            **kwargs,
        )

        return model

    def _extract_usage(self, response: Any) -> UsageMetadata:
        """Extract usage metadata from LangChain response.

        Args:
            response: LangChain AIMessage or similar response object.

        Returns:
            UsageMetadata with token counts and cost info.
        """
        # LangChain stores usage in response_metadata or usage_metadata attribute
        usage_data: dict[str, Any] = {}

        # Try response_metadata first (newer LangChain versions)
        if hasattr(response, "response_metadata"):
            metadata = response.response_metadata or {}
            # Anthropic/OpenAI style: token_usage or usage
            usage_data = metadata.get("token_usage") or metadata.get("usage") or {}

        # Fall back to usage_metadata attribute
        if not usage_data and hasattr(response, "usage_metadata"):
            usage_data = response.usage_metadata or {}

        input_tokens = usage_data.get("input_tokens") or usage_data.get("prompt_tokens") or 0
        output_tokens = usage_data.get("output_tokens") or usage_data.get("completion_tokens") or 0
        total_tokens = usage_data.get("total_tokens") or (input_tokens + output_tokens)

        # Extract cache tokens if available (Anthropic)
        cache_read = usage_data.get("cache_read_input_tokens") or usage_data.get("cache_read_tokens")
        cache_creation = usage_data.get("cache_creation_input_tokens") or usage_data.get("cache_creation_tokens")

        return UsageMetadata(
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            total_tokens=int(total_tokens),
            cache_read_tokens=int(cache_read) if cache_read else None,
            cache_creation_tokens=int(cache_creation) if cache_creation else None,
            model=self._config.model_name,
        )

    async def ainvoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM asynchronously.

        Args:
            messages: List of unified Message objects.

        Returns:
            LLMResponse with content, usage metadata, and raw response.

        Raises:
            PortError: If the invocation fails.
        """
        # Convert unified messages to LangChain format
        lc_messages = self._converter.to_provider(messages)

        # Invoke the underlying model
        response = await self._model.ainvoke(lc_messages)

        # Extract content - handle both structured and text responses
        if self._structured_schema is not None:
            # Structured output - response may be the parsed model itself
            content = str(response) if not hasattr(response, "content") else str(response.content)
        else:
            content = str(response.content) if hasattr(response, "content") else str(response)

        # Extract usage metadata
        usage = self._extract_usage(response)

        return LLMResponse(
            content=content,
            usage=usage,
            raw=response,
        )

    def invoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM synchronously.

        This is a convenience wrapper around ainvoke() for sync code.
        Uses the shared async portal if available, otherwise falls back
        to asyncio.run() with proper event loop handling.

        Args:
            messages: List of unified Message objects.

        Returns:
            LLMResponse with content, usage metadata, and raw response.

        Raises:
            PortError: If the invocation fails.
        """
        from karenina.benchmark.verification.batch_runner import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            # Use the shared BlockingPortal for proper event loop management
            return portal.call(self.ainvoke, messages)

        # No portal available - check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - use ThreadPoolExecutor to avoid
            # nested event loop issues

            def run_in_thread() -> LLMResponse:
                return asyncio.run(self.ainvoke(messages))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=300)  # 5 minute timeout

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.ainvoke(messages))

    def with_structured_output(self, schema: type[BaseModel]) -> LangChainLLMAdapter:
        """Return a new adapter configured for structured output.

        The returned adapter uses LangChain's with_structured_output() to
        constrain the LLM's output format to match the provided schema.

        Args:
            schema: A Pydantic model class defining the output structure.

        Returns:
            A new LangChainLLMAdapter instance configured for structured output.

        Example:
            >>> class Answer(BaseModel):
            ...     value: str
            ...     confidence: float
            >>> structured = adapter.with_structured_output(Answer)
        """
        # Create structured model via LangChain's with_structured_output
        structured_model = self._model.with_structured_output(schema)

        return LangChainLLMAdapter(
            model_config=self._config,
            _structured_schema=schema,
            _underlying_model=structured_model,
        )


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify LangChainLLMAdapter implements LLMPort protocol."""
    # This is a static check - if this fails, it means the adapter
    # doesn't properly implement the protocol
    adapter_instance: LLMPort = None  # type: ignore[assignment]

    # The following would fail mypy if the protocol wasn't properly implemented
    # We don't actually run this, it's just for static analysis
    _ = adapter_instance  # Suppress unused warning
