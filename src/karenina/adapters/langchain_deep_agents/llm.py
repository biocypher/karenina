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
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from karenina.ports import LLMResponse, Message
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.usage import UsageMetadata

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

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare adapter capabilities.

        Returns:
            PortCapabilities with system_prompt=True and structured_output=True.
        """
        return PortCapabilities(
            supports_system_prompt=True,
            supports_structured_output=True,
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

    async def aclose(self) -> None:
        """Close underlying resources.

        No resources to clean up: the LangChain model is created fresh
        per ainvoke() call. Provided for interface consistency with other
        adapters.
        """
