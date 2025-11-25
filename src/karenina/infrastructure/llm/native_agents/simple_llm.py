"""Simple native LLM wrapper for non-MCP calls.

This module provides the NativeSimpleLLM class that uses OpenAI/Anthropic SDKs
directly for simple LLM calls (no tool calling). This bypasses LangChain entirely
while providing a compatible interface.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NativeSimpleLLMResponse:
    """Response from native simple LLM call."""

    content: str
    usage: dict[str, Any] = field(default_factory=dict)


class NativeSimpleLLM:
    """Simple LLM wrapper using native SDKs without tool calling.

    This class provides a LangChain-compatible interface (invoke/ainvoke) while
    using OpenAI or Anthropic SDKs directly. Useful when you want to bypass
    LangChain overhead for simple LLM calls.

    Supports:
    - OpenAI: gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini, o1, o3, etc.
    - Anthropic: claude-sonnet-4, claude-haiku-4.5, etc.
    """

    client: Any  # Type: AsyncOpenAI | AsyncAnthropic

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.0,
        api_key: str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the native simple LLM.

        Args:
            provider: "openai" or "anthropic"
            model: Model name (e.g., "gpt-4.1-mini", "claude-sonnet-4-20250514")
            temperature: Model temperature (default: 0.0)
            api_key: Optional explicit API key (falls back to env vars)
            system_prompt: Optional system prompt
            **kwargs: Additional model parameters (e.g., max_tokens)
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.extra_kwargs = kwargs

        if provider == "openai":
            self._init_openai(api_key)
        elif provider == "anthropic":
            self._init_anthropic(api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'.")

    def _init_openai(self, api_key: str | None) -> None:
        """Initialize OpenAI client."""
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for native OpenAI calls. Install it with: pip install openai>=1.0.0"
            ) from e

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key explicitly."
            )

        self.client = openai.AsyncOpenAI(api_key=resolved_key)

    def _init_anthropic(self, api_key: str | None) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for native Anthropic calls. "
                "Install it with: pip install anthropic>=0.25.0"
            ) from e

        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or pass api_key explicitly."
            )

        self.client = anthropic.AsyncAnthropic(api_key=resolved_key)

    async def ainvoke(self, messages: list[Any]) -> Any:
        """Async invoke the LLM.

        Args:
            messages: List of messages (LangChain BaseMessage or dict format)

        Returns:
            LangChain-compatible AIMessage
        """
        # Convert messages to native format
        native_messages = self._convert_messages(messages)

        if self.provider == "openai":
            return await self._ainvoke_openai(native_messages)
        else:
            return await self._ainvoke_anthropic(native_messages)

    def invoke(self, messages: list[Any]) -> Any:
        """Sync invoke the LLM.

        Args:
            messages: List of messages (LangChain BaseMessage or dict format)

        Returns:
            LangChain-compatible AIMessage
        """
        try:
            asyncio.get_running_loop()
            # Already in async context, use thread pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(self.ainvoke(messages)))
                return future.result(timeout=120)
        except RuntimeError:
            # No event loop, safe to use asyncio.run
            return asyncio.run(self.ainvoke(messages))

    def _convert_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Convert LangChain messages to native format."""
        native_messages = []

        for msg in messages:
            if isinstance(msg, dict):
                # Already in dict format
                native_messages.append(msg)
            elif hasattr(msg, "type") and hasattr(msg, "content"):
                # LangChain message
                role_map = {"system": "system", "human": "user", "ai": "assistant"}
                role = role_map.get(msg.type, msg.type)
                native_messages.append({"role": role, "content": msg.content})
            else:
                # Unknown format, try to convert
                native_messages.append({"role": "user", "content": str(msg)})

        return native_messages

    async def _ainvoke_openai(self, messages: list[dict[str, Any]]) -> Any:
        """Invoke OpenAI API."""
        from langchain_core.messages import AIMessage

        # Build request
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        # Add extra kwargs (e.g., max_tokens)
        for key, value in self.extra_kwargs.items():
            if key not in request_kwargs:
                request_kwargs[key] = value

        # Call OpenAI API
        response = await self.client.chat.completions.create(**request_kwargs)

        # Extract content and usage
        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        # Return LangChain-compatible AIMessage
        return AIMessage(
            content=content,
            response_metadata={
                "model": self.model,
                "usage": usage,
            },
        )

    async def _ainvoke_anthropic(self, messages: list[dict[str, Any]]) -> Any:
        """Invoke Anthropic API."""
        from langchain_core.messages import AIMessage

        # Anthropic requires system prompt separate from messages
        system_content = self.system_prompt or ""
        conversation = []

        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                conversation.append(msg)

        # Build request
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.extra_kwargs.get("max_tokens", 4096),
            "messages": conversation,
            "temperature": self.temperature,
        }

        # Add system prompt if present
        if system_content:
            request_kwargs["system"] = system_content

        # Add extra kwargs (excluding max_tokens which we handle above)
        for key, value in self.extra_kwargs.items():
            if key not in request_kwargs and key != "max_tokens":
                request_kwargs[key] = value

        # Call Anthropic API
        response = await self.client.messages.create(**request_kwargs)

        # Extract content
        content = ""
        for block in response.content:
            if block.type == "text":
                content = block.text
                break

        # Extract usage
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens or 0,
                "output_tokens": response.usage.output_tokens or 0,
                "total_tokens": (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0),
            }

        # Return LangChain-compatible AIMessage
        return AIMessage(
            content=content,
            response_metadata={
                "model": self.model,
                "usage": usage,
            },
        )
