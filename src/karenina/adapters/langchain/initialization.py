"""LangChain model initialization for the adapter layer.

Internal module providing unified model initialization across multiple interfaces.
External code should use the LLMPort interface via get_llm() instead.

Supported interfaces:
    - langchain: Standard LangChain init_chat_model
    - openrouter: OpenRouter API
    - openai_endpoint: OpenAI-compatible endpoints (vLLM, Ollama, etc.)

Note: Manual interface is handled separately via ManualAgentAdapter in
karenina.adapters.manual. The LangChain adapter layer does not handle manual traces.

For agent creation with MCP tools, use LangChainAgentAdapter instead.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.chat_models import init_chat_model
from pydantic import SecretStr

from karenina.ports import AdapterUnavailableError

from .models import ChatOpenAIEndpoint, ChatOpenRouter

logger = logging.getLogger(__name__)


def init_chat_model_unified(
    model: str,
    provider: str | None = None,
    interface: str = "langchain",
    endpoint_base_url: str | None = None,
    endpoint_api_key: str | SecretStr | None = None,
    **kwargs: Any,
) -> Any:
    """Initialize a chat model using the unified interface.

    Internal function for LangChain adapter layer. External code should use
    get_llm() (LLMPort) or get_agent() (AgentPort) instead.

    Args:
        model: Model name. Examples by provider:
            - Anthropic: "claude-sonnet-4", "claude-sonnet-4.5", "claude-haiku-4.5"
            - OpenAI: "gpt-4.1", "gpt-4.1-mini", "gpt-5.2"
            - Google: "gemini-2.5-flash", "gemini-3-pro", "gemini-3-flash"
        provider: Model provider (e.g., "google_genai", "openai", "anthropic").
            Optional for openrouter and openai_endpoint interfaces.
        interface: Interface for initialization. One of:
            "langchain" (default), "openrouter", "openai_endpoint"
        endpoint_base_url: Base URL for openai_endpoint interface. Required.
        endpoint_api_key: API key for openai_endpoint interface. Required.
            Must be explicitly provided; does NOT read from environment.
        **kwargs: Additional args passed to underlying model (temperature, etc.)

    Returns:
        Initialized chat model instance.

    Raises:
        ValueError: If required args missing for the specified interface.

    Examples:
        >>> # LangChain (default)
        >>> model = init_chat_model_unified("gemini-2.5-flash", "google_genai")

        >>> # OpenRouter
        >>> model = init_chat_model_unified("gpt-4.1-mini", interface="openrouter")

        >>> # OpenAI-compatible endpoint (vLLM, Ollama, etc.)
        >>> model = init_chat_model_unified(
        ...     "glm-4.7",
        ...     interface="openai_endpoint",
        ...     endpoint_base_url="http://localhost:11434/v1",
        ...     endpoint_api_key="your-api-key",
        ... )
    """
    # Filter out karenina-specific parameters that shouldn't be passed to underlying models
    # max_context_tokens is used by middleware (summarization trigger), not by LangChain models
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != "max_context_tokens"}

    if interface == "langchain":
        return init_chat_model(model=model, model_provider=provider, **filtered_kwargs)

    if interface == "openrouter":
        return ChatOpenRouter(model=model, **filtered_kwargs)

    if interface == "openai_endpoint":
        if endpoint_base_url is None:
            raise AdapterUnavailableError(
                "endpoint_base_url is required for openai_endpoint interface",
                reason="missing_endpoint_base_url",
            )
        if endpoint_api_key is None:
            raise AdapterUnavailableError(
                "endpoint_api_key is required for openai_endpoint interface. "
                "Pass the API key explicitly - this interface does not read from environment.",
                reason="missing_endpoint_api_key",
            )
        return ChatOpenAIEndpoint(
            base_url=endpoint_base_url,
            openai_api_key=endpoint_api_key,
            model=model,
            **filtered_kwargs,
        )

    # Unknown interface - should not happen if AdapterFactory routes correctly
    logger.warning(
        f"Unknown interface '{interface}' in LangChain adapter; "
        f"falling back to init_chat_model. Check AdapterFactory routing."
    )
    return init_chat_model(model=model, model_provider=provider, **filtered_kwargs)
