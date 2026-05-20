"""Model initialization for the Deep Agents adapter.

Adapted from the LangChain adapter's init_chat_model_unified. Creates a
BaseChatModel instance that can be passed to create_deep_agent(model=...).

This is a separate copy (not a cross-adapter import) to maintain adapter
isolation per karenina conventions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.chat_models import init_chat_model
from pydantic import SecretStr

from karenina.ports import AdapterUnavailableError

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig

logger = logging.getLogger(__name__)


def _normalize_openai_endpoint_url(base_url: str) -> str:
    """Normalize OpenAI-compatible endpoint URLs to include /v1."""

    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def create_chat_model(model_config: ModelConfig, **kwargs: Any) -> Any:
    """Create a LangChain BaseChatModel from a karenina ModelConfig.

    Args:
        model_config: Configuration specifying model name and provider.
        **kwargs: Additional arguments passed to init_chat_model.

    Returns:
        Initialized BaseChatModel instance.

    Raises:
        AdapterUnavailableError: If required configuration is missing.
    """
    if not model_config.model_name:
        raise AdapterUnavailableError(
            "model_name is required for langchain_deep_agents interface",
            reason="missing_model_name",
        )

    if not model_config.model_provider:
        raise AdapterUnavailableError(
            "model_provider is required for langchain_deep_agents interface",
            reason="missing_model_provider",
        )

    model_kwargs: dict[str, Any] = {}
    if model_config.temperature is not None:
        model_kwargs["temperature"] = model_config.temperature
    if model_config.request_timeout is not None:
        if model_config.model_provider == "anthropic":
            model_kwargs["default_request_timeout"] = model_config.request_timeout
        else:
            model_kwargs["request_timeout"] = model_config.request_timeout
    if model_config.endpoint_base_url is not None:
        if model_config.endpoint_api_key is None:
            raise AdapterUnavailableError(
                "endpoint_api_key is required when endpoint_base_url is set for langchain_deep_agents",
                reason="missing_endpoint_api_key",
            )
        model_kwargs["base_url"] = _normalize_openai_endpoint_url(model_config.endpoint_base_url)
        endpoint_api_key = model_config.endpoint_api_key
        model_kwargs["api_key"] = (
            endpoint_api_key.get_secret_value() if isinstance(endpoint_api_key, SecretStr) else endpoint_api_key
        )

    model_kwargs.update(kwargs)

    # Suppress SDK-level retries unless a caller explicitly opts in. The
    # single-turn LLM/parser paths use RetryExecutor at the adapter layer.
    # Agent loops instead need model-call retries inside LangGraph, because
    # retrying the entire agent can repeat workspace side effects.
    model_kwargs.setdefault("max_retries", 0)

    return init_chat_model(
        model=model_config.model_name,
        model_provider=model_config.model_provider,
        **model_kwargs,
    )
