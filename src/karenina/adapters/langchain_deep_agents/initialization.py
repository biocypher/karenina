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

from karenina.ports import AdapterUnavailableError

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig

logger = logging.getLogger(__name__)


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
        model_kwargs["request_timeout"] = model_config.request_timeout

    model_kwargs.update(kwargs)

    return init_chat_model(
        model=model_config.model_name,
        model_provider=model_config.model_provider,
        **model_kwargs,
    )
