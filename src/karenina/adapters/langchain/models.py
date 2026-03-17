"""LangChain model classes for custom endpoints.

This module contains custom LangChain model classes for providers that need
special handling (OpenRouter, custom OpenAI-compatible endpoints).

These classes are adapter-internal and should not be imported directly.
Use the LLMPort interface via get_llm() instead.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr

logger = logging.getLogger(__name__)


def _normalize_openai_endpoint_url(base_url: str) -> str:
    """Ensure base URL ends with /v1 for OpenAI-compatible endpoints.

    The OpenAI API expects requests to paths like /v1/chat/completions.
    This function normalizes URLs to ensure they end with /v1.

    Args:
        base_url: The user-provided base URL

    Returns:
        Normalized URL ending with /v1

    Examples:
        >>> _normalize_openai_endpoint_url("http://localhost:8000")
        'http://localhost:8000/v1'
        >>> _normalize_openai_endpoint_url("http://localhost:8000/")
        'http://localhost:8000/v1'
        >>> _normalize_openai_endpoint_url("http://localhost:8000/v1")
        'http://localhost:8000/v1'
        >>> _normalize_openai_endpoint_url("http://localhost:8000/v1/")
        'http://localhost:8000/v1'
    """
    # Remove trailing slashes for consistent handling
    url = base_url.rstrip("/")

    # Check if URL already ends with /v1 (case-insensitive)
    if re.search(r"/v1$", url, re.IGNORECASE):
        return url

    # Append /v1
    normalized = f"{url}/v1"
    logger.debug(f"Normalized OpenAI endpoint URL: {base_url} -> {normalized}")
    return normalized


class ChatOpenRouter(ChatOpenAI):
    """LangChain ChatOpenAI wrapper for OpenRouter API.

    OpenRouter provides a unified API for accessing multiple LLM providers.
    This class automatically configures the correct base URL and API key
    handling for OpenRouter.

    The API key is read from the OPENROUTER_API_KEY environment variable
    if not explicitly provided.
    """

    openai_api_key: SecretStr | None = Field(alias="api_key", default=None)

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self, openai_api_key: str | None = None, **kwargs: Any) -> None:
        openai_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(openai_api_key) if openai_api_key else None,
            **kwargs,
        )


class ChatOpenAIEndpoint(ChatOpenAI):
    """ChatOpenAI wrapper for user-provided custom endpoints.

    Unlike ChatOpenRouter, this does NOT automatically read from environment.
    API key must be explicitly provided by the user.

    This class is used for connecting to OpenAI-compatible endpoints like:
    - vLLM servers
    - SGLang servers
    - Ollama with OpenAI compatibility layer
    - Text Generation Inference (TGI)
    - Local LLM servers
    """

    openai_api_key: SecretStr | None = Field(alias="api_key", default=None)

    @property
    def lc_secrets(self) -> dict[str, str]:
        # Return empty dict - we don't want LangChain trying to read from env
        return {}

    def __init__(
        self,
        base_url: str,
        openai_api_key: str | SecretStr | None = None,
        **kwargs: Any,
    ) -> None:
        # Do NOT fallback to environment - require explicit API key
        if openai_api_key is None:
            raise ValueError(
                "API key is required for openai_endpoint interface. "
                "This interface does not automatically read from environment variables."
            )

        if isinstance(openai_api_key, str):
            openai_api_key = SecretStr(openai_api_key)

        # Normalize URL to ensure it ends with /v1
        normalized_url = _normalize_openai_endpoint_url(base_url)

        super().__init__(
            base_url=normalized_url,
            api_key=openai_api_key,
            **kwargs,
        )
