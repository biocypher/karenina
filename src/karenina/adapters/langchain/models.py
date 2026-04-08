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

import httpx
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr

logger = logging.getLogger(__name__)

# Httpx pool sizing for ChatOpenAI-derived custom models.
#
# These bound the underlying httpx connection pool so that, under concurrent
# streaming load, sockets in CLOSE_WAIT cannot accumulate without limit and
# pool waits cannot block forever. The pool deadline turns silent waits into
# httpx.PoolTimeout exceptions, which ErrorRegistry classifies as TIMEOUT and
# RetryExecutor can rescue. See issue 194 for the lsof-confirmed root cause.
_HTTPX_DEFAULT_MAX_CONNECTIONS = 32
_HTTPX_DEFAULT_MAX_KEEPALIVE = 16
_HTTPX_DEFAULT_POOL_TIMEOUT_S = 30.0
_HTTPX_DEFAULT_CONNECT_TIMEOUT_S = 10.0
# Per-chunk read timeout: this is NOT a wall-clock budget for the whole
# response, it is the maximum gap between successive bytes/chunks. Streaming
# generations can run for many minutes, but individual token-to-token gaps
# (even during heavy "thinking" phases) should be a few seconds at most. A
# silent stream death (vLLM sent FIN but no ReadTimeout fires because we
# never asked for one) was the second wedge mode observed in issue 194 after
# the pool fix landed: the pool fix bounded socket counts but a stuck
# ESTABLISHED stream still hung the agent for the full agent_timeout. This
# converts that hang into a retryable httpx.ReadTimeout within ~2 minutes,
# which RetryExecutor classifies as TIMEOUT and rescues automatically.
_HTTPX_DEFAULT_READ_TIMEOUT_S = 120.0


def _build_default_request_timeout() -> httpx.Timeout:
    """Build the per-request timeout that bounds read and connect waits.

    The openai SDK overrides the underlying httpx client's default Timeout
    on every request via ``client.send(request, timeout=...)``, which means
    setting read/connect on the http client alone has no effect on real
    traffic. We pass this Timeout via langchain-openai's ``request_timeout``
    field instead: that maps to the openai SDK's ``timeout`` parameter,
    which is then used for ``client.send`` and DOES bound each request.

    Pool deadline is still meaningful here: it bounds the wait when a
    request needs to acquire a connection from the underlying pool, before
    any send call is even reached.
    """
    return httpx.Timeout(
        connect=_HTTPX_DEFAULT_CONNECT_TIMEOUT_S,
        read=_HTTPX_DEFAULT_READ_TIMEOUT_S,
        write=None,
        pool=_HTTPX_DEFAULT_POOL_TIMEOUT_S,
    )


def _build_httpx_clients() -> tuple[httpx.Client, httpx.AsyncClient]:
    """Build sync and async httpx clients with bounded pool and read timeouts.

    Returns:
        A tuple of (sync_client, async_client). Both are configured with the
        module-level default Limits and Timeout. The read timeout on the
        client itself is mostly a defense-in-depth measure: the openai SDK
        overrides per-request timeouts via ``client.send(request, timeout=)``
        so the effective per-request bound also has to be set via
        ``request_timeout`` on the model class. See _build_default_request_timeout.
    """
    limits = httpx.Limits(
        max_connections=_HTTPX_DEFAULT_MAX_CONNECTIONS,
        max_keepalive_connections=_HTTPX_DEFAULT_MAX_KEEPALIVE,
    )
    timeout = _build_default_request_timeout()
    return httpx.Client(limits=limits, timeout=timeout), httpx.AsyncClient(limits=limits, timeout=timeout)


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
        # Inject bounded httpx clients unless the caller already supplied them.
        # See _build_httpx_clients docstring and issue 194 for rationale.
        if "http_client" not in kwargs and "http_async_client" not in kwargs:
            sync_client, async_client = _build_httpx_clients()
            kwargs["http_client"] = sync_client
            kwargs["http_async_client"] = async_client
        # Inject the per-request timeout that openai SDK actually honors.
        # Without this, openai's per-request timeout override (~600s) defeats
        # the read timeout we tried to set on the http client.
        if "request_timeout" not in kwargs and "timeout" not in kwargs:
            kwargs["request_timeout"] = _build_default_request_timeout()
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

        # Inject bounded httpx clients unless the caller already supplied them.
        # See _build_httpx_clients docstring and issue 194 for rationale.
        if "http_client" not in kwargs and "http_async_client" not in kwargs:
            sync_client, async_client = _build_httpx_clients()
            kwargs["http_client"] = sync_client
            kwargs["http_async_client"] = async_client
        # Inject the per-request timeout that openai SDK actually honors.
        # Without this, openai's per-request timeout override (~600s) defeats
        # the read timeout we tried to set on the http client.
        if "request_timeout" not in kwargs and "timeout" not in kwargs:
            kwargs["request_timeout"] = _build_default_request_timeout()

        super().__init__(
            base_url=normalized_url,
            api_key=openai_api_key,
            **kwargs,
        )
