"""Tests for ChatOpenRouter and ChatOpenAIEndpoint custom model classes."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from karenina.adapters.langchain.models import (
    _HTTPX_DEFAULT_CONNECT_TIMEOUT_S,
    _HTTPX_DEFAULT_MAX_CONNECTIONS,
    _HTTPX_DEFAULT_MAX_KEEPALIVE,
    _HTTPX_DEFAULT_POOL_TIMEOUT_S,
    _HTTPX_DEFAULT_READ_TIMEOUT_S,
)


class TestChatOpenRouter:
    """Tests for ChatOpenRouter custom model class."""

    def test_chat_openrouter_initialization(self) -> None:
        """Test ChatOpenRouter initialization."""
        from karenina.adapters.langchain.models import ChatOpenRouter

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            model = ChatOpenRouter(model="gpt-4", temperature=0.5)

            assert model.model_name == "gpt-4"
            assert model.temperature == 0.5

    def test_chat_openrouter_lc_secrets(self) -> None:
        """Test ChatOpenRouter lc_secrets property."""
        from karenina.adapters.langchain.models import ChatOpenRouter

        model = ChatOpenRouter(model="gpt-4", openai_api_key="test-key")
        secrets = model.lc_secrets
        assert secrets == {"openai_api_key": "OPENROUTER_API_KEY"}


class TestChatOpenAIEndpoint:
    """Tests for ChatOpenAIEndpoint custom model class."""

    def test_chat_openai_endpoint_requires_api_key(self) -> None:
        """Test that ChatOpenAIEndpoint requires an API key."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        with pytest.raises(ValueError, match="API key is required"):
            ChatOpenAIEndpoint(base_url="http://localhost:8000")

    def test_chat_openai_endpoint_requires_explicit_api_key(self) -> None:
        """Test that ChatOpenAIEndpoint does NOT read from environment."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}),
            pytest.raises(ValueError, match="API key is required"),
        ):
            ChatOpenAIEndpoint(base_url="http://localhost:8000")

    def test_chat_openai_endpoint_initialization_with_key(self) -> None:
        """Test ChatOpenAIEndpoint initialization with explicit API key."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        model = ChatOpenAIEndpoint(
            base_url="http://localhost:8000",
            openai_api_key="explicit-key",
        )

        assert model is not None

    def test_chat_openai_endpoint_lc_secrets_empty(self) -> None:
        """Test ChatOpenAIEndpoint lc_secrets returns empty dict."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        model = ChatOpenAIEndpoint(
            base_url="http://localhost:8000",
            openai_api_key="test-key",
        )
        secrets = model.lc_secrets
        assert secrets == {}

    def test_endpoint_base_url_mode_auto_v1_appends_v1(self) -> None:
        """Default mode normalizes the base URL to end with /v1."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        model = ChatOpenAIEndpoint(
            base_url="https://api.z.ai/api/coding/paas/v4",
            openai_api_key="k",
        )
        assert str(model.openai_api_base).rstrip("/").endswith("/v4/v1")

    def test_endpoint_base_url_mode_raw_preserves_url(self) -> None:
        """raw mode uses the base URL as given (no /v1 append).

        Needed for OpenAI-compatible endpoints not served at /v1, such as the
        z.ai coding endpoint at .../v4.
        """
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        model = ChatOpenAIEndpoint(
            base_url="https://api.z.ai/api/coding/paas/v4",
            openai_api_key="k",
            endpoint_base_url_mode="raw",
        )
        assert str(model.openai_api_base).rstrip("/") == "https://api.z.ai/api/coding/paas/v4"

    def test_endpoint_base_url_mode_invalid_rejected(self) -> None:
        """An unknown endpoint_base_url_mode is rejected."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        with pytest.raises(ValueError, match="endpoint_base_url_mode"):
            ChatOpenAIEndpoint(
                base_url="http://localhost:8000",
                openai_api_key="k",
                endpoint_base_url_mode="bogus",
            )


def _assert_bounded_httpx_client(client: httpx.AsyncClient | httpx.Client) -> None:
    """Assert that an httpx client has bounded limits and a pool timeout.

    See _build_httpx_clients in models.py and issue 194 for rationale.
    """
    assert client is not None
    # Pool deadline must be set so silent waits raise PoolTimeout instead
    # of blocking forever.
    assert client.timeout.pool == _HTTPX_DEFAULT_POOL_TIMEOUT_S
    assert client.timeout.connect == _HTTPX_DEFAULT_CONNECT_TIMEOUT_S
    # Read timeout is a per-chunk gap budget, not a total response budget,
    # so it bounds silent stream death without interrupting normal streaming.
    assert client.timeout.read == _HTTPX_DEFAULT_READ_TIMEOUT_S
    # Write left at None: prompts are short and never streamed.
    assert client.timeout.write is None
    # Pool size must be bounded so CLOSE_WAIT cannot accumulate without limit.
    pool = client._transport._pool  # noqa: SLF001 (private access intentional for test assertions)
    assert pool._max_connections == _HTTPX_DEFAULT_MAX_CONNECTIONS
    assert pool._max_keepalive_connections == _HTTPX_DEFAULT_MAX_KEEPALIVE


def _assert_bounded_request_timeout(model: object) -> None:
    """Assert that the model has a per-request httpx.Timeout that openai SDK honors.

    The openai SDK overrides per-request timeouts via ``client.send(request,
    timeout=...)``, so the http_client's default timeout alone is not enough.
    The model must also expose ``request_timeout`` set to an httpx.Timeout
    with bounded read and pool deadlines.
    """
    request_timeout = getattr(model, "request_timeout", None)
    assert isinstance(request_timeout, httpx.Timeout), f"expected httpx.Timeout, got {type(request_timeout).__name__}"
    assert request_timeout.read == _HTTPX_DEFAULT_READ_TIMEOUT_S
    assert request_timeout.connect == _HTTPX_DEFAULT_CONNECT_TIMEOUT_S
    assert request_timeout.pool == _HTTPX_DEFAULT_POOL_TIMEOUT_S
    assert request_timeout.write is None


class TestHttpxPoolConfig:
    """Tests that custom model classes inject bounded httpx clients.

    Regression coverage for issue 194: under concurrent streaming load,
    the default httpx pool failed to release vLLM connections (CLOSE_WAIT
    accumulation), causing requests to block forever waiting for a slot.
    The custom model classes now inject explicit Limits and Timeout(pool=)
    so that the pool wait raises PoolTimeout instead of hanging.
    """

    def test_chat_openai_endpoint_injects_bounded_httpx_clients(self) -> None:
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        model = ChatOpenAIEndpoint(
            base_url="http://localhost:8000",
            openai_api_key="test-key",
            model="dummy",
            max_retries=0,
        )
        _assert_bounded_httpx_client(model.http_async_client)
        _assert_bounded_httpx_client(model.http_client)
        _assert_bounded_request_timeout(model)

    def test_chat_openrouter_injects_bounded_httpx_clients(self) -> None:
        from karenina.adapters.langchain.models import ChatOpenRouter

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            model = ChatOpenRouter(model="gpt-4", max_retries=0)
        _assert_bounded_httpx_client(model.http_async_client)
        _assert_bounded_httpx_client(model.http_client)
        _assert_bounded_request_timeout(model)

    def test_caller_supplied_http_clients_are_respected(self) -> None:
        """If the caller passes their own http clients, do not override them."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        custom_async = httpx.AsyncClient()
        custom_sync = httpx.Client()
        model = ChatOpenAIEndpoint(
            base_url="http://localhost:8000",
            openai_api_key="test-key",
            model="dummy",
            max_retries=0,
            http_client=custom_sync,
            http_async_client=custom_async,
        )
        assert model.http_async_client is custom_async
        assert model.http_client is custom_sync

    def test_caller_supplied_request_timeout_is_respected(self) -> None:
        """If the caller passes their own request_timeout, do not override it."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        model = ChatOpenAIEndpoint(
            base_url="http://localhost:8000",
            openai_api_key="test-key",
            model="dummy",
            max_retries=0,
            request_timeout=42.0,
        )
        assert model.request_timeout == 42.0
