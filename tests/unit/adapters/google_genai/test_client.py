"""Tests for google-genai client construction and the per-loop client holder."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from google.auth.exceptions import DefaultCredentialsError
from pydantic import SecretStr

from karenina.adapters.google_genai.client import GenAIClientHolder, build_client
from karenina.adapters.google_genai.errors import AUTH_HINT, GeminiAuthError
from karenina.adapters.google_genai.settings import GoogleGenAISettings
from karenina.ports import AdapterUnavailableError
from tests.unit.adapters.google_genai._fakes import make_config


def _settings(**overrides: Any) -> GoogleGenAISettings:
    return GoogleGenAISettings.from_model_config(make_config(**overrides))


@pytest.mark.unit
class TestBuildClient:
    def test_vertex_client_with_sdk_retries_off(self) -> None:
        settings = _settings(request_timeout=30.0, extra_kwargs={"project": "proj"})
        with patch("google.genai.Client") as client_cls:
            build_client(settings)
        kwargs = client_cls.call_args.kwargs
        assert kwargs["vertexai"] is True
        assert kwargs["project"] == "proj"
        assert kwargs["location"] == "global"
        http_options = kwargs["http_options"]
        # RetryExecutor is the only retry layer (design D6).
        assert http_options.retry_options.attempts == 1
        assert http_options.timeout == 30_000
        assert http_options.base_url is None

    def test_gemini_api_client_uses_key_and_base_url(self) -> None:
        settings = _settings(endpoint_base_url="https://gw.example.org", endpoint_api_key=SecretStr("k1"))
        with patch("google.genai.Client") as client_cls:
            build_client(settings)
        kwargs = client_cls.call_args.kwargs
        assert kwargs["api_key"] == "k1"
        assert "vertexai" not in kwargs
        assert kwargs["http_options"].base_url == "https://gw.example.org"
        assert kwargs["http_options"].retry_options.attempts == 1

    def test_gemini_api_key_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        settings = _settings(extra_kwargs={"backend": "gemini_api"})
        with patch("google.genai.Client") as client_cls:
            build_client(settings)
        assert client_cls.call_args.kwargs["api_key"] == "env-key"

    def test_gemini_api_without_key_is_unavailable(self) -> None:
        settings = _settings(extra_kwargs={"backend": "gemini_api"})
        with pytest.raises(AdapterUnavailableError, match="API key"):
            build_client(settings)

    def test_missing_google_login_maps_to_auth_error(self) -> None:
        with (
            patch("google.genai.Client", side_effect=DefaultCredentialsError("no ADC")),
            pytest.raises(GeminiAuthError, match="gcloud auth application-default login"),
        ):
            build_client(_settings())
        assert "gcloud auth application-default login" in AUTH_HINT

    def test_sdk_value_error_maps_to_unavailable(self) -> None:
        with (
            patch("google.genai.Client", side_effect=ValueError("bad args")),
            pytest.raises(AdapterUnavailableError, match="bad args"),
        ):
            build_client(_settings())


@pytest.mark.unit
class TestClientHolder:
    async def test_one_client_per_event_loop(self) -> None:
        holder = GenAIClientHolder(_settings())
        built: list[MagicMock] = []

        def fake_build(_settings: Any) -> MagicMock:
            client = MagicMock()
            built.append(client)
            return client

        with patch("karenina.adapters.google_genai.client.build_client", side_effect=fake_build):
            first = holder.get()
            assert holder.get() is first

            async def other_loop_get() -> Any:
                return holder.get()

            other = await asyncio.to_thread(lambda: asyncio.run(other_loop_get()))
        assert other is not first
        assert len(built) == 2

    async def test_aclose_closes_the_running_loops_client(self) -> None:
        holder = GenAIClientHolder(_settings())
        client = MagicMock()

        async def aclose() -> None:
            client.aio_closed = True

        client.aio.aclose = aclose
        with patch("karenina.adapters.google_genai.client.build_client", return_value=client):
            holder.get()
        await holder.aclose()
        assert client.aio_closed is True
        client.close.assert_called_once()
        await holder.aclose()  # safe to call twice
        client.close.assert_called_once()
