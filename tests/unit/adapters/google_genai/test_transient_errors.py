"""Transient failures outside the API status codes are retried, not permanent.

A response body or stream chunk that is not JSON (the SDK's
``UnknownApiResponseError``, a ``ValueError``) and a network failure while
refreshing Google credentials are transient. Real credential failures stay
permanent and keep the "run gcloud auth application-default login" hint.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from google.auth.exceptions import DefaultCredentialsError, RefreshError, TransportError
from google.genai import errors as genai_errors

from karenina.adapters.google_genai.client import build_client
from karenina.adapters.google_genai.errors import (
    AUTH_HINT,
    GeminiAuthError,
    GeminiConnectionError,
    GeminiIncompleteResponseError,
    build_error_registry,
    translate_auth_error,
)
from karenina.adapters.google_genai.settings import GoogleGenAISettings
from karenina.utils.errors import ErrorCategory
from tests.unit.adapters.google_genai._fakes import FakeModels, make_config, response, run_generate, text_part

NOT_JSON = genai_errors.UnknownApiResponseError("Failed to parse response as JSON. Raw response: <html>500</html>")


@pytest.mark.unit
class TestAuthErrorTranslation:
    @pytest.mark.parametrize(
        "error",
        [TransportError("connection reset"), RefreshError("temporary failure", retryable=True)],
    )
    def test_network_failures_are_connection_errors(self, error: Exception) -> None:
        translated = translate_auth_error(error)
        assert isinstance(translated, GeminiConnectionError)
        assert AUTH_HINT not in str(translated)
        assert build_error_registry().classify(translated) is ErrorCategory.CONNECTION

    @pytest.mark.parametrize(
        "error",
        [RefreshError("invalid_grant: Token has been expired or revoked."), DefaultCredentialsError("no ADC")],
    )
    def test_credential_failures_stay_permanent(self, error: Exception) -> None:
        translated = translate_auth_error(error)
        assert isinstance(translated, GeminiAuthError)
        assert "run gcloud auth application-default login" in str(translated)
        assert build_error_registry().classify(translated) is ErrorCategory.PERMANENT


@pytest.mark.unit
class TestDuringCalls:
    async def test_non_json_response_is_retried(self) -> None:
        models = FakeModels(responses=[NOT_JSON, response([text_part("ok")])])
        assert (await run_generate(models)).text == "ok"
        assert len(models.calls) == 2

    async def test_non_json_stream_chunk_is_retried(self) -> None:
        models = FakeModels(
            streams=[[response([text_part("a")], finish=None), NOT_JSON], [response([text_part("ok")])]]
        )
        assert (await run_generate(models, stream=True)).text == "ok"

    async def test_persistent_non_json_fails_as_incomplete(self) -> None:
        models = FakeModels(responses=[NOT_JSON] * 3)
        with pytest.raises(GeminiIncompleteResponseError, match="not JSON"):
            await run_generate(models)
        assert len(models.calls) == 3

    async def test_token_refresh_network_failure_is_retried(self) -> None:
        models = FakeModels(responses=[TransportError("Connection aborted"), response([text_part("ok")])])
        assert (await run_generate(models)).text == "ok"
        assert len(models.calls) == 2

    async def test_expired_login_is_not_retried(self) -> None:
        models = FakeModels(responses=[RefreshError("Reauthentication is needed."), response([text_part("never")])])
        with pytest.raises(GeminiAuthError, match="gcloud auth application-default login"):
            await run_generate(models)
        assert len(models.calls) == 1


@pytest.mark.unit
class TestDuringClientConstruction:
    def test_network_failure_loading_credentials(self) -> None:
        settings = GoogleGenAISettings.from_model_config(make_config())
        with (
            patch("google.genai.Client", side_effect=TransportError("metadata server unreachable")),
            pytest.raises(GeminiConnectionError),
        ):
            build_client(settings)


@pytest.mark.unit
class TestTransportErrors:
    """aiohttp and httpx failures become port errors inside the attempt, so they never escape raw."""

    async def test_dropped_connection_is_retried(self) -> None:
        import aiohttp

        models = FakeModels(responses=[aiohttp.ServerDisconnectedError(), response([text_part("ok")])])
        assert (await run_generate(models)).text == "ok"
        assert len(models.calls) == 2

    async def test_persistent_dropped_connection_fails_as_a_connection_error(self) -> None:
        import aiohttp

        models = FakeModels(responses=[aiohttp.ServerDisconnectedError()] * 3)
        with pytest.raises(GeminiConnectionError, match="Server disconnected") as info:
            await run_generate(models)
        assert isinstance(info.value.__cause__, aiohttp.ServerDisconnectedError)

    async def test_transport_timeout_becomes_a_timeout(self) -> None:
        import aiohttp

        models = FakeModels(responses=[aiohttp.ServerTimeoutError("read timed out"), response([text_part("ok")])])
        assert (await run_generate(models)).text == "ok"
        assert len(models.calls) == 2


@pytest.mark.unit
class TestPipelineRegistry:
    """The pipeline's generic registry classifies the adapter's errors the way the adapter does."""

    @pytest.mark.parametrize(
        ("error", "category"),
        [
            (GeminiIncompleteResponseError("Gemini stopped with no text"), ErrorCategory.SERVER_ERROR),
            (GeminiConnectionError("Network failure talking to Gemini"), ErrorCategory.CONNECTION),
            (GeminiAuthError("login expired, timeout"), ErrorCategory.PERMANENT),
        ],
    )
    def test_declared_category_wins_in_a_generic_registry(self, error: Exception, category: ErrorCategory) -> None:
        from karenina.utils.errors import ErrorRegistry

        assert ErrorRegistry().classify(error) is category

    def test_permanent_400_mentioning_500_stays_permanent(self) -> None:
        from karenina.adapters.google_genai.errors import GeminiClientError
        from karenina.utils.errors import ErrorRegistry

        error = GeminiClientError(
            "Gemini API error 400 INVALID_ARGUMENT: input token count (1500001) exceeds the limit"
        )
        assert ErrorRegistry().classify(error) is ErrorCategory.PERMANENT
