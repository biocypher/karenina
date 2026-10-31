"""Tests for the shared model call: errors, finish reasons, retries, streaming, limiter."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from karenina.adapters.google_genai import calls
from karenina.adapters.google_genai.calls import (
    build_generate_config,
    classify_finish_reason,
    generate,
    outcome_from_response,
    raise_for_llm_finish,
)
from karenina.adapters.google_genai.client import GenAIClientHolder
from karenina.adapters.google_genai.errors import (
    GeminiAuthError,
    GeminiClientError,
    GeminiFinishReasonError,
    GeminiIncompleteResponseError,
    GeminiRateLimitError,
    GeminiServerError,
    build_error_registry,
    translate_api_error,
)
from karenina.adapters.google_genai.settings import GoogleGenAISettings
from karenina.ports import PortError
from karenina.utils.errors import ErrorCategory
from karenina.utils.retry_policy import RetryExecutor
from tests.unit.adapters.google_genai._fakes import (
    Delay,
    FakeModels,
    call_part,
    fast_policy,
    install,
    make_config,
    response,
    search_grounding,
    text_part,
    thought_part,
    usage,
)


def _client_error(code: int, status: str, message: str = "bad request") -> genai_errors.ClientError:
    return genai_errors.ClientError(code, {"error": {"code": code, "message": message, "status": status}})


def _server_error(code: int = 500) -> genai_errors.ServerError:
    return genai_errors.ServerError(code, {"error": {"code": code, "message": "boom", "status": "INTERNAL"}})


async def _generate(
    models: FakeModels,
    *,
    stream: bool = False,
    request_timeout: float | None = None,
    **policy: int,
) -> calls.CallOutcome:
    settings = GoogleGenAISettings.from_model_config(make_config())
    holder = GenAIClientHolder(settings)
    install(holder, models)
    return await generate(
        holder,
        model="gemini-3.8-flash",
        contents=[types.Content(role="user", parts=[types.Part(text="Q")])],
        config=build_generate_config(settings, system_instruction=None),
        stream=stream,
        request_timeout=request_timeout,
        retry_executor=RetryExecutor(fast_policy(**policy), build_error_registry()),
    )


@pytest.mark.unit
class TestErrorClassification:
    @pytest.mark.parametrize(
        ("error", "expected_type", "category"),
        [
            (_client_error(429, "RESOURCE_EXHAUSTED"), GeminiRateLimitError, ErrorCategory.RATE_LIMIT),
            (
                _client_error(400, "INVALID_ARGUMENT", "timeout 500 connection"),
                GeminiClientError,
                ErrorCategory.PERMANENT,
            ),
            (_client_error(403, "PERMISSION_DENIED"), GeminiClientError, ErrorCategory.PERMANENT),
            (_client_error(404, "NOT_FOUND"), GeminiClientError, ErrorCategory.PERMANENT),
            (_server_error(500), GeminiServerError, ErrorCategory.SERVER_ERROR),
            (_server_error(503), GeminiServerError, ErrorCategory.SERVER_ERROR),
        ],
    )
    def test_api_errors(self, error: Any, expected_type: type, category: ErrorCategory) -> None:
        translated = translate_api_error(error)
        assert type(translated) is expected_type
        assert isinstance(translated, PortError)
        assert translated.code == error.code
        assert build_error_registry().classify(translated) is category

    def test_other_adapter_errors(self) -> None:
        registry = build_error_registry()
        assert registry.classify(GeminiIncompleteResponseError("cut")) is ErrorCategory.SERVER_ERROR
        assert registry.classify(GeminiAuthError("login")) is ErrorCategory.PERMANENT
        assert registry.classify(GeminiFinishReasonError("x", finish_reason="SAFETY")) is ErrorCategory.PERMANENT
        assert registry.classify(_server_error()) is ErrorCategory.SERVER_ERROR  # untranslated SDK error

    def test_transport_errors_are_retryable(self) -> None:
        registry = build_error_registry()
        assert registry.classify(aiohttp.ServerDisconnectedError()) is ErrorCategory.CONNECTION
        assert registry.classify(aiohttp.ClientPayloadError("payload not completed")) is ErrorCategory.CONNECTION
        assert registry.classify(aiohttp.ServerTimeoutError()) is ErrorCategory.TIMEOUT
        assert registry.classify(TimeoutError()) is ErrorCategory.TIMEOUT


@pytest.mark.unit
class TestFinishReasons:
    @pytest.mark.parametrize(
        ("reason", "kind"),
        [
            ("STOP", "ok"),
            ("MAX_TOKENS", "max_tokens"),
            ("TOO_MANY_TOOL_CALLS", "tool_limit"),
            ("MALFORMED_FUNCTION_CALL", "bad_tool_call"),
            ("UNEXPECTED_TOOL_CALL", "bad_tool_call"),
            ("SAFETY", "blocked"),
            ("RECITATION", "blocked"),
            ("PROHIBITED_CONTENT", "blocked"),
            ("BLOCKLIST", "blocked"),
            ("SPII", "blocked"),
            ("OTHER", "blocked"),
            ("A_REASON_FROM_A_NEWER_API", "blocked"),
        ],
    )
    def test_classification(self, reason: str, kind: str) -> None:
        assert classify_finish_reason(reason) == kind

    def test_reason_names_are_read_from_sdk_enums(self) -> None:
        # TOO_MANY_TOOL_CALLS is missing from some SDK enums: it still parses by name.
        outcome = outcome_from_response(response([text_part("x")], finish="TOO_MANY_TOOL_CALLS"))
        assert outcome.finish_reason == "TOO_MANY_TOOL_CALLS"

    def test_llm_rules(self) -> None:
        raise_for_llm_finish(outcome_from_response(response([text_part("ok")], finish="STOP")))
        raise_for_llm_finish(outcome_from_response(response([text_part("cut")], finish="MAX_TOKENS")))
        for reason in ("TOO_MANY_TOOL_CALLS", "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "SAFETY", "OTHER"):
            outcome = outcome_from_response(response([text_part("")], finish=reason, finish_message="why"))
            with pytest.raises(GeminiFinishReasonError, match=reason) as excinfo:
                raise_for_llm_finish(outcome)
            assert excinfo.value.finish_reason == reason
            assert excinfo.value.finish_message == "why"


@pytest.mark.unit
class TestOutcome:
    def test_non_streamed_content_is_the_api_object(self) -> None:
        raw = response([thought_part("hmm", signature=b"s"), text_part("Answer"), call_part("f", {"a": 1})])
        outcome = outcome_from_response(raw)
        assert outcome.content is raw.candidates[0].content  # appended to the history unchanged
        assert outcome.text == "Answer"
        assert outcome.has_function_calls is True
        assert outcome.model_version == "gemini-3.8-flash-001"

    def test_empty_content_is_none(self) -> None:
        assert outcome_from_response(response([], finish="STOP")).content is None


@pytest.mark.unit
class TestGenerate:
    async def test_streamed_chunks_are_accumulated(self) -> None:
        chunks = [
            response([thought_part("Think")], finish=None, usage_metadata=usage(10, 1)),
            response(
                [text_part("Hel")],
                finish=None,
                usage_metadata=usage(10, 2),
                grounding=search_grounding(["q1"], [("T", "u")]),
            ),
            response([text_part("lo", signature=b"sig")], finish="STOP", usage_metadata=usage(10, 3, thoughts=4)),
        ]
        outcome = await _generate(FakeModels(streams=[chunks]), stream=True)
        assert outcome.text == "Hello"
        assert outcome.finish_reason == "STOP"
        assert outcome.usage_metadata.thoughts_token_count == 4  # last chunk carries the totals
        assert outcome.grounding is not None and outcome.grounding.queries == ["q1"]
        assert outcome.content.role == "model"
        assert [p.text for p in outcome.content.parts] == ["Think", "Hel", "lo"]
        assert outcome.content.parts[2].thought_signature == b"sig"

    async def test_stream_without_finish_reason_is_retried_then_fails(self) -> None:
        cut = [response([text_part("partial")], finish=None)]
        models = FakeModels(streams=[cut, cut, cut])
        with pytest.raises(GeminiIncompleteResponseError, match="without a finish reason"):
            await _generate(models, stream=True, server_error=2)
        assert len(models.calls) == 3

    async def test_stream_without_finish_reason_recovers_on_retry(self) -> None:
        models = FakeModels(streams=[[response([text_part("partial")], finish=None)], [response([text_part("full")])]])
        outcome = await _generate(models, stream=True)
        assert outcome.text == "full"
        assert len(models.calls) == 2

    async def test_server_error_is_retried(self) -> None:
        models = FakeModels(responses=[_server_error(), response([text_part("ok")])])
        assert (await _generate(models)).text == "ok"
        assert len(models.calls) == 2

    async def test_rate_limit_is_retried(self) -> None:
        models = FakeModels(responses=[_client_error(429, "RESOURCE_EXHAUSTED"), response([text_part("ok")])])
        assert (await _generate(models)).text == "ok"

    async def test_client_error_is_not_retried(self) -> None:
        models = FakeModels(responses=[_client_error(400, "INVALID_ARGUMENT"), response([text_part("never")])])
        with pytest.raises(GeminiClientError) as excinfo:
            await _generate(models)
        assert excinfo.value.code == 400
        assert len(models.calls) == 1

    async def test_dropped_stream_is_retried(self) -> None:
        models = FakeModels(
            streams=[
                [response([text_part("a")], finish=None), aiohttp.ClientPayloadError("not completed")],
                [response([text_part("ok")])],
            ]
        )
        assert (await _generate(models, stream=True)).text == "ok"

    async def test_blocked_prompt_is_permanent(self) -> None:
        models = FakeModels(responses=[response(None, block_reason="SAFETY"), response([text_part("never")])])
        with pytest.raises(GeminiFinishReasonError, match="blocked the prompt: SAFETY"):
            await _generate(models)
        assert len(models.calls) == 1

    async def test_request_timeout_guard_is_retried(self) -> None:
        # The first attempt sleeps past the guard and is cancelled before it
        # takes a response, so the retry receives the scripted answer.
        models = FakeModels(responses=[Delay(5), response([text_part("ok")])])
        outcome = await _generate(models, request_timeout=0.05, timeout=1)
        assert outcome.text == "ok"
        assert len(models.calls) == 2

    async def test_sdk_value_error_is_a_client_error(self) -> None:
        models = FakeModels(responses=[ValueError("seed parameter is not supported in Gemini API.")])
        with pytest.raises(GeminiClientError, match="seed parameter"):
            await _generate(models)

    async def test_each_attempt_holds_one_limiter_permit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        borrows: list[str] = []

        class FakeLimiter:
            @contextlib.asynccontextmanager
            async def borrow(self) -> AsyncIterator[None]:
                borrows.append("enter")
                try:
                    yield
                finally:
                    borrows.append("exit")

        monkeypatch.setattr(calls, "get_global_llm_limiter", lambda: FakeLimiter())
        models = FakeModels(responses=[_server_error(), response([text_part("ok")])])
        await _generate(models)
        assert borrows == ["enter", "exit", "enter", "exit"]


@pytest.mark.unit
class TestBuildGenerateConfig:
    def test_defaults(self) -> None:
        settings = GoogleGenAISettings.from_model_config(make_config(temperature=0.2, max_tokens=4096))
        config = build_generate_config(settings, system_instruction="Be brief.")
        assert config.automatic_function_calling.disable is True
        assert config.temperature == 0.2
        assert config.max_output_tokens == 4096
        assert config.system_instruction == "Be brief."
        assert config.thinking_config is None
        assert config.tools is None
        assert config.tool_config is None  # AUTO

    def test_thinking_tools_schema_and_passthrough(self) -> None:
        settings = GoogleGenAISettings.from_model_config(
            make_config(
                extra_kwargs={"include_thoughts": True, "thinking_level": "LOW", "generate_content_config": {"seed": 3}}
            )
        )
        search = types.Tool(google_search=types.GoogleSearch())
        config = build_generate_config(
            settings, system_instruction=None, tools=[search], response_json_schema={"type": "object"}
        )
        assert config.thinking_config.include_thoughts is True
        assert config.thinking_config.thinking_level.name == "LOW"
        assert config.tools == [search]
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema == {"type": "object"}
        assert config.seed == 3
