"""Tests for GoogleGenAILLMAdapter: text, structured output, streaming."""

from __future__ import annotations

import asyncio
import json

import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

from karenina.adapters.google_genai.errors import GeminiFinishReasonError, GeminiIncompleteResponseError
from karenina.adapters.google_genai.llm import GoogleGenAILLMAdapter
from karenina.exceptions import StreamingTimeoutError
from karenina.ports import LLMPort, Message, ParseError, PortError
from tests.unit.adapters.google_genai._fakes import (
    Delay,
    FakeModels,
    fast_policy,
    install,
    make_config,
    response,
    search_grounding,
    text_part,
    thought_part,
    usage,
)


class Verdict(BaseModel):
    """A small structured output schema."""

    backed: bool = Field(description="Whether the citation backs the claim")
    reason: str


class Nested(BaseModel):
    """Schema with $defs, to exercise the schema fallback."""

    verdicts: list[Verdict]


def _llm(models: FakeModels, **overrides: object) -> GoogleGenAILLMAdapter:
    adapter = GoogleGenAILLMAdapter(make_config(**overrides))
    install(adapter, models)
    return adapter


@pytest.mark.unit
class TestTextCalls:
    async def test_ainvoke_returns_text_without_thoughts(self) -> None:
        raw = response(
            [thought_part("Pondering."), text_part("Paris")],
            usage_metadata=usage(20, 2, thoughts=7, total=29),
        )
        models = FakeModels(responses=[raw])
        result = await _llm(models, extra_kwargs={"include_thoughts": True}).ainvoke(
            [Message.system("Be brief."), Message.user("Capital?")]
        )
        assert result.content == "Paris"
        assert result.raw is raw
        assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.total_tokens) == (20, 9, 29)
        assert result.usage.reasoning_tokens == 7
        config = models.calls[0]["config"]
        assert config.system_instruction == "Be brief."
        assert config.thinking_config.include_thoughts is True

    async def test_builtin_tools_apply_and_searches_are_counted(self) -> None:
        models = FakeModels(responses=[response([text_part("Found.")], grounding=search_grounding(["a", "b"]))])
        result = await _llm(models, extra_kwargs={"builtin_tools": ["google_search"]}).ainvoke([Message.user("Q")])
        assert models.calls[0]["config"].tools[0].google_search is not None
        assert result.usage.web_search_queries == 2

    async def test_gemini_api_backend_streams(self) -> None:
        models = FakeModels(
            streams=[[response([text_part("He")], finish=None), response([text_part("llo")], finish="STOP")]]
        )
        result = await _llm(models, endpoint_base_url="https://gw.example.org").ainvoke([Message.user("Q")])
        assert models.calls[0]["kind"] == "stream"
        assert result.content == "Hello"

    @pytest.mark.parametrize(
        "reason", ["TOO_MANY_TOOL_CALLS", "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "SAFETY", "OTHER"]
    )
    async def test_stops_without_an_answer_raise_port_errors(self, reason: str) -> None:
        models = FakeModels(responses=[response([], finish=reason)])
        with pytest.raises(PortError, match=reason) as excinfo:
            await _llm(models).ainvoke([Message.user("Q")])
        assert isinstance(excinfo.value, GeminiFinishReasonError)

    async def test_max_tokens_keeps_the_content(self) -> None:
        models = FakeModels(responses=[response([text_part("Cut sho")], finish="MAX_TOKENS")])
        assert (await _llm(models).ainvoke([Message.user("Q")])).content == "Cut sho"

    async def test_sync_invoke(self) -> None:
        models = FakeModels(responses=[response([text_part("sync")])])
        adapter = _llm(models)
        result = await asyncio.to_thread(adapter.invoke, [Message.user("Q")])
        assert result.content == "sync"

    def test_capabilities(self) -> None:
        caps = GoogleGenAILLMAdapter(make_config()).capabilities
        assert caps.supports_structured_output is True
        assert caps.supports_streaming is True


@pytest.mark.unit
class TestStructuredOutput:
    async def test_json_schema_route_and_json_content(self) -> None:
        models = FakeModels(responses=[response([text_part('{"backed": true, "reason": "abstract says so"}')])])
        adapter = _llm(models)
        structured = adapter.with_structured_output(Verdict)
        result = await structured.ainvoke([Message.user("Judge")])

        assert isinstance(structured, LLMPort)
        assert structured is not adapter
        assert adapter._structured_schema is None
        assert isinstance(result.raw, Verdict) and result.raw.backed is True
        assert json.loads(result.content) == {"backed": True, "reason": "abstract says so"}
        config = models.calls[0]["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema == Verdict.model_json_schema()

    async def test_validation_retry_with_feedback_and_summed_usage(self) -> None:
        models = FakeModels(
            responses=[
                response([text_part('{"backed": "maybe"}')], usage_metadata=usage(100, 10, total=110)),
                response([text_part('{"backed": false, "reason": "no"}')], usage_metadata=usage(130, 12, total=142)),
            ]
        )
        result = await _llm(models).with_structured_output(Verdict, max_retries=2).ainvoke([Message.user("Judge")])
        assert result.raw == Verdict(backed=False, reason="no")
        assert result.usage.input_tokens == 230
        assert result.usage.total_tokens == 252
        feedback = models.calls[1]["contents"][-1].parts[0].text
        assert "PREVIOUS ATTEMPT FAILED" in feedback

    async def test_code_fenced_json_is_accepted(self) -> None:
        fenced = '```json\n{"backed": true, "reason": "ok"}\n```'
        models = FakeModels(responses=[response([text_part(fenced)])])
        result = await _llm(models).with_structured_output(Verdict).ainvoke([Message.user("Judge")])
        assert result.raw.backed is True

    async def test_parse_error_after_all_retries(self) -> None:
        models = FakeModels(responses=[response([text_part("not json")]) for _ in range(3)])
        with pytest.raises(ParseError, match="after 3 attempts") as excinfo:
            await _llm(models).with_structured_output(Verdict, max_retries=2).ainvoke([Message.user("Judge")])
        assert excinfo.value.raw_response == "not json"

    async def test_truncated_json_at_max_tokens_is_a_parse_error(self) -> None:
        models = FakeModels(responses=[response([text_part('{"backed": tr')], finish="MAX_TOKENS")])
        with pytest.raises(ParseError):
            await _llm(models).with_structured_output(Verdict, max_retries=0).ainvoke([Message.user("Judge")])

    async def test_rejected_json_schema_falls_back_to_the_sdk_conversion(self) -> None:
        rejected = genai_errors.ClientError(
            400, {"error": {"code": 400, "message": "Invalid JSON schema: $ref", "status": "INVALID_ARGUMENT"}}
        )
        valid = '{"verdicts": [{"backed": true, "reason": "r"}]}'
        models = FakeModels(responses=[rejected, response([text_part(valid)]), response([text_part(valid)])])
        structured = _llm(models).with_structured_output(Nested)
        assert (await structured.ainvoke([Message.user("Judge")])).raw.verdicts[0].backed is True
        assert models.calls[0]["config"].response_json_schema is not None
        assert models.calls[1]["config"].response_json_schema is None
        assert models.calls[1]["config"].response_schema is not None
        await structured.ainvoke([Message.user("Again")])  # the fallback mode sticks
        assert models.calls[2]["config"].response_schema is not None

    async def test_other_client_errors_are_not_retried_with_feedback(self) -> None:
        forbidden = genai_errors.ClientError(
            403, {"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}}
        )
        models = FakeModels(responses=[forbidden])
        with pytest.raises(PortError, match="403"):
            await _llm(models).with_structured_output(Verdict).ainvoke([Message.user("Judge")])
        assert len(models.calls) == 1


@pytest.mark.unit
class TestStreaming:
    async def test_astream_yields_text_without_thoughts(self) -> None:
        chunks = [
            response([thought_part("hmm")], finish=None, usage_metadata=usage(10, 1)),
            response([text_part("Hel")], finish=None, usage_metadata=usage(10, 2)),
            response([text_part("lo")], finish="STOP", usage_metadata=usage(10, 3, total=13)),
        ]
        models = FakeModels(streams=[chunks])
        async with _llm(models).astream([Message.user("Q")]) as sr:
            received = [chunk async for chunk in sr]
        assert received == ["Hel", "lo"]
        assert sr.accumulated_content == "Hello"
        assert sr.usage.total_tokens == 13
        assert sr.is_complete is True

    async def test_astream_without_finish_reason_raises(self) -> None:
        models = FakeModels(streams=[[response([text_part("cut")], finish=None)]])
        with pytest.raises(GeminiIncompleteResponseError):
            async with _llm(models).astream([Message.user("Q")]) as sr:
                async for _ in sr:
                    pass

    async def test_stream_invoke_returns_partial_content_on_timeout(self) -> None:
        chunks = [response([text_part("Partial ")], finish=None), response([text_part("text")], finish=None), Delay(5)]
        models = FakeModels(streams=[chunks])
        adapter = _llm(models)
        result = await asyncio.to_thread(adapter.stream_invoke, [Message.user("Q")], 0.2)
        assert result.content == "Partial text"
        assert result.is_partial is True
        assert result.usage_unavailable is True

    async def test_stream_invoke_complete(self) -> None:
        models = FakeModels(streams=[[response([text_part("Whole.")], finish="STOP")]])
        adapter = _llm(models)
        result = await asyncio.to_thread(adapter.stream_invoke, [Message.user("Q")], 5.0)
        assert result.content == "Whole."
        assert result.is_partial is False
        assert result.usage_unavailable is False

    async def test_stream_invoke_timeout_without_content_raises_after_retries(self) -> None:
        models = FakeModels(streams=[[Delay(5)], [Delay(5)]])
        adapter = _llm(models, retry_policy=fast_policy(timeout=1))
        with pytest.raises(StreamingTimeoutError):
            await asyncio.to_thread(adapter.stream_invoke, [Message.user("Q")], 0.1)
        assert len(models.calls) == 2

    async def test_stream_invoke_retries_an_incomplete_stream(self) -> None:
        models = FakeModels(
            streams=[[response([text_part("cut")], finish=None)], [response([text_part("Full.")], finish="STOP")]]
        )
        adapter = _llm(models)
        result = await asyncio.to_thread(adapter.stream_invoke, [Message.user("Q")], 5.0)
        assert result.content == "Full."


@pytest.mark.unit
class TestLifecycle:
    async def test_structured_adapter_shares_the_client(self) -> None:
        adapter = GoogleGenAILLMAdapter(make_config())
        structured = adapter.with_structured_output(Verdict)
        assert structured._client is adapter._client
        await structured.aclose()
        await adapter.aclose()
