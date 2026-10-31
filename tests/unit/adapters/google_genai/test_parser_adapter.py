"""Tests for GoogleGenAIParserAdapter."""

from __future__ import annotations

import asyncio

import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel

from karenina.adapters.google_genai.errors import GeminiClientError
from karenina.adapters.google_genai.parser import GoogleGenAIParserAdapter
from karenina.ports import Message, ParseError
from tests.unit.adapters.google_genai._fakes import (
    FakeModels,
    fast_policy,
    install,
    make_config,
    response,
    text_part,
    usage,
)


class Extracted(BaseModel):
    gene: str
    is_oncogene: bool


def _parser(models: FakeModels, **overrides: object) -> GoogleGenAIParserAdapter:
    parser = GoogleGenAIParserAdapter(make_config(**overrides))
    install(parser._llm_adapter, models)
    return parser


MESSAGES = [Message.system("Extract the fields."), Message.user("BCL2 is an anti-apoptotic oncogene.")]


@pytest.mark.unit
class TestParser:
    async def test_parse_with_usage(self) -> None:
        models = FakeModels(
            responses=[
                response([text_part('{"gene": "BCL2", "is_oncogene": true}')], usage_metadata=usage(50, 9, total=59))
            ]
        )
        result = await _parser(models).aparse_to_pydantic(MESSAGES, Extracted)
        assert result.parsed == Extracted(gene="BCL2", is_oncogene=True)
        assert result.usage.total_tokens == 59
        assert models.calls[0]["config"].system_instruction == "Extract the fields."

    async def test_server_side_tools_stay_off_for_parsing(self) -> None:
        models = FakeModels(responses=[response([text_part('{"gene": "TP53", "is_oncogene": false}')])])
        parser = _parser(models, extra_kwargs={"builtin_tools": ["google_search", "url_context"]})
        await parser.aparse_to_pydantic(MESSAGES, Extracted)
        assert models.calls[0]["config"].tools is None

    async def test_usage_is_summed_over_validation_retries(self) -> None:
        models = FakeModels(
            responses=[
                response([text_part('{"gene": "BCL2"}')], usage_metadata=usage(50, 5, total=55)),
                response([text_part('{"gene": "BCL2", "is_oncogene": true}')], usage_metadata=usage(60, 6, total=66)),
            ]
        )
        result = await _parser(models).aparse_to_pydantic(MESSAGES, Extracted)
        assert result.usage.total_tokens == 121

    async def test_parse_error_after_retries(self) -> None:
        models = FakeModels(responses=[response([text_part("nope")]) for _ in range(3)])
        with pytest.raises(ParseError):
            await _parser(models).aparse_to_pydantic(MESSAGES, Extracted)
        assert len(models.calls) == 3  # max_retries=2 by default

    async def test_port_errors_pass_through(self) -> None:
        denied = genai_errors.ClientError(
            403, {"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}}
        )
        with pytest.raises(GeminiClientError):
            await _parser(FakeModels(responses=[denied])).aparse_to_pydantic(MESSAGES, Extracted)

    async def test_retryable_errors_keep_their_type(self) -> None:
        models = FakeModels(responses=[ConnectionError("refused")] * 3)
        parser = _parser(models, retry_policy=fast_policy(connection=2))
        with pytest.raises(ConnectionError):
            await parser.aparse_to_pydantic(MESSAGES, Extracted)

    async def test_sync_parse(self) -> None:
        models = FakeModels(responses=[response([text_part('{"gene": "MYC", "is_oncogene": true}')])])
        parser = _parser(models)
        result = await asyncio.to_thread(parser.parse_to_pydantic, MESSAGES, Extracted)
        assert result.parsed.gene == "MYC"

    async def test_capabilities_and_aclose(self) -> None:
        parser = GoogleGenAIParserAdapter(make_config())
        assert parser.capabilities.supports_structured_output is True
        await parser.aclose()
        await parser.aclose()
