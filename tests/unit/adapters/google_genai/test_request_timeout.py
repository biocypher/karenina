"""The request timeout travels with each attempt, so escalation lengthens calls.

The SDK applies ``HttpOptions.timeout`` as a per-request total timeout. A
timeout set once on the client would cap every retry at the base value, so
the adapter passes the attempt's (possibly escalated) timeout in the
request's own ``http_options``.
"""

from __future__ import annotations

import pytest

from karenina.adapters.google_genai.calls import with_request_timeout
from karenina.adapters.google_genai.llm import GoogleGenAILLMAdapter
from karenina.ports import Message
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy, TimeoutEscalationConfig
from tests.unit.adapters.google_genai._fakes import (
    Delay,
    FakeModels,
    install,
    make_config,
    response,
    run_generate,
    text_part,
)


def _escalating_policy() -> RetryPolicy:
    no_wait = CategoryRetryConfig(max_attempts=2, backoff_min=0.0, backoff_max=0.0)
    return RetryPolicy(
        timeout=no_wait,
        timeout_escalation=TimeoutEscalationConfig(strategy="multiplicative", multiplier=2.0),
    )


@pytest.mark.unit
class TestPerAttemptTimeout:
    async def test_each_request_carries_the_timeout(self) -> None:
        models = FakeModels(responses=[response([text_part("ok")])])
        await run_generate(models, request_timeout=12.5)
        assert models.calls[0]["config"].http_options.timeout == 12_500

    async def test_escalated_retries_get_longer_http_timeouts(self) -> None:
        models = FakeModels(responses=[Delay(5), Delay(5), response([text_part("ok")])])
        outcome = await run_generate(models, request_timeout=0.05, retry_policy=_escalating_policy())
        assert outcome.text == "ok"
        assert [call["config"].http_options.timeout for call in models.calls] == [50, 100, 200]

    async def test_no_timeout_leaves_the_config_alone(self) -> None:
        models = FakeModels(responses=[response([text_part("ok")])])
        await run_generate(models, request_timeout=None)
        assert models.calls[0]["config"].http_options is None

    def test_with_request_timeout_copies_the_config(self) -> None:
        from google.genai import types

        config = types.GenerateContentConfig(temperature=0.2)
        copied = with_request_timeout(config, 3.0)
        assert copied is not config
        assert copied.http_options.timeout == 3000
        assert copied.temperature == 0.2
        assert config.http_options is None
        assert with_request_timeout(config, None) is config

    async def test_stream_establishment_carries_the_timeout(self) -> None:
        models = FakeModels(streams=[[response([text_part("Hi")])]])
        adapter = GoogleGenAILLMAdapter(make_config(request_timeout=7.0))
        install(adapter, models)
        async with adapter.astream([Message.user("Q")]) as sr:
            async for _ in sr:
                pass
        assert models.calls[0]["config"].http_options.timeout == 7000
