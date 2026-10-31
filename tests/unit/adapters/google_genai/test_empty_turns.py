"""An empty Gemini turn is never a successful answer.

Gemini sometimes ends a turn with STOP and no parts, notably right after
function responses. Such a call is retried as incomplete, and a run that
still ends without any answer text fails, so an agentic judge never scores
a trace that holds no assessment.
"""

from __future__ import annotations

import asyncio

import pytest

from karenina.adapters.google_genai.agent import GoogleGenAIAgentAdapter
from karenina.adapters.google_genai.errors import GeminiIncompleteResponseError
from karenina.adapters.google_genai.llm import GoogleGenAILLMAdapter
from karenina.ports import AgentExecutionError, AgentResponseError, Message, Tool
from tests.unit.adapters.google_genai._fakes import (
    FakeModels,
    call_part,
    install,
    make_config,
    response,
    run_generate,
    search_grounding,
    text_part,
    thought_part,
    usage,
)

LOOKUP = Tool(name="lookup", description="Look up a record", input_schema={"type": "object"})


def _empty_stop() -> object:
    return response([], finish="STOP")


def _agent(models: FakeModels, **overrides: object) -> GoogleGenAIAgentAdapter:
    adapter = GoogleGenAIAgentAdapter(make_config(**overrides))
    install(adapter, models)
    return adapter


def _llm(models: FakeModels, **overrides: object) -> GoogleGenAILLMAdapter:
    adapter = GoogleGenAILLMAdapter(make_config(**overrides))
    install(adapter, models)
    return adapter


@pytest.mark.unit
class TestEmptyStopCall:
    async def test_empty_stop_is_retried(self) -> None:
        models = FakeModels(responses=[_empty_stop(), response([text_part("Answer.")])])
        outcome = await run_generate(models)
        assert outcome.text == "Answer."
        assert len(models.calls) == 2

    async def test_persistent_empty_stop_fails_as_incomplete(self) -> None:
        models = FakeModels(responses=[_empty_stop() for _ in range(3)])
        with pytest.raises(GeminiIncompleteResponseError, match="no text and no function call"):
            await run_generate(models)
        assert len(models.calls) == 3  # first call plus the two server-error retries

    async def test_thought_only_stop_is_incomplete(self) -> None:
        models = FakeModels(responses=[response([thought_part("Thinking...")]), response([text_part("Done.")])])
        assert (await run_generate(models)).text == "Done."
        assert len(models.calls) == 2

    async def test_search_without_text_is_incomplete(self) -> None:
        searched = response([], grounding=search_grounding(["q"]))
        models = FakeModels(responses=[searched, response([text_part("Done.")])])
        assert (await run_generate(models)).text == "Done."

    async def test_empty_stop_in_a_stream_is_retried(self) -> None:
        models = FakeModels(streams=[[response([thought_part("hmm")], finish="STOP")], [response([text_part("Ok.")])]])
        assert (await run_generate(models, stream=True)).text == "Ok."

    async def test_function_call_without_text_is_complete(self) -> None:
        models = FakeModels(responses=[response([call_part("lookup")])])
        outcome = await run_generate(models)
        assert outcome.has_function_calls is True
        assert len(models.calls) == 1


@pytest.mark.unit
class TestAgentEmptyTurns:
    async def test_empty_final_turn_after_function_responses_is_retried(self) -> None:
        models = FakeModels(
            responses=[response([call_part("lookup", {"id": "1"})]), _empty_stop(), response([text_part("Verdict.")])]
        )
        result = await _agent(models).arun([Message.user("Q")], tools=[LOOKUP])
        assert result.final_response == "Verdict."
        assert result.turns == 2
        assert len(models.calls) == 3

    async def test_persistent_empty_final_turn_fails_the_run(self) -> None:
        models = FakeModels(
            responses=[
                response([text_part("Let me check."), call_part("lookup", {"id": "1"})]),
                *[_empty_stop() for _ in range(3)],
            ]
        )
        with pytest.raises(AgentExecutionError, match="no text and no function call"):
            await _agent(models).arun([Message.user("Q")], tools=[LOOKUP])

    async def test_no_answer_at_all_fails_the_run(self) -> None:
        models = FakeModels(responses=[_empty_stop() for _ in range(3)])
        with pytest.raises(AgentExecutionError, match="GeminiIncompleteResponseError"):
            await _agent(models).arun([Message.user("Q")])

    async def test_empty_max_tokens_final_turn_fails_the_run(self) -> None:
        models = FakeModels(
            responses=[
                response([text_part("Let me check."), call_part("lookup", {"id": "1"})]),
                response([thought_part("long thinking")], finish="MAX_TOKENS"),
            ]
        )
        with pytest.raises(AgentResponseError, match="empty final turn"):
            await _agent(models).arun([Message.user("Q")], tools=[LOOKUP])

    async def test_too_many_tool_calls_without_text_fails_the_run(self) -> None:
        models = FakeModels(
            responses=[response([], finish="TOO_MANY_TOOL_CALLS", grounding=search_grounding(["q1", "q2"]))]
        )
        adapter = _agent(models, extra_kwargs={"builtin_tools": ["google_search"]})
        with pytest.raises(AgentResponseError, match="TOO_MANY_TOOL_CALLS"):
            await adapter.arun([Message.user("Q")])

    async def test_too_many_tool_calls_with_text_is_a_limit(self) -> None:
        models = FakeModels(
            responses=[
                response(
                    [text_part("Partial assessment: likely fabricated.")],
                    finish="TOO_MANY_TOOL_CALLS",
                    grounding=search_grounding(["q1", "q2"]),
                )
            ]
        )
        result = await _agent(models, extra_kwargs={"builtin_tools": ["google_search"]}).arun([Message.user("Q")])
        assert result.limit_reached is True
        assert result.final_response == "Partial assessment: likely fabricated."
        assert "TOO_MANY_TOOL_CALLS" in result.raw_trace
        assert result.usage.web_search_queries == 2


@pytest.mark.unit
class TestLLMEmptyTurns:
    async def test_ainvoke_retries_an_empty_stop(self) -> None:
        models = FakeModels(responses=[_empty_stop(), response([text_part("Paris")])])
        assert (await _llm(models).ainvoke([Message.user("Q")])).content == "Paris"

    async def test_astream_empty_stop_raises(self) -> None:
        models = FakeModels(streams=[[response([], finish="STOP")]])
        with pytest.raises(GeminiIncompleteResponseError, match="no text"):
            async with _llm(models).astream([Message.user("Q")]) as sr:
                async for _ in sr:
                    pass

    async def test_stream_invoke_retries_an_empty_stop(self) -> None:
        models = FakeModels(streams=[[response([], finish="STOP")], [response([text_part("Whole.")])]])
        adapter = _llm(models)
        result = await asyncio.to_thread(adapter.stream_invoke, [Message.user("Q")], 5.0)
        assert result.content == "Whole."


@pytest.mark.unit
class TestRetriedAttemptsAreBilled:
    async def test_usage_and_searches_of_a_retried_attempt_are_kept(self) -> None:
        searched = response([], grounding=search_grounding(["q1", "q2"]), usage_metadata=usage(100, 3))
        models = FakeModels(responses=[searched, response([text_part("Done.")], usage_metadata=usage(120, 7))])
        outcome = await run_generate(models)
        assert outcome.text == "Done."
        assert outcome.search_query_count == 2
        assert outcome.usage_metadata.prompt_token_count == 220
        assert outcome.usage_metadata.candidates_token_count == 10

    async def test_a_first_time_success_keeps_the_sdk_usage_object(self) -> None:
        only = usage(10, 5)
        outcome = await run_generate(FakeModels(responses=[response([text_part("ok")], usage_metadata=only)]))
        assert outcome.usage_metadata is only
        assert outcome.search_query_count == 0
