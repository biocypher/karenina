"""Tests for GoogleGenAIAgentAdapter (mocked genai client and MCP sessions)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from google.genai import types

from karenina.adapters.google_genai import agent as agent_module
from karenina.adapters.google_genai.agent import GoogleGenAIAgentAdapter
from karenina.ports import (
    AgentConfig,
    AgentExecutionError,
    AgentResponseError,
    AgentResult,
    AgentTimeoutError,
    Message,
    Role,
    ThinkingContent,
    Tool,
    ToolResultContent,
    ToolUseContent,
)
from tests.unit.adapters.google_genai._fakes import (
    Delay,
    FakeClient,
    FakeModels,
    call_part,
    fast_policy,
    install,
    make_config,
    response,
    search_grounding,
    text_part,
    thought_part,
    url_context,
    usage,
)

LOOKUP = Tool(
    name="lookup",
    description="Look up a record",
    input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
)


def _agent(models: FakeModels, **config_overrides: Any) -> GoogleGenAIAgentAdapter:
    adapter = GoogleGenAIAgentAdapter(make_config(**config_overrides))
    install(adapter, models)
    return adapter


def _tool_names(call: dict[str, Any]) -> list[str]:
    """Names of the tools sent in one recorded call."""
    names: list[str] = []
    for tool in call["config"].tools or []:
        if tool.function_declarations:
            names.extend(d.name for d in tool.function_declarations)
        if tool.google_search is not None:
            names.append("google_search")
        if tool.url_context is not None:
            names.append("url_context")
    return names


@pytest.mark.unit
class TestBasics:
    async def test_no_tools_is_a_single_turn(self) -> None:
        models = FakeModels(responses=[response([text_part("Paris.")], usage_metadata=usage(12, 3, total=15))])
        result = await _agent(models).arun([Message.system("Be brief."), Message.user("Capital of France?")])

        assert isinstance(result, AgentResult)
        assert result.final_response == "Paris."
        assert result.turns == 1
        assert result.limit_reached is False
        assert result.timeout_reached is False
        assert result.actual_model == "gemini-3.8-flash-001"
        assert result.session_id is None
        assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.total_tokens) == (12, 3, 15)
        assert result.usage.model == "gemini-3.8-flash"
        assert "--- AI Message ---\nParis." in result.raw_trace
        (call,) = models.calls
        assert call["config"].system_instruction == "Be brief."
        assert call["config"].automatic_function_calling.disable is True
        assert call["config"].tools is None

    def test_capabilities(self) -> None:
        caps = GoogleGenAIAgentAdapter(make_config()).capabilities
        assert caps.supports_system_prompt is True
        assert caps.supports_file_tools is False
        assert caps.supports_code_execution is False

    def test_unknown_extra_kwargs_fail_at_construction(self) -> None:
        with pytest.raises(ValueError, match="Unknown extra_kwargs"):
            GoogleGenAIAgentAdapter(make_config(extra_kwargs={"max_searches": 3}))

    async def test_system_prompt_fallbacks(self) -> None:
        models = FakeModels(responses=[response([text_part("a")]), response([text_part("b")])])
        adapter = _agent(models, system_prompt="model-config prompt")
        await adapter.arun([Message.user("Q")], config=AgentConfig(system_prompt="agent-config prompt"))
        await adapter.arun([Message.user("Q")])
        assert models.calls[0]["config"].system_instruction == "agent-config prompt"
        assert models.calls[1]["config"].system_instruction == "model-config prompt"

    async def test_sync_run_without_a_loop(self) -> None:
        models = FakeModels(responses=[response([text_part("sync ok")])])
        adapter = _agent(models)
        result = await asyncio.to_thread(adapter.run, [Message.user("Q")])
        assert result.final_response == "sync ok"

    async def test_aclose_closes_the_client(self) -> None:
        models = FakeModels(responses=[response([text_part("x")])])
        fake_client = FakeClient(models)
        adapter = GoogleGenAIAgentAdapter(make_config())
        with patch("karenina.adapters.google_genai.client.build_client", return_value=fake_client):
            await adapter.arun([Message.user("Q")])
        await adapter.aclose()
        fake_client.aio.aclose.assert_awaited_once()
        fake_client.close.assert_called_once()
        await adapter.aclose()  # safe to call twice


@pytest.mark.unit
class TestToolLoop:
    async def test_two_turn_loop_keeps_the_model_content_unchanged(self) -> None:
        first = response(
            [thought_part("Need the record.", signature=b"sig-1"), call_part("lookup", {"id": "42"})],
            usage_metadata=usage(100, 10, thoughts=20, cached=0),
        )
        second = response([text_part("Record 42 is fine.")], usage_metadata=usage(150, 8, cached=90))
        models = FakeModels(responses=[first, second])
        result = await _agent(models, extra_kwargs={"include_thoughts": True}).arun(
            [Message.user("Check 42")], tools=[LOOKUP]
        )

        assert result.turns == 2
        assert result.final_response == "Record 42 is fine."
        # The second request carries the first model Content object itself (thought signatures intact).
        second_contents = models.calls[1]["contents"]
        assert second_contents[1] is first.candidates[0].content
        responses_turn = second_contents[2]
        assert responses_turn.role == "user"
        function_response = responses_turn.parts[0].function_response
        assert function_response.name == "lookup"
        assert "lookup" in function_response.response["result"]

        assistant, tool_message, final = result.trace_messages
        thinking = assistant.content[0]
        assert isinstance(thinking, ThinkingContent)
        assert thinking.signature is not None
        assert assistant.tool_calls == [ToolUseContent(id="call_1_0", name="lookup", input={"id": "42"})]
        assert assistant.usage_metadata == {
            "input_tokens": 100,
            "output_tokens": 30,
            "cache_read_input_tokens": 0,
            "reasoning_tokens": 20,
        }
        assert tool_message.role == Role.TOOL
        assert tool_message.content[0].tool_use_id == "call_1_0"
        assert final.text == "Record 42 is fine."
        # Usage summed over both calls, cached and thinking tokens included.
        assert result.usage.input_tokens == 250
        assert result.usage.output_tokens == 38
        assert result.usage.cache_read_tokens == 90
        assert result.usage.reasoning_tokens == 20
        assert "--- Tool Message (call_id: call_1_0) ---" in result.raw_trace

    async def test_api_call_id_is_echoed_in_the_response(self) -> None:
        models = FakeModels(
            responses=[response([call_part("lookup", {"id": "1"}, call_id="api-1")]), response([text_part("done")])]
        )
        result = await _agent(models).arun([Message.user("Q")], tools=[LOOKUP])
        assert models.calls[1]["contents"][2].parts[0].function_response.id == "api-1"
        assert result.trace_messages[0].tool_calls[0].id == "api-1"

    async def test_mcp_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = SimpleNamespace(
            call_tool=AsyncMock(
                side_effect=[
                    SimpleNamespace(content=[SimpleNamespace(text="PMID 1: real paper")], isError=False),
                    RuntimeError("mirror down"),
                ]
            )
        )
        mcp_tools = [
            SimpleNamespace(name="europepmc search", description="Search", inputSchema={"type": "object"}),
            SimpleNamespace(name="europepmc_get", description="Get", inputSchema={"type": "object"}),
            SimpleNamespace(name="hidden", description="Not allowed", inputSchema={"type": "object"}),
        ]
        connected: dict[str, Any] = {}

        async def fake_connect(_stack: Any, servers: dict[str, Any]) -> dict[str, Any]:
            connected.update(servers)
            return {"epmc": session}

        async def fake_tools(_sessions: dict[str, Any]) -> list[tuple[str, Any, Any]]:
            return [("epmc", session, tool) for tool in mcp_tools]

        monkeypatch.setattr(agent_module, "connect_all_mcp_servers", fake_connect)
        monkeypatch.setattr(agent_module, "get_all_mcp_tools", fake_tools)
        models = FakeModels(
            responses=[
                response([call_part("europepmc_search", {"q": "x"}), call_part("europepmc_get", {"id": "1"})]),
                response([text_part("Verdict: backed.")]),
            ]
        )
        adapter = _agent(
            models,
            mcp_tool_filter=["europepmc search", "europepmc_get"],
            mcp_tool_description_overrides={"europepmc_get": "Fetch one record"},
            mcp_http_timeout=15.0,
            mcp_sse_read_timeout=90.0,
        )
        result = await adapter.arun(
            [Message.user("Check")], mcp_servers={"epmc": {"type": "http", "url": "http://localhost:1/mcp"}}
        )

        assert connected["epmc"]["timeout"] == 15.0
        assert connected["epmc"]["sse_read_timeout"] == 90.0
        declarations = models.calls[0]["config"].tools[0].function_declarations
        assert [d.name for d in declarations] == ["europepmc_search", "europepmc_get"]
        assert declarations[1].description == "Fetch one record"
        assert session.call_tool.await_args_list[0].args == ("europepmc search", {"q": "x"})
        trace_names = [t.name for t in result.trace_messages[0].tool_calls]
        assert trace_names == ["europepmc search", "europepmc_get"]
        error_result = result.trace_messages[2].content[0]
        assert isinstance(error_result, ToolResultContent)
        assert error_result.is_error is True
        sent = models.calls[1]["contents"][2].parts[1].function_response.response
        assert "mirror down" in sent["error"]
        assert result.final_response == "Verdict: backed."

    async def test_duplicate_sanitised_names_fail_before_any_call(self) -> None:
        models = FakeModels(responses=[response([text_part("never")])])
        tools = [
            Tool(name="get data", description="a", input_schema={}),
            Tool(name="get/data", description="b", input_schema={}),
        ]
        with pytest.raises(AgentExecutionError, match="get_data"):
            await _agent(models).arun([Message.user("Q")], tools=tools)
        assert models.calls == []

    async def test_tool_results_are_truncated(self) -> None:
        session = SimpleNamespace(
            call_tool=AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(text="x" * 500)], isError=False))
        )

        async def fake_connect(_stack: Any, _servers: dict[str, Any]) -> dict[str, Any]:
            return {"s": session}

        async def fake_tools(_sessions: dict[str, Any]) -> list[tuple[str, Any, Any]]:
            return [("s", session, SimpleNamespace(name="dump", description="d", inputSchema={}))]

        models = FakeModels(responses=[response([call_part("dump")]), response([text_part("ok")])])
        adapter = _agent(models)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(agent_module, "connect_all_mcp_servers", fake_connect)
            mp.setattr(agent_module, "get_all_mcp_tools", fake_tools)
            result = await adapter.arun(
                [Message.user("Q")],
                mcp_servers={"s": {"type": "http", "url": "http://x/mcp"}},
                config=AgentConfig(extra={"max_tool_result_chars": 100}),
            )
        sent = models.calls[1]["contents"][2].parts[0].function_response.response["result"]
        assert sent.startswith("x" * 100)
        assert "[... truncated 400 characters]" in sent
        assert result.trace_messages[1].content[0].content == sent


@pytest.mark.unit
class TestServerSideTools:
    async def test_grounding_pairs_come_before_the_turns_text(self) -> None:
        models = FakeModels(
            responses=[
                response(
                    [text_part("Answer with sources.")],
                    grounding=search_grounding(["q1", "q2"], [("Paper", "https://src")]),
                    url_metadata=url_context(("https://page", "URL_RETRIEVAL_STATUS_SUCCESS")),
                )
            ]
        )
        result = await _agent(models, extra_kwargs={"builtin_tools": ["google_search", "url_context"]}).arun(
            [Message.user("Q")]
        )
        search_use, search_result, url_use, url_result, answer = result.trace_messages
        assert search_use.tool_calls[0].name == "google_search"
        assert search_use.tool_calls[0].input == {"queries": ["q1", "q2"]}
        assert json.loads(search_result.content[0].content)["sources"] == [{"title": "Paper", "uri": "https://src"}]
        assert url_use.tool_calls[0].input == {"urls": ["https://page"]}
        assert answer.text == "Answer with sources."
        assert result.usage.web_search_queries == 2
        assert answer.usage_metadata["web_search_queries"] == 2
        assert _tool_names(models.calls[0]) == ["google_search", "url_context"]

    async def test_search_budget_removes_google_search_after_it_is_reached(self) -> None:
        models = FakeModels(
            responses=[
                response([call_part("lookup", {"id": "1"})], grounding=search_grounding(["a", "b", "c"])),
                response([text_part("done")]),
            ]
        )
        adapter = _agent(models, extra_kwargs={"builtin_tools": ["google_search"], "max_search_queries": 2})
        result = await adapter.arun([Message.user("Q")], tools=[LOOKUP])

        assert _tool_names(models.calls[0]) == ["lookup", "google_search"]
        assert _tool_names(models.calls[1]) == ["lookup"]  # budget spent: search removed
        # Soft budget: the first call went past it and the real count is reported.
        assert result.usage.web_search_queries == 3
        assert "Google Search budget of 2 queries reached after 3 queries" in result.raw_trace

    async def test_budget_override_per_run(self) -> None:
        models = FakeModels(responses=[response([text_part("no search")])])
        adapter = _agent(models, extra_kwargs={"builtin_tools": ["google_search"]})
        result = await adapter.arun([Message.user("Q")], config=AgentConfig(extra={"max_search_queries": 0}))
        assert _tool_names(models.calls[0]) == []
        assert result.usage.web_search_queries is None

    async def test_budget_below_limit_keeps_search(self) -> None:
        models = FakeModels(
            responses=[
                response([call_part("lookup")], grounding=search_grounding(["a"])),
                response([text_part("done")]),
            ]
        )
        adapter = _agent(models, extra_kwargs={"builtin_tools": ["google_search"], "max_search_queries": 5})
        result = await adapter.arun([Message.user("Q")], tools=[LOOKUP])
        assert _tool_names(models.calls[1]) == ["lookup", "google_search"]
        assert result.usage.web_search_queries == 1
        assert "budget" not in result.raw_trace


@pytest.mark.unit
class TestFinishReasons:
    async def test_too_many_tool_calls_without_any_trace_fails(self) -> None:
        # More cases (text kept as a limit, searches only) are in test_empty_turns.py.
        models = FakeModels(responses=[response([], finish="TOO_MANY_TOOL_CALLS")])
        with pytest.raises(AgentResponseError, match="TOO_MANY_TOOL_CALLS"):
            await _agent(models).arun([Message.user("Q")])

    @pytest.mark.parametrize("reason", ["MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"])
    async def test_bad_tool_call_gets_a_corrective_note(self, reason: str) -> None:
        models = FakeModels(responses=[response([], finish=reason), response([text_part("Recovered.")])])
        result = await _agent(models).arun([Message.user("Q")], tools=[LOOKUP])

        assert result.turns == 2
        assert result.final_response == "Recovered."
        retry_contents = models.calls[1]["contents"]
        assert [c.role for c in retry_contents] == ["user"]
        assert retry_contents[0].parts[0].text == "Q"
        assert reason in retry_contents[0].parts[1].text
        notes = [m for m in result.trace_messages if m.role == Role.USER]
        assert len(notes) == 1 and reason in notes[0].text

    @pytest.mark.parametrize("reason", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "OTHER"])
    async def test_policy_stops_raise(self, reason: str) -> None:
        models = FakeModels(responses=[response([], finish=reason)])
        with pytest.raises(AgentExecutionError, match=reason):
            await _agent(models).arun([Message.user("Q")])

    async def test_max_tokens_keeps_the_text_with_a_note(self) -> None:
        models = FakeModels(responses=[response([text_part("Truncated ans")], finish="MAX_TOKENS")])
        result = await _agent(models).arun([Message.user("Q")])
        assert result.final_response == "Truncated ans"
        assert "MAX_TOKENS" in result.raw_trace
        assert result.limit_reached is False


@pytest.mark.unit
class TestLimits:
    async def test_turn_limit(self) -> None:
        models = FakeModels(responses=[response([call_part("lookup", {"id": str(i)})]) for i in range(5)])
        result = await _agent(models).arun([Message.user("Q")], tools=[LOOKUP], config=AgentConfig(max_turns=2))
        assert result.turns == 2
        assert result.limit_reached is True
        assert len(models.calls) == 2
        assert result.raw_trace.endswith("[Note: Turn limit reached - partial response shown]")
        assert result.trace_messages[-1].role == Role.TOOL

    async def test_timeout_keeps_the_partial_trace(self) -> None:
        models = FakeModels(
            responses=[
                response([text_part("Looking it up."), call_part("lookup", {"id": "1"})], usage_metadata=usage(40, 4)),
                Delay(5),
                response([text_part("never")]),
            ]
        )
        result = await _agent(models).arun([Message.user("Q")], tools=[LOOKUP], config=AgentConfig(timeout=0.2))
        assert result.timeout_reached is True
        assert result.limit_reached is False
        assert result.turns == 2
        assert result.final_response == "Looking it up."
        assert [m.role for m in result.trace_messages] == [Role.ASSISTANT, Role.TOOL]
        assert result.usage.input_tokens == 40
        assert "[Note: Agent timed out - partial response shown]" in result.raw_trace

    async def test_timeout_without_any_trace_raises(self) -> None:
        models = FakeModels(responses=[Delay(5), response([text_part("never")])])
        with pytest.raises(AgentTimeoutError, match="no messages"):
            await _agent(models).arun([Message.user("Q")], config=AgentConfig(timeout=0.1))

    @pytest.mark.parametrize("hung_stage", ["connect", "list_tools"])
    async def test_hung_mcp_setup_is_bounded_by_the_run_timeout(
        self, hung_stage: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def connect(_stack: Any, _servers: dict[str, Any]) -> dict[str, Any]:
            if hung_stage == "connect":
                await asyncio.sleep(5)
            return {"s": object()}

        async def list_tools(_sessions: dict[str, Any]) -> list[tuple[str, Any, Any]]:
            await asyncio.sleep(5)
            return []

        monkeypatch.setattr(agent_module, "connect_all_mcp_servers", connect)
        monkeypatch.setattr(agent_module, "get_all_mcp_tools", list_tools)
        models = FakeModels(responses=[response([text_part("never")])])
        with pytest.raises(AgentTimeoutError, match="no messages"):
            await _agent(models).arun(
                [Message.user("Q")],
                mcp_servers={"s": {"type": "http", "url": "http://localhost:1/mcp"}},
                config=AgentConfig(timeout=0.1),
            )
        assert models.calls == []

    async def test_per_call_timeout_is_not_mistaken_for_the_run_timeout(self) -> None:
        models = FakeModels(responses=[Delay(5), Delay(5), response([text_part("never")])])
        adapter = _agent(models, request_timeout=0.05, retry_policy=fast_policy(timeout=1))
        with pytest.raises(AgentExecutionError, match="TimeoutError"):
            await adapter.arun([Message.user("Q")], config=AgentConfig(timeout=30))
        assert len(models.calls) == 2


@pytest.mark.unit
def test_streamed_agent_turns_use_the_stream_route() -> None:
    models = FakeModels(
        streams=[
            [
                response([thought_part("thinking")], finish=None),
                response([call_part("lookup", {"id": "1"})], finish="STOP"),
            ],
            [response([text_part("Fin")], finish=None), response([text_part("al.")], finish="STOP")],
        ]
    )
    adapter = _agent(models, endpoint_base_url="https://gw.example.org", endpoint_api_key=None)
    result = asyncio.run(adapter.arun([Message.user("Q")], tools=[LOOKUP]))
    assert [c["kind"] for c in models.calls] == ["stream", "stream"]
    assert result.final_response == "Final."
    history_turn = models.calls[1]["contents"][1]
    assert isinstance(history_turn, types.Content)
    assert [bool(p.thought) for p in history_turn.parts] == [True, False]
