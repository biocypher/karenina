"""Conformance tests for the google_genai adapter (cold, fake genai client).

The shared port suites in this package (test_agent_port, test_llm_port,
test_parser_port, test_message_converter, test_trace_format) are written
against the langchain_deep_agents fixtures and message types. This module
applies the same checks to the google_genai adapter, with a fake
``client.aio.models`` in place of the network.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytest.importorskip("google.genai", reason="google-genai not installed")

from google.genai import types  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from karenina.adapters.google_genai.agent import GoogleGenAIAgentAdapter  # noqa: E402
from karenina.adapters.google_genai.llm import GoogleGenAILLMAdapter  # noqa: E402
from karenina.adapters.google_genai.messages import extract_system_instruction, from_contents, to_contents  # noqa: E402
from karenina.adapters.google_genai.parser import GoogleGenAIParserAdapter  # noqa: E402
from karenina.ports import (  # noqa: E402
    AgentConfig,
    AgentPort,
    AgentResult,
    LLMPort,
    Message,
    ParserPort,
    Role,
    Tool,
    ToolUseContent,
)
from karenina.ports.capabilities import PortCapabilities  # noqa: E402
from tests.unit.adapters.google_genai._fakes import (  # noqa: E402
    FakeModels,
    call_part,
    install,
    make_config,
    response,
    text_part,
)

LOOKUP = Tool(name="lookup", description="Look up a record", input_schema={"type": "object"})


class _Schema(BaseModel):
    value: str = Field(description="test value")


@pytest.fixture
def agent_adapter() -> GoogleGenAIAgentAdapter:
    return GoogleGenAIAgentAdapter(make_config())


@pytest.fixture
def llm_adapter() -> GoogleGenAILLMAdapter:
    return GoogleGenAILLMAdapter(make_config())


@pytest.fixture
def parser_adapter() -> GoogleGenAIParserAdapter:
    return GoogleGenAIParserAdapter(make_config())


def _run_tool_loop(adapter: GoogleGenAIAgentAdapter) -> AgentResult:
    """One tool call, its result, then the final answer."""
    models = FakeModels(
        responses=[
            response([text_part("Searching..."), call_part("lookup", {"q": "test"}, call_id="c1")]),
            response([text_part("The capital of France is Paris.")]),
        ]
    )
    install(adapter, models)
    return asyncio.run(adapter.arun([Message.user("What is the capital of France?")], tools=[LOOKUP]))


@pytest.mark.unit
class TestGoogleGenAIAgentPortConformance:
    def test_isinstance_check(self, agent_adapter: GoogleGenAIAgentAdapter) -> None:
        assert isinstance(agent_adapter, AgentPort)

    @pytest.mark.parametrize("method", ["arun", "run"])
    def test_signatures(self, agent_adapter: GoogleGenAIAgentAdapter, method: str) -> None:
        params = list(inspect.signature(getattr(agent_adapter, method)).parameters)
        assert {"messages", "tools", "mcp_servers", "config"} <= set(params)

    def test_aclose_does_not_raise(self, agent_adapter: GoogleGenAIAgentAdapter) -> None:
        asyncio.run(agent_adapter.aclose())

    def test_capabilities(self, agent_adapter: GoogleGenAIAgentAdapter) -> None:
        assert isinstance(agent_adapter.capabilities, PortCapabilities)

    def test_arun_returns_a_valid_agent_result(self, agent_adapter: GoogleGenAIAgentAdapter) -> None:
        install(agent_adapter, FakeModels(responses=[response([text_part("Paris.")])]))
        result = asyncio.run(agent_adapter.arun([Message.user("Capital of France?")], config=AgentConfig(max_turns=5)))
        assert isinstance(result, AgentResult)
        assert isinstance(result.final_response, str) and result.final_response
        assert isinstance(result.raw_trace, str)
        assert isinstance(result.trace_messages, list)
        assert isinstance(result.turns, int) and result.turns >= 0
        assert isinstance(result.limit_reached, bool)

    def test_agent_result_usage_is_valid(self, agent_adapter: GoogleGenAIAgentAdapter) -> None:
        result = _run_tool_loop(agent_adapter)
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0
        assert result.usage.total_tokens > 0
        assert result.usage.model is not None


@pytest.mark.unit
class TestGoogleGenAILLMPortConformance:
    def test_isinstance_check(self, llm_adapter: GoogleGenAILLMAdapter) -> None:
        assert isinstance(llm_adapter, LLMPort)

    @pytest.mark.parametrize("method", ["ainvoke", "invoke"])
    def test_signatures(self, llm_adapter: GoogleGenAILLMAdapter, method: str) -> None:
        assert "messages" in inspect.signature(getattr(llm_adapter, method)).parameters

    def test_capabilities(self, llm_adapter: GoogleGenAILLMAdapter) -> None:
        caps = llm_adapter.capabilities
        assert isinstance(caps, PortCapabilities)
        assert isinstance(caps.supports_system_prompt, bool)
        assert isinstance(caps.supports_structured_output, bool)

    def test_with_structured_output_returns_a_new_llm_port(self, llm_adapter: GoogleGenAILLMAdapter) -> None:
        structured = llm_adapter.with_structured_output(_Schema)
        assert isinstance(structured, LLMPort)
        assert structured is not llm_adapter

    def test_with_structured_output_preserves_the_original(self, llm_adapter: GoogleGenAILLMAdapter) -> None:
        before = getattr(llm_adapter, "_structured_schema", None)
        llm_adapter.with_structured_output(_Schema)
        assert getattr(llm_adapter, "_structured_schema", None) == before

    def test_structured_content_is_json(self, llm_adapter: GoogleGenAILLMAdapter) -> None:
        install(llm_adapter, FakeModels(responses=[response([text_part('{"value": "x"}')])]))
        result = asyncio.run(llm_adapter.with_structured_output(_Schema).ainvoke([Message.user("Q")]))
        assert _Schema.model_validate_json(result.content) == _Schema(value="x")


@pytest.mark.unit
class TestGoogleGenAIParserPortConformance:
    def test_isinstance_check(self, parser_adapter: GoogleGenAIParserAdapter) -> None:
        assert isinstance(parser_adapter, ParserPort)

    @pytest.mark.parametrize("method", ["aparse_to_pydantic", "parse_to_pydantic"])
    def test_signatures(self, parser_adapter: GoogleGenAIParserAdapter, method: str) -> None:
        params = inspect.signature(getattr(parser_adapter, method)).parameters
        assert "messages" in params and "schema" in params

    def test_capabilities(self, parser_adapter: GoogleGenAIParserAdapter) -> None:
        caps = parser_adapter.capabilities
        assert isinstance(caps, PortCapabilities)
        assert isinstance(caps.supports_system_prompt, bool)
        assert isinstance(caps.supports_structured_output, bool)

    def test_parse_returns_model_and_usage(self, parser_adapter: GoogleGenAIParserAdapter) -> None:
        install(parser_adapter._llm_adapter, FakeModels(responses=[response([text_part('{"value": "y"}')])]))
        result = asyncio.run(parser_adapter.aparse_to_pydantic([Message.user("Q")], _Schema))
        assert result.parsed == _Schema(value="y")
        assert result.usage.total_tokens >= 0


@pytest.mark.unit
class TestGoogleGenAIMessageConverterConformance:
    def test_user_message_roundtrip(self) -> None:
        contents = to_contents([Message.user("Hello, world!")])
        restored = from_contents(contents)
        assert len(restored) == 1
        assert restored[0].role == Role.USER
        assert restored[0].text == "Hello, world!"

    def test_assistant_message_roundtrip(self) -> None:
        restored = from_contents([types.Content(role="model", parts=[types.Part(text="The answer is 42.")])])
        assert len(restored) == 1
        assert restored[0].role == Role.ASSISTANT
        assert "42" in restored[0].text

    def test_system_message_extracted_separately(self) -> None:
        messages = [Message.system("Be helpful."), Message.user("Hi")]
        contents = to_contents(messages)
        assert extract_system_instruction(messages) == "Be helpful."
        assert all("Be helpful." not in (p.text or "") for c in contents for p in c.parts)
        assert contents[0].parts[0].text == "Hi"

    def test_tool_message_roundtrip(self) -> None:
        messages = [
            Message.assistant(tool_calls=[ToolUseContent(id="call_123", name="lookup", input={})]),
            Message.tool_result("call_123", "tool output data"),
        ]
        restored = from_contents(to_contents(messages))
        assert restored[1].role == Role.TOOL
        assert restored[1].content[0].tool_use_id == "call_123"
        assert restored[1].content[0].content == "tool output data"

    def test_empty_messages(self) -> None:
        assert to_contents([]) == []
        assert extract_system_instruction([]) is None


@pytest.mark.unit
class TestGoogleGenAITraceFormatConformance:
    def test_raw_trace_uses_the_standard_delimiters(self, agent_adapter: GoogleGenAIAgentAdapter) -> None:
        trace = _run_tool_loop(agent_adapter).raw_trace
        assert "--- AI Message ---" in trace
        assert "Tool Calls:" in trace
        assert "lookup" in trace
        assert "--- Tool Message (call_id: c1) ---" in trace
        assert "--- Tool Call ---" not in trace
        assert "--- Tool Result ---" not in trace

    def test_trace_keeps_message_order(self, agent_adapter: GoogleGenAIAgentAdapter) -> None:
        trace = _run_tool_loop(agent_adapter).raw_trace
        first = trace.index("Searching...")
        tool = trace.index("--- Tool Message (call_id: c1) ---")
        final = trace.index("The capital of France is Paris.")
        assert first < tool < final

    def test_empty_messages_produce_an_empty_trace(self) -> None:
        from karenina.adapters.google_genai.trace import build_raw_trace

        assert build_raw_trace([]) == ""
