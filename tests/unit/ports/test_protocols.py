"""Unit tests for Protocol classes in the ports module.

Tests cover:
- LLMPort Protocol: isinstance checks with proper implementation
- AgentPort Protocol: isinstance checks with proper implementation
- ParserPort Protocol: isinstance checks with proper implementation
- Verification that objects missing required methods fail isinstance checks
- Runtime checkable behavior verification
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from karenina.ports import (
    AgentConfig,
    AgentPort,
    AgentResult,
    LLMPort,
    LLMResponse,
    MCPServerConfig,
    Message,
    ParsePortResult,
    ParserPort,
    Tool,
    UsageMetadata,
)
from karenina.ports.capabilities import PortCapabilities

# ruff: noqa: ARG002 - unused arguments are expected in mock implementations


# =============================================================================
# Mock Implementations for Testing
# =============================================================================


class MockLLMPort:
    """Mock implementation of LLMPort for testing isinstance checks.

    Implements all required members: capabilities, ainvoke, invoke, with_structured_output.
    """

    @property
    def capabilities(self) -> PortCapabilities:
        return PortCapabilities()

    async def ainvoke(self, messages: list[Message]) -> LLMResponse:
        """Mock async invocation."""
        return LLMResponse(
            content="Mock response",
            usage=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    def invoke(self, messages: list[Message]) -> LLMResponse:
        """Mock sync invocation."""
        return LLMResponse(
            content="Mock response",
            usage=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    def with_structured_output(self, schema: type[BaseModel]) -> LLMPort:
        """Mock structured output configuration."""
        return self


class MockAgentPort:
    """Mock implementation of AgentPort for testing isinstance checks.

    Implements all required methods: arun, run.
    """

    async def arun(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Mock async agent execution."""
        return AgentResult(
            final_response="Mock response",
            raw_trace="--- AI Message ---\nMock response",
            trace_messages=[Message.assistant("Mock response")],
            usage=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
            turns=1,
            limit_reached=False,
        )

    def run(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Mock sync agent execution."""
        return AgentResult(
            final_response="Mock response",
            raw_trace="--- AI Message ---\nMock response",
            trace_messages=[Message.assistant("Mock response")],
            usage=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
            turns=1,
            limit_reached=False,
        )


class MockParserPort:
    """Mock implementation of ParserPort for testing isinstance checks.

    Implements all required members: capabilities, aparse_to_pydantic, parse_to_pydantic.
    """

    @property
    def capabilities(self) -> PortCapabilities:
        return PortCapabilities()

    async def aparse_to_pydantic(self, response: str, schema: type[BaseModel]) -> ParsePortResult:
        """Mock async parsing."""
        return ParsePortResult(parsed=schema())

    def parse_to_pydantic(self, response: str, schema: type[BaseModel]) -> ParsePortResult:
        """Mock sync parsing."""
        return ParsePortResult(parsed=schema())


# =============================================================================
# Incomplete Implementations (Missing Methods)
# =============================================================================


class IncompleteLLMPort:
    """LLM implementation missing with_structured_output method."""

    async def ainvoke(self, messages: list[Message]) -> LLMResponse:
        return LLMResponse(
            content="Response",
            usage=UsageMetadata(),
        )

    def invoke(self, messages: list[Message]) -> LLMResponse:
        return LLMResponse(
            content="Response",
            usage=UsageMetadata(),
        )

    # Missing: with_structured_output


class IncompleteAgentPort:
    """Agent implementation missing run method."""

    async def arun(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        return AgentResult(
            final_response="Response",
            raw_trace="",
            trace_messages=[],
            usage=UsageMetadata(),
            turns=1,
            limit_reached=False,
        )

    # Missing: run


class IncompleteParserPort:
    """Parser implementation missing parse_to_pydantic method."""

    async def aparse_to_pydantic(self, response: str, schema: type[BaseModel]) -> ParsePortResult:
        return ParsePortResult(parsed=schema())

    # Missing: parse_to_pydantic


class NotAPort:
    """Class that doesn't implement any port methods."""

    def unrelated_method(self) -> str:
        return "Not a port"


# =============================================================================
# LLMPort Protocol Tests
# =============================================================================


@pytest.mark.unit
class TestLLMPort:
    """Tests for the LLMPort Protocol."""

    def test_isinstance_returns_true_for_mock_llm_port(self) -> None:
        """Test that isinstance returns True for proper LLMPort implementation."""
        mock = MockLLMPort()
        assert isinstance(mock, LLMPort)

    def test_isinstance_returns_false_for_incomplete_llm_port(self) -> None:
        """Test that isinstance returns False when with_structured_output is missing."""
        incomplete = IncompleteLLMPort()
        assert not isinstance(incomplete, LLMPort)

    def test_isinstance_returns_false_for_unrelated_class(self) -> None:
        """Test that isinstance returns False for unrelated classes."""
        not_port = NotAPort()
        assert not isinstance(not_port, LLMPort)

    def test_isinstance_returns_false_for_none(self) -> None:
        """Test that isinstance returns False for None."""
        assert not isinstance(None, LLMPort)

    def test_isinstance_returns_false_for_dict(self) -> None:
        """Test that isinstance returns False for dict."""
        assert not isinstance({}, LLMPort)

    def test_llm_port_is_runtime_checkable(self) -> None:
        """Verify LLMPort has runtime_checkable behavior."""
        # This confirms the protocol is properly decorated with @runtime_checkable
        # by verifying isinstance checks work at all (they would fail without it)
        mock = MockLLMPort()
        result = isinstance(mock, LLMPort)
        assert result is True


@pytest.mark.unit
class TestLLMPortMethodSignatures:
    """Tests verifying LLMPort method signatures work correctly."""

    @pytest.mark.asyncio
    async def test_mock_llm_port_ainvoke_returns_llm_response(self) -> None:
        """Test that MockLLMPort.ainvoke returns LLMResponse."""
        mock = MockLLMPort()
        messages = [Message.user("Hello")]
        response = await mock.ainvoke(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Mock response"
        assert isinstance(response.usage, UsageMetadata)

    def test_mock_llm_port_invoke_returns_llm_response(self) -> None:
        """Test that MockLLMPort.invoke returns LLMResponse."""
        mock = MockLLMPort()
        messages = [Message.user("Hello")]
        response = mock.invoke(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Mock response"

    def test_mock_llm_port_with_structured_output_returns_llm_port(self) -> None:
        """Test that MockLLMPort.with_structured_output returns LLMPort."""

        class TestSchema(BaseModel):
            value: str = "test"

        mock = MockLLMPort()
        result = mock.with_structured_output(TestSchema)

        assert isinstance(result, LLMPort)


# =============================================================================
# AgentPort Protocol Tests
# =============================================================================


@pytest.mark.unit
class TestAgentPort:
    """Tests for the AgentPort Protocol."""

    def test_isinstance_returns_true_for_mock_agent_port(self) -> None:
        """Test that isinstance returns True for proper AgentPort implementation."""
        mock = MockAgentPort()
        assert isinstance(mock, AgentPort)

    def test_isinstance_returns_false_for_incomplete_agent_port(self) -> None:
        """Test that isinstance returns False when run is missing."""
        incomplete = IncompleteAgentPort()
        assert not isinstance(incomplete, AgentPort)

    def test_isinstance_returns_false_for_unrelated_class(self) -> None:
        """Test that isinstance returns False for unrelated classes."""
        not_port = NotAPort()
        assert not isinstance(not_port, AgentPort)

    def test_isinstance_returns_false_for_none(self) -> None:
        """Test that isinstance returns False for None."""
        assert not isinstance(None, AgentPort)

    def test_agent_port_is_runtime_checkable(self) -> None:
        """Verify AgentPort has runtime_checkable behavior."""
        mock = MockAgentPort()
        result = isinstance(mock, AgentPort)
        assert result is True


@pytest.mark.unit
class TestAgentPortMethodSignatures:
    """Tests verifying AgentPort method signatures work correctly."""

    @pytest.mark.asyncio
    async def test_mock_agent_port_arun_returns_agent_result(self) -> None:
        """Test that MockAgentPort.arun returns AgentResult."""
        mock = MockAgentPort()
        messages = [Message.user("Hello")]
        result = await mock.arun(messages)

        assert isinstance(result, AgentResult)
        assert result.final_response == "Mock response"
        assert isinstance(result.trace_messages, list)
        assert isinstance(result.usage, UsageMetadata)
        assert result.turns == 1
        assert result.limit_reached is False

    def test_mock_agent_port_run_returns_agent_result(self) -> None:
        """Test that MockAgentPort.run returns AgentResult."""
        mock = MockAgentPort()
        messages = [Message.user("Hello")]
        result = mock.run(messages)

        assert isinstance(result, AgentResult)
        assert result.final_response == "Mock response"

    @pytest.mark.asyncio
    async def test_mock_agent_port_arun_with_tools(self) -> None:
        """Test that MockAgentPort.arun accepts tools parameter."""
        mock = MockAgentPort()
        messages = [Message.user("Search for something")]
        tools = [
            Tool(
                name="search",
                description="Search tool",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]

        result = await mock.arun(messages, tools=tools)
        assert isinstance(result, AgentResult)

    @pytest.mark.asyncio
    async def test_mock_agent_port_arun_with_config(self) -> None:
        """Test that MockAgentPort.arun accepts config parameter."""
        mock = MockAgentPort()
        messages = [Message.user("Hello")]
        config = AgentConfig(max_turns=10, timeout=30.0)

        result = await mock.arun(messages, config=config)
        assert isinstance(result, AgentResult)


# =============================================================================
# ParserPort Protocol Tests
# =============================================================================


@pytest.mark.unit
class TestParserPort:
    """Tests for the ParserPort Protocol."""

    def test_isinstance_returns_true_for_mock_parser_port(self) -> None:
        """Test that isinstance returns True for proper ParserPort implementation."""
        mock = MockParserPort()
        assert isinstance(mock, ParserPort)

    def test_isinstance_returns_false_for_incomplete_parser_port(self) -> None:
        """Test that isinstance returns False when parse_to_pydantic is missing."""
        incomplete = IncompleteParserPort()
        assert not isinstance(incomplete, ParserPort)

    def test_isinstance_returns_false_for_unrelated_class(self) -> None:
        """Test that isinstance returns False for unrelated classes."""
        not_port = NotAPort()
        assert not isinstance(not_port, ParserPort)

    def test_isinstance_returns_false_for_none(self) -> None:
        """Test that isinstance returns False for None."""
        assert not isinstance(None, ParserPort)

    def test_parser_port_is_runtime_checkable(self) -> None:
        """Verify ParserPort has runtime_checkable behavior."""
        mock = MockParserPort()
        result = isinstance(mock, ParserPort)
        assert result is True


@pytest.mark.unit
class TestParserPortMethodSignatures:
    """Tests verifying ParserPort method signatures work correctly."""

    @pytest.mark.asyncio
    async def test_mock_parser_port_aparse_returns_parse_port_result(self) -> None:
        """Test that MockParserPort.aparse_to_pydantic returns a ParsePortResult."""

        class TestSchema(BaseModel):
            value: str = "parsed"

        mock = MockParserPort()
        result = await mock.aparse_to_pydantic("Some response text", TestSchema)

        assert isinstance(result, ParsePortResult)
        assert isinstance(result.parsed, BaseModel)

    def test_mock_parser_port_parse_returns_parse_port_result(self) -> None:
        """Test that MockParserPort.parse_to_pydantic returns a ParsePortResult."""

        class TestSchema(BaseModel):
            value: str = "parsed"

        mock = MockParserPort()
        result = mock.parse_to_pydantic("Some response text", TestSchema)

        assert isinstance(result, ParsePortResult)
        assert isinstance(result.parsed, BaseModel)


# =============================================================================
# Cross-Protocol Tests
# =============================================================================


@pytest.mark.unit
class TestCrossProtocolBehavior:
    """Tests for cross-protocol behaviors and type checking."""

    def test_mock_llm_port_is_not_agent_port(self) -> None:
        """Test that MockLLMPort doesn't satisfy AgentPort."""
        mock = MockLLMPort()
        assert not isinstance(mock, AgentPort)

    def test_mock_llm_port_is_not_parser_port(self) -> None:
        """Test that MockLLMPort doesn't satisfy ParserPort."""
        mock = MockLLMPort()
        assert not isinstance(mock, ParserPort)

    def test_mock_agent_port_is_not_llm_port(self) -> None:
        """Test that MockAgentPort doesn't satisfy LLMPort."""
        mock = MockAgentPort()
        assert not isinstance(mock, LLMPort)

    def test_mock_agent_port_is_not_parser_port(self) -> None:
        """Test that MockAgentPort doesn't satisfy ParserPort."""
        mock = MockAgentPort()
        assert not isinstance(mock, ParserPort)

    def test_mock_parser_port_is_not_llm_port(self) -> None:
        """Test that MockParserPort doesn't satisfy LLMPort."""
        mock = MockParserPort()
        assert not isinstance(mock, LLMPort)

    def test_mock_parser_port_is_not_agent_port(self) -> None:
        """Test that MockParserPort doesn't satisfy AgentPort."""
        mock = MockParserPort()
        assert not isinstance(mock, AgentPort)


# =============================================================================
# Protocol Method Existence Tests
# =============================================================================


@pytest.mark.unit
class TestProtocolMethodExistence:
    """Tests verifying protocol methods exist and are callable."""

    def test_llm_port_has_required_methods(self) -> None:
        """Test that MockLLMPort has all required LLMPort methods."""
        mock = MockLLMPort()

        assert hasattr(mock, "ainvoke")
        assert callable(mock.ainvoke)

        assert hasattr(mock, "invoke")
        assert callable(mock.invoke)

        assert hasattr(mock, "with_structured_output")
        assert callable(mock.with_structured_output)

    def test_agent_port_has_required_methods(self) -> None:
        """Test that MockAgentPort has all required AgentPort methods."""
        mock = MockAgentPort()

        assert hasattr(mock, "run")
        assert callable(mock.run)

        assert hasattr(mock, "arun")
        assert callable(mock.arun)

    def test_parser_port_has_required_methods(self) -> None:
        """Test that MockParserPort has all required ParserPort methods."""
        mock = MockParserPort()

        assert hasattr(mock, "aparse_to_pydantic")
        assert callable(mock.aparse_to_pydantic)

        assert hasattr(mock, "parse_to_pydantic")
        assert callable(mock.parse_to_pydantic)
