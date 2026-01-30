"""Tests for Claude Tool adapter initialization, protocol compliance, and factory integration.

Tests ClaudeToolLLMAdapter, ClaudeToolAgentAdapter, and ClaudeToolParserAdapter.
"""

from __future__ import annotations

from typing import Any

import pytest


class TestClaudeToolAdapterInitialization:
    """Tests for Claude Tool adapter initialization."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a mock ModelConfig for claude_tool interface."""
        from karenina.schemas.workflow.models import ModelConfig

        return ModelConfig(
            id="test-claude-tool",
            model_name="claude-haiku-4-5",
            model_provider="anthropic",
            interface="claude_tool",
        )

    def test_llm_adapter_initialization(self, model_config: Any) -> None:
        """Test ClaudeToolLLMAdapter can be initialized."""
        from karenina.adapters.claude_tool import ClaudeToolLLMAdapter

        adapter = ClaudeToolLLMAdapter(model_config)

        assert adapter._config == model_config
        assert adapter._structured_schema is None
        assert adapter._max_retries == 0

    def test_llm_adapter_stores_model_config(self, model_config: Any) -> None:
        """Test LLM adapter stores model configuration."""
        from karenina.adapters.claude_tool import ClaudeToolLLMAdapter

        adapter = ClaudeToolLLMAdapter(model_config)

        assert adapter._config.model_name == "claude-haiku-4-5"
        assert adapter._config.model_provider == "anthropic"
        assert adapter._config.interface == "claude_tool"

    def test_agent_adapter_initialization(self, model_config: Any) -> None:
        """Test ClaudeToolAgentAdapter can be initialized."""
        from karenina.adapters.claude_tool import ClaudeToolAgentAdapter

        adapter = ClaudeToolAgentAdapter(model_config)

        assert adapter._config == model_config
        assert adapter._async_client is None

    def test_parser_adapter_initialization(self, model_config: Any) -> None:
        """Test ClaudeToolParserAdapter can be initialized."""
        from karenina.adapters.claude_tool import ClaudeToolParserAdapter

        adapter = ClaudeToolParserAdapter(model_config)

        assert adapter._config == model_config
        assert adapter._llm_adapter is not None
        assert adapter._max_retries == 2

    def test_parser_adapter_custom_max_retries(self, model_config: Any) -> None:
        """Test parser adapter respects custom max_retries."""
        from karenina.adapters.claude_tool import ClaudeToolParserAdapter

        adapter = ClaudeToolParserAdapter(model_config, max_retries=5)

        assert adapter._max_retries == 5

    def test_llm_adapter_with_structured_output(self, model_config: Any) -> None:
        """Test LLM adapter with_structured_output returns new instance."""
        from pydantic import BaseModel

        from karenina.adapters.claude_tool import ClaudeToolLLMAdapter

        class TestSchema(BaseModel):
            value: str

        adapter = ClaudeToolLLMAdapter(model_config)
        structured_adapter = adapter.with_structured_output(TestSchema)

        assert structured_adapter is not adapter
        assert structured_adapter._structured_schema == TestSchema
        assert structured_adapter._max_retries == 3  # Default

    def test_llm_adapter_with_structured_output_custom_retries(self, model_config: Any) -> None:
        """Test with_structured_output respects max_retries parameter."""
        from pydantic import BaseModel

        from karenina.adapters.claude_tool import ClaudeToolLLMAdapter

        class TestSchema(BaseModel):
            value: str

        adapter = ClaudeToolLLMAdapter(model_config)
        structured_adapter = adapter.with_structured_output(TestSchema, max_retries=5)

        assert structured_adapter._max_retries == 5

    def test_llm_adapter_clients_lazy_initialized(self, model_config: Any) -> None:
        """Test LLM adapter clients are lazily initialized."""
        from karenina.adapters.claude_tool import ClaudeToolLLMAdapter

        adapter = ClaudeToolLLMAdapter(model_config)

        assert adapter._client is None
        assert adapter._async_client is None


class TestProtocolCompliance:
    """Tests for protocol compliance of Claude Tool adapters."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a ModelConfig for claude_tool interface."""
        from karenina.schemas.workflow.models import ModelConfig

        return ModelConfig(
            id="test-claude-tool",
            model_name="claude-haiku-4-5",
            model_provider="anthropic",
            interface="claude_tool",
        )

    def test_llm_adapter_implements_llm_port(self, model_config: Any) -> None:
        """Test ClaudeToolLLMAdapter implements LLMPort protocol."""
        from karenina.adapters.claude_tool import ClaudeToolLLMAdapter
        from karenina.ports import LLMPort

        adapter = ClaudeToolLLMAdapter(model_config)

        assert isinstance(adapter, LLMPort)

    def test_agent_adapter_implements_agent_port(self, model_config: Any) -> None:
        """Test ClaudeToolAgentAdapter implements AgentPort protocol."""
        from karenina.adapters.claude_tool import ClaudeToolAgentAdapter
        from karenina.ports import AgentPort

        adapter = ClaudeToolAgentAdapter(model_config)

        assert isinstance(adapter, AgentPort)

    def test_parser_adapter_implements_parser_port(self, model_config: Any) -> None:
        """Test ClaudeToolParserAdapter implements ParserPort protocol."""
        from karenina.adapters.claude_tool import ClaudeToolParserAdapter
        from karenina.ports import ParserPort

        adapter = ClaudeToolParserAdapter(model_config)

        assert isinstance(adapter, ParserPort)

    def test_llm_adapter_has_invoke_method(self, model_config: Any) -> None:
        """Test LLM adapter has invoke method."""
        from karenina.adapters.claude_tool import ClaudeToolLLMAdapter

        adapter = ClaudeToolLLMAdapter(model_config)

        assert hasattr(adapter, "invoke")
        assert callable(adapter.invoke)

    def test_llm_adapter_has_ainvoke_method(self, model_config: Any) -> None:
        """Test LLM adapter has ainvoke method."""
        from karenina.adapters.claude_tool import ClaudeToolLLMAdapter

        adapter = ClaudeToolLLMAdapter(model_config)

        assert hasattr(adapter, "ainvoke")
        assert callable(adapter.ainvoke)

    def test_agent_adapter_has_run_method(self, model_config: Any) -> None:
        """Test agent adapter has run method."""
        from karenina.adapters.claude_tool import ClaudeToolAgentAdapter

        adapter = ClaudeToolAgentAdapter(model_config)

        assert hasattr(adapter, "run")
        assert callable(adapter.run)

    def test_agent_adapter_has_run_sync_method(self, model_config: Any) -> None:
        """Test agent adapter has run_sync method."""
        from karenina.adapters.claude_tool import ClaudeToolAgentAdapter

        adapter = ClaudeToolAgentAdapter(model_config)

        assert hasattr(adapter, "run_sync")
        assert callable(adapter.run_sync)

    def test_parser_adapter_has_parse_to_pydantic_method(self, model_config: Any) -> None:
        """Test parser adapter has parse_to_pydantic method."""
        from karenina.adapters.claude_tool import ClaudeToolParserAdapter

        adapter = ClaudeToolParserAdapter(model_config)

        assert hasattr(adapter, "parse_to_pydantic")
        assert callable(adapter.parse_to_pydantic)

    def test_parser_adapter_has_aparse_to_pydantic_method(self, model_config: Any) -> None:
        """Test parser adapter has aparse_to_pydantic method."""
        from karenina.adapters.claude_tool import ClaudeToolParserAdapter

        adapter = ClaudeToolParserAdapter(model_config)

        assert hasattr(adapter, "aparse_to_pydantic")
        assert callable(adapter.aparse_to_pydantic)


class TestFactoryIntegration:
    """Tests for factory integration with Claude Tool adapters."""

    @pytest.fixture
    def claude_tool_config(self) -> Any:
        """Create a ModelConfig for claude_tool interface."""
        from karenina.schemas.workflow.models import ModelConfig

        return ModelConfig(
            id="test-claude-tool",
            model_name="claude-haiku-4-5",
            model_provider="anthropic",
            interface="claude_tool",
        )

    def test_factory_routes_llm_to_claude_tool_adapter(self, claude_tool_config: Any) -> None:
        """Test factory routes claude_tool to ClaudeToolLLMAdapter."""
        from karenina.adapters.factory import get_llm

        llm = get_llm(claude_tool_config)

        assert llm is not None
        assert type(llm).__name__ == "ClaudeToolLLMAdapter"

    def test_factory_routes_agent_to_claude_tool_adapter(self, claude_tool_config: Any) -> None:
        """Test factory routes claude_tool to ClaudeToolAgentAdapter."""
        from karenina.adapters.factory import get_agent

        agent = get_agent(claude_tool_config)

        assert agent is not None
        assert type(agent).__name__ == "ClaudeToolAgentAdapter"

    def test_factory_routes_parser_to_claude_tool_adapter(self, claude_tool_config: Any) -> None:
        """Test factory routes claude_tool to ClaudeToolParserAdapter."""
        from karenina.adapters.factory import get_parser

        parser = get_parser(claude_tool_config)

        assert parser is not None
        assert type(parser).__name__ == "ClaudeToolParserAdapter"


class TestRegistration:
    """Tests for adapter registration."""

    def test_claude_tool_registered_in_registry(self) -> None:
        """Test claude_tool is registered in AdapterRegistry."""
        # Ensure registration module is imported
        import karenina.adapters.claude_tool.registration  # noqa: F401
        from karenina.adapters.registry import AdapterRegistry

        assert "claude_tool" in AdapterRegistry.get_interfaces()

    def test_claude_tool_spec_has_correct_properties(self) -> None:
        """Test registered spec has correct properties."""
        # Ensure registration module is imported
        import karenina.adapters.claude_tool.registration  # noqa: F401
        from karenina.adapters.registry import AdapterRegistry

        spec = AdapterRegistry.get_spec("claude_tool")

        assert spec is not None
        assert spec.interface == "claude_tool"
        assert spec.supports_mcp is True
        assert spec.supports_tools is True
        assert spec.fallback_interface == "langchain"

    def test_availability_checker_returns_availability(self) -> None:
        """Test availability checker returns AdapterAvailability."""
        from karenina.adapters.claude_tool.registration import _check_availability

        result = _check_availability()

        # Should always have these attributes
        assert hasattr(result, "available")
        assert hasattr(result, "reason")
        assert isinstance(result.available, bool)
        assert isinstance(result.reason, str)

    def test_availability_checker_with_anthropic_installed(self) -> None:
        """Test availability checker when anthropic is installed."""
        from karenina.adapters.claude_tool.registration import _check_availability

        # anthropic should be installed in test environment
        result = _check_availability()

        assert result.available is True
        assert "anthropic" in result.reason.lower()
