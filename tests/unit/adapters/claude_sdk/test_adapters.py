"""Tests for Claude SDK adapter initialization, protocol compliance, and factory integration.

Tests ClaudeSDKLLMAdapter, ClaudeSDKAgentAdapter, and ClaudeSDKParserAdapter.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


class TestClaudeSDKAdapterInitialization:
    """Tests for Claude SDK adapter initialization."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a mock ModelConfig for claude_agent_sdk interface."""
        from karenina.schemas.config import ModelConfig

        return ModelConfig(
            id="test-claude-sdk",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="claude_agent_sdk",
        )

    def test_llm_adapter_initialization(self, model_config: Any) -> None:
        """Test ClaudeSDKLLMAdapter can be initialized."""
        from karenina.adapters.claude_agent_sdk import ClaudeSDKLLMAdapter

        adapter = ClaudeSDKLLMAdapter(model_config)

        assert adapter._config == model_config
        assert adapter._converter is not None

    def test_agent_adapter_initialization(self, model_config: Any) -> None:
        """Test ClaudeSDKAgentAdapter can be initialized."""
        from karenina.adapters.claude_agent_sdk import ClaudeSDKAgentAdapter

        adapter = ClaudeSDKAgentAdapter(model_config)

        assert adapter._config == model_config
        assert adapter._converter is not None

    def test_parser_adapter_initialization(self, model_config: Any) -> None:
        """Test ClaudeSDKParserAdapter can be initialized."""
        from karenina.adapters.claude_agent_sdk import ClaudeSDKParserAdapter

        adapter = ClaudeSDKParserAdapter(model_config)

        assert adapter._config == model_config

    def test_llm_adapter_with_structured_output(self, model_config: Any) -> None:
        """Test LLM adapter with_structured_output returns new instance."""
        from pydantic import BaseModel

        from karenina.adapters.claude_agent_sdk import ClaudeSDKLLMAdapter

        class TestSchema(BaseModel):
            value: str

        adapter = ClaudeSDKLLMAdapter(model_config)
        structured_adapter = adapter.with_structured_output(TestSchema)

        assert structured_adapter is not adapter
        assert structured_adapter._structured_schema == TestSchema


class TestProtocolCompliance:
    """Tests for protocol compliance of Claude SDK adapters."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a ModelConfig for claude_agent_sdk interface."""
        from karenina.schemas.config import ModelConfig

        return ModelConfig(
            id="test-claude-sdk",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="claude_agent_sdk",
        )

    def test_llm_adapter_implements_llm_port(self, model_config: Any) -> None:
        """Test ClaudeSDKLLMAdapter implements LLMPort protocol."""
        from karenina.adapters.claude_agent_sdk import ClaudeSDKLLMAdapter
        from karenina.ports import LLMPort

        adapter = ClaudeSDKLLMAdapter(model_config)

        assert isinstance(adapter, LLMPort)

    def test_agent_adapter_implements_agent_port(self, model_config: Any) -> None:
        """Test ClaudeSDKAgentAdapter implements AgentPort protocol."""
        from karenina.adapters.claude_agent_sdk import ClaudeSDKAgentAdapter
        from karenina.ports import AgentPort

        adapter = ClaudeSDKAgentAdapter(model_config)

        assert isinstance(adapter, AgentPort)

    def test_parser_adapter_implements_parser_port(self, model_config: Any) -> None:
        """Test ClaudeSDKParserAdapter implements ParserPort protocol."""
        from karenina.adapters.claude_agent_sdk import ClaudeSDKParserAdapter
        from karenina.ports import ParserPort

        adapter = ClaudeSDKParserAdapter(model_config)

        assert isinstance(adapter, ParserPort)


class TestFactoryIntegration:
    """Tests for factory integration with Claude SDK adapters."""

    @pytest.fixture
    def claude_sdk_config(self) -> Any:
        """Create a ModelConfig for claude_agent_sdk interface."""
        from karenina.schemas.config import ModelConfig

        return ModelConfig(
            id="test-claude-sdk",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="claude_agent_sdk",
        )

    def test_factory_routes_llm_to_sdk_adapter(self, claude_sdk_config: Any) -> None:
        """Test factory routes claude_agent_sdk to ClaudeSDKLLMAdapter."""
        from karenina.adapters.factory import get_llm

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/claude"

            llm = get_llm(claude_sdk_config)

            assert llm is not None
            assert type(llm).__name__ == "ClaudeSDKLLMAdapter"

    def test_factory_routes_agent_to_sdk_adapter(self, claude_sdk_config: Any) -> None:
        """Test factory routes claude_agent_sdk to ClaudeSDKAgentAdapter."""
        from karenina.adapters.factory import get_agent

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/claude"

            agent = get_agent(claude_sdk_config)

            assert agent is not None
            assert type(agent).__name__ == "ClaudeSDKAgentAdapter"

    def test_factory_routes_parser_to_sdk_adapter(self, claude_sdk_config: Any) -> None:
        """Test factory routes claude_agent_sdk to ClaudeSDKParserAdapter."""
        from karenina.adapters.factory import get_parser

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/claude"

            parser = get_parser(claude_sdk_config)

            assert parser is not None
            assert type(parser).__name__ == "ClaudeSDKParserAdapter"
