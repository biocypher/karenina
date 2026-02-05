"""Tests for LangChainAgentAdapter.

Tests agent execution with mocked infrastructure.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from karenina.adapters.langchain import LangChainAgentAdapter
from karenina.ports import Message, Tool


class TestLangChainAgentAdapter:
    """Tests for LangChainAgentAdapter."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a mock ModelConfig."""
        from karenina.schemas.workflow.models import ModelConfig

        return ModelConfig(
            id="test-agent",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="langchain",
        )

    def test_agent_initialization(self, model_config: Any) -> None:
        """Test agent adapter initialization."""
        adapter = LangChainAgentAdapter(model_config)

        assert adapter._config == model_config
        assert adapter._converter is not None

    def test_convert_mcp_servers_to_urls_http(self, model_config: Any) -> None:
        """Test MCP server conversion for HTTP transport."""
        adapter = LangChainAgentAdapter(model_config)

        mcp_servers = {
            "filesystem": {
                "type": "http",
                "url": "https://mcp.example.com/fs",
            }
        }
        result = adapter._convert_mcp_servers_to_urls(mcp_servers)

        assert result == {"filesystem": "https://mcp.example.com/fs"}

    def test_convert_mcp_servers_to_urls_stdio_raises(self, model_config: Any) -> None:
        """Test that stdio transport raises AgentExecutionError."""
        from karenina.ports import AgentExecutionError

        adapter = LangChainAgentAdapter(model_config)

        mcp_servers = {
            "local": {
                "type": "stdio",
                "command": "node",
                "args": ["server.js"],
            }
        }

        with pytest.raises(AgentExecutionError) as exc_info:
            adapter._convert_mcp_servers_to_urls(mcp_servers)

        assert "stdio transport" in str(exc_info.value)

    def test_convert_mcp_servers_to_urls_none(self, model_config: Any) -> None:
        """Test MCP server conversion with None input."""
        adapter = LangChainAgentAdapter(model_config)
        result = adapter._convert_mcp_servers_to_urls(None)
        assert result is None

    def test_convert_tools_to_names(self, model_config: Any) -> None:
        """Test tool list to name list conversion."""
        adapter = LangChainAgentAdapter(model_config)

        tools = [
            Tool(
                name="search",
                description="Search the web",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
            Tool(
                name="calculator",
                description="Do math",
                input_schema={"type": "object", "properties": {"expr": {"type": "string"}}},
            ),
        ]
        result = adapter._convert_tools_to_names(tools)

        assert result == ["search", "calculator"]

    def test_convert_tools_to_names_none(self, model_config: Any) -> None:
        """Test tool conversion with None input."""
        adapter = LangChainAgentAdapter(model_config)
        result = adapter._convert_tools_to_names(None)
        assert result is None

    def test_extract_usage_from_ai_messages(self, model_config: Any) -> None:
        """Test usage extraction from AIMessage list."""
        from karenina.adapters.langchain.usage import extract_usage_cumulative

        msg1 = AIMessage(content="First response")
        msg1.response_metadata = {"usage": {"input_tokens": 100, "output_tokens": 50}}

        msg2 = AIMessage(content="Second response")
        msg2.response_metadata = {"usage": {"input_tokens": 80, "output_tokens": 40}}

        messages: list[Any] = [
            HumanMessage(content="User message"),
            msg1,
            msg2,
        ]

        usage = extract_usage_cumulative(messages, model_name=model_config.model_name)

        assert usage.input_tokens == 180  # 100 + 80
        assert usage.output_tokens == 90  # 50 + 40
        assert usage.total_tokens == 270

    def test_count_turns(self) -> None:
        """Test turn counting from messages."""
        from karenina.adapters.langchain.usage import count_agent_turns

        messages: list[Any] = [
            HumanMessage(content="Question 1"),
            AIMessage(content="Answer 1"),
            HumanMessage(content="Question 2"),
            AIMessage(content="Answer 2"),
            AIMessage(content="Answer 3"),
        ]

        turns = count_agent_turns(messages)
        assert turns == 3  # 3 AIMessages

    @pytest.mark.asyncio
    async def test_run_requires_mcp_servers(self, model_config: Any) -> None:
        """Test that agent run requires MCP servers."""
        from karenina.ports.errors import AgentExecutionError

        adapter = LangChainAgentAdapter(model_config)

        with pytest.raises(AgentExecutionError, match="AgentPort requires MCP servers"):
            await adapter.arun([Message.user("Test question")])

    @pytest.mark.asyncio
    async def test_run_basic(self, model_config: Any) -> None:
        """Test basic agent run with mocked infrastructure."""
        with (
            patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init_model,
            patch("karenina.adapters.langchain.middleware.build_agent_middleware") as mock_middleware,
            patch("karenina.adapters.langchain.mcp.acreate_mcp_client_and_tools") as mock_mcp,
            patch("langchain.agents.create_agent") as mock_create_agent,
            patch("langgraph.checkpoint.memory.InMemorySaver"),
            patch("karenina.adapters.langchain.trace.harmonize_agent_response") as mock_harmonize,
            patch("karenina.adapters.langchain.trace.extract_final_ai_message_from_response") as mock_extract,
        ):
            mock_base_model = MagicMock()
            mock_init_model.return_value = mock_base_model

            mock_middleware.return_value = []

            mock_mcp.return_value = (None, [MagicMock()])  # (client, tools)

            ai_msg = AIMessage(content="Final answer")
            ai_msg.response_metadata = {"usage": {"input_tokens": 50, "output_tokens": 25}}

            mock_response = {
                "messages": [
                    HumanMessage(content="Test question"),
                    ai_msg,
                ]
            }

            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value=mock_response)
            mock_create_agent.return_value = mock_agent

            mock_harmonize.return_value = "--- User Message ---\nTest question\n\n--- AI Message ---\nFinal answer"
            mock_extract.return_value = ("Final answer", None)

            adapter = LangChainAgentAdapter(model_config)
            result = await adapter.arun(
                [Message.user("Test question")],
                mcp_servers={"test": {"type": "http", "url": "http://localhost:8080"}},
            )

            assert result.final_response == "Final answer"
            assert result.raw_trace is not None
            assert result.turns == 1
            assert result.limit_reached is False
            assert result.usage.input_tokens == 50
            assert result.usage.output_tokens == 25
