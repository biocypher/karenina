"""Integration tests for LangChain adapters.

This module tests the LangChain adapter implementations:
- LangChainMessageConverter: Round-trip message conversion
- LangChainLLMAdapter: LLM invocation with mocked model
- LangChainParserAdapter: Structured output parsing
- LangChainAgentAdapter: Agent execution with mocked infrastructure

Tests use mocks to avoid live API calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel

from karenina.adapters.langchain import (
    LangChainAgentAdapter,
    LangChainLLMAdapter,
    LangChainMessageConverter,
    LangChainParserAdapter,
)
from karenina.ports import (
    Message,
    Role,
    ToolResultContent,
    ToolUseContent,
    UsageMetadata,
)

# =============================================================================
# LangChainMessageConverter Tests
# =============================================================================


class TestLangChainMessageConverter:
    """Tests for LangChainMessageConverter."""

    @pytest.fixture
    def converter(self) -> LangChainMessageConverter:
        """Create a converter instance."""
        return LangChainMessageConverter()

    # -------------------------------------------------------------------------
    # to_provider tests
    # -------------------------------------------------------------------------

    def test_to_provider_system_message(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of system message to LangChain format."""
        messages = [Message.system("You are a helpful assistant")]
        result = converter.to_provider(messages)

        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are a helpful assistant"

    def test_to_provider_user_message(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of user message to LangChain format."""
        messages = [Message.user("Hello, world!")]
        result = converter.to_provider(messages)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello, world!"

    def test_to_provider_assistant_message_text_only(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of assistant message with text only."""
        messages = [Message.assistant("I am a helpful assistant")]
        result = converter.to_provider(messages)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "I am a helpful assistant"
        assert result[0].tool_calls == []

    def test_to_provider_assistant_message_with_tool_calls(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of assistant message with tool calls."""
        tool_call = ToolUseContent(
            id="call_123",
            name="search",
            input={"query": "test"},
        )
        messages = [Message.assistant("Let me search for that", tool_calls=[tool_call])]
        result = converter.to_provider(messages)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "Let me search for that"
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0]["id"] == "call_123"
        assert result[0].tool_calls[0]["name"] == "search"
        assert result[0].tool_calls[0]["args"] == {"query": "test"}

    def test_to_provider_tool_result_message(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of tool result message to LangChain format."""
        messages = [Message.tool_result(tool_use_id="call_123", content="Search results here")]
        result = converter.to_provider(messages)

        assert len(result) == 1
        assert isinstance(result[0], ToolMessage)
        assert result[0].content == "Search results here"
        assert result[0].tool_call_id == "call_123"

    def test_to_provider_multiple_messages(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of a conversation with multiple message types."""
        messages = [
            Message.system("You are helpful"),
            Message.user("Hi"),
            Message.assistant("Hello!"),
        ]
        result = converter.to_provider(messages)

        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)

    def test_to_provider_empty_list(self, converter: LangChainMessageConverter) -> None:
        """Test conversion of empty message list."""
        result = converter.to_provider([])
        assert result == []

    # -------------------------------------------------------------------------
    # from_provider tests
    # -------------------------------------------------------------------------

    def test_from_provider_system_message(self, converter: LangChainMessageConverter) -> None:
        """Test conversion from LangChain SystemMessage."""
        lc_messages: list[BaseMessage] = [SystemMessage(content="You are helpful")]
        result = converter.from_provider(lc_messages)

        assert len(result) == 1
        assert result[0].role == Role.SYSTEM
        assert result[0].text == "You are helpful"

    def test_from_provider_human_message(self, converter: LangChainMessageConverter) -> None:
        """Test conversion from LangChain HumanMessage."""
        lc_messages: list[BaseMessage] = [HumanMessage(content="Hello")]
        result = converter.from_provider(lc_messages)

        assert len(result) == 1
        assert result[0].role == Role.USER
        assert result[0].text == "Hello"

    def test_from_provider_ai_message_text_only(self, converter: LangChainMessageConverter) -> None:
        """Test conversion from LangChain AIMessage with text only."""
        lc_messages: list[BaseMessage] = [AIMessage(content="I can help with that")]
        result = converter.from_provider(lc_messages)

        assert len(result) == 1
        assert result[0].role == Role.ASSISTANT
        assert result[0].text == "I can help with that"
        assert result[0].tool_calls == []

    def test_from_provider_ai_message_with_tool_calls_dict(self, converter: LangChainMessageConverter) -> None:
        """Test conversion from LangChain AIMessage with tool calls (dict format)."""
        lc_messages: list[BaseMessage] = [
            AIMessage(
                content="Searching...",
                tool_calls=[{"id": "call_456", "name": "search", "args": {"q": "test"}}],
            )
        ]
        result = converter.from_provider(lc_messages)

        assert len(result) == 1
        assert result[0].role == Role.ASSISTANT
        assert result[0].text == "Searching..."
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].id == "call_456"
        assert result[0].tool_calls[0].name == "search"
        assert result[0].tool_calls[0].input == {"q": "test"}

    def test_from_provider_tool_message(self, converter: LangChainMessageConverter) -> None:
        """Test conversion from LangChain ToolMessage."""
        lc_messages: list[BaseMessage] = [ToolMessage(content="Result data", tool_call_id="call_789")]
        result = converter.from_provider(lc_messages)

        assert len(result) == 1
        assert result[0].role == Role.TOOL
        assert len(result[0].content) == 1
        tool_result = result[0].content[0]
        assert isinstance(tool_result, ToolResultContent)
        assert tool_result.tool_use_id == "call_789"
        assert tool_result.content == "Result data"

    # -------------------------------------------------------------------------
    # Round-trip tests
    # -------------------------------------------------------------------------

    def test_roundtrip_system_message(self, converter: LangChainMessageConverter) -> None:
        """Test round-trip conversion preserves system message."""
        original = [Message.system("Test system prompt")]
        lc = converter.to_provider(original)
        roundtrip = converter.from_provider(lc)

        assert len(roundtrip) == 1
        assert roundtrip[0].role == original[0].role
        assert roundtrip[0].text == original[0].text

    def test_roundtrip_user_message(self, converter: LangChainMessageConverter) -> None:
        """Test round-trip conversion preserves user message."""
        original = [Message.user("User question")]
        lc = converter.to_provider(original)
        roundtrip = converter.from_provider(lc)

        assert len(roundtrip) == 1
        assert roundtrip[0].role == original[0].role
        assert roundtrip[0].text == original[0].text

    def test_roundtrip_assistant_with_tool_calls(self, converter: LangChainMessageConverter) -> None:
        """Test round-trip conversion preserves assistant message with tool calls."""
        tool_call = ToolUseContent(
            id="tc_001",
            name="calculator",
            input={"expression": "2+2"},
        )
        original = [Message.assistant("Computing...", tool_calls=[tool_call])]
        lc = converter.to_provider(original)
        roundtrip = converter.from_provider(lc)

        assert len(roundtrip) == 1
        assert roundtrip[0].role == Role.ASSISTANT
        assert roundtrip[0].text == "Computing..."
        assert len(roundtrip[0].tool_calls) == 1
        assert roundtrip[0].tool_calls[0].id == "tc_001"
        assert roundtrip[0].tool_calls[0].name == "calculator"
        assert roundtrip[0].tool_calls[0].input == {"expression": "2+2"}

    def test_roundtrip_full_conversation(self, converter: LangChainMessageConverter) -> None:
        """Test round-trip conversion of full conversation flow."""
        tool_call = ToolUseContent(id="tc_1", name="search", input={"q": "BCL2"})
        original = [
            Message.system("You are a biology expert"),
            Message.user("What is BCL2?"),
            Message.assistant("Let me search for that", tool_calls=[tool_call]),
            Message.tool_result(tool_use_id="tc_1", content="BCL2 is a gene..."),
            Message.assistant("BCL2 is a proto-oncogene..."),
        ]

        lc = converter.to_provider(original)
        roundtrip = converter.from_provider(lc)

        assert len(roundtrip) == 5
        assert roundtrip[0].role == Role.SYSTEM
        assert roundtrip[1].role == Role.USER
        assert roundtrip[2].role == Role.ASSISTANT
        assert roundtrip[3].role == Role.TOOL
        assert roundtrip[4].role == Role.ASSISTANT


# =============================================================================
# LangChainLLMAdapter Tests
# =============================================================================


class TestLangChainLLMAdapter:
    """Tests for LangChainLLMAdapter."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a mock ModelConfig."""
        from karenina.schemas.workflow.models import ModelConfig

        return ModelConfig(
            id="test-model",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="langchain",
        )

    def test_adapter_initialization(self, model_config: Any) -> None:
        """Test adapter can be initialized with ModelConfig."""
        # Patch at the source module since it's lazy imported inside the method
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            adapter = LangChainLLMAdapter(model_config)

            assert adapter._config == model_config
            assert adapter._converter is not None
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_ainvoke_basic(self, model_config: Any) -> None:
        """Test basic async invocation."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            # Set up mock model
            mock_response = MagicMock()
            mock_response.content = "Hello! I'm happy to help."
            mock_response.response_metadata = {"usage": {"input_tokens": 10, "output_tokens": 20}}

            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(return_value=mock_response)
            mock_init.return_value = mock_model

            adapter = LangChainLLMAdapter(model_config)
            result = await adapter.ainvoke([Message.user("Hello")])

            assert result.content == "Hello! I'm happy to help."
            assert result.usage.input_tokens == 10
            assert result.usage.output_tokens == 20
            assert result.raw == mock_response

    @pytest.mark.asyncio
    async def test_ainvoke_with_usage_metadata(self, model_config: Any) -> None:
        """Test usage metadata extraction from response."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_response = MagicMock()
            mock_response.content = "Response text"
            mock_response.response_metadata = {
                "token_usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 25,
                }
            }

            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(return_value=mock_response)
            mock_init.return_value = mock_model

            adapter = LangChainLLMAdapter(model_config)
            result = await adapter.ainvoke([Message.system("System"), Message.user("User")])

            assert result.usage.input_tokens == 100
            assert result.usage.output_tokens == 50
            assert result.usage.total_tokens == 150
            assert result.usage.cache_read_tokens == 25

    def test_with_structured_output(self, model_config: Any) -> None:
        """Test with_structured_output returns new adapter instance."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_model = MagicMock()
            mock_structured_model = MagicMock()
            mock_model.with_structured_output = MagicMock(return_value=mock_structured_model)
            mock_init.return_value = mock_model

            class TestSchema(BaseModel):
                value: str

            adapter = LangChainLLMAdapter(model_config)
            structured_adapter = adapter.with_structured_output(TestSchema)

            assert structured_adapter is not adapter
            assert structured_adapter._structured_schema == TestSchema
            assert structured_adapter._model == mock_structured_model


# =============================================================================
# LangChainParserAdapter Tests
# =============================================================================


class TestLangChainParserAdapter:
    """Tests for LangChainParserAdapter."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a mock ModelConfig."""
        from karenina.schemas.workflow.models import ModelConfig

        return ModelConfig(
            id="test-parser",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="langchain",
        )

    def test_parser_initialization(self, model_config: Any) -> None:
        """Test parser adapter can be initialized."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            parser = LangChainParserAdapter(model_config)

            assert parser._config == model_config
            assert parser._llm_adapter is not None

    def test_json_extraction_with_markdown_fences(self) -> None:
        """Test JSON extraction from markdown fences via shared utility."""
        from karenina.utils.json_extraction import extract_json_from_response

        # Test with json fence
        text = '```json\n{"value": "test"}\n```'
        result = extract_json_from_response(text)
        assert result == '{"value": "test"}'

        # Test without fence (direct JSON)
        text = '{"value": "test"}'
        result = extract_json_from_response(text)
        assert result == '{"value": "test"}'

    def test_json_extraction_from_mixed_text(self) -> None:
        """Test JSON extraction from mixed text via shared utility."""
        from karenina.utils.json_extraction import extract_json_from_response

        # Test embedded JSON
        text = 'Here is the result: {"gene": "BCL2", "score": 0.95} as requested.'
        result = extract_json_from_response(text)
        assert '{"gene": "BCL2", "score": 0.95}' in result

        # Test no JSON
        text = "No JSON here"
        with pytest.raises(ValueError, match="Could not extract JSON"):
            extract_json_from_response(text)

    def test_parse_response_content_direct_json(self, model_config: Any) -> None:
        """Test direct JSON parsing."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            parser = LangChainParserAdapter(model_config)

            class SimpleSchema(BaseModel):
                name: str
                value: int

            content = '{"name": "test", "value": 42}'
            result = parser._parse_response_content(content, SimpleSchema)

            assert result.name == "test"
            assert result.value == 42

    def test_parse_response_content_with_markdown_fences(self, model_config: Any) -> None:
        """Test parsing JSON wrapped in markdown fences."""
        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            parser = LangChainParserAdapter(model_config)

            class SimpleSchema(BaseModel):
                name: str

            content = '```json\n{"name": "test"}\n```'
            result = parser._parse_response_content(content, SimpleSchema)

            assert result.name == "test"


# =============================================================================
# LangChainAgentAdapter Tests
# =============================================================================


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
        from karenina.ports import Tool

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

        # Create mock AIMessages with usage metadata
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
            await adapter.run([Message.user("Test question")])

    @pytest.mark.asyncio
    async def test_run_basic(self, model_config: Any) -> None:
        """Test basic agent run with mocked infrastructure."""
        with (
            patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init_model,
            patch("karenina.adapters.langchain.middleware.build_agent_middleware") as mock_middleware,
            patch("karenina.infrastructure.llm.mcp_utils.create_mcp_client_and_tools") as mock_mcp,
            patch("langchain.agents.create_agent") as mock_create_agent,
            patch("langgraph.checkpoint.memory.InMemorySaver"),
            patch("karenina.infrastructure.llm.mcp_utils.harmonize_agent_response") as mock_harmonize,
            patch("karenina.infrastructure.llm.mcp_utils.extract_final_ai_message_from_response") as mock_extract,
        ):
            # Set up mock model
            mock_base_model = MagicMock()
            mock_init_model.return_value = mock_base_model

            # Set up mock middleware
            mock_middleware.return_value = []

            # Set up mock MCP tools
            mock_mcp.return_value = (None, [MagicMock()])  # (client, tools)

            # Set up mock agent
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
            result = await adapter.run(
                [Message.user("Test question")],
                mcp_servers={"test": {"type": "http", "url": "http://localhost:8080"}},
            )

            assert result.final_response == "Final answer"
            assert result.raw_trace is not None
            assert result.turns == 1
            assert result.limit_reached is False
            assert result.usage.input_tokens == 50
            assert result.usage.output_tokens == 25


# =============================================================================
# Usage Metadata Tests
# =============================================================================


class TestUsageMetadataExtraction:
    """Tests for usage metadata extraction patterns."""

    def test_usage_metadata_dataclass(self) -> None:
        """Test UsageMetadata creation and fields."""
        usage = UsageMetadata(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cache_read_tokens=25,
            cache_creation_tokens=10,
            model="claude-sonnet-4",
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cache_read_tokens == 25
        assert usage.cache_creation_tokens == 10
        assert usage.model == "claude-sonnet-4"

    def test_usage_metadata_optional_fields(self) -> None:
        """Test UsageMetadata with only required fields."""
        usage = UsageMetadata(
            input_tokens=50,
            output_tokens=25,
            total_tokens=75,
        )

        assert usage.input_tokens == 50
        assert usage.output_tokens == 25
        assert usage.total_tokens == 75
        assert usage.cache_read_tokens is None
        assert usage.cache_creation_tokens is None
        assert usage.model is None


# =============================================================================
# ChatOpenRouter and ChatOpenAIEndpoint Model Tests
# =============================================================================


class TestChatOpenRouter:
    """Tests for ChatOpenRouter custom model class."""

    def test_chat_openrouter_initialization(self) -> None:
        """Test ChatOpenRouter initialization."""
        from karenina.adapters.langchain.models import ChatOpenRouter

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            model = ChatOpenRouter(model="gpt-4", temperature=0.5)

            assert model.model_name == "gpt-4"
            assert model.temperature == 0.5

    def test_chat_openrouter_lc_secrets(self) -> None:
        """Test ChatOpenRouter lc_secrets property."""
        from karenina.adapters.langchain.models import ChatOpenRouter

        model = ChatOpenRouter(model="gpt-4", openai_api_key="test-key")
        secrets = model.lc_secrets
        assert secrets == {"openai_api_key": "OPENROUTER_API_KEY"}


class TestChatOpenAIEndpoint:
    """Tests for ChatOpenAIEndpoint custom model class."""

    def test_chat_openai_endpoint_requires_api_key(self) -> None:
        """Test that ChatOpenAIEndpoint requires an API key."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        with pytest.raises(ValueError, match="API key is required"):
            ChatOpenAIEndpoint(base_url="http://localhost:8000")

    def test_chat_openai_endpoint_requires_explicit_api_key(self) -> None:
        """Test that ChatOpenAIEndpoint does NOT read from environment."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        # Even with env var set, it should still fail if no explicit key
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}),
            pytest.raises(ValueError, match="API key is required"),
        ):
            ChatOpenAIEndpoint(base_url="http://localhost:8000")

    def test_chat_openai_endpoint_initialization_with_key(self) -> None:
        """Test ChatOpenAIEndpoint initialization with explicit API key."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        model = ChatOpenAIEndpoint(
            base_url="http://localhost:8000",
            openai_api_key="explicit-key",
        )

        assert model is not None

    def test_chat_openai_endpoint_lc_secrets_empty(self) -> None:
        """Test ChatOpenAIEndpoint lc_secrets returns empty dict."""
        from karenina.adapters.langchain.models import ChatOpenAIEndpoint

        model = ChatOpenAIEndpoint(
            base_url="http://localhost:8000",
            openai_api_key="test-key",
        )
        secrets = model.lc_secrets
        assert secrets == {}
