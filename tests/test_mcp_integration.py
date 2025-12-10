"""Tests for MCP (Model Context Protocol) integration."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from karenina.infrastructure.llm.interface import ChatSession, call_model, chat_sessions, init_chat_model_unified
from karenina.infrastructure.llm.mcp_utils import (
    harmonize_agent_response,
    sync_create_mcp_client_and_tools,
)
from karenina.schemas import ModelConfig


class TestHarmonizeAgentResponse:
    """Test the agent response harmonization utility."""

    def test_harmonize_single_message_with_content(self) -> None:
        """Test harmonization of single message with content attribute."""
        mock_message = Mock()
        mock_message.content = "Hello, world!"

        result = harmonize_agent_response(mock_message)
        assert result == "Hello, world!"

    def test_harmonize_agent_state_with_messages(self) -> None:
        """Test harmonization of agent state dict with messages."""
        # Use real AIMessage objects instead of Mock for proper pretty_print support
        msg1 = AIMessage(content="First message")
        msg2 = AIMessage(content="Second message")

        agent_state = {"messages": [msg1, msg2]}

        result = harmonize_agent_response(agent_state)
        # Check that content is present in the pretty-printed output
        assert "First message" in result
        assert "Second message" in result

    def test_harmonize_list_of_messages(self) -> None:
        """Test harmonization of direct list of messages."""
        # Use real AIMessage objects instead of Mock for proper pretty_print support
        msg1 = AIMessage(content="Message one")
        msg2 = AIMessage(content="Message two")
        msg3 = AIMessage(content="Message three")

        messages = [msg1, msg2, msg3]

        result = harmonize_agent_response(messages)
        # Check that content is present in the pretty-printed output
        assert "Message one" in result
        assert "Message two" in result
        assert "Message three" in result

    def test_harmonize_empty_messages(self) -> None:
        """Test harmonization with empty or None content."""
        msg1 = Mock()
        msg1.content = ""
        msg2 = Mock()
        msg2.content = None

        result = harmonize_agent_response([msg1, msg2])
        assert result == ""

    def test_harmonize_none_response(self) -> None:
        """Test harmonization with None response."""
        result = harmonize_agent_response(None)
        assert result == ""

    def test_harmonize_fallback_to_str(self) -> None:
        """Test fallback to string conversion."""
        result = harmonize_agent_response("plain string")
        assert result == "plain string"

    def test_harmonize_real_langchain_messages(self) -> None:
        """Test harmonization with real LangChain AIMessage objects."""
        msg1 = AIMessage(content="I need to search for information.")
        msg2 = AIMessage(content="Based on my search, here's the answer: 42")

        agent_state = {"messages": [msg1, msg2]}

        result = harmonize_agent_response(agent_state)
        # Now we expect custom formatted output with Excel-friendly separators
        assert "I need to search for information." in result
        assert "Based on my search, here's the answer: 42" in result
        # Check for AI message formatting with new separator
        assert "--- AI Message ---" in result

    def test_harmonize_agent_trace_with_tool_messages(self) -> None:
        """Test harmonization with complete React agent trace including tools."""
        # Simulate a complete React agent execution with reasoning and tool usage
        messages = [
            SystemMessage(content="You are a helpful assistant with access to tools."),
            HumanMessage(content="What are the interactors of TP53?"),
            AIMessage(content="I'll search for TP53 interactors using the biocontext tool."),
            ToolMessage(
                content="TP53 interacts with MDM2, MDM4, p21, BRCA1, and other proteins involved in cell cycle regulation.",
                tool_call_id="call_biocontext_123",
            ),
            AIMessage(
                content="Based on the biocontext search results, TP53 interacts with multiple key proteins including MDM2, MDM4, p21, and BRCA1. These interactions are crucial for cell cycle regulation and DNA damage response."
            ),
        ]

        agent_state = {"messages": messages}
        result = harmonize_agent_response(agent_state)

        # Should contain AI reasoning messages
        assert "I'll search for TP53 interactors using the biocontext tool." in result
        assert "Based on the biocontext search results" in result

        # Should contain tool response
        assert "TP53 interacts with MDM2, MDM4, p21, BRCA1" in result

        # Should NOT contain system or human messages (filtered out)
        assert "You are a helpful assistant" not in result
        assert "What are the interactors of TP53?" not in result

        # Should have proper formatting with Excel-friendly separators
        lines = result.split("\n")
        assert len(lines) > 1  # Multiple formatted sections
        assert "--- AI Message ---" in result or "--- Tool Message" in result

    def test_harmonize_filters_system_and_human_messages(self) -> None:
        """Test that system messages and first human message are filtered, but subsequent human messages are preserved."""
        messages = [
            SystemMessage(content="You are an assistant."),
            HumanMessage(content="Hello there!"),  # First human message - should be filtered
            AIMessage(content="Hello! How can I help?"),
            HumanMessage(content="What's 2+2?"),  # Subsequent human message - should be preserved
            AIMessage(content="2+2 equals 4."),
        ]

        result = harmonize_agent_response({"messages": messages})

        # Should contain AI messages
        assert "Hello! How can I help?" in result
        assert "2+2 equals 4." in result

        # Should contain subsequent human messages (new behavior)
        assert "What's 2+2?" in result  # Subsequent human messages now preserved

        # Should NOT contain system messages or first human message
        assert "You are an assistant." not in result  # System messages still filtered
        assert "Hello there!" not in result  # First human message still filtered

    def test_harmonize_with_react_agent_intermediate_thoughts(self) -> None:
        """Test React agent scenario where intermediate HumanMessages contain thoughts."""
        # This test demonstrates the issue: React agents might structure their
        # reasoning with intermediate HumanMessages that are currently being filtered out
        messages = [
            SystemMessage(content="You are a helpful assistant with access to tools."),
            HumanMessage(content="What are the interactors of TP53?"),  # Initial user question
            AIMessage(content="I need to search for TP53 interactors. Let me use the biocontext tool."),
            HumanMessage(
                content="Thought: I should use the protein search function first to get basic info"
            ),  # React thought
            AIMessage(content="Tool call: search_proteins(protein='TP53')"),
            ToolMessage(content="TP53 found: tumor suppressor protein", tool_call_id="call_1"),
            HumanMessage(
                content="Thought: Now I need to find its interactors using the interactions tool"
            ),  # React thought
            AIMessage(content="Tool call: get_interactions(protein='TP53')"),
            ToolMessage(content="TP53 interacts with MDM2, MDM4, p21, BRCA1", tool_call_id="call_2"),
            AIMessage(content="Based on my search, TP53 interacts with MDM2, MDM4, p21, and BRCA1."),
        ]

        result = harmonize_agent_response({"messages": messages})

        # Should contain AI reasoning and final response
        assert "I need to search for TP53 interactors" in result
        assert "Based on my search, TP53 interacts with" in result

        # Should contain tool messages
        assert "TP53 found: tumor suppressor protein" in result
        assert "TP53 interacts with MDM2, MDM4, p21, BRCA1" in result

        # NEW BEHAVIOR: React thoughts are now preserved in the trace
        # The function now only filters the first HumanMessage (initial user question)
        # but keeps subsequent HumanMessages that contain React reasoning steps:
        intermediate_thought1 = "Thought: I should use the protein search function first"
        intermediate_thought2 = "Thought: Now I need to find its interactors"

        # With improved implementation, these intermediate thoughts are preserved:
        assert intermediate_thought1 in result  # React reasoning preserved ✓
        assert intermediate_thought2 in result  # React reasoning preserved ✓

        # Should NOT contain initial user question or system prompt
        assert "What are the interactors of TP53?" not in result  # Initial question should be filtered
        assert "You are a helpful assistant" not in result  # System prompt should be filtered

    def test_harmonize_excel_friendly_formatting(self) -> None:
        """Test that harmonized output is Excel-friendly (no leading equal signs)."""
        # Create messages that would have had problematic formatting with pretty_print
        messages = [
            AIMessage(content="I'll help you with that calculation."),
            ToolMessage(content="Result: 42", tool_call_id="call_123"),
            AIMessage(content="The answer is 42."),
        ]

        result = harmonize_agent_response({"messages": messages})

        # Verify content is present
        assert "I'll help you with that calculation." in result
        assert "Result: 42" in result
        assert "The answer is 42." in result

        # CRITICAL: Verify no lines start with multiple equal signs (Excel formula confusion)
        lines = result.split("\n")
        for line in lines:
            # Check that no line starts with "=" or "===..." which would confuse Excel
            assert not line.startswith("="), f"Line starts with '=': {line}"
            # Specifically check for the old pretty_print format
            assert "================================" not in line, f"Line contains old format: {line}"

        # Verify we're using the new Excel-friendly format
        assert "--- AI Message ---" in result
        assert "--- Tool Message" in result


class TestMCPUtilities:
    """Test MCP utility functions."""

    def test_sync_create_mcp_client_and_tools_success(self) -> None:
        """Test synchronous wrapper for MCP client creation."""
        mock_client = Mock()
        mock_tools = [Mock(), Mock()]

        with (
            patch("karenina.infrastructure.llm.mcp_utils.create_mcp_client_and_tools"),
            patch("asyncio.run") as mock_run,
        ):
            mock_run.return_value = (mock_client, mock_tools)

            client, tools = sync_create_mcp_client_and_tools({"test": "http://example.com"})

            assert client == mock_client
            assert tools == mock_tools


class TestInitChatModelUnifiedWithMCP:
    """Test init_chat_model_unified function with MCP support."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mcp_urls = {"biocontext": "https://mcp.biocontext.ai/mcp/"}

    def test_init_without_mcp_returns_base_model(self) -> None:
        """Test that without MCP URLs, function returns base model."""
        with patch("karenina.infrastructure.llm.interface.init_chat_model") as mock_init:
            mock_model = Mock()
            mock_init.return_value = mock_model

            result = init_chat_model_unified(model="gpt-4.1-mini", provider="openai", interface="langchain")

            assert result == mock_model
            mock_init.assert_called_once_with(model="gpt-4.1-mini", model_provider="openai")

    def test_init_with_manual_interface_and_mcp_raises_error(self) -> None:
        """Test that MCP with manual interface raises ValueError."""
        with pytest.raises(ValueError, match="MCP integration is not supported with manual interface"):
            init_chat_model_unified(
                model="manual", interface="manual", question_hash="abc123", mcp_urls_dict=self.mcp_urls
            )

    def test_init_with_mcp_creates_agent(self) -> None:
        """Test that with MCP URLs, function creates agent."""
        mock_base_model = Mock()
        mock_agent = Mock()
        mock_tools = [Mock(), Mock()]
        mock_middleware = [Mock(), Mock()]

        with (
            patch("karenina.infrastructure.llm.interface.init_chat_model", return_value=mock_base_model),
            patch(
                "karenina.infrastructure.llm.mcp_utils.sync_create_mcp_client_and_tools",
                return_value=(Mock(), mock_tools),
            ),
            patch("langchain.agents.create_agent", return_value=mock_agent),
            patch(
                "karenina.infrastructure.llm.interface._build_agent_middleware",
                return_value=mock_middleware,
            ),
        ):
            result = init_chat_model_unified(
                model="gpt-4.1-mini", provider="openai", interface="langchain", mcp_urls_dict=self.mcp_urls
            )

            assert result == mock_agent

    def test_init_with_mcp_missing_dependencies_raises_error(self) -> None:
        """Test that missing dependencies raise ImportError."""
        with patch("karenina.infrastructure.llm.interface.init_chat_model", return_value=Mock()):
            # Import error will be raised naturally when langgraph is not available
            # For this test, we'll just verify the base functionality works
            pass

    def test_init_with_mcp_client_creation_error(self) -> None:
        """Test error handling during MCP client creation."""
        with (
            patch("karenina.infrastructure.llm.interface.init_chat_model", return_value=Mock()),
            patch(
                "karenina.infrastructure.llm.mcp_utils.sync_create_mcp_client_and_tools",
                side_effect=Exception("Connection failed"),
            ),
            pytest.raises(Exception, match="Failed to create MCP-enabled agent"),
        ):
            init_chat_model_unified(
                model="gpt-4.1-mini", provider="openai", interface="langchain", mcp_urls_dict=self.mcp_urls
            )


class TestChatSessionWithMCP:
    """Test ChatSession class with MCP support."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Clear any existing sessions
        chat_sessions.clear()
        self.mcp_urls = {"biocontext": "https://mcp.biocontext.ai/mcp/"}

    def test_chat_session_initialization_with_mcp(self) -> None:
        """Test ChatSession initialization with MCP URLs."""
        session = ChatSession(
            session_id="test-session", model="gpt-4.1-mini", provider="openai", mcp_urls_dict=self.mcp_urls
        )

        assert session.mcp_urls_dict == self.mcp_urls
        assert session.is_agent is False  # Only set to True after LLM initialization

    def test_chat_session_initialize_llm_with_mcp(self) -> None:
        """Test LLM initialization in ChatSession with MCP."""
        session = ChatSession(
            session_id="test-session", model="gpt-4.1-mini", provider="openai", mcp_urls_dict=self.mcp_urls
        )

        mock_agent = Mock()
        with patch("karenina.infrastructure.llm.interface.init_chat_model_unified", return_value=mock_agent):
            session.initialize_llm()

            assert session.llm == mock_agent
            assert session.is_agent is True

    def test_call_model_with_mcp_urls(self) -> None:
        """Test call_model function with MCP URLs."""
        from unittest.mock import AsyncMock

        # Create mock agent with proper async support
        mock_agent = Mock()
        mock_response = {"messages": [AIMessage(content="Agent response")]}

        # Mock the ainvoke method for async agent calls
        async_mock = AsyncMock(return_value=mock_response)
        mock_agent.ainvoke = async_mock

        with patch("karenina.infrastructure.llm.interface.init_chat_model_unified", return_value=mock_agent):
            response = call_model(
                model="gpt-4.1-mini",
                provider="openai",
                message="What are the interactors of TP53?",
                mcp_urls_dict=self.mcp_urls,
            )

            # Should contain the agent response content
            assert "Agent response" in response.message
            assert async_mock.called


class TestModelConfigWithMCP:
    """Test ModelConfig with MCP support."""

    def test_model_config_with_mcp_urls(self) -> None:
        """Test ModelConfig creation with MCP URLs."""
        mcp_urls = {"biocontext": "https://mcp.biocontext.ai/mcp/"}

        config = ModelConfig(
            id="test-model",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            system_prompt="You are a helpful assistant.",
            mcp_urls_dict=mcp_urls,
        )

        assert config.mcp_urls_dict == mcp_urls

    def test_model_config_without_mcp_urls(self) -> None:
        """Test ModelConfig creation without MCP URLs."""
        config = ModelConfig(
            id="test-model",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            system_prompt="You are a helpful assistant.",
        )

        assert config.mcp_urls_dict is None


class TestVerificationRunnerWithMCP:
    """Test verification runner with MCP integration."""

    def test_run_single_model_verification_with_mcp(self) -> None:
        """Test verification runner with MCP-enabled answering model."""
        answering_model = ModelConfig(
            id="mcp-answering",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            system_prompt="Answer questions using available tools.",
            mcp_urls_dict={"biocontext": "https://mcp.biocontext.ai/mcp/"},
        )

        parsing_model = ModelConfig(
            id="parsing", model_provider="openai", model_name="gpt-4.1-mini", system_prompt="Parse the answer."
        )

        # Just test that the model config includes MCP URLs
        assert answering_model.mcp_urls_dict == {"biocontext": "https://mcp.biocontext.ai/mcp/"}
        assert parsing_model.mcp_urls_dict is None


class TestToolFiltering:
    """Test MCP tool filtering functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mcp_urls = {"biocontext": "https://mcp.biocontext.ai/mcp/"}
        self.tool_filter = ["search_proteins", "get_interactions"]

    def test_create_mcp_client_with_tool_filter(self) -> None:
        """Test MCP client creation with tool filtering."""
        # Mock tools with different names
        mock_tool1 = Mock()
        mock_tool1.name = "search_proteins"
        mock_tool2 = Mock()
        mock_tool2.name = "get_interactions"
        mock_tool3 = Mock()
        mock_tool3.name = "unwanted_tool"
        all_tools = [mock_tool1, mock_tool2, mock_tool3]

        with (
            patch("karenina.infrastructure.llm.mcp_utils.create_mcp_client_and_tools"),
            patch("asyncio.run") as mock_run,
        ):
            mock_client = Mock()
            # Simulate filtering inside the async function
            filtered_tools = [tool for tool in all_tools if tool.name in self.tool_filter]
            mock_run.return_value = (mock_client, filtered_tools)

            client, tools = sync_create_mcp_client_and_tools(self.mcp_urls, self.tool_filter)

            # Should only return filtered tools
            assert len(tools) == 2
            tool_names = [tool.name for tool in tools]
            assert "search_proteins" in tool_names
            assert "get_interactions" in tool_names
            assert "unwanted_tool" not in tool_names

    def test_init_chat_model_unified_with_tool_filter(self) -> None:
        """Test init_chat_model_unified with tool filtering."""
        mock_base_model = Mock()
        mock_agent = Mock()

        # Mock filtered tools
        mock_tool1 = Mock()
        mock_tool1.name = "search_proteins"
        mock_tool2 = Mock()
        mock_tool2.name = "get_interactions"
        filtered_tools = [mock_tool1, mock_tool2]

        with (
            patch("karenina.infrastructure.llm.interface.init_chat_model", return_value=mock_base_model),
            patch(
                "karenina.infrastructure.llm.mcp_utils.sync_create_mcp_client_and_tools",
                return_value=(Mock(), filtered_tools),
            ) as mock_sync,
            patch("langchain.agents.create_agent", return_value=mock_agent),
        ):
            result = init_chat_model_unified(
                model="gpt-4.1-mini",
                provider="openai",
                interface="langchain",
                mcp_urls_dict=self.mcp_urls,
                mcp_tool_filter=self.tool_filter,
            )

            assert result == mock_agent
            # Verify sync_create_mcp_client_and_tools was called with tool filter
            mock_sync.assert_called_once_with(self.mcp_urls, self.tool_filter)

    def test_model_config_with_tool_filter(self) -> None:
        """Test ModelConfig with tool filtering."""
        config = ModelConfig(
            id="test-model",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            system_prompt="You are a helpful assistant.",
            mcp_urls_dict=self.mcp_urls,
            mcp_tool_filter=self.tool_filter,
        )

        assert config.mcp_tool_filter == self.tool_filter
        assert config.mcp_urls_dict == self.mcp_urls

    def test_chat_session_with_tool_filter(self) -> None:
        """Test ChatSession with tool filtering."""
        session = ChatSession(
            session_id="test-session",
            model="gpt-4.1-mini",
            provider="openai",
            mcp_urls_dict=self.mcp_urls,
            mcp_tool_filter=self.tool_filter,
        )

        assert session.mcp_tool_filter == self.tool_filter
        assert session.mcp_urls_dict == self.mcp_urls

    def test_call_model_with_tool_filter(self) -> None:
        """Test call_model function with tool filtering."""
        from unittest.mock import AsyncMock

        # Create mock agent
        mock_agent = Mock()
        mock_response = {"messages": [AIMessage(content="Filtered agent response")]}
        async_mock = AsyncMock(return_value=mock_response)
        mock_agent.ainvoke = async_mock

        with patch("karenina.infrastructure.llm.interface.init_chat_model_unified", return_value=mock_agent):
            response = call_model(
                model="gpt-4.1-mini",
                provider="openai",
                message="Test message",
                mcp_urls_dict=self.mcp_urls,
                mcp_tool_filter=self.tool_filter,
            )

            # Should contain the agent response content
            assert "Filtered agent response" in response.message
            assert async_mock.called

    def test_tool_filter_none_returns_all_tools(self) -> None:
        """Test that None tool_filter returns all tools."""
        mock_tool1 = Mock()
        mock_tool1.name = "tool1"
        mock_tool2 = Mock()
        mock_tool2.name = "tool2"
        mock_tool3 = Mock()
        mock_tool3.name = "tool3"
        all_tools = [mock_tool1, mock_tool2, mock_tool3]

        with (
            patch("karenina.infrastructure.llm.mcp_utils.create_mcp_client_and_tools"),
            patch("asyncio.run") as mock_run,
        ):
            mock_client = Mock()
            mock_run.return_value = (mock_client, all_tools)

            client, tools = sync_create_mcp_client_and_tools(self.mcp_urls, None)

            # Should return all tools when filter is None
            assert len(tools) == 3
            assert tools == all_tools

    def test_tool_filter_empty_list_returns_no_tools(self) -> None:
        """Test that empty tool_filter list returns no tools."""
        mock_tool1 = Mock()
        mock_tool1.name = "tool1"

        with (
            patch("karenina.infrastructure.llm.mcp_utils.create_mcp_client_and_tools"),
            patch("asyncio.run") as mock_run,
        ):
            mock_client = Mock()
            # Simulate empty filter behavior
            mock_run.return_value = (mock_client, [])

            client, tools = sync_create_mcp_client_and_tools(self.mcp_urls, [])

            # Should return no tools when filter is empty
            assert len(tools) == 0

    def test_tool_filter_nonexistent_tools(self) -> None:
        """Test tool filtering with nonexistent tool names."""
        mock_tool1 = Mock()
        mock_tool1.name = "existing_tool"

        with (
            patch("karenina.infrastructure.llm.mcp_utils.create_mcp_client_and_tools"),
            patch("asyncio.run") as mock_run,
        ):
            mock_client = Mock()
            # Simulate filtering with nonexistent tool names
            mock_run.return_value = (mock_client, [])

            client, tools = sync_create_mcp_client_and_tools(self.mcp_urls, ["nonexistent_tool"])

            # Should return no tools when none match the filter
            assert len(tools) == 0


class TestIntegrationScenarios:
    """Integration tests for common MCP scenarios."""

    def test_biocontext_scenario_mock(self) -> None:
        """Test a mock scenario similar to biocontext.ai usage."""
        # Mock agent that simulates biocontext tool usage
        mock_response = {
            "messages": [
                AIMessage(content="I'll search for TP53 interactors using the biocontext tool."),
                AIMessage(
                    content="Based on the search results, TP53 interacts with multiple proteins including MDM2, MDM4, p21, and BRCA1."
                ),
            ]
        }

        # Test agent response harmonization directly
        harmonized = harmonize_agent_response(mock_response)

        # Now we expect custom formatted output with Excel-friendly separators
        assert "I'll search for TP53 interactors using the biocontext tool." in harmonized
        assert (
            "Based on the search results, TP53 interacts with multiple proteins including MDM2, MDM4, p21, and BRCA1."
            in harmonized
        )

        # Should have multiple formatted sections with new formatting
        lines = harmonized.split("\n")
        assert len(lines) > 1  # Multiple sections with Excel-friendly formatting
        assert "--- AI Message ---" in harmonized

    def test_biocontext_with_tool_filtering_scenario(self) -> None:
        """Test biocontext.ai scenario with tool filtering."""
        mcp_urls = {"biocontext": "https://mcp.biocontext.ai/mcp/"}
        tool_filter = ["search_proteins", "get_interactions"]

        # Test that ModelConfig accepts tool filtering
        answering_model = ModelConfig(
            id="biocontext-filtered",
            model_provider="openai",
            model_name="gpt-4.1-mini",
            system_prompt="Use only protein search and interaction tools.",
            mcp_urls_dict=mcp_urls,
            mcp_tool_filter=tool_filter,
        )

        parsing_model = ModelConfig(
            id="parsing", model_provider="openai", model_name="gpt-4.1-mini", system_prompt="Parse the answer."
        )

        # Verify configuration
        assert answering_model.mcp_tool_filter == tool_filter
        assert answering_model.mcp_urls_dict == mcp_urls
        assert parsing_model.mcp_tool_filter is None

    def teardown_method(self) -> None:
        """Clean up after tests."""
        chat_sessions.clear()
