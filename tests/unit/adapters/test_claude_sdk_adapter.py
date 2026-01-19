"""Unit tests for Claude Agent SDK adapters.

This module tests the Claude Agent SDK adapter implementations:
- ClaudeSDKMessageConverter: Message conversion to/from SDK format
- convert_mcp_config: MCP configuration conversion
- extract_sdk_usage: Usage metadata extraction
- sdk_messages_to_raw_trace: Trace string formatting
- sdk_messages_to_trace_messages: Structured trace formatting
- wrap_sdk_error: SDK exception wrapping

Tests use mocks to avoid requiring the Claude SDK or live API calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from karenina.adapters.claude_agent_sdk import (
    ClaudeSDKMessageConverter,
    convert_mcp_config,
    extract_sdk_usage,
    wrap_sdk_error,
)
from karenina.adapters.claude_agent_sdk.mcp import (
    convert_and_validate_mcp_config,
    validate_mcp_config,
)
from karenina.ports import (
    Message,
    UsageMetadata,
)
from karenina.ports.errors import (
    AdapterUnavailableError,
    AgentExecutionError,
    AgentResponseError,
    AgentTimeoutError,
)

# =============================================================================
# ClaudeSDKMessageConverter Tests
# =============================================================================


class TestClaudeSDKMessageConverter:
    """Tests for ClaudeSDKMessageConverter."""

    @pytest.fixture
    def converter(self) -> ClaudeSDKMessageConverter:
        """Create a converter instance."""
        return ClaudeSDKMessageConverter()

    # -------------------------------------------------------------------------
    # to_prompt_string tests
    # -------------------------------------------------------------------------

    def test_to_prompt_string_single_user_message(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test converting a single user message to prompt string."""
        messages = [Message.user("What is BCL2?")]
        result = converter.to_prompt_string(messages)

        assert result == "What is BCL2?"

    def test_to_prompt_string_multiple_user_messages(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test converting multiple user messages to prompt string."""
        messages = [
            Message.user("First question"),
            Message.user("Second question"),
        ]
        result = converter.to_prompt_string(messages)

        assert result == "First question\n\nSecond question"

    def test_to_prompt_string_ignores_system_messages(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test that system messages are not included in prompt string."""
        messages = [
            Message.system("You are a helpful assistant"),
            Message.user("Hello"),
        ]
        result = converter.to_prompt_string(messages)

        assert result == "Hello"
        assert "helpful assistant" not in result

    def test_to_prompt_string_ignores_assistant_messages(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test that assistant messages are not included in prompt string."""
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there!"),
            Message.user("Follow-up question"),
        ]
        result = converter.to_prompt_string(messages)

        assert result == "Hello\n\nFollow-up question"
        assert "Hi there" not in result

    def test_to_prompt_string_empty_list(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test converting empty message list."""
        result = converter.to_prompt_string([])
        assert result == ""

    def test_to_prompt_string_no_user_messages(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test converting when there are no user messages."""
        messages = [
            Message.system("System prompt"),
            Message.assistant("Response"),
        ]
        result = converter.to_prompt_string(messages)

        assert result == ""

    # -------------------------------------------------------------------------
    # extract_system_prompt tests
    # -------------------------------------------------------------------------

    def test_extract_system_prompt_single(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test extracting a single system prompt."""
        messages = [Message.system("You are a biology expert")]
        result = converter.extract_system_prompt(messages)

        assert result == "You are a biology expert"

    def test_extract_system_prompt_multiple(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test extracting multiple system prompts."""
        messages = [
            Message.system("Be helpful"),
            Message.system("Be concise"),
        ]
        result = converter.extract_system_prompt(messages)

        assert result == "Be helpful\n\nBe concise"

    def test_extract_system_prompt_ignores_other_roles(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test that non-system messages are not extracted."""
        messages = [
            Message.system("System prompt"),
            Message.user("User message"),
            Message.assistant("Assistant response"),
        ]
        result = converter.extract_system_prompt(messages)

        assert result == "System prompt"
        assert "User message" not in result
        assert "Assistant response" not in result

    def test_extract_system_prompt_no_system_messages(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test extraction when no system messages present."""
        messages = [Message.user("Hello")]
        result = converter.extract_system_prompt(messages)

        assert result is None

    def test_extract_system_prompt_empty_list(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test extraction with empty message list."""
        result = converter.extract_system_prompt([])

        assert result is None

    # -------------------------------------------------------------------------
    # from_provider tests (with SDK not installed)
    # -------------------------------------------------------------------------

    def test_from_provider_empty_list(self, converter: ClaudeSDKMessageConverter) -> None:
        """Test converting empty SDK message list."""
        result = converter.from_provider([])

        # With empty input, should return empty list
        assert result == []


# =============================================================================
# MCP Config Conversion Tests
# =============================================================================


class TestConvertMcpConfig:
    """Tests for convert_mcp_config function."""

    def test_http_url_conversion(self) -> None:
        """Test HTTP URL -> type='http' (NOT 'sse')."""
        config = convert_mcp_config({"api": "https://mcp.example.com/mcp/"})

        assert "api" in config
        api_config = config["api"]
        assert api_config.get("type") == "http"
        assert api_config.get("url") == "https://mcp.example.com/mcp/"

    def test_https_url_conversion(self) -> None:
        """Test HTTPS URL conversion."""
        config = convert_mcp_config({"secure": "https://secure.example.com/api"})

        secure_config = config["secure"]
        assert secure_config.get("type") == "http"
        assert secure_config.get("url") == "https://secure.example.com/api"

    def test_http_not_sse(self) -> None:
        """CRITICAL: Verify streamable_http maps to type='http' NOT 'sse'."""
        config = convert_mcp_config({"biocontext": "https://mcp.biocontext.ai/mcp/"})

        # This is the critical assertion from the PRD
        biocontext_config = config["biocontext"]
        assert biocontext_config.get("type") == "http"
        assert biocontext_config.get("type") != "sse"

    def test_http_url_with_auth_headers(self) -> None:
        """Test HTTP URL with auth headers."""
        auth_headers = {"Authorization": "Bearer token123"}
        config = convert_mcp_config(
            {"api": "https://mcp.example.com/"},
            auth_headers=auth_headers,
        )

        api_config = config["api"]
        assert api_config.get("type") == "http"
        assert api_config.get("url") == "https://mcp.example.com/"
        assert api_config.get("headers") == {"Authorization": "Bearer token123"}

    def test_command_with_args(self) -> None:
        """Test command with arguments is properly split."""
        config = convert_mcp_config({"github": "npx -y @modelcontextprotocol/server-github"})

        github_config = config["github"]
        assert "command" in github_config
        assert github_config.get("command") == "npx"
        assert github_config.get("args") == ["-y", "@modelcontextprotocol/server-github"]

    def test_simple_command_path(self) -> None:
        """Test simple command/path without args."""
        config = convert_mcp_config({"local": "/usr/local/bin/mcp-server"})

        local_config = config["local"]
        assert local_config.get("command") == "/usr/local/bin/mcp-server"
        assert "args" not in local_config

    def test_mixed_config(self) -> None:
        """Test mixed config with HTTP and stdio servers."""
        config = convert_mcp_config(
            {
                "biocontext": "https://mcp.biocontext.ai/mcp/",
                "github": "npx -y @mcp/server-github",
                "local": "/path/to/server",
            }
        )

        # HTTP URL
        biocontext_config = config["biocontext"]
        assert biocontext_config.get("type") == "http"
        assert biocontext_config.get("url") == "https://mcp.biocontext.ai/mcp/"

        # Command with args
        github_config = config["github"]
        assert github_config.get("command") == "npx"
        assert github_config.get("args") == ["-y", "@mcp/server-github"]

        # Simple path
        local_config = config["local"]
        assert local_config.get("command") == "/path/to/server"


class TestValidateMcpConfig:
    """Tests for validate_mcp_config function."""

    def test_valid_http_config(self) -> None:
        """Test validation of valid HTTP config."""
        config: dict[str, Any] = {"api": {"type": "http", "url": "https://example.com/mcp/"}}
        errors = validate_mcp_config(config)

        assert errors == []

    def test_valid_stdio_config(self) -> None:
        """Test validation of valid stdio config."""
        config: dict[str, Any] = {"local": {"command": "/path/to/server", "args": ["-v"]}}
        errors = validate_mcp_config(config)

        assert errors == []

    def test_http_config_missing_url(self) -> None:
        """Test validation fails when HTTP config missing url."""
        config: dict[str, Any] = {"bad": {"type": "http"}}
        errors = validate_mcp_config(config)

        assert len(errors) == 1
        assert "missing 'url' field" in errors[0]

    def test_stdio_config_missing_command(self) -> None:
        """Test validation catches missing command field."""
        config: dict[str, Any] = {"bad": {"type": "stdio"}}  # This has type but not command
        errors = validate_mcp_config(config)

        # Should have error about unknown type (since it's not http/sse/sdk)
        assert len(errors) >= 1

    def test_invalid_url_scheme(self) -> None:
        """Test validation catches invalid URL scheme."""
        config: dict[str, Any] = {"bad": {"type": "http", "url": "ftp://example.com"}}
        errors = validate_mcp_config(config)

        assert len(errors) == 1
        assert "http://" in errors[0] or "https://" in errors[0]

    def test_unknown_type(self) -> None:
        """Test validation catches unknown type."""
        config: dict[str, Any] = {"bad": {"type": "unknown"}}
        errors = validate_mcp_config(config)

        assert len(errors) == 1
        assert "Unknown type" in errors[0]

    def test_invalid_headers_type(self) -> None:
        """Test validation catches invalid headers type."""
        config: dict[str, Any] = {"bad": {"type": "http", "url": "https://example.com", "headers": "not-a-dict"}}
        errors = validate_mcp_config(config)

        assert len(errors) == 1
        assert "headers" in errors[0]

    def test_invalid_args_type(self) -> None:
        """Test validation catches invalid args type."""
        config: dict[str, Any] = {"bad": {"command": "/path/to/server", "args": "not-a-list"}}
        errors = validate_mcp_config(config)

        assert len(errors) == 1
        assert "args" in errors[0]


class TestConvertAndValidateMcpConfig:
    """Tests for convert_and_validate_mcp_config function."""

    def test_valid_conversion(self) -> None:
        """Test valid conversion passes validation."""
        config = convert_and_validate_mcp_config({"api": "https://mcp.example.com/"})

        api_config = config["api"]
        assert api_config.get("type") == "http"
        assert api_config.get("url") == "https://mcp.example.com/"

    def test_conversion_always_produces_valid_config(self) -> None:
        """Test that convert_mcp_config always produces valid configs."""
        # All possible input types should produce valid SDK configs
        test_configs = {
            "http_url": "https://example.com/mcp/",
            "http_insecure": "http://localhost:8080/mcp/",
            "cmd_with_args": "npx -y @mcp/server",
            "simple_cmd": "/usr/bin/mcp-server",
        }

        # Should not raise any errors
        config = convert_and_validate_mcp_config(test_configs)

        # All should be present
        assert len(config) == 4


# =============================================================================
# Usage Extraction Tests
# =============================================================================


class TestExtractSdkUsage:
    """Tests for extract_sdk_usage function."""

    def test_basic_usage_extraction(self) -> None:
        """Test basic usage extraction from mock ResultMessage."""
        # Create mock ResultMessage
        mock_result = MagicMock()
        mock_result.usage = {
            "input_tokens": 100,
            "output_tokens": 50,
        }
        mock_result.total_cost_usd = 0.0015

        usage = extract_sdk_usage(mock_result)

        assert isinstance(usage, UsageMetadata)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cost_usd == 0.0015

    def test_usage_with_cache_tokens(self) -> None:
        """Test usage extraction with cache tokens."""
        mock_result = MagicMock()
        mock_result.usage = {
            "input_tokens": 200,
            "output_tokens": 100,
            "cache_read_input_tokens": 150,
            "cache_creation_input_tokens": 25,
        }
        mock_result.total_cost_usd = 0.003

        usage = extract_sdk_usage(mock_result)

        assert usage.input_tokens == 200
        assert usage.output_tokens == 100
        assert usage.total_tokens == 300
        assert usage.cache_read_tokens == 150
        assert usage.cache_creation_tokens == 25
        assert usage.cost_usd == 0.003

    def test_usage_with_model(self) -> None:
        """Test usage extraction with model parameter."""
        mock_result = MagicMock()
        mock_result.usage = {"input_tokens": 50, "output_tokens": 25}
        mock_result.total_cost_usd = None

        usage = extract_sdk_usage(mock_result, model="claude-sonnet-4-20250514")

        assert usage.model == "claude-sonnet-4-20250514"

    def test_usage_with_missing_data(self) -> None:
        """Test usage extraction handles missing data gracefully."""
        mock_result = MagicMock()
        mock_result.usage = None  # No usage data
        mock_result.total_cost_usd = None

        usage = extract_sdk_usage(mock_result)

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cost_usd is None
        assert usage.cache_read_tokens is None
        assert usage.cache_creation_tokens is None

    def test_usage_with_empty_usage_dict(self) -> None:
        """Test usage extraction with empty usage dict."""
        mock_result = MagicMock()
        mock_result.usage = {}
        mock_result.total_cost_usd = 0.001

        usage = extract_sdk_usage(mock_result)

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cost_usd == 0.001


# =============================================================================
# Error Wrapping Tests
# =============================================================================


class TestWrapSdkError:
    """Tests for wrap_sdk_error function."""

    def test_cli_not_found_error(self) -> None:
        """Test CLINotFoundError wrapping."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "CLINotFoundError"
        mock_error.cli_path = "/usr/local/bin/claude"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AdapterUnavailableError)
        assert "Claude Code CLI not found" in result.message
        assert result.fallback_interface == "langchain"

    def test_cli_connection_error(self) -> None:
        """Test CLIConnectionError wrapping."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "CLIConnectionError"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentExecutionError)
        assert "connect" in result.message.lower()

    def test_process_error_timeout_124(self) -> None:
        """Test ProcessError with exit code 124 (timeout command)."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "ProcessError"
        mock_error.exit_code = 124
        mock_error.stderr = "Timeout exceeded"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentTimeoutError)
        assert "timed out" in result.message.lower()

    def test_process_error_timeout_137(self) -> None:
        """Test ProcessError with exit code 137 (SIGKILL)."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "ProcessError"
        mock_error.exit_code = 137
        mock_error.stderr = "Killed"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentTimeoutError)

    def test_process_error_general(self) -> None:
        """Test ProcessError with other exit codes."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "ProcessError"
        mock_error.exit_code = 1
        mock_error.stderr = "Something went wrong"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentExecutionError)
        assert "exit code 1" in result.message

    def test_process_error_truncates_long_stderr(self) -> None:
        """Test ProcessError truncates very long stderr."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "ProcessError"
        mock_error.exit_code = 1
        mock_error.stderr = "x" * 1000  # Very long stderr

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentExecutionError)
        assert len(result.message) < 1000  # Should be truncated

    def test_cli_json_decode_error(self) -> None:
        """Test CLIJSONDecodeError wrapping."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "CLIJSONDecodeError"
        mock_error.line = "invalid json here"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentResponseError)
        assert "JSON" in result.message

    def test_cli_json_decode_error_long_line(self) -> None:
        """Test CLIJSONDecodeError truncates long line info."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "CLIJSONDecodeError"
        mock_error.line = "x" * 500  # Long line

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentResponseError)
        assert "..." in result.message  # Should be truncated

    def test_unknown_exception(self) -> None:
        """Test wrapping of unknown exception types."""
        mock_error = MagicMock()
        mock_error.__class__.__name__ = "SomeRandomError"

        result = wrap_sdk_error(mock_error)

        assert isinstance(result, AgentExecutionError)
        assert "SomeRandomError" in result.message

    def test_standard_exception(self) -> None:
        """Test wrapping of standard Python exceptions."""
        error = ValueError("Invalid value provided")

        result = wrap_sdk_error(error)

        assert isinstance(result, AgentExecutionError)
        assert "ValueError" in result.message


# =============================================================================
# Adapter Initialization Tests
# =============================================================================


class TestClaudeSDKAdapterInitialization:
    """Tests for Claude SDK adapter initialization."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a mock ModelConfig for claude_agent_sdk interface."""
        from karenina.schemas.workflow.models import ModelConfig

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


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Tests for protocol compliance of Claude SDK adapters."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a ModelConfig for claude_agent_sdk interface."""
        from karenina.schemas.workflow.models import ModelConfig

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


# =============================================================================
# Factory Integration Tests
# =============================================================================


class TestFactoryIntegration:
    """Tests for factory integration with Claude SDK adapters."""

    @pytest.fixture
    def claude_sdk_config(self) -> Any:
        """Create a ModelConfig for claude_agent_sdk interface."""
        from karenina.schemas.workflow.models import ModelConfig

        return ModelConfig(
            id="test-claude-sdk",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="claude_agent_sdk",
        )

    def test_factory_routes_llm_to_sdk_adapter(self, claude_sdk_config: Any) -> None:
        """Test factory routes claude_agent_sdk to ClaudeSDKLLMAdapter."""
        from unittest.mock import patch

        from karenina.adapters.factory import get_llm

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/claude"

            llm = get_llm(claude_sdk_config)

            assert llm is not None
            assert type(llm).__name__ == "ClaudeSDKLLMAdapter"

    def test_factory_routes_agent_to_sdk_adapter(self, claude_sdk_config: Any) -> None:
        """Test factory routes claude_agent_sdk to ClaudeSDKAgentAdapter."""
        from unittest.mock import patch

        from karenina.adapters.factory import get_agent

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/claude"

            agent = get_agent(claude_sdk_config)

            assert agent is not None
            assert type(agent).__name__ == "ClaudeSDKAgentAdapter"

    def test_factory_routes_parser_to_sdk_adapter(self, claude_sdk_config: Any) -> None:
        """Test factory routes claude_agent_sdk to ClaudeSDKParserAdapter."""
        from unittest.mock import patch

        from karenina.adapters.factory import get_parser

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/claude"

            parser = get_parser(claude_sdk_config)

            assert parser is not None
            assert type(parser).__name__ == "ClaudeSDKParserAdapter"
