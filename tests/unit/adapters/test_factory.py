"""Tests for adapter factory module.

This module tests the adapter factory functions:
- check_adapter_available: Verifies adapter infrastructure
- get_llm: Creates LLMPort implementations
- get_agent: Creates AgentPort implementations
- get_parser: Creates ParserPort implementations

Tests cover:
- Interface routing (langchain, openrouter, openai_endpoint, manual)
- Availability checking (including claude_agent_sdk for future support)
- Fallback behavior
- Error handling

Note: claude_agent_sdk interface is not yet supported in ModelConfig (planned for PR3),
so tests for that interface use mocked configs to test factory routing logic.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from karenina.adapters.factory import (
    INTERFACE_CLAUDE_AGENT_SDK,
    INTERFACE_MANUAL,
    LANGCHAIN_ROUTED_INTERFACES,
    AdapterAvailability,
    check_adapter_available,
    get_agent,
    get_llm,
    get_parser,
)
from karenina.ports import AdapterUnavailableError

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def langchain_model_config() -> Any:
    """Create a ModelConfig with langchain interface."""
    from karenina.schemas.workflow.models import ModelConfig

    return ModelConfig(
        id="test-langchain",
        model_name="claude-sonnet-4-20250514",
        model_provider="anthropic",
        interface="langchain",
    )


@pytest.fixture
def openrouter_model_config() -> Any:
    """Create a ModelConfig with openrouter interface."""
    from karenina.schemas.workflow.models import ModelConfig

    return ModelConfig(
        id="test-openrouter",
        model_name="anthropic/claude-3-sonnet",
        model_provider="openrouter",
        interface="openrouter",
    )


@pytest.fixture
def openai_endpoint_model_config() -> Any:
    """Create a ModelConfig with openai_endpoint interface."""
    from karenina.schemas.workflow.models import ModelConfig

    return ModelConfig(
        id="test-openai-endpoint",
        model_name="gpt-4",
        model_provider="openai",
        interface="openai_endpoint",
    )


@pytest.fixture
def claude_sdk_model_config() -> Any:
    """Create a mock ModelConfig for claude_agent_sdk interface.

    Note: claude_agent_sdk is not yet a valid interface in ModelConfig schema
    (planned for PR3), so we use a mock to test factory routing logic.
    """
    mock_config = MagicMock()
    mock_config.interface = "claude_agent_sdk"
    mock_config.id = "test-claude-sdk"
    mock_config.model_name = "claude-sonnet-4-20250514"
    mock_config.model_provider = "anthropic"
    return mock_config


@pytest.fixture
def manual_model_config() -> Any:
    """Create a mock ModelConfig for manual interface.

    Note: Real ModelConfig requires manual_traces for manual interface,
    so we use a mock to isolate factory tests from ModelConfig validation.
    """
    mock_config = MagicMock()
    mock_config.interface = "manual"
    mock_config.id = "test-manual"
    mock_config.model_name = "manual"
    mock_config.model_provider = "manual"
    return mock_config


@pytest.fixture
def unknown_interface_model_config() -> Any:
    """Create a mock ModelConfig with an unknown interface.

    Note: Real ModelConfig validates interface values, so we use a mock
    to test factory behavior with unknown interfaces.
    """
    mock_config = MagicMock()
    mock_config.interface = "nonexistent_interface"
    mock_config.id = "test-unknown"
    mock_config.model_name = "unknown-model"
    mock_config.model_provider = "unknown"
    return mock_config


# =============================================================================
# AdapterAvailability Tests
# =============================================================================


class TestAdapterAvailability:
    """Tests for AdapterAvailability dataclass."""

    def test_availability_available(self) -> None:
        """Test AdapterAvailability with available adapter."""
        availability = AdapterAvailability(
            available=True,
            reason="LangChain is installed",
        )

        assert availability.available is True
        assert availability.reason == "LangChain is installed"
        assert availability.fallback_interface is None

    def test_availability_unavailable_with_fallback(self) -> None:
        """Test AdapterAvailability with unavailable adapter and fallback."""
        availability = AdapterAvailability(
            available=False,
            reason="Claude CLI not found",
            fallback_interface="langchain",
        )

        assert availability.available is False
        assert availability.reason == "Claude CLI not found"
        assert availability.fallback_interface == "langchain"


# =============================================================================
# check_adapter_available Tests
# =============================================================================


class TestCheckAdapterAvailable:
    """Tests for check_adapter_available function."""

    def test_langchain_available(self) -> None:
        """Test check_adapter_available for langchain returns available=True."""
        # langchain_core is installed in the test environment
        result = check_adapter_available("langchain")

        assert result.available is True
        assert "LangChain" in result.reason
        assert result.fallback_interface is None

    def test_openrouter_available(self) -> None:
        """Test check_adapter_available for openrouter routes through LangChain."""
        result = check_adapter_available("openrouter")

        assert result.available is True
        assert "LangChain" in result.reason

    def test_openai_endpoint_available(self) -> None:
        """Test check_adapter_available for openai_endpoint routes through LangChain."""
        result = check_adapter_available("openai_endpoint")

        assert result.available is True
        assert "LangChain" in result.reason

    def test_langchain_unavailable_when_not_installed(self) -> None:
        """Test check_adapter_available returns unavailable when langchain not installed."""
        with (
            patch.dict("sys.modules", {"langchain_core": None}),
            patch("karenina.adapters.factory.check_adapter_available") as mock_check,
        ):
            # Force re-check by patching the import
            mock_check.return_value = AdapterAvailability(
                available=False,
                reason="LangChain packages not installed",
                fallback_interface=None,
            )
            result = mock_check("langchain")

            assert result.available is False
            assert "not installed" in result.reason

    def test_claude_agent_sdk_with_cli_installed(self) -> None:
        """Test check_adapter_available for claude_agent_sdk with CLI present."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/claude"
            result = check_adapter_available("claude_agent_sdk")

            assert result.available is True
            assert "Claude CLI found" in result.reason
            mock_which.assert_called_once_with("claude")

    def test_claude_agent_sdk_without_cli_installed(self) -> None:
        """Test check_adapter_available for claude_agent_sdk without CLI."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            result = check_adapter_available("claude_agent_sdk")

            assert result.available is False
            assert "not found" in result.reason
            assert result.fallback_interface == "langchain"

    def test_manual_always_available(self) -> None:
        """Test check_adapter_available for manual interface always returns True."""
        result = check_adapter_available("manual")

        assert result.available is True
        assert "pre-recorded" in result.reason

    def test_unknown_interface_unavailable(self) -> None:
        """Test check_adapter_available for unknown interface returns unavailable."""
        result = check_adapter_available("nonexistent_interface")

        assert result.available is False
        assert "Unknown interface" in result.reason
        assert result.fallback_interface == "langchain"


# =============================================================================
# get_llm Tests
# =============================================================================


class TestGetLLM:
    """Tests for get_llm factory function."""

    def test_get_llm_langchain(self, langchain_model_config: Any) -> None:
        """Test get_llm routes langchain to LangChainLLMAdapter."""
        with patch("karenina.infrastructure.llm.interface.init_chat_model_unified") as mock_init:
            mock_init.return_value = MagicMock()

            llm = get_llm(langchain_model_config)

            assert llm is not None
            assert type(llm).__name__ == "LangChainLLMAdapter"

    def test_get_llm_openrouter(self, openrouter_model_config: Any) -> None:
        """Test get_llm routes openrouter to LangChainLLMAdapter."""
        with patch("karenina.infrastructure.llm.interface.init_chat_model_unified") as mock_init:
            mock_init.return_value = MagicMock()

            llm = get_llm(openrouter_model_config)

            assert llm is not None
            assert type(llm).__name__ == "LangChainLLMAdapter"

    def test_get_llm_openai_endpoint(self, openai_endpoint_model_config: Any) -> None:
        """Test get_llm routes openai_endpoint to LangChainLLMAdapter."""
        with patch("karenina.infrastructure.llm.interface.init_chat_model_unified") as mock_init:
            mock_init.return_value = MagicMock()

            llm = get_llm(openai_endpoint_model_config)

            assert llm is not None
            assert type(llm).__name__ == "LangChainLLMAdapter"

    def test_get_llm_manual_returns_none(self, manual_model_config: Any) -> None:
        """Test get_llm returns None for manual interface."""
        llm = get_llm(manual_model_config)

        assert llm is None

    def test_get_llm_claude_sdk_fallback_to_langchain(self, claude_sdk_model_config: Any) -> None:
        """Test get_llm falls back to LangChain when Claude SDK unavailable."""
        with (
            patch("shutil.which") as mock_which,
            patch("karenina.infrastructure.llm.interface.init_chat_model_unified") as mock_init,
        ):
            mock_which.return_value = None  # Claude CLI not installed
            mock_init.return_value = MagicMock()

            # With auto_fallback=True (default), should fall back to LangChain
            llm = get_llm(claude_sdk_model_config, auto_fallback=True)

            assert llm is not None
            assert type(llm).__name__ == "LangChainLLMAdapter"

    def test_get_llm_claude_sdk_no_fallback_raises(self, claude_sdk_model_config: Any) -> None:
        """Test get_llm raises AdapterUnavailableError when auto_fallback=False."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None  # Claude CLI not installed

            with pytest.raises(AdapterUnavailableError) as exc_info:
                get_llm(claude_sdk_model_config, auto_fallback=False)

            assert "not available" in str(exc_info.value)
            assert exc_info.value.fallback_interface == "langchain"

    def test_get_llm_unknown_interface_fallback(self, unknown_interface_model_config: Any) -> None:
        """Test get_llm with unknown interface falls back to langchain."""
        with patch("karenina.infrastructure.llm.interface.init_chat_model_unified") as mock_init:
            mock_init.return_value = MagicMock()

            llm = get_llm(unknown_interface_model_config, auto_fallback=True)

            assert llm is not None
            assert type(llm).__name__ == "LangChainLLMAdapter"

    def test_get_llm_unknown_interface_no_fallback_raises(self, unknown_interface_model_config: Any) -> None:
        """Test get_llm with unknown interface raises when auto_fallback=False."""
        with pytest.raises(AdapterUnavailableError) as exc_info:
            get_llm(unknown_interface_model_config, auto_fallback=False)

        assert "Unknown interface" in exc_info.value.reason


# =============================================================================
# get_agent Tests
# =============================================================================


class TestGetAgent:
    """Tests for get_agent factory function."""

    def test_get_agent_langchain(self, langchain_model_config: Any) -> None:
        """Test get_agent routes langchain to LangChainAgentAdapter."""
        agent = get_agent(langchain_model_config)

        assert agent is not None
        assert type(agent).__name__ == "LangChainAgentAdapter"

    def test_get_agent_openrouter(self, openrouter_model_config: Any) -> None:
        """Test get_agent routes openrouter to LangChainAgentAdapter."""
        agent = get_agent(openrouter_model_config)

        assert agent is not None
        assert type(agent).__name__ == "LangChainAgentAdapter"

    def test_get_agent_openai_endpoint(self, openai_endpoint_model_config: Any) -> None:
        """Test get_agent routes openai_endpoint to LangChainAgentAdapter."""
        agent = get_agent(openai_endpoint_model_config)

        assert agent is not None
        assert type(agent).__name__ == "LangChainAgentAdapter"

    def test_get_agent_manual_returns_none(self, manual_model_config: Any) -> None:
        """Test get_agent returns None for manual interface."""
        agent = get_agent(manual_model_config)

        assert agent is None

    def test_get_agent_claude_sdk_fallback_to_langchain(self, claude_sdk_model_config: Any) -> None:
        """Test get_agent falls back to LangChain when Claude SDK unavailable."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None  # Claude CLI not installed

            agent = get_agent(claude_sdk_model_config, auto_fallback=True)

            assert agent is not None
            assert type(agent).__name__ == "LangChainAgentAdapter"

    def test_get_agent_claude_sdk_no_fallback_raises(self, claude_sdk_model_config: Any) -> None:
        """Test get_agent raises AdapterUnavailableError when auto_fallback=False."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None  # Claude CLI not installed

            with pytest.raises(AdapterUnavailableError) as exc_info:
                get_agent(claude_sdk_model_config, auto_fallback=False)

            assert "not available" in str(exc_info.value)


# =============================================================================
# get_parser Tests
# =============================================================================


class TestGetParser:
    """Tests for get_parser factory function."""

    def test_get_parser_langchain(self, langchain_model_config: Any) -> None:
        """Test get_parser routes langchain to LangChainParserAdapter."""
        with patch("karenina.infrastructure.llm.interface.init_chat_model_unified") as mock_init:
            mock_init.return_value = MagicMock()

            parser = get_parser(langchain_model_config)

            assert parser is not None
            assert type(parser).__name__ == "LangChainParserAdapter"

    def test_get_parser_openrouter(self, openrouter_model_config: Any) -> None:
        """Test get_parser routes openrouter to LangChainParserAdapter."""
        with patch("karenina.infrastructure.llm.interface.init_chat_model_unified") as mock_init:
            mock_init.return_value = MagicMock()

            parser = get_parser(openrouter_model_config)

            assert parser is not None
            assert type(parser).__name__ == "LangChainParserAdapter"

    def test_get_parser_openai_endpoint(self, openai_endpoint_model_config: Any) -> None:
        """Test get_parser routes openai_endpoint to LangChainParserAdapter."""
        with patch("karenina.infrastructure.llm.interface.init_chat_model_unified") as mock_init:
            mock_init.return_value = MagicMock()

            parser = get_parser(openai_endpoint_model_config)

            assert parser is not None
            assert type(parser).__name__ == "LangChainParserAdapter"

    def test_get_parser_manual_returns_none(self, manual_model_config: Any) -> None:
        """Test get_parser returns None for manual interface."""
        parser = get_parser(manual_model_config)

        assert parser is None

    def test_get_parser_claude_sdk_fallback_to_langchain(self, claude_sdk_model_config: Any) -> None:
        """Test get_parser falls back to LangChain when Claude SDK unavailable."""
        with (
            patch("shutil.which") as mock_which,
            patch("karenina.infrastructure.llm.interface.init_chat_model_unified") as mock_init,
        ):
            mock_which.return_value = None  # Claude CLI not installed
            mock_init.return_value = MagicMock()

            parser = get_parser(claude_sdk_model_config, auto_fallback=True)

            assert parser is not None
            assert type(parser).__name__ == "LangChainParserAdapter"

    def test_get_parser_claude_sdk_no_fallback_raises(self, claude_sdk_model_config: Any) -> None:
        """Test get_parser raises AdapterUnavailableError when auto_fallback=False."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None  # Claude CLI not installed

            with pytest.raises(AdapterUnavailableError) as exc_info:
                get_parser(claude_sdk_model_config, auto_fallback=False)

            assert "not available" in str(exc_info.value)


# =============================================================================
# Interface Constants Tests
# =============================================================================


class TestInterfaceConstants:
    """Tests for module-level interface constants."""

    def test_langchain_routed_interfaces(self) -> None:
        """Test LANGCHAIN_ROUTED_INTERFACES contains expected values."""
        assert "langchain" in LANGCHAIN_ROUTED_INTERFACES
        assert "openrouter" in LANGCHAIN_ROUTED_INTERFACES
        assert "openai_endpoint" in LANGCHAIN_ROUTED_INTERFACES
        assert len(LANGCHAIN_ROUTED_INTERFACES) == 3

    def test_interface_claude_agent_sdk_constant(self) -> None:
        """Test INTERFACE_CLAUDE_AGENT_SDK has expected value."""
        assert INTERFACE_CLAUDE_AGENT_SDK == "claude_agent_sdk"

    def test_interface_manual_constant(self) -> None:
        """Test INTERFACE_MANUAL has expected value."""
        assert INTERFACE_MANUAL == "manual"


# =============================================================================
# Error Type Tests
# =============================================================================


class TestAdapterUnavailableError:
    """Tests for AdapterUnavailableError."""

    def test_error_attributes(self) -> None:
        """Test AdapterUnavailableError has expected attributes."""
        error = AdapterUnavailableError(
            message="Test error message",
            reason="Test reason",
            fallback_interface="langchain",
        )

        assert str(error) == "Test error message"
        assert error.reason == "Test reason"
        assert error.fallback_interface == "langchain"

    def test_error_without_fallback(self) -> None:
        """Test AdapterUnavailableError without fallback interface."""
        error = AdapterUnavailableError(
            message="No fallback available",
            reason="Infrastructure missing",
        )

        assert str(error) == "No fallback available"
        assert error.reason == "Infrastructure missing"
        assert error.fallback_interface is None
