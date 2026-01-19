"""Tests for extract_partial_agent_state function.

This function recovers partial agent state when LangGraph execution limits are hit.
It tries 4 methods in order:
1. Exception's state attribute
2. Checkpointer's get_state method
3. Exception's messages attribute
4. Fallback to input messages
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from karenina.adapters.langchain import extract_partial_agent_state


class TestExtractPartialAgentState:
    """Tests for extract_partial_agent_state function."""

    @pytest.fixture
    def input_messages(self) -> list[Any]:
        """Create sample input messages."""
        return [
            HumanMessage(content="What is BCL2?"),
        ]

    @pytest.fixture
    def accumulated_messages(self) -> list[Any]:
        """Create sample accumulated messages (partial trace)."""
        return [
            HumanMessage(content="What is BCL2?"),
            AIMessage(content="Let me search for that..."),
            ToolMessage(content="BCL2 is a gene...", tool_call_id="call_1"),
            AIMessage(content="BCL2 is a proto-oncogene that..."),
        ]

    # -------------------------------------------------------------------------
    # Method 1: Exception with state attribute
    # -------------------------------------------------------------------------

    def test_extracts_from_exception_state_dict(
        self, input_messages: list[Any], accumulated_messages: list[Any]
    ) -> None:
        """Test extraction when exception has state attribute with dict."""
        mock_agent = MagicMock()
        mock_agent.checkpointer = None

        exception = Exception("GraphRecursionError: limit reached")
        exception.state = {"messages": accumulated_messages}

        result = extract_partial_agent_state(mock_agent, input_messages, exception)

        assert result == {"messages": accumulated_messages}
        assert len(result["messages"]) == 4

    def test_extracts_from_exception_state_non_dict_falls_back(self, input_messages: list[Any]) -> None:
        """Test that non-dict exception.state falls back to input messages."""
        mock_agent = MagicMock()
        mock_agent.checkpointer = None

        exception = Exception("GraphRecursionError")
        exception.state = "not a dict"
        result = extract_partial_agent_state(mock_agent, input_messages, exception)

        # Should return input messages wrapped in dict since state isn't a dict
        assert result == {"messages": input_messages}

    # -------------------------------------------------------------------------
    # Method 2: Checkpointer's get_state method
    # -------------------------------------------------------------------------

    def test_extracts_from_checkpointer(self, input_messages: list[Any], accumulated_messages: list[Any]) -> None:
        """Test extraction from agent checkpointer."""
        mock_state = MagicMock()
        mock_state.values = {"messages": accumulated_messages}

        mock_agent = MagicMock()
        mock_agent.checkpointer = MagicMock()  # Not None
        mock_agent.get_state = MagicMock(return_value=mock_state)

        exception = Exception("GraphRecursionError")
        # No state attribute on exception

        config = {"configurable": {"thread_id": "test-thread"}}
        result = extract_partial_agent_state(mock_agent, input_messages, exception, config)

        assert result == {"messages": accumulated_messages}
        mock_agent.get_state.assert_called_once_with(config)

    def test_checkpointer_uses_default_config_when_none_provided(
        self, input_messages: list[Any], accumulated_messages: list[Any]
    ) -> None:
        """Test that default config is used when none provided."""
        mock_state = MagicMock()
        mock_state.values = {"messages": accumulated_messages}

        mock_agent = MagicMock()
        mock_agent.checkpointer = MagicMock()
        mock_agent.get_state = MagicMock(return_value=mock_state)

        exception = Exception("GraphRecursionError")

        result = extract_partial_agent_state(mock_agent, input_messages, exception)

        assert result == {"messages": accumulated_messages}
        mock_agent.get_state.assert_called_once_with({"configurable": {"thread_id": "default"}})

    def test_checkpointer_error_continues_to_next_method(
        self, input_messages: list[Any], accumulated_messages: list[Any]
    ) -> None:
        """Test that checkpointer errors are handled gracefully."""
        mock_agent = MagicMock()
        mock_agent.checkpointer = MagicMock()
        mock_agent.get_state = MagicMock(side_effect=RuntimeError("Checkpointer failed"))

        exception = Exception("GraphRecursionError")
        exception.messages = accumulated_messages
        result = extract_partial_agent_state(mock_agent, input_messages, exception)

        # Should fall through to Method 3 (exception.messages)
        assert result == {"messages": accumulated_messages}

    def test_checkpointer_returns_empty_state_continues(
        self, input_messages: list[Any], accumulated_messages: list[Any]
    ) -> None:
        """Test handling when checkpointer returns empty state."""
        mock_agent = MagicMock()
        mock_agent.checkpointer = MagicMock()
        mock_agent.get_state = MagicMock(return_value=None)

        exception = Exception("GraphRecursionError")
        exception.messages = accumulated_messages
        result = extract_partial_agent_state(mock_agent, input_messages, exception)

        # Should fall through to Method 3
        assert result == {"messages": accumulated_messages}

    # -------------------------------------------------------------------------
    # Method 3: Exception's messages attribute
    # -------------------------------------------------------------------------

    def test_extracts_from_exception_messages(self, input_messages: list[Any], accumulated_messages: list[Any]) -> None:
        """Test extraction from exception.messages attribute."""
        mock_agent = MagicMock()
        mock_agent.checkpointer = None

        exception = Exception("GraphRecursionError")
        exception.messages = accumulated_messages
        result = extract_partial_agent_state(mock_agent, input_messages, exception)

        assert result == {"messages": accumulated_messages}

    # -------------------------------------------------------------------------
    # Method 4: Fallback to input messages
    # -------------------------------------------------------------------------

    def test_fallback_to_input_messages(self, input_messages: list[Any]) -> None:
        """Test fallback when no recovery method works."""
        mock_agent = MagicMock()
        mock_agent.checkpointer = None

        exception = Exception("GraphRecursionError")
        # No state or messages attributes

        result = extract_partial_agent_state(mock_agent, input_messages, exception)

        assert result == {"messages": input_messages}

    def test_fallback_with_failed_checkpointer(self, input_messages: list[Any]) -> None:
        """Test fallback when checkpointer exists but fails."""
        mock_agent = MagicMock()
        mock_agent.checkpointer = MagicMock()
        mock_agent.get_state = MagicMock(side_effect=RuntimeError("Failed"))

        exception = Exception("GraphRecursionError")
        # No state or messages attributes

        result = extract_partial_agent_state(mock_agent, input_messages, exception)

        assert result == {"messages": input_messages}

    # -------------------------------------------------------------------------
    # Priority order tests
    # -------------------------------------------------------------------------

    def test_exception_state_takes_priority_over_checkpointer(
        self, input_messages: list[Any], accumulated_messages: list[Any]
    ) -> None:
        """Test that exception.state is checked before checkpointer."""
        different_messages = [AIMessage(content="Different")]

        mock_state = MagicMock()
        mock_state.values = {"messages": different_messages}

        mock_agent = MagicMock()
        mock_agent.checkpointer = MagicMock()
        mock_agent.get_state = MagicMock(return_value=mock_state)

        exception = Exception("GraphRecursionError")
        exception.state = {"messages": accumulated_messages}
        result = extract_partial_agent_state(mock_agent, input_messages, exception)

        # Should use exception.state, not checkpointer
        assert result == {"messages": accumulated_messages}
        mock_agent.get_state.assert_not_called()

    def test_checkpointer_takes_priority_over_exception_messages(
        self, input_messages: list[Any], accumulated_messages: list[Any]
    ) -> None:
        """Test that checkpointer is checked before exception.messages."""
        different_messages = [AIMessage(content="From checkpointer")]

        mock_state = MagicMock()
        mock_state.values = {"messages": different_messages}

        mock_agent = MagicMock()
        mock_agent.checkpointer = MagicMock()
        mock_agent.get_state = MagicMock(return_value=mock_state)

        exception = Exception("GraphRecursionError")
        exception.messages = accumulated_messages
        result = extract_partial_agent_state(mock_agent, input_messages, exception)

        # Should use checkpointer, not exception.messages
        assert result == {"messages": different_messages}
