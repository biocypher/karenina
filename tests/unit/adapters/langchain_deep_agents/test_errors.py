"""Tests for Deep Agents error wrapping."""

from __future__ import annotations

import pytest

from karenina.adapters.langchain_deep_agents.errors import wrap_deep_agents_error
from karenina.ports.errors import AgentExecutionError, AgentResponseError


@pytest.mark.unit
class TestWrapDeepAgentsError:
    def test_parse_alone_does_not_trigger_response_error(self):
        """'parse' alone should NOT map to AgentResponseError (093 regression)."""
        error = RuntimeError("failed to parse MCP configuration")
        mapped, was_limit = wrap_deep_agents_error(error)
        assert not isinstance(mapped, AgentResponseError)
        assert isinstance(mapped, AgentExecutionError)

    def test_parse_output_triggers_response_error(self):
        """'parse' + 'output' should map to AgentResponseError."""
        error = RuntimeError("failed to parse output from agent")
        mapped, was_limit = wrap_deep_agents_error(error)
        assert isinstance(mapped, AgentResponseError)
        assert was_limit is False

    def test_output_format_triggers_response_error(self):
        """'output' + 'format' should map to AgentResponseError."""
        error = RuntimeError("output format error in response")
        mapped, was_limit = wrap_deep_agents_error(error)
        assert isinstance(mapped, AgentResponseError)
        assert was_limit is False

    def test_recursion_limit_detected(self):
        """Recursion limit errors should set limit_reached=True."""
        error = RuntimeError("Hit recursion limit")
        mapped, was_limit = wrap_deep_agents_error(error)
        assert isinstance(mapped, AgentExecutionError)
        assert was_limit is True
