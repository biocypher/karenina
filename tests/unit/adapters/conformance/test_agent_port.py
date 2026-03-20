"""Conformance tests for AgentPort implementations.

Validates that AgentPort adapters correctly satisfy the protocol,
produce valid AgentResult objects, and handle the aclose() lifecycle.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from karenina.ports import AgentConfig, AgentPort, AgentResult, Message


@pytest.mark.unit
class TestAgentPortConformance:
    """Verify AgentPort protocol compliance for Deep Agents adapter."""

    def test_isinstance_check(self, deep_agents_agent_adapter):
        """Adapter must pass runtime_checkable isinstance."""
        assert isinstance(deep_agents_agent_adapter, AgentPort)

    def test_arun_signature(self, deep_agents_agent_adapter):
        """arun() must accept messages, tools, mcp_servers, config."""
        sig = inspect.signature(deep_agents_agent_adapter.arun)
        params = list(sig.parameters.keys())
        assert "messages" in params
        assert "tools" in params
        assert "mcp_servers" in params
        assert "config" in params

    def test_run_signature(self, deep_agents_agent_adapter):
        """run() must have the same parameters as arun()."""
        sig = inspect.signature(deep_agents_agent_adapter.run)
        params = list(sig.parameters.keys())
        assert "messages" in params
        assert "tools" in params
        assert "mcp_servers" in params
        assert "config" in params

    def test_aclose_does_not_raise(self, deep_agents_agent_adapter):
        """aclose() must be callable without error."""
        asyncio.run(deep_agents_agent_adapter.aclose())

    @pytest.mark.asyncio
    async def test_arun_returns_agent_result(
        self,
        deep_agents_agent_adapter,
        mock_deep_agents_agent_result,  # noqa: ARG002
    ):
        """arun() must return a valid AgentResult with all required fields."""
        result = await deep_agents_agent_adapter.arun(
            messages=[Message.user("What is the capital of France?")],
            config=AgentConfig(max_turns=5),
        )

        assert isinstance(result, AgentResult)
        assert isinstance(result.final_response, str)
        assert len(result.final_response) > 0
        assert isinstance(result.raw_trace, str)
        assert isinstance(result.trace_messages, list)
        assert isinstance(result.turns, int)
        assert result.turns >= 0
        assert isinstance(result.limit_reached, bool)

    @pytest.mark.asyncio
    async def test_agent_result_usage_valid(
        self,
        deep_agents_agent_adapter,
        mock_deep_agents_agent_result,  # noqa: ARG002
    ):
        """AgentResult.usage must have non-negative token counts."""
        result = await deep_agents_agent_adapter.arun(
            messages=[Message.user("Hello")],
            config=AgentConfig(max_turns=5),
        )

        assert result.usage.input_tokens >= 0
        assert result.usage.output_tokens >= 0
        assert result.usage.total_tokens >= 0
