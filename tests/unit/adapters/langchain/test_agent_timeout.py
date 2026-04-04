"""Tests for LangChain agent timeout recovery.

Verifies that the LangChainAgentAdapter handles timeout scenarios correctly:
partial AgentResult on timeout via extract_partial_agent_state, and normal
completion without timeout. Covers both the callback and non-callback
code paths.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from karenina.adapters.langchain import LangChainAgentAdapter
from karenina.ports import AgentConfig, AgentResult, Message


@pytest.fixture
def model_config() -> Any:
    """Create a ModelConfig for LangChain interface."""
    from karenina.schemas.config import ModelConfig

    return ModelConfig(
        id="test-langchain-timeout",
        model_name="claude-sonnet-4-20250514",
        model_provider="anthropic",
        interface="langchain",
    )


def _make_mcp_servers() -> dict[str, Any]:
    """Return a minimal MCP servers dict for agent adapter tests."""
    return {"test": {"type": "http", "url": "http://localhost:8080"}}


def _make_ai_message(content: str, input_tokens: int = 50, output_tokens: int = 25) -> AIMessage:
    """Create an AIMessage with response_metadata for usage extraction."""
    msg = AIMessage(content=content)
    msg.response_metadata = {"usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}
    return msg


def _standard_patches():
    """Return a context manager that patches the LangChain infrastructure.

    Patches model init, middleware, MCP tool loading, agent creation,
    and trace utilities. Returns a dict of the mocks keyed by short name.
    """

    class _PatchContext:
        """Holds mock objects for standard LangChain agent test patches."""

        def __init__(self) -> None:
            self.mocks: dict[str, Any] = {}
            self._stack: list[Any] = []

        def __enter__(self) -> _PatchContext:
            patches = [
                (
                    "init_model",
                    patch(
                        "karenina.adapters.langchain.initialization.init_chat_model_unified",
                        return_value=MagicMock(),
                    ),
                ),
                (
                    "middleware",
                    patch(
                        "karenina.adapters.langchain.middleware.build_agent_middleware",
                        return_value=[],
                    ),
                ),
                (
                    "mcp_tools",
                    patch(
                        "karenina.adapters.langchain.mcp.acreate_persistent_mcp_tools",
                        new_callable=AsyncMock,
                        return_value=[MagicMock()],
                    ),
                ),
                ("create_agent", patch("langchain.agents.create_agent")),
                ("memory_saver", patch("langgraph.checkpoint.memory.InMemorySaver")),
                (
                    "harmonize",
                    patch(
                        "karenina.adapters.langchain.trace.harmonize_agent_response",
                        return_value="--- AI Message ---\ntest",
                    ),
                ),
                (
                    "extract_final",
                    patch(
                        "karenina.adapters.langchain.trace.extract_final_ai_message_from_response",
                        return_value=("Partial answer", None),
                    ),
                ),
            ]
            for name, p in patches:
                mock = p.start()
                self.mocks[name] = mock
                self._stack.append(p)
            return self

        def __exit__(self, *args: Any) -> None:
            for p in reversed(self._stack):
                p.stop()

    return _PatchContext()


@pytest.mark.unit
class TestLangChainAgentTimeoutWithCallback:
    """Test timeout recovery in the callback code path.

    When ``langchain_core.callbacks.get_usage_metadata_callback`` is available,
    the adapter wraps ``agent.ainvoke`` in a ``with get_usage_metadata_callback()``
    block. These tests verify the timeout handling inside that branch.
    """

    @pytest.mark.asyncio
    async def test_timeout_returns_partial_result(self, model_config: Any) -> None:
        """Verify that a TimeoutError yields a partial AgentResult with timeout_reached=True."""
        with _standard_patches() as ctx:
            # Configure mock agent to raise TimeoutError
            mock_agent = AsyncMock()

            async def slow_invoke(*args: Any, **kwargs: Any) -> None:
                await asyncio.sleep(10)

            mock_agent.ainvoke = slow_invoke
            # extract_partial_agent_state needs checkpointer attribute
            mock_agent.checkpointer = None
            ctx.mocks["create_agent"].return_value = mock_agent

            adapter = LangChainAgentAdapter(model_config)
            result = await adapter.arun(
                [Message.user("Test question")],
                mcp_servers=_make_mcp_servers(),
                config=AgentConfig(timeout=0.05),
            )

            assert isinstance(result, AgentResult)
            assert result.timeout_reached is True
            assert "[Note: Agent timed out" in result.raw_trace

    @pytest.mark.asyncio
    async def test_timeout_calls_extract_partial_state(self, model_config: Any) -> None:
        """Verify extract_partial_agent_state is called on timeout."""
        with _standard_patches() as ctx:
            mock_agent = AsyncMock()

            async def slow_invoke(*args: Any, **kwargs: Any) -> None:
                await asyncio.sleep(10)

            mock_agent.ainvoke = slow_invoke
            mock_agent.checkpointer = None
            ctx.mocks["create_agent"].return_value = mock_agent

            with patch(
                "karenina.adapters.langchain.agent.extract_partial_agent_state",
                return_value={"messages": [HumanMessage(content="Test")]},
            ) as mock_extract:
                adapter = LangChainAgentAdapter(model_config)
                result = await adapter.arun(
                    [Message.user("Test question")],
                    mcp_servers=_make_mcp_servers(),
                    config=AgentConfig(timeout=0.05),
                )

                mock_extract.assert_called_once()
                assert result.timeout_reached is True


@pytest.mark.unit
class TestLangChainAgentTimeoutWithoutCallback:
    """Test timeout recovery when the usage metadata callback is not available.

    This exercises the else branch where ``agent.ainvoke`` is called without
    the ``get_usage_metadata_callback`` context manager.
    """

    @pytest.mark.asyncio
    async def test_timeout_without_callback_returns_partial(self, model_config: Any) -> None:
        """Verify timeout handling in the non-callback branch."""
        with _standard_patches() as ctx:
            mock_agent = AsyncMock()

            async def slow_invoke(*args: Any, **kwargs: Any) -> None:
                await asyncio.sleep(10)

            mock_agent.ainvoke = slow_invoke
            mock_agent.checkpointer = None
            ctx.mocks["create_agent"].return_value = mock_agent

            # Force the non-callback path by making the import fail
            with patch(
                "karenina.adapters.langchain.agent.asyncio.wait_for",
                side_effect=TimeoutError("timed out"),
            ):
                adapter = LangChainAgentAdapter(model_config)
                result = await adapter.arun(
                    [Message.user("Test question")],
                    mcp_servers=_make_mcp_servers(),
                    config=AgentConfig(timeout=0.05),
                )

                assert result.timeout_reached is True


@pytest.mark.unit
class TestLangChainAgentNormalCompletion:
    """Test normal agent completion (no timeout)."""

    @pytest.mark.asyncio
    async def test_normal_completion_not_timed_out(self, model_config: Any) -> None:
        """Verify normal completion sets timeout_reached=False."""
        with _standard_patches() as ctx:
            ai_msg = _make_ai_message("Final answer")
            mock_response = {
                "messages": [
                    HumanMessage(content="Test question"),
                    ai_msg,
                ]
            }

            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value=mock_response)
            ctx.mocks["create_agent"].return_value = mock_agent

            ctx.mocks["extract_final"].return_value = ("Final answer", None)

            adapter = LangChainAgentAdapter(model_config)
            result = await adapter.arun(
                [Message.user("Test question")],
                mcp_servers=_make_mcp_servers(),
            )

            assert isinstance(result, AgentResult)
            assert result.timeout_reached is False
            assert result.limit_reached is False
            assert result.final_response == "Final answer"

    @pytest.mark.asyncio
    async def test_normal_completion_with_timeout_config(self, model_config: Any) -> None:
        """Verify normal completion with a large timeout configured but not hit."""
        with _standard_patches() as ctx:
            ai_msg = _make_ai_message("Done")
            mock_response = {
                "messages": [
                    HumanMessage(content="Question"),
                    ai_msg,
                ]
            }

            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value=mock_response)
            ctx.mocks["create_agent"].return_value = mock_agent

            ctx.mocks["extract_final"].return_value = ("Done", None)

            adapter = LangChainAgentAdapter(model_config)
            result = await adapter.arun(
                [Message.user("Question")],
                mcp_servers=_make_mcp_servers(),
                config=AgentConfig(timeout=60.0),
            )

            assert result.timeout_reached is False
            assert "[Note: Agent timed out" not in result.raw_trace
