"""Tests for the codex error wrapping matrix."""

from __future__ import annotations

import pytest

from karenina.adapters.codex_sdk.errors import wrap_codex_error
from karenina.ports import (
    AdapterUnavailableError,
    AgentExecutionError,
    AgentResponseError,
    AgentTimeoutError,
)


def _named_exception(name: str, message: str = "boom") -> Exception:
    """Build an exception whose type name matches a codex SDK error class."""
    return type(name, (Exception,), {})(message)


class TestWrapCodexError:
    def test_missing_binary_maps_to_adapter_unavailable(self) -> None:
        error, limit = wrap_codex_error(FileNotFoundError("No such file or directory: 'codex'"))
        assert isinstance(error, AdapterUnavailableError)
        assert error.fallback_interface == "langchain"
        assert limit is False

    def test_timeout_maps_to_agent_timeout(self) -> None:
        error, limit = wrap_codex_error(TimeoutError("took too long"))
        assert isinstance(error, AgentTimeoutError)
        assert limit is False

    def test_asyncio_timeout_maps_to_agent_timeout(self) -> None:
        error, _ = wrap_codex_error(TimeoutError())
        assert isinstance(error, AgentTimeoutError)

    @pytest.mark.parametrize("name", ["ServerBusyError", "RetryLimitExceededError"])
    def test_overload_errors_are_execution_errors(self, name: str) -> None:
        error, limit = wrap_codex_error(_named_exception(name, "server overloaded"))
        assert isinstance(error, AgentExecutionError)
        assert not isinstance(error, AgentTimeoutError)
        assert "transient" in str(error)
        assert limit is False

    def test_transport_closed_is_execution_error(self) -> None:
        error, _ = wrap_codex_error(_named_exception("TransportClosedError"))
        assert isinstance(error, AgentExecutionError)

    @pytest.mark.parametrize(
        "name",
        [
            "JsonRpcError",
            "CodexRpcError",
            "ParseError",
            "InvalidRequestError",
            "MethodNotFoundError",
            "InvalidParamsError",
            "InternalRpcError",
        ],
    )
    def test_jsonrpc_errors_map_to_response_error(self, name: str) -> None:
        error, limit = wrap_codex_error(_named_exception(name, "rpc failure"))
        assert isinstance(error, AgentResponseError)
        assert name in str(error)
        assert limit is False

    @pytest.mark.parametrize(
        "message",
        [
            "hit the turn limit for this thread",
            "model context window exceeded",
            "recursion depth reached",
        ],
    )
    def test_limit_messages_set_limit_reached(self, message: str) -> None:
        error, limit = wrap_codex_error(RuntimeError(message))
        assert isinstance(error, AgentExecutionError)
        assert limit is True
        assert error.limit_reached is True

    def test_unknown_error_falls_back_to_execution_error(self) -> None:
        error, limit = wrap_codex_error(ValueError("something odd"))
        assert isinstance(error, AgentExecutionError)
        assert "ValueError" in str(error)
        assert limit is False
