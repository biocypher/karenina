"""Timeout passthrough in the canonical MCP connector."""

from __future__ import annotations

from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from karenina.utils.mcp.client import connect_mcp_session


def _mock_transport() -> tuple[MagicMock, MagicMock]:
    transport_cm = AsyncMock()
    transport_cm.__aenter__.return_value = (MagicMock(), MagicMock(), MagicMock())
    session_cm = AsyncMock()
    session = AsyncMock()
    session_cm.__aenter__.return_value = session
    return transport_cm, session_cm


@pytest.mark.unit
class TestConnectMcpSessionTimeouts:
    async def test_passes_timeouts_to_transport(self) -> None:
        transport_cm, session_cm = _mock_transport()
        with (
            patch("mcp.client.streamable_http.streamablehttp_client", return_value=transport_cm) as transport,
            patch("mcp.ClientSession", return_value=session_cm),
        ):
            async with AsyncExitStack() as stack:
                await connect_mcp_session(
                    stack,
                    {
                        "type": "http",
                        "url": "http://localhost:1/mcp",
                        "timeout": 240.0,
                        "sse_read_timeout": 600.0,
                    },
                )
        assert transport.call_args.kwargs["timeout"] == 240.0
        assert transport.call_args.kwargs["sse_read_timeout"] == 600.0

    async def test_omits_timeouts_when_unset(self) -> None:
        transport_cm, session_cm = _mock_transport()
        with (
            patch("mcp.client.streamable_http.streamablehttp_client", return_value=transport_cm) as transport,
            patch("mcp.ClientSession", return_value=session_cm),
        ):
            async with AsyncExitStack() as stack:
                await connect_mcp_session(stack, {"type": "http", "url": "http://localhost:1/mcp"})
        assert "timeout" not in transport.call_args.kwargs
        assert "sse_read_timeout" not in transport.call_args.kwargs


@pytest.mark.unit
class TestTimeoutPlumbing:
    def test_build_llm_kwargs_forwards_timeouts(self) -> None:
        from karenina.adapters.factory import build_llm_kwargs
        from karenina.schemas.config.models import ModelConfig

        config = ModelConfig(
            id="m",
            model_provider="openai",
            model_name="gpt-test",
            mcp_urls_dict={"otp": "http://localhost:1/mcp"},
            mcp_http_timeout=240.0,
            mcp_sse_read_timeout=600.0,
        )
        kwargs = build_llm_kwargs(config)
        assert kwargs["mcp_http_timeout"] == 240.0
        assert kwargs["mcp_sse_read_timeout"] == 600.0
