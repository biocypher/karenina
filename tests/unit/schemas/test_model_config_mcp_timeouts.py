"""ModelConfig MCP timeout fields."""

from __future__ import annotations

import pytest

from karenina.schemas.config.models import ModelConfig


@pytest.mark.unit
class TestModelConfigMcpTimeouts:
    def test_defaults_are_none(self) -> None:
        config = ModelConfig(id="m", model_provider="openai", model_name="gpt-test")
        assert config.mcp_http_timeout is None
        assert config.mcp_sse_read_timeout is None

    def test_accepts_float_values(self) -> None:
        config = ModelConfig(
            id="m",
            model_provider="openai",
            model_name="gpt-test",
            mcp_http_timeout=240.0,
            mcp_sse_read_timeout=600.0,
        )
        assert config.mcp_http_timeout == 240.0
        assert config.mcp_sse_read_timeout == 600.0
