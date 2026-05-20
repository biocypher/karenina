"""Tests for ModelRetryMiddleware on_failure mapping in build_agent_middleware.

karenina's ModelRetryConfig exposes ``on_failure: Literal["continue", "raise"]``
but LangChain's ModelRetryMiddleware checks ``on_failure == "error"`` and treats
anything else as ``"continue"``. ToolRetryMiddleware has back-compat mapping for
``"raise"`` upstream; ModelRetryMiddleware does not. Without translation in the
karenina adapter, ``on_failure="raise"`` silently falls through to continue,
producing the AIMessage placeholder ("Model call failed after N attempts ...")
instead of propagating the exception. See daily note
``2026-04-25-mcp-connection-error-misclassification.md``.
"""

from __future__ import annotations

from typing import Any

import pytest

from karenina.adapters.langchain.middleware import build_agent_middleware
from karenina.schemas.config import AgentMiddlewareConfig


def _build_config(on_failure: str) -> AgentMiddlewareConfig:
    cfg = AgentMiddlewareConfig()
    cfg.summarization.enabled = False
    cfg.model_retry.on_failure = on_failure
    return cfg


def _extract_model_retry(middleware: list[Any]) -> Any:
    from langchain.agents.middleware import ModelRetryMiddleware

    instances = [m for m in middleware if isinstance(m, ModelRetryMiddleware)]
    assert len(instances) == 1, f"expected 1 ModelRetryMiddleware, got {len(instances)}"
    return instances[0]


@pytest.mark.unit
class TestModelRetryOnFailureMapping:
    """build_agent_middleware must translate karenina's "raise" into LangChain's "error"."""

    def test_raise_is_translated_to_error(self) -> None:
        middleware = build_agent_middleware(config=_build_config("raise"))
        retry = _extract_model_retry(middleware)
        assert retry.on_failure == "error"

    def test_continue_is_passed_through(self) -> None:
        middleware = build_agent_middleware(config=_build_config("continue"))
        retry = _extract_model_retry(middleware)
        assert retry.on_failure == "continue"
