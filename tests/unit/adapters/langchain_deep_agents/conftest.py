"""Shared fixtures for LangChain Deep Agents adapter tests."""

from __future__ import annotations

import pytest

pytest.importorskip("deepagents", reason="deepagents not installed")
pytest.importorskip("langchain_core", reason="langchain-core not installed")

from karenina.schemas.config import ModelConfig


@pytest.fixture
def deep_agents_model_config() -> ModelConfig:
    """ModelConfig configured for langchain_deep_agents interface."""
    return ModelConfig(
        id="claude-sonnet-4-20250514",
        model_name="claude-sonnet-4-20250514",
        model_provider="anthropic",
        interface="langchain_deep_agents",
        temperature=0.0,
    )
