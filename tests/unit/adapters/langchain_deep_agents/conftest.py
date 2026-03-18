"""Shared fixtures for LangChain Deep Agents adapter tests."""

from __future__ import annotations

import pytest

from karenina.schemas.config import ModelConfig


@pytest.fixture
def deep_agents_model_config() -> ModelConfig:
    """ModelConfig configured for langchain_deep_agents interface."""
    return ModelConfig(
        id="test-deep-agents",
        model_name="claude-sonnet-4-20250514",
        model_provider="anthropic",
        interface="langchain_deep_agents",
        temperature=0.0,
    )
