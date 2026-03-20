"""Shared fixtures for adapter conformance tests.

Provides fixtures that create adapter instances for each registered interface.
Conformance tests use these to validate protocol compliance across all adapters.

Adapters that are unavailable (e.g., missing SDK) are skipped automatically.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from karenina.adapters.registry import AdapterRegistry
from karenina.schemas.config import ModelConfig


@pytest.fixture
def all_registered_interfaces() -> list[str]:
    """Return all registered interface names."""
    return sorted(AdapterRegistry.get_interfaces())


@pytest.fixture
def mock_model_config_for_interface():
    """Factory fixture: create a ModelConfig for a given interface."""

    def _factory(interface: str) -> ModelConfig:
        return ModelConfig(
            id=f"test-{interface}",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface=interface,
            temperature=0.0,
        )

    return _factory


# --- Deep Agents adapter fixtures (mocked) ---
# These fixtures require deepagents and langchain_core to be installed.
# Tests that use them will be skipped if the packages are missing.


def _skip_if_no_deep_agents():
    """Skip if deepagents or langchain_core are not installed."""
    pytest.importorskip("deepagents", reason="deepagents not installed")
    pytest.importorskip("langchain_core", reason="langchain-core not installed")


@pytest.fixture
def deep_agents_agent_adapter():
    """DeepAgentsAgentAdapter with mocked create_deep_agent."""
    _skip_if_no_deep_agents()
    from karenina.adapters.langchain_deep_agents.agent import DeepAgentsAgentAdapter

    config = ModelConfig(
        id="test-da",
        model_name="claude-sonnet-4-20250514",
        model_provider="anthropic",
        interface="langchain_deep_agents",
        temperature=0.0,
    )
    return DeepAgentsAgentAdapter(config)


@pytest.fixture
def deep_agents_llm_adapter():
    """DeepAgentsLLMAdapter instance."""
    _skip_if_no_deep_agents()
    from karenina.adapters.langchain_deep_agents.llm import DeepAgentsLLMAdapter

    config = ModelConfig(
        id="test-da",
        model_name="claude-sonnet-4-20250514",
        model_provider="anthropic",
        interface="langchain_deep_agents",
        temperature=0.0,
    )
    return DeepAgentsLLMAdapter(config)


@pytest.fixture
def deep_agents_parser_adapter():
    """DeepAgentsParserAdapter instance."""
    _skip_if_no_deep_agents()
    from karenina.adapters.langchain_deep_agents.parser import DeepAgentsParserAdapter

    config = ModelConfig(
        id="test-da",
        model_name="claude-sonnet-4-20250514",
        model_provider="anthropic",
        interface="langchain_deep_agents",
        temperature=0.0,
    )
    return DeepAgentsParserAdapter(config)


@pytest.fixture
def deep_agents_message_converter():
    """DeepAgentsMessageConverter instance."""
    _skip_if_no_deep_agents()
    from karenina.adapters.langchain_deep_agents.messages import DeepAgentsMessageConverter

    return DeepAgentsMessageConverter()


@pytest.fixture
def mock_deep_agents_agent_result(monkeypatch):
    """Monkeypatch create_deep_agent and create_chat_model for mocked arun()."""
    _skip_if_no_deep_agents()
    from langchain_core.messages import AIMessage

    mock_result = {
        "messages": [AIMessage(content="The answer is Paris.")],
        "is_last_step": False,
    }
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(return_value=mock_result)

    monkeypatch.setattr(
        "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
        lambda **_kwargs: mock_agent,
    )
    monkeypatch.setattr(
        "karenina.adapters.langchain_deep_agents.agent.create_chat_model",
        lambda _config, **_kw: MagicMock(),
    )
    return mock_result
