"""Tests for DeepAgentsLLMAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from karenina.adapters.langchain_deep_agents.llm import DeepAgentsLLMAdapter
from karenina.ports import LLMResponse, Message


@pytest.mark.unit
class TestDeepAgentsLLMAdapter:
    @pytest.mark.asyncio
    async def test_ainvoke_returns_llm_response(self, deep_agents_model_config, monkeypatch):
        """Test that ainvoke produces a valid LLMResponse."""
        from langchain_core.messages import AIMessage

        mock_response = AIMessage(content="Response text.")
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.llm.create_chat_model",
            lambda _config, **_kw: mock_model,
        )

        adapter = DeepAgentsLLMAdapter(deep_agents_model_config)
        result = await adapter.ainvoke([Message.user("Hello")])

        assert isinstance(result, LLMResponse)
        assert "Response text" in result.content

    def test_capabilities_include_system_prompt(self, deep_agents_model_config):
        """Capabilities should declare system prompt and structured output support."""
        adapter = DeepAgentsLLMAdapter(deep_agents_model_config)
        caps = adapter.capabilities
        assert caps.supports_system_prompt is True
        assert caps.supports_structured_output is True

    def test_with_structured_output_returns_new_adapter(self, deep_agents_model_config):
        """with_structured_output should return a new adapter with the schema set."""
        from pydantic import BaseModel, Field

        class Answer(BaseModel):
            value: str = Field(description="The answer")

        adapter = DeepAgentsLLMAdapter(deep_agents_model_config)
        structured = adapter.with_structured_output(Answer)

        assert isinstance(structured, DeepAgentsLLMAdapter)
        assert structured._structured_schema == Answer
        assert adapter._structured_schema is None  # Original unchanged

    def test_with_structured_output_warns_on_max_retries(self, deep_agents_model_config, caplog):
        """with_structured_output should warn when max_retries is provided."""
        import logging

        from pydantic import BaseModel, Field

        class Answer(BaseModel):
            value: str = Field(description="The answer")

        adapter = DeepAgentsLLMAdapter(deep_agents_model_config)

        with caplog.at_level(logging.WARNING):
            adapter.with_structured_output(Answer, max_retries=5)

        assert "max_retries" in caplog.text
        assert "ignored" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_aclose_exists_and_is_noop(self, deep_agents_model_config):
        """aclose() should exist and complete without error."""
        adapter = DeepAgentsLLMAdapter(deep_agents_model_config)
        await adapter.aclose()  # Should not raise


@pytest.mark.unit
class TestDeepAgentsSDKRetrySuppression:
    """Tests for SDK max_retries=0 in deep agents adapter."""

    def test_create_chat_model_passes_max_retries_zero(self, deep_agents_model_config, monkeypatch):
        """create_chat_model passes max_retries=0 to suppress SDK-level retries."""
        captured_kwargs: dict = {}

        def _capture_init(*, model, model_provider, **kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.initialization.init_chat_model",
            _capture_init,
        )

        from karenina.adapters.langchain_deep_agents.initialization import create_chat_model

        create_chat_model(deep_agents_model_config)

        assert captured_kwargs.get("max_retries") == 0

    def test_create_chat_model_allows_explicit_max_retries_override(self, deep_agents_model_config, monkeypatch):
        """Agent loops can opt into SDK model-call retries without changing LLM/parser defaults."""
        captured_kwargs: dict = {}

        def _capture_init(*, model, model_provider, **kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.initialization.init_chat_model",
            _capture_init,
        )

        from karenina.adapters.langchain_deep_agents.initialization import create_chat_model

        create_chat_model(deep_agents_model_config, max_retries=3)

        assert captured_kwargs.get("max_retries") == 3
