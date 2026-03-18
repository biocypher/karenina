"""Tests for DeepAgentsParserAdapter."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from karenina.adapters.langchain_deep_agents.parser import DeepAgentsParserAdapter
from karenina.ports import Message
from karenina.ports.parser import ParsePortResult


class SimpleAnswer(BaseModel):
    value: str = Field(description="The answer")


@pytest.mark.unit
class TestDeepAgentsParserAdapter:
    @pytest.mark.asyncio
    async def test_aparse_to_pydantic_with_structured_output(self, deep_agents_model_config, monkeypatch):
        """Test parsing with structured output returning a dict (include_raw=True format)."""
        from langchain_core.messages import AIMessage

        # include_raw=True returns {"raw": AIMessage, "parsed": dict, "parsing_error": None}
        mock_structured_model = MagicMock()
        mock_structured_model.ainvoke = AsyncMock(
            return_value={
                "raw": AIMessage(content='{"value": "42"}'),
                "parsed": {"value": "42"},
                "parsing_error": None,
            }
        )

        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured_model)

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.parser.create_chat_model",
            lambda _config, **_kw: mock_model,
        )

        adapter = DeepAgentsParserAdapter(deep_agents_model_config)
        result = await adapter.aparse_to_pydantic(
            [Message.user("Parse this")],
            SimpleAnswer,
        )

        assert isinstance(result, ParsePortResult)
        assert isinstance(result.parsed, SimpleAnswer)
        assert result.parsed.value == "42"

    @pytest.mark.asyncio
    async def test_aparse_to_pydantic_with_pydantic_response(self, deep_agents_model_config, monkeypatch):
        """Test parsing when structured output returns a Pydantic model directly."""
        from langchain_core.messages import AIMessage

        answer = SimpleAnswer(value="hello")
        mock_structured_model = MagicMock()
        mock_structured_model.ainvoke = AsyncMock(
            return_value={
                "raw": AIMessage(content='{"value": "hello"}'),
                "parsed": answer,
                "parsing_error": None,
            }
        )

        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured_model)

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.parser.create_chat_model",
            lambda _config, **_kw: mock_model,
        )

        adapter = DeepAgentsParserAdapter(deep_agents_model_config)
        result = await adapter.aparse_to_pydantic(
            [Message.user("Parse this")],
            SimpleAnswer,
        )

        assert result.parsed.value == "hello"

    @pytest.mark.asyncio
    async def test_aparse_fallback_to_text_extraction(self, deep_agents_model_config, monkeypatch):
        """Test fallback to text JSON extraction when structured output fails."""
        from langchain_core.messages import AIMessage

        answer_json = json.dumps({"value": "fallback"})
        mock_text_response = AIMessage(content=answer_json)

        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(side_effect=Exception("Not supported"))
        mock_model.ainvoke = AsyncMock(return_value=mock_text_response)

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.parser.create_chat_model",
            lambda _config, **_kw: mock_model,
        )

        adapter = DeepAgentsParserAdapter(deep_agents_model_config)
        result = await adapter.aparse_to_pydantic(
            [Message.user("Parse this")],
            SimpleAnswer,
        )

        assert result.parsed.value == "fallback"

    def test_capabilities_supports_structured_output(self, deep_agents_model_config):
        """Capabilities should declare structured output support."""
        adapter = DeepAgentsParserAdapter(deep_agents_model_config)
        caps = adapter.capabilities
        assert caps.supports_structured_output is True
