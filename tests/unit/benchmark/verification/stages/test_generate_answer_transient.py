"""Tests for GenerateAnswerStage error category classification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.benchmark.verification.stages.pipeline.generate_answer import GenerateAnswerStage
from karenina.schemas.config import ModelConfig
from karenina.utils.errors import ErrorCategory


def _make_context() -> VerificationContext:
    model = ModelConfig(
        id="test",
        model_name="test-model",
        interface="openai_endpoint",
        endpoint_base_url="http://localhost:8000",
        endpoint_api_key="EMPTY",
    )
    ctx = VerificationContext(
        question_id="q1",
        question_text="What is 2+2?",
        raw_answer="4",
        template_code="class Answer: pass",
        answering_model=model,
        parsing_model=model,
        template_id="tpl1",
    )
    return ctx


@pytest.mark.unit
class TestGenerateAnswerErrorCategoryClassification:
    @patch("karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm")
    def test_adapter_init_connection_error_category(self, mock_get_llm: MagicMock) -> None:
        """When get_llm() raises ConnectionError, error_category should be CONNECTION."""
        mock_get_llm.side_effect = ConnectionError("connection refused")

        ctx = _make_context()
        stage = GenerateAnswerStage()
        stage.execute(ctx)

        assert ctx.completed_without_errors is False
        assert ctx.error_category == ErrorCategory.CONNECTION

    @patch("karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm")
    def test_adapter_call_connection_error_category(self, mock_get_llm: MagicMock) -> None:
        """When llm.stream_invoke() raises ConnectionError, error_category should be CONNECTION."""
        mock_llm = MagicMock()
        mock_llm.capabilities = MagicMock(supports_streaming=True)
        mock_llm.stream_invoke.side_effect = ConnectionError("connection reset")
        mock_get_llm.return_value = mock_llm

        ctx = _make_context()
        stage = GenerateAnswerStage()
        stage.execute(ctx)

        assert ctx.completed_without_errors is False
        assert ctx.error_category == ErrorCategory.CONNECTION

    @patch("karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm")
    def test_adapter_call_value_error_permanent_category(self, mock_get_llm: MagicMock) -> None:
        """When llm.stream_invoke() raises ValueError, error_category should be PERMANENT."""
        mock_llm = MagicMock()
        mock_llm.capabilities = MagicMock(supports_streaming=True)
        mock_llm.stream_invoke.side_effect = ValueError("invalid prompt")
        mock_get_llm.return_value = mock_llm

        ctx = _make_context()
        stage = GenerateAnswerStage()
        stage.execute(ctx)

        assert ctx.completed_without_errors is False
        assert ctx.error_category == ErrorCategory.PERMANENT
