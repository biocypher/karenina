"""Tests for GenerateAnswerStage system prompt deduplication."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.benchmark.verification.stages.pipeline.generate_answer import GenerateAnswerStage
from karenina.ports.messages import Message, Role
from karenina.schemas.config import ModelConfig


def _make_context(system_prompt: str, conversation_history: list[Message] | None = None) -> VerificationContext:
    """Build a minimal VerificationContext with the given system_prompt."""
    model = ModelConfig(
        id="test",
        model_name="test",
        interface="openai_endpoint",
        endpoint_base_url="http://localhost:8000",
        endpoint_api_key="EMPTY",
        system_prompt=system_prompt,
    )
    ctx = VerificationContext(
        question_id="q1",
        template_id="t1",
        question_text="What is BCL2?",
        template_code="class Answer(BaseAnswer):\n    answer: str",
        answering_model=model,
        parsing_model=model,
    )
    if conversation_history is not None:
        ctx.set_artifact("conversation_history", conversation_history)
    return ctx


@pytest.mark.unit
class TestSystemPromptDedup:
    def test_prepends_system_when_no_history(self) -> None:
        """With empty history, system prompt should be prepended."""
        ctx = _make_context("Be helpful.", conversation_history=[])
        captured_messages = []

        class FakeLLM:
            capabilities = MagicMock(supports_streaming=False)

            def invoke(self, messages):
                captured_messages.extend(messages)
                resp = MagicMock()
                resp.content = "BCL2 is a protein."
                resp.is_partial = False
                resp.usage_unavailable = False
                resp.usage = None
                return resp

        with patch("karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm", return_value=FakeLLM()):
            stage = GenerateAnswerStage()
            stage.execute(ctx)

        assert any(m.role == Role.SYSTEM for m in captured_messages), "System prompt should be present"

    def test_skips_system_when_history_has_one(self) -> None:
        """With system message in history, should NOT prepend a new one."""
        history = [
            Message.system("Be helpful."),
            Message.user("Prior question"),
            Message.assistant("Prior answer"),
        ]
        ctx = _make_context("Be helpful.", conversation_history=history)
        captured_messages = []

        class FakeLLM:
            capabilities = MagicMock(supports_streaming=False)

            def invoke(self, messages):
                captured_messages.extend(messages)
                resp = MagicMock()
                resp.content = "BCL2 is a protein."
                resp.is_partial = False
                resp.usage_unavailable = False
                resp.usage = None
                return resp

        with patch("karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm", return_value=FakeLLM()):
            stage = GenerateAnswerStage()
            stage.execute(ctx)

        system_messages = [m for m in captured_messages if m.role == Role.SYSTEM]
        assert len(system_messages) == 1, "Should have exactly one system message (from history, not a new prepend)"
        assert system_messages[0].text == "Be helpful."
