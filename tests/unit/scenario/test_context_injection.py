"""Tests for conversation history injection into GenerateAnswerStage."""

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.ports.messages import Message
from karenina.schemas.config import ModelConfig


def _make_context(conversation_history=None, **overrides):
    """Create a minimal VerificationContext for testing."""
    defaults = {
        "question_id": "test_q",
        "template_id": "test_t",
        "question_text": "What is X?",
        "template_code": "class Answer: pass",
        "answering_model": ModelConfig(id="test-model", model_name="test-model", model_provider="anthropic"),
        "parsing_model": ModelConfig(id="test-model", model_name="test-model", model_provider="anthropic"),
    }
    defaults.update(overrides)
    ctx = VerificationContext(**defaults)
    if conversation_history is not None:
        ctx.set_artifact("conversation_history", conversation_history)
    return ctx


@pytest.mark.unit
class TestConversationHistoryInjection:
    def test_history_artifact_read(self):
        history = [
            Message.user("Prior question"),
            Message.assistant("Prior answer"),
        ]
        ctx = _make_context(conversation_history=history)
        retrieved = ctx.get_artifact("conversation_history")
        assert retrieved is not None
        assert len(retrieved) == 2

    def test_no_history_artifact_returns_none(self):
        ctx = _make_context()
        assert ctx.get_artifact("conversation_history") is None
