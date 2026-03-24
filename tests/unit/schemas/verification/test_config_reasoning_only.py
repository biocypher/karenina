"""Tests for deep_judgment_mode field in VerificationConfig and VerificationContext."""

import pytest

from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification.config import VerificationConfig


def _parsing_model() -> ModelConfig:
    return ModelConfig(
        id="test-parser",
        model_name="claude-3-5-sonnet-20241022",
        model_provider="anthropic",
    )


@pytest.mark.unit
class TestDeepJudgmentModeConfig:
    """Tests for deep_judgment_mode field on VerificationConfig."""

    def test_default_is_disabled(self):
        """deep_judgment_mode defaults to 'disabled'."""
        config = VerificationConfig(
            parsing_models=[_parsing_model()],
            parsing_only=True,
        )
        assert config.deep_judgment_mode == "disabled"

    def test_can_be_set_to_reasoning_only(self):
        """deep_judgment_mode can be set to 'reasoning_only'."""
        config = VerificationConfig(
            parsing_models=[_parsing_model()],
            parsing_only=True,
            deep_judgment_mode="reasoning_only",
        )
        assert config.deep_judgment_mode == "reasoning_only"

    def test_can_be_set_to_full(self):
        """deep_judgment_mode='full' is the standard deep judgment path."""
        config = VerificationConfig(
            parsing_models=[_parsing_model()],
            parsing_only=True,
            deep_judgment_mode="full",
        )
        assert config.deep_judgment_mode == "full"


@pytest.mark.unit
class TestVerificationContextDeepJudgmentMode:
    """Tests for deep_judgment_mode mirror field on VerificationContext."""

    def test_context_default_is_disabled(self):
        """VerificationContext.deep_judgment_mode defaults to 'disabled'."""
        from karenina.benchmark.verification.stages.core.base import VerificationContext

        context = VerificationContext(
            question_id="q1",
            template_id="t1",
            question_text="What is the answer?",
            template_code="class Answer(BaseAnswer): pass",
            answering_model=ModelConfig(id="a", model_name="claude-3-5-sonnet-20241022", model_provider="anthropic"),
            parsing_model=_parsing_model(),
        )
        assert context.deep_judgment_mode == "disabled"

    def test_context_can_be_set_to_reasoning_only(self):
        """VerificationContext.deep_judgment_mode can be set to 'reasoning_only'."""
        from karenina.benchmark.verification.stages.core.base import VerificationContext

        context = VerificationContext(
            question_id="q1",
            template_id="t1",
            question_text="What is the answer?",
            template_code="class Answer(BaseAnswer): pass",
            answering_model=ModelConfig(id="a", model_name="claude-3-5-sonnet-20241022", model_provider="anthropic"),
            parsing_model=_parsing_model(),
            deep_judgment_mode="reasoning_only",
        )
        assert context.deep_judgment_mode == "reasoning_only"
