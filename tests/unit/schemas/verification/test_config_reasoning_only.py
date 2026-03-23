"""Tests for deep_judgment_reasoning_only field in VerificationConfig and VerificationContext."""

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
class TestDeepJudgmentReasoningOnlyConfig:
    """Tests for deep_judgment_reasoning_only field on VerificationConfig."""

    def test_default_is_false(self):
        """deep_judgment_reasoning_only defaults to False."""
        config = VerificationConfig(
            parsing_models=[_parsing_model()],
            parsing_only=True,
        )
        assert config.deep_judgment_reasoning_only is False

    def test_can_be_set_to_true_with_deep_judgment_enabled(self):
        """reasoning_only=True is valid alongside deep_judgment_enabled=True."""
        config = VerificationConfig(
            parsing_models=[_parsing_model()],
            parsing_only=True,
            deep_judgment_enabled=True,
            deep_judgment_reasoning_only=True,
        )
        assert config.deep_judgment_reasoning_only is True
        assert config.deep_judgment_enabled is True

    def test_reasoning_only_true_without_deep_judgment_enabled_is_noop(self):
        """reasoning_only=True with deep_judgment_enabled=False does not raise."""
        config = VerificationConfig(
            parsing_models=[_parsing_model()],
            parsing_only=True,
            deep_judgment_enabled=False,
            deep_judgment_reasoning_only=True,
        )
        assert config.deep_judgment_reasoning_only is True
        assert config.deep_judgment_enabled is False

    def test_reasoning_only_false_with_deep_judgment_enabled(self):
        """reasoning_only=False with deep_judgment_enabled=True is the normal deep judgment path."""
        config = VerificationConfig(
            parsing_models=[_parsing_model()],
            parsing_only=True,
            deep_judgment_enabled=True,
            deep_judgment_reasoning_only=False,
        )
        assert config.deep_judgment_reasoning_only is False
        assert config.deep_judgment_enabled is True


@pytest.mark.unit
class TestVerificationContextReasoningOnly:
    """Tests for deep_judgment_reasoning_only mirror field on VerificationContext."""

    def test_context_default_is_false(self):
        """VerificationContext.deep_judgment_reasoning_only defaults to False."""
        from karenina.benchmark.verification.stages.core.base import VerificationContext

        context = VerificationContext(
            question_id="q1",
            template_id="t1",
            question_text="What is the answer?",
            template_code="class Answer(BaseAnswer): pass",
            answering_model=ModelConfig(id="a", model_name="claude-3-5-sonnet-20241022", model_provider="anthropic"),
            parsing_model=_parsing_model(),
        )
        assert context.deep_judgment_reasoning_only is False

    def test_context_can_be_set_to_true(self):
        """VerificationContext.deep_judgment_reasoning_only can be set to True."""
        from karenina.benchmark.verification.stages.core.base import VerificationContext

        context = VerificationContext(
            question_id="q1",
            template_id="t1",
            question_text="What is the answer?",
            template_code="class Answer(BaseAnswer): pass",
            answering_model=ModelConfig(id="a", model_name="claude-3-5-sonnet-20241022", model_provider="anthropic"),
            parsing_model=_parsing_model(),
            deep_judgment_reasoning_only=True,
        )
        assert context.deep_judgment_reasoning_only is True
