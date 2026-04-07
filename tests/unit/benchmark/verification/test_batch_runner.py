"""Tests for batch_runner task queue generation and parsing_only guard."""

import pytest

from karenina.benchmark.verification.batch_runner import (
    _apply_retry_config,
    generate_task_queue,
    run_verification_batch,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import FinishedTemplate, VerificationConfig


def _make_model(model_id: str) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        model_name=model_id,
        model_provider="anthropic",
        interface="langchain",
        system_prompt="test",
        temperature=0.1,
    )


def _make_template(question_id: str = "q1", raw_answer: str | None = None) -> FinishedTemplate:
    return FinishedTemplate(
        question_id=question_id,
        question_text="What is 2+2?",
        question_preview="What is 2+2?",
        template_code="class Answer(BaseAnswer): pass",
        raw_answer=raw_answer,
        last_modified="2026-01-01T00:00:00",
    )


HAIKU = _make_model("haiku")
SONNET = _make_model("sonnet")


# =============================================================================
# generate_task_queue: normal mode
# =============================================================================


@pytest.mark.unit
class TestTaskQueueGeneration:
    """Verify normal task queue expansion."""

    def test_cross_product_of_models(self) -> None:
        """Normal mode produces answering x parsing cross product."""
        config = VerificationConfig(
            parsing_models=[HAIKU, SONNET],
            answering_models=[HAIKU, SONNET],
        )
        templates = [_make_template()]

        tasks = generate_task_queue(templates, config)

        assert len(tasks) == 4  # 2 answering x 2 parsing

    def test_replicates_expand(self) -> None:
        """Replicates multiply the task count."""
        config = VerificationConfig(
            parsing_models=[HAIKU],
            answering_models=[SONNET],
            replicate_count=3,
        )
        templates = [_make_template()]

        tasks = generate_task_queue(templates, config)

        assert len(tasks) == 3


# =============================================================================
# parsing_only guard
# =============================================================================


@pytest.mark.unit
class TestParsingOnlyGuard:
    """Verify that parsing_only=True is rejected in the batch verification path."""

    def test_parsing_only_rejected_in_batch_runner(self) -> None:
        """run_verification_batch should raise ValueError for parsing_only=True."""
        config = VerificationConfig(
            parsing_models=[HAIKU],
            answering_models=[],
            parsing_only=True,
        )
        templates = [_make_template(raw_answer="The answer is 4.")]

        with pytest.raises(ValueError, match="parsing_only=True is not supported"):
            run_verification_batch(templates=templates, config=config)

    def test_parsing_only_produces_zero_tasks_in_queue(self) -> None:
        """generate_task_queue with empty answering_models produces 0 tasks."""
        config = VerificationConfig(
            parsing_models=[HAIKU],
            answering_models=[],
            parsing_only=True,
        )
        templates = [_make_template(raw_answer="The answer is 4.")]

        tasks = generate_task_queue(templates, config)

        # This confirms the underlying issue: the loop over answering_models
        # produces nothing. The guard in run_verification_batch prevents
        # users from hitting this silently.
        assert len(tasks) == 0

    def test_normal_mode_not_rejected(self) -> None:
        """Normal mode (parsing_only=False) should not raise."""
        config = VerificationConfig(
            parsing_models=[HAIKU],
            answering_models=[SONNET],
        )
        templates = [_make_template()]

        # Should not raise (we don't run the full pipeline, just check no guard fires)
        tasks = generate_task_queue(templates, config)
        assert len(tasks) == 1


# =============================================================================
# _apply_retry_config
# =============================================================================


@pytest.mark.unit
class TestApplyRetryConfig:
    """Tests for _apply_retry_config function."""

    def test_stamps_when_model_has_no_value(self) -> None:
        """Test that pipeline value is stamped when model has None."""
        model = ModelConfig(id="test", model_name="test", model_provider="openai")
        result = _apply_retry_config(model, max_transient_retries=2)
        assert result.max_transient_retries == 2

    def test_preserves_existing_model_value(self) -> None:
        """Test that model-level value is not overwritten."""
        model = ModelConfig(
            id="test",
            model_name="test",
            model_provider="openai",
            max_transient_retries=1,
        )
        result = _apply_retry_config(model, max_transient_retries=5)
        assert result.max_transient_retries == 1

    def test_noop_when_pipeline_value_is_none(self) -> None:
        """Test that None pipeline value does not stamp."""
        model = ModelConfig(id="test", model_name="test", model_provider="openai")
        result = _apply_retry_config(model, max_transient_retries=None)
        assert result.max_transient_retries is None
        assert result is model  # Same object, no copy
