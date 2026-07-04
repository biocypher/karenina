"""Tests for runner evaluation_mode auto-upgrade removal."""

import logging
from unittest.mock import patch

import pytest

from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric


@pytest.mark.unit
class TestRunnerAutoUpgrade:
    """Verify that runner no longer auto-upgrades template_only to template_and_rubric."""

    def test_template_only_with_rubric_warns_and_does_not_upgrade(self, caplog: pytest.LogCaptureFixture) -> None:
        """Rubric traits with template_only should warn, not upgrade."""
        from karenina.benchmark.verification.runner import run_single_model_verification
        from karenina.schemas.config import ModelConfig

        rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="clarity", kind="boolean", description="Is the response clear?")]
        )
        answering = ModelConfig(id="test", model_name="test", model_provider="openai", system_prompt="You are helpful.")
        parsing = ModelConfig(id="test", model_name="test", model_provider="openai", system_prompt="Parse.")

        # Mock StageOrchestrator to avoid running the full pipeline
        with (
            patch("karenina.benchmark.verification.runner.StageOrchestrator") as mock_orch,
            caplog.at_level(logging.WARNING),
        ):
            mock_orch.from_config.return_value.execute.return_value = None
            run_single_model_verification(
                question_id="q1",
                question_text="What is 2+2?",
                template_code="class Answer(BaseAnswer): value: str",
                answering_model=answering,
                parsing_model=parsing,
                evaluation_mode="template_only",
                rubric=rubric,
            )

        assert any("template_only" in r.message and "Rubric" in r.message for r in caplog.records)
        # Verify evaluation_mode was NOT changed: from_config should receive "template_only"
        call_kwargs = mock_orch.from_config.call_args
        assert call_kwargs.kwargs.get("evaluation_mode", call_kwargs[1].get("evaluation_mode")) == "template_only"

    def test_template_only_without_rubric_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """No rubric traits should produce no warning."""
        from karenina.benchmark.verification.runner import run_single_model_verification
        from karenina.schemas.config import ModelConfig

        answering = ModelConfig(id="test", model_name="test", model_provider="openai", system_prompt="You are helpful.")
        parsing = ModelConfig(id="test", model_name="test", model_provider="openai", system_prompt="Parse.")

        with (
            patch("karenina.benchmark.verification.runner.StageOrchestrator") as mock_orch,
            caplog.at_level(logging.WARNING),
        ):
            mock_orch.from_config.return_value.execute.return_value = None
            run_single_model_verification(
                question_id="q1",
                question_text="What is 2+2?",
                template_code="class Answer(BaseAnswer): value: str",
                answering_model=answering,
                parsing_model=parsing,
                evaluation_mode="template_only",
            )

        assert not any("template_only" in r.message for r in caplog.records)
