"""Tests for rubric_only mode validation warning."""

import logging

import pytest

from karenina.benchmark.verification.runner import run_single_model_verification
from karenina.schemas.config.models import ModelConfig


@pytest.mark.unit
class TestRubricOnlyWarning:
    """Warning when rubric_only mode is set but no rubric traits are provided."""

    def test_rubric_only_no_traits_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """rubric_only with no rubric should log a warning before pipeline runs."""
        # Patch StageOrchestrator to prevent actual pipeline execution
        import karenina.benchmark.verification.runner as runner_module

        mock_orchestrator = type(
            "MockOrch",
            (),
            {
                "from_config": classmethod(lambda cls, **_kw: cls()),
                "execute": lambda _self, _ctx: None,
            },
        )
        monkeypatch.setattr(runner_module, "StageOrchestrator", mock_orchestrator)

        answering = ModelConfig(id="gpt-4", interface="langchain", model_name="gpt-4")
        parsing = ModelConfig(id="gpt-4", interface="langchain", model_name="gpt-4")

        with caplog.at_level(logging.WARNING):
            run_single_model_verification(
                question_id="abc123",
                question_text="What is 2+2?",
                template_code="class Answer: pass",
                answering_model=answering,
                parsing_model=parsing,
                evaluation_mode="rubric_only",
                # No rubric provided
            )

        assert "rubric_only" in caplog.text
        assert "no rubric traits provided" in caplog.text
