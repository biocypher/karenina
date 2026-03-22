"""Tests for issue 104: warn when evaluation_mode='rubric_only' in scenarios."""

from __future__ import annotations

import pytest

from karenina.scenario.builder import Scenario
from karenina.scenario.manager import ScenarioManager
from karenina.schemas.config import ModelConfig
from karenina.schemas.scenario.types import END
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.result_components import VerificationResultMetadata, VerificationResultTemplate


def _make_question(text: str = "What?"):
    from karenina.schemas.entities import Question

    return Question(question=text, raw_answer="Y", answer_template="class Answer: pass")


def _make_model(name: str = "claude", provider: str = "anthropic") -> ModelConfig:
    return ModelConfig(id=name, model_name=name, model_provider=provider)


def _make_mock_vr() -> VerificationResult:
    """Create a mock VerificationResult that causes the loop to exit."""
    from karenina.schemas.verification.model_identity import ModelIdentity

    identity = ModelIdentity.from_model_config(_make_model(), role="answering")
    metadata = VerificationResultMetadata(
        question_id="q1",
        template_id="t1",
        completed_without_errors=False,  # causes loop exit
        question_text="What?",
        answering=identity,
        parsing=identity,
        execution_time=0.0,
        timestamp="2024-01-01",
        result_id="r1",
    )
    template = VerificationResultTemplate(raw_llm_response="")
    return VerificationResult(metadata=metadata, template=template)


@pytest.mark.unit
class TestEvaluationModeWarning:
    """Issue 104: ScenarioManager should warn on rubric_only."""

    def test_warns_on_rubric_only_evaluation_mode(self, monkeypatch):
        """Passing evaluation_mode='rubric_only' triggers UserWarning."""
        s = Scenario("test")
        s.add_node("ask", question=_make_question())
        s.add_edge("ask", END)
        s.set_entry("ask")
        defn = s.validate()

        config = VerificationConfig(
            answering_models=[_make_model()],
            parsing_models=[_make_model()],
            evaluation_mode="rubric_only",
        )

        mock_vr = _make_mock_vr()

        # Monkeypatch _run_turn to avoid real pipeline execution
        monkeypatch.setattr(
            ScenarioManager,
            "_run_turn",
            lambda _self, **_kw: (mock_vr, None, None, None),
        )

        manager = ScenarioManager()
        with pytest.warns(UserWarning, match="rubric_only"):
            manager.run(
                scenario=defn,
                config=config,
                base_answering_model=_make_model(),
                base_parsing_model=_make_model(),
            )

    def test_no_warning_on_template_only(self, monkeypatch):
        """evaluation_mode='template_only' does not trigger warning."""
        import warnings

        s = Scenario("test")
        s.add_node("ask", question=_make_question())
        s.add_edge("ask", END)
        s.set_entry("ask")
        defn = s.validate()

        config = VerificationConfig(
            answering_models=[_make_model()],
            parsing_models=[_make_model()],
            evaluation_mode="template_only",
        )

        mock_vr = _make_mock_vr()

        monkeypatch.setattr(
            ScenarioManager,
            "_run_turn",
            lambda _self, **_kw: (mock_vr, None, None, None),
        )

        manager = ScenarioManager()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            manager.run(
                scenario=defn,
                config=config,
                base_answering_model=_make_model(),
                base_parsing_model=_make_model(),
            )
