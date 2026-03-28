"""Tests for issue 106: state_update snapshot/restore on failure."""

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


def _make_mock_vr(completed: bool = True) -> VerificationResult:
    """Create a mock VerificationResult."""
    from karenina.schemas.verification.model_identity import ModelIdentity

    identity = ModelIdentity.from_model_config(_make_model(), role="answering")
    metadata = VerificationResultMetadata(
        question_id="q1",
        template_id="t1",
        completed_without_errors=completed,
        question_text="What?",
        answering=identity,
        parsing=identity,
        execution_time=0.0,
        timestamp="2024-01-01",
        result_id="r1",
    )
    template = VerificationResultTemplate(raw_llm_response="test response")
    return VerificationResult(metadata=metadata, template=template)


@pytest.mark.unit
class TestStateUpdateSafety:
    """Issue 106: snapshot/restore state.accumulated on state_update failure."""

    def test_corrupted_state_is_restored_on_state_update_failure(self, monkeypatch):
        """state_update that mutates in-place then raises should not corrupt state."""

        def bad_state_update(acc, parsed):
            acc["corrupted_key"] = "bad_value"
            raise ValueError("intentional failure after in-place mutation")

        s = Scenario("test")
        s.add_node("ask", question=_make_question())
        s.add_edge("ask", END)
        s.set_entry("ask")
        defn = s.validate()

        # Inject the callable directly onto the node (bypassing builder string compilation)
        defn.nodes["ask"].state_update = bad_state_update

        config = VerificationConfig(
            answering_models=[_make_model()],
            parsing_models=[_make_model()],
        )

        # Mock pipeline to return a successful VR (so state_update runs)
        # then exit on next iteration via turn limit
        mock_vr = _make_mock_vr(completed=True)
        monkeypatch.setattr(
            ScenarioManager,
            "_run_turn",
            lambda _self, **_kw: (mock_vr, None, None, None),
        )

        manager = ScenarioManager()
        # Set turn limit to 1 so we exit after one turn
        config_dict = config.model_dump()
        config_dict["scenario_turn_limit"] = 1
        limited_config = VerificationConfig(**config_dict)

        result = manager.run(
            scenario=defn,
            config=limited_config,
            base_answering_model=_make_model(),
            base_parsing_model=_make_model(),
        )

        # The key assertion: corrupted_key should NOT be in accumulated
        assert "corrupted_key" not in result.final_state.accumulated
