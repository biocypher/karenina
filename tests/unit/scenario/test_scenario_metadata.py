"""Tests for issue 163: scenario metadata plumbing to VerificationResult."""

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.benchmark.verification.stages.pipeline.finalize_result import FinalizeResultStage
from karenina.schemas.config import ModelConfig


def _make_model(name: str = "test", provider: str = "test") -> ModelConfig:
    return ModelConfig(id=name, model_name=name, model_provider=provider)


@pytest.mark.unit
class TestScenarioMetadataPlumbing:
    """Issue 163: scenario fields must flow from VerificationContext to VerificationResultMetadata."""

    def test_verification_context_accepts_scenario_fields(self):
        """VerificationContext can be constructed with scenario_id, scenario_node, scenario_path."""
        ctx = VerificationContext(
            question_id="q1",
            template_id="t1",
            question_text="What is X?",
            template_code="",
            answering_model=_make_model(),
            parsing_model=_make_model(),
            scenario_id="my_scenario",
            scenario_node="ask",
            scenario_turn=0,
            scenario_path=["ask"],
        )
        assert ctx.scenario_id == "my_scenario"
        assert ctx.scenario_node == "ask"
        assert ctx.scenario_turn == 0
        assert ctx.scenario_path == ["ask"]

    def test_finalize_transfers_scenario_metadata(self):
        """FinalizeResultStage transfers scenario fields to VerificationResultMetadata."""
        ctx = VerificationContext(
            question_id="q1",
            template_id="t1",
            question_text="What is X?",
            template_code="",
            answering_model=_make_model(),
            parsing_model=_make_model(),
            scenario_id="my_scenario",
            scenario_node="ask",
            scenario_turn=0,
            scenario_path=["ask"],
        )
        ctx.set_result_field("timestamp", "2024-01-01T00:00:00")
        ctx.set_result_field("execution_time", 1.0)

        stage = FinalizeResultStage()
        stage.execute(ctx)

        result = ctx.get_artifact("final_result")
        assert result.metadata.scenario_id == "my_scenario"
        assert result.metadata.scenario_node == "ask"
        assert result.metadata.scenario_turn == 0
        assert result.metadata.scenario_path == ["ask"]

    def test_scenario_fields_default_to_none(self):
        """Non-scenario contexts have None for all scenario fields."""
        ctx = VerificationContext(
            question_id="q1",
            template_id="t1",
            question_text="What is X?",
            template_code="",
            answering_model=_make_model(),
            parsing_model=_make_model(),
        )
        ctx.set_result_field("timestamp", "2024-01-01T00:00:00")
        ctx.set_result_field("execution_time", 1.0)

        stage = FinalizeResultStage()
        stage.execute(ctx)

        result = ctx.get_artifact("final_result")
        assert result.metadata.scenario_id is None
        assert result.metadata.scenario_node is None
        assert result.metadata.scenario_turn is None
        assert result.metadata.scenario_path is None
