"""Tests for scenario linking metadata on VerificationResultMetadata."""

import pytest

from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import VerificationResultMetadata


@pytest.mark.unit
class TestScenarioMetadataFields:
    def test_fields_default_to_none(self):
        identity = ModelIdentity(
            interface="test",
            model_name="test",
        )
        m = VerificationResultMetadata(
            question_id="q1",
            template_id="t1",
            completed_without_errors=True,
            question_text="What?",
            answering=identity,
            parsing=identity,
            execution_time=1.0,
            timestamp="2026-01-01",
            result_id="abc123",
        )
        assert m.scenario_id is None
        assert m.scenario_node is None
        assert m.scenario_turn is None
        assert m.scenario_path is None

    def test_scenario_fields_set(self):
        identity = ModelIdentity(
            interface="test",
            model_name="test",
        )
        m = VerificationResultMetadata(
            question_id="q1",
            template_id="t1",
            completed_without_errors=True,
            question_text="What?",
            answering=identity,
            parsing=identity,
            execution_time=1.0,
            timestamp="2026-01-01",
            result_id="abc123",
            scenario_id="drug_pipeline",
            scenario_node="ask",
            scenario_turn=0,
            scenario_path=["ask"],
        )
        assert m.scenario_id == "drug_pipeline"
        assert m.scenario_node == "ask"
        assert m.scenario_turn == 0
        assert m.scenario_path == ["ask"]

    def test_serialization_roundtrip(self):
        identity = ModelIdentity(
            interface="test",
            model_name="test",
        )
        m = VerificationResultMetadata(
            question_id="q1",
            template_id="t1",
            completed_without_errors=True,
            question_text="What?",
            answering=identity,
            parsing=identity,
            execution_time=1.0,
            timestamp="2026-01-01",
            result_id="abc123",
            scenario_id="test_scenario",
            scenario_node="node1",
            scenario_turn=2,
            scenario_path=["ask", "probe", "node1"],
        )
        data = m.model_dump()
        restored = VerificationResultMetadata.model_validate(data)
        assert restored.scenario_id == "test_scenario"
        assert restored.scenario_path == ["ask", "probe", "node1"]
