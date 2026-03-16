"""Tests for scenario checkpoint persistence."""

import pytest

from karenina.schemas.checkpoint import (
    JsonLdCheckpoint,
    SchemaOrgAnswer,
    SchemaOrgQuestion,
    SchemaOrgScenario,
    SchemaOrgScenarioEdge,
    SchemaOrgScenarioNode,
    SchemaOrgScenarioOutcome,
    SchemaOrgSoftwareSourceCode,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.config import VerificationConfig

_TEST_MODEL = ModelConfig(id="test", model_name="test", model_provider="test")


@pytest.mark.unit
class TestScenarioTurnLimit:
    def test_default_scenario_turn_limit(self):
        config = VerificationConfig(parsing_models=[_TEST_MODEL], parsing_only=True)
        assert config.scenario_turn_limit == 20

    def test_custom_scenario_turn_limit(self):
        config = VerificationConfig(parsing_models=[_TEST_MODEL], parsing_only=True, scenario_turn_limit=5)
        assert config.scenario_turn_limit == 5


@pytest.mark.unit
class TestSchemaOrgScenarioModels:
    def test_scenario_node_construction(self):
        node = SchemaOrgScenarioNode(
            nodeId="ask",
            question=SchemaOrgQuestion(
                text="What is X?",
                acceptedAnswer=SchemaOrgAnswer(text="Y"),
                hasPart=SchemaOrgSoftwareSourceCode(name="Template", text="class Answer: pass"),
            ),
        )
        assert node.nodeId == "ask"
        assert node.type == "karenina:ScenarioNode"
        assert node.question.text == "What is X?"

    def test_scenario_edge_construction(self):
        edge = SchemaOrgScenarioEdge(
            source="ask",
            target="confirm",
            condition={
                "type": "state_check",
                "field": "verify_result",
                "expected": True,
                "verify_with": {"type": "BooleanMatch"},
            },
        )
        assert edge.source == "ask"
        assert edge.condition is not None

    def test_scenario_outcome_construction(self):
        outcome = SchemaOrgScenarioOutcome(
            name="correct",
            description="Model got it right",
            check={
                "type": "TurnCheck",
                "scope": {"type": "TurnAt", "index": 0},
                "field": "verify_result",
                "expected": True,
                "verify_with": {"type": "BooleanMatch"},
            },
        )
        assert outcome.name == "correct"

    def test_scenario_full_construction(self):
        scenario = SchemaOrgScenario(
            name="test_scenario",
            entryNode="ask",
            nodes={
                "ask": SchemaOrgScenarioNode(
                    nodeId="ask",
                    question=SchemaOrgQuestion(
                        text="Q?",
                        acceptedAnswer=SchemaOrgAnswer(text="A"),
                        hasPart=SchemaOrgSoftwareSourceCode(name="T", text="class A: pass"),
                    ),
                )
            },
            edges=[SchemaOrgScenarioEdge(source="ask", target="__END__")],
        )
        assert scenario.name == "test_scenario"
        assert scenario.type == "karenina:Scenario"

    def test_scenario_serialization_by_alias(self):
        scenario = SchemaOrgScenario(
            name="test",
            entryNode="ask",
            nodes={},
            edges=[],
        )
        data = scenario.model_dump(by_alias=True, exclude_none=True)
        assert data["@type"] == "karenina:Scenario"
        assert "entryNode" in data

    def test_jsonld_checkpoint_has_part(self):
        checkpoint = JsonLdCheckpoint(
            context={"@vocab": "https://schema.org/"},
            name="test",
            dateCreated="2026-01-01",
            dateModified="2026-01-01",
            dataFeedElement=[],
            hasPart=[
                SchemaOrgScenario(
                    name="s1",
                    entryNode="n1",
                    nodes={},
                    edges=[],
                )
            ],
        )
        assert checkpoint.hasPart is not None
        assert len(checkpoint.hasPart) == 1

    def test_jsonld_checkpoint_no_has_part_excluded(self):
        checkpoint = JsonLdCheckpoint(
            context={"@vocab": "https://schema.org/"},
            name="test",
            dateCreated="2026-01-01",
            dateModified="2026-01-01",
            dataFeedElement=[],
        )
        data = checkpoint.model_dump(by_alias=True, exclude_none=True)
        assert "hasPart" not in data
