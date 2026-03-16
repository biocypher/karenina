"""Tests for scenario checkpoint persistence."""

import pytest

from karenina.scenario import END, Scenario
from karenina.scenario.checkpoint import scenario_to_schema_org
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
from karenina.schemas.entities import Question
from karenina.schemas.primitives import BooleanMatch
from karenina.schemas.primitives.composition import AllOf
from karenina.schemas.primitives.scope import TurnAt
from karenina.schemas.scenario.checks import TurnCheck
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


def _build_branching_scenario():
    """Build a branching scenario for testing (mirrors sycophancy example)."""
    q1 = Question(question="What is X?", raw_answer="Y", answer_template="class Answer: pass")
    q2 = Question(question="Are you sure?", raw_answer="Yes", answer_template="class Answer: pass")
    q3 = Question(question="Actually, X is Z.", raw_answer="No, X is Y", answer_template="class Answer: pass")

    s = Scenario("test_branch", description="Branch test")
    s.add_node("ask", question=q1)
    s.add_node("challenge", question=q2)
    s.add_node("correct", question=q3)

    s.add_edge("ask", "challenge", when={"verify_result": True})
    s.add_edge("ask", "correct")
    s.add_edge("challenge", END)
    s.add_edge("correct", END)
    s.set_entry("ask")

    s.add_outcome(
        "both_pass",
        AllOf(
            conditions=[
                TurnCheck(scope=TurnAt(index=0), field="verify_result", expected=True, verify_with=BooleanMatch()),
                TurnCheck(scope=TurnAt(index=1), field="verify_result", expected=True, verify_with=BooleanMatch()),
            ]
        ),
        description="Both turns pass",
    )

    return s.validate()


@pytest.mark.unit
class TestScenarioToSchemaOrg:
    def test_converts_basic_scenario(self):
        defn = _build_branching_scenario()
        schema = scenario_to_schema_org(defn)
        assert schema.name == "test_branch"
        assert schema.entryNode == "ask"
        assert schema.type == "karenina:Scenario"

    def test_converts_nodes_with_questions(self):
        defn = _build_branching_scenario()
        schema = scenario_to_schema_org(defn)
        assert "ask" in schema.nodes
        assert schema.nodes["ask"].question.text == "What is X?"
        assert schema.nodes["ask"].question.acceptedAnswer.text == "Y"
        assert "class Answer" in schema.nodes["ask"].question.hasPart.text

    def test_converts_edges_with_conditions(self):
        defn = _build_branching_scenario()
        schema = scenario_to_schema_org(defn)
        conditional_edges = [e for e in schema.edges if e.condition is not None]
        assert len(conditional_edges) >= 1
        cond = conditional_edges[0].condition
        assert cond["field"] == "verify_result"
        assert cond["verify_with"]["type"] == "BooleanMatch"

    def test_converts_outcome_criteria(self):
        defn = _build_branching_scenario()
        schema = scenario_to_schema_org(defn)
        assert len(schema.outcomeCriteria) == 1
        outcome = schema.outcomeCriteria[0]
        assert outcome.name == "both_pass"
        assert outcome.check is not None
        assert outcome.check["type"] == "AllOf"

    def test_verify_with_primitive_type_preserved_in_edges(self):
        """Critical: verify_with must include the primitive class name."""
        defn = _build_branching_scenario()
        schema = scenario_to_schema_org(defn)
        cond_edge = [e for e in schema.edges if e.condition is not None][0]
        assert cond_edge.condition["verify_with"]["type"] == "BooleanMatch"

    def test_verify_with_primitive_type_preserved_in_outcomes(self):
        """Critical: verify_with inside outcome checks must include class name."""
        defn = _build_branching_scenario()
        schema = scenario_to_schema_org(defn)
        check = schema.outcomeCriteria[0].check
        for cond in check["conditions"]:
            assert cond["verify_with"]["type"] == "BooleanMatch"
