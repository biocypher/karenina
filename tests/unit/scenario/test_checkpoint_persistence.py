"""Tests for scenario checkpoint persistence."""

import json as json_mod
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from karenina.benchmark.benchmark import Benchmark
from karenina.scenario import END, Scenario
from karenina.scenario.checkpoint import scenario_to_schema_org, schema_org_to_scenario
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


@pytest.mark.unit
class TestSchemaOrgToScenario:
    def test_roundtrip_basic(self):
        defn = _build_branching_scenario()
        schema = scenario_to_schema_org(defn)
        restored = schema_org_to_scenario(schema)
        assert restored.name == defn.name
        assert restored.entry_node == defn.entry_node
        assert sorted(restored.nodes.keys()) == sorted(defn.nodes.keys())

    def test_roundtrip_preserves_questions(self):
        defn = _build_branching_scenario()
        schema = scenario_to_schema_org(defn)
        restored = schema_org_to_scenario(schema)
        assert restored.nodes["ask"].question.question == "What is X?"
        assert restored.nodes["ask"].question.raw_answer == "Y"
        assert "class Answer" in restored.nodes["ask"].question.answer_template

    def test_roundtrip_preserves_edges(self):
        defn = _build_branching_scenario()
        schema = scenario_to_schema_org(defn)
        restored = schema_org_to_scenario(schema)
        assert len(restored.edges) == len(defn.edges)

    def test_roundtrip_preserves_edge_conditions(self):
        defn = _build_branching_scenario()
        schema = scenario_to_schema_org(defn)
        restored = schema_org_to_scenario(schema)
        conditional_edges = [e for e in restored.edges if e.condition is not None]
        assert len(conditional_edges) >= 1
        cond = conditional_edges[0].condition
        # Must be a proper StateCheck, not a dict
        assert hasattr(cond, "field")
        assert cond.field == "verify_result"
        # Critical: verify_with must be concrete BooleanMatch, not base VerificationPrimitive
        assert type(cond.verify_with).__name__ == "BooleanMatch"

    def test_roundtrip_preserves_outcome_checks(self):
        defn = _build_branching_scenario()
        schema = scenario_to_schema_org(defn)
        restored = schema_org_to_scenario(schema)
        assert len(restored.outcome_criteria) == 1
        criterion = restored.outcome_criteria[0]
        assert criterion.name == "both_pass"
        assert criterion.check is not None

    def test_roundtrip_callable_source_recompiled(self):
        """state_update should be recompiled from source and be callable."""
        q = Question(question="Q?", raw_answer="A", answer_template="class Answer: pass")
        s = Scenario("callable_test")
        s.add_node("ask", question=q, state_update="lambda acc, p: {**acc, 'x': 1}")
        s.add_edge("ask", END)
        s.set_entry("ask")
        defn = s.validate()

        schema = scenario_to_schema_org(defn)
        restored = schema_org_to_scenario(schema)

        # state_update should be recompiled and callable
        assert restored.nodes["ask"].state_update is not None
        result = restored.nodes["ask"].state_update({}, {})
        assert result == {"x": 1}


@pytest.mark.unit
class TestBenchmarkScenarioCheckpoint:
    def test_add_scenario_writes_to_checkpoint(self):
        b = Benchmark("test")
        defn = _build_branching_scenario()
        b.add_scenario(defn)
        assert b._base._checkpoint.hasPart is not None
        assert len(b._base._checkpoint.hasPart) == 1
        assert b._base._checkpoint.hasPart[0].name == "test_branch"

    def test_add_scenario_sets_benchmark_type(self):
        b = Benchmark("test")
        b.add_scenario(_build_branching_scenario())
        props = b._base._checkpoint.additionalProperty or []
        types = [p for p in props if p.name == "benchmark_type"]
        assert len(types) == 1
        assert types[0].value == "scenario"

    def test_add_scenario_benchmark_type_set_once(self):
        """Adding multiple scenarios should not duplicate the benchmark_type flag."""
        b = Benchmark("test")
        q = Question(question="Q2?", raw_answer="A2", answer_template="class Answer: pass")
        s2 = Scenario("second")
        s2.add_node("n", question=q)
        s2.add_edge("n", END)
        s2.set_entry("n")

        b.add_scenario(_build_branching_scenario())
        b.add_scenario(s2)
        props = b._base._checkpoint.additionalProperty or []
        types = [p for p in props if p.name == "benchmark_type"]
        assert len(types) == 1

    def test_remove_scenario_clears_checkpoint(self):
        b = Benchmark("test")
        b.add_scenario(_build_branching_scenario())
        b.remove_scenario("test_branch")
        assert b._base._checkpoint.hasPart is None
        props = b._base._checkpoint.additionalProperty or []
        types = [p for p in props if p.name == "benchmark_type"]
        assert len(types) == 0

    def test_remove_scenario_partial(self):
        """Removing one of two scenarios should keep hasPart and benchmark_type."""
        b = Benchmark("test")
        q = Question(question="Q2?", raw_answer="A2", answer_template="class Answer: pass")
        s2 = Scenario("second")
        s2.add_node("n", question=q)
        s2.add_edge("n", END)
        s2.set_entry("n")

        b.add_scenario(_build_branching_scenario())
        b.add_scenario(s2)
        b.remove_scenario("test_branch")

        assert b._base._checkpoint.hasPart is not None
        assert len(b._base._checkpoint.hasPart) == 1
        assert b._base._checkpoint.hasPart[0].name == "second"
        props = b._base._checkpoint.additionalProperty or []
        types = [p for p in props if p.name == "benchmark_type"]
        assert len(types) == 1  # still present


@pytest.mark.unit
class TestBenchmarkSaveLoadRoundtrip:
    def test_scenario_survives_save_load(self):
        b = Benchmark("test_save")
        b.add_scenario(_build_branching_scenario())

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonld"
            b.save(path)

            loaded = Benchmark.load(path)
            assert loaded.is_scenario_benchmark
            assert len(loaded.get_scenarios()) == 1
            restored = loaded.get_scenario("test_branch")
            assert sorted(restored.nodes.keys()) == ["ask", "challenge", "correct"]
            assert restored.entry_node == "ask"

    def test_edge_conditions_survive_roundtrip(self):
        b = Benchmark("test_edges")
        b.add_scenario(_build_branching_scenario())

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonld"
            b.save(path)
            loaded = Benchmark.load(path)
            restored = loaded.get_scenario("test_branch")
            conditional_edges = [e for e in restored.edges if e.condition is not None]
            assert len(conditional_edges) >= 1
            assert type(conditional_edges[0].condition.verify_with).__name__ == "BooleanMatch"

    def test_outcome_criteria_survive_roundtrip(self):
        b = Benchmark("test_outcomes")
        b.add_scenario(_build_branching_scenario())

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonld"
            b.save(path)
            loaded = Benchmark.load(path)
            restored = loaded.get_scenario("test_branch")
            assert len(restored.outcome_criteria) == 1
            assert restored.outcome_criteria[0].name == "both_pass"
            assert restored.outcome_criteria[0].check is not None

    def test_backward_compat_no_scenarios(self):
        b = Benchmark("question_bench")
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonld"
            b.save(path)
            loaded = Benchmark.load(path)
            assert not loaded.is_scenario_benchmark
            assert loaded.get_scenarios() == []

    def test_homogeneous_enforcement_after_load(self):
        b = Benchmark("scenario_bench")
        b.add_scenario(_build_branching_scenario())

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonld"
            b.save(path)
            loaded = Benchmark.load(path)
            with pytest.raises(ValueError, match="Cannot add standalone questions"):
                loaded.add_question("What?", raw_answer="Y", answer_template="class A: pass")

    def test_malformed_has_part_raises(self):
        """Checkpoint with invalid hasPart data should raise on load."""
        b = Benchmark("malformed")
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonld"
            b.save(path)

            with open(path) as f:
                data = json_mod.load(f)
            # Inject malformed scenario (missing entryNode)
            data["hasPart"] = [{"@type": "karenina:Scenario", "name": "bad", "nodes": {}, "edges": []}]
            data.setdefault("additionalProperty", []).append(
                {"@type": "PropertyValue", "name": "benchmark_type", "value": "scenario"}
            )
            with open(path, "w") as f:
                json_mod.dump(data, f)

            with pytest.raises(ValueError, match="Invalid JSON-LD"):
                Benchmark.load(path)

    def test_mixed_checkpoint_rejected(self):
        """Hand-edited checkpoint with both questions and scenarios should fail."""
        b = Benchmark("mixed")
        b.add_question("What?", raw_answer="Y", answer_template="class A: pass")

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonld"
            b.save(path)

            with open(path) as f:
                data = json_mod.load(f)
            # Hand-inject hasPart into a question benchmark
            data["hasPart"] = [
                SchemaOrgScenario(
                    name="injected",
                    entryNode="n",
                    nodes={},
                    edges=[],
                ).model_dump(by_alias=True)
            ]
            with open(path, "w") as f:
                json_mod.dump(data, f)

            with pytest.raises(ValueError, match="both questions and scenarios"):
                Benchmark.load(path)


@pytest.mark.unit
class TestBenchmarkScenarioProperties:
    def test_is_empty_false_for_scenario_benchmark(self):
        b = Benchmark("test")
        b.add_scenario(_build_branching_scenario())
        assert not b.is_empty

    def test_is_empty_true_for_empty_benchmark(self):
        b = Benchmark("test")
        assert b.is_empty

    def test_scenario_count(self):
        b = Benchmark("test")
        assert b.scenario_count == 0
        b.add_scenario(_build_branching_scenario())
        assert b.scenario_count == 1

    def test_len_returns_scenario_count(self):
        b = Benchmark("test")
        b.add_scenario(_build_branching_scenario())
        assert len(b) == 1

    def test_clone_preserves_scenarios(self):
        b = Benchmark("test")
        b.add_scenario(_build_branching_scenario())
        cloned = b.clone()
        assert cloned.is_scenario_benchmark
        assert len(cloned.get_scenarios()) == 1
        assert cloned.get_scenario("test_branch").entry_node == "ask"
