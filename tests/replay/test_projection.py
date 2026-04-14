"""Unit + integration tests for ScenarioReplayBuilder (R3 projection)."""

from __future__ import annotations

import pytest

from karenina.benchmark import Benchmark
from karenina.replay import ReplayKey
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities.question import Question
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode
from karenina.schemas.verification.config import VerificationConfig


def _minimal_benchmark() -> Benchmark:
    """Two-scenario benchmark, single node each, for builder tests."""
    q1 = Question(question="What is X?", raw_answer="X", keywords=[])
    q2 = Question(question="What is Y?", raw_answer="Y", keywords=[])
    defn1 = ScenarioDefinition(
        name="s1",
        entry_node="ask",
        nodes={"ask": ScenarioNode(node_id="ask", question=q1)},
        edges=[ScenarioEdge(source="ask", target=END)],
        outcome_criteria=[],
    )
    defn2 = ScenarioDefinition(
        name="s2",
        entry_node="ask",
        nodes={"ask": ScenarioNode(node_id="ask", question=q2)},
        edges=[ScenarioEdge(source="ask", target=END)],
        outcome_criteria=[],
    )
    bm = Benchmark.create(name="proj-test", version="1.0.0")
    bm.add_scenario(defn1)
    bm.add_scenario(defn2)
    return bm


def _default_config() -> VerificationConfig:
    return VerificationConfig(
        answering_models=[
            ModelConfig(id="gpt-5", model_name="gpt-5", model_provider="openai"),
        ],
        parsing_models=[
            ModelConfig(id="sonnet", model_name="claude", model_provider="anthropic"),
        ],
    )


@pytest.mark.unit
class TestProjectionDataTypes:
    def test_unmatched_target_requires_reason(self):
        from karenina.replay.projection import UnmatchedTarget

        target = UnmatchedTarget(
            scenario_id="s",
            node_id="n",
            question_id=None,
            answering_model_id=None,
            reason="missing_scenario",
        )
        assert target.reason == "missing_scenario"

    def test_unmatched_target_rejects_unknown_reason(self):
        from pydantic import ValidationError

        from karenina.replay.projection import UnmatchedTarget

        with pytest.raises(ValidationError):
            UnmatchedTarget(
                scenario_id="s",
                node_id="n",
                question_id=None,
                answering_model_id=None,
                reason="bogus",
            )

    def test_unmatched_target_forbids_extra_fields(self):
        from pydantic import ValidationError

        from karenina.replay.projection import UnmatchedTarget

        with pytest.raises(ValidationError):
            UnmatchedTarget(
                scenario_id="s",
                node_id="n",
                question_id=None,
                answering_model_id=None,
                reason="missing_node",
                bogus="x",
            )

    def test_orphan_entry_requires_reason(self):
        from karenina.replay.projection import OrphanEntry

        o = OrphanEntry(
            question_id="q",
            answering_model_id="m",
            reason="no_target_scenario",
        )
        assert o.reason == "no_target_scenario"

    def test_projection_report_matched_is_length_of_projected_keys(self):
        from karenina.replay.projection import ProjectionReport

        report = ProjectionReport(
            projected_keys=[
                ReplayKey(question_id="q", scenario_id="s1", scenario_node="ask"),
                ReplayKey(question_id="q", scenario_id="s2", scenario_node="ask"),
            ],
            unmatched_targets=[],
            orphan_qa_entries=[],
            duplicate_targets=[],
        )
        assert report.matched == 2

    def test_projection_report_matched_zero_when_empty(self):
        from karenina.replay.projection import ProjectionReport

        report = ProjectionReport(
            projected_keys=[],
            unmatched_targets=[],
            orphan_qa_entries=[],
            duplicate_targets=[],
        )
        assert report.matched == 0


@pytest.mark.unit
class TestScenarioReplayBuilderCtor:
    def test_ctor_accepts_benchmark_and_config(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        builder = ScenarioReplayBuilder(_minimal_benchmark(), config=_default_config())
        assert builder is not None

    def test_ctor_rejects_none_benchmark(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        with pytest.raises(TypeError, match="benchmark"):
            ScenarioReplayBuilder(None, config=_default_config())

    def test_ctor_rejects_none_config(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        with pytest.raises(TypeError, match="config"):
            ScenarioReplayBuilder(_minimal_benchmark(), config=None)

    def test_ctor_rejects_empty_answering_models(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        # parsing_only=True lets VerificationConfig accept an empty
        # answering_models list, so the builder's own guard is the one
        # under test here.
        bad = VerificationConfig(
            answering_models=[],
            parsing_models=[ModelConfig(id="p", model_name="p", model_provider="anthropic")],
            parsing_only=True,
        )
        with pytest.raises(ValueError, match="answering_models"):
            ScenarioReplayBuilder(_minimal_benchmark(), config=bad)

    def test_ctor_default_miss_policy_is_strict(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        builder = ScenarioReplayBuilder(_minimal_benchmark(), config=_default_config())
        assert builder.miss_policy == "strict"

    def test_ctor_accepts_custom_miss_policy(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        builder = ScenarioReplayBuilder(
            _minimal_benchmark(),
            config=_default_config(),
            miss_policy="fall_through",
        )
        assert builder.miss_policy == "fall_through"
