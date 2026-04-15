"""Validate-phase tests for ScenarioReplayBuilder (R3 projection).

Covers target resolution, duplicate-target detection, orphan detection,
and the validate() purity contract. Separated from test_projection.py
to keep each test module under the 800-line guidance.
"""

from __future__ import annotations

import pytest

from karenina.benchmark import Benchmark
from karenina.replay import ReplayEntry, ReplayKey, ReplayStore
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities.question import Question
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.types import END, ModelOverride, ScenarioEdge, ScenarioNode
from karenina.schemas.verification.config import VerificationConfig
from karenina.utils.checkpoint import generate_question_id


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


def _qa_store(entries: list[tuple[ReplayKey, ReplayEntry]] | None = None) -> ReplayStore:
    store = ReplayStore(miss_policy="fall_through")
    for key, entry in entries or []:
        store.register(key, entry)
    return store


def _qa_entry_for(question_text: str, model_display: str) -> tuple[ReplayKey, ReplayEntry]:
    return (
        ReplayKey(
            question_id=generate_question_id(question_text),
            answering_model_id=model_display,
        ),
        ReplayEntry(raw_trace=f"canned:{question_text}"),
    )


def _answering_display(cfg: VerificationConfig) -> str:
    from karenina.schemas.verification.model_identity import ModelIdentity

    return ModelIdentity.from_model_config(
        cfg.answering_models[0],
        role="answering",
    ).display_string


@pytest.mark.unit
class TestValidateCore:
    def test_matches_single_target(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        model_display = _answering_display(cfg)
        qa_store = _qa_store([_qa_entry_for("What is X?", model_display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa_store, target_nodes=["ask"], scenarios=["s1"])

        report = builder.validate()
        assert report.matched == 1
        assert report.projected_keys[0].scenario_id == "s1"
        assert report.projected_keys[0].scenario_node == "ask"
        assert report.projected_keys[0].answering_model_id == model_display
        assert report.projected_keys[0].visit_index is None
        assert report.projected_keys[0].replicate is None

    def test_reports_missing_scenario(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        qa_store = _qa_store([_qa_entry_for("What is X?", _answering_display(cfg))])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa_store, target_nodes=["ask"], scenarios=["nonexistent"])

        report = builder.validate()
        assert any(t.reason == "missing_scenario" and t.scenario_id == "nonexistent" for t in report.unmatched_targets)

    def test_reports_missing_node(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        qa_store = _qa_store([_qa_entry_for("What is X?", _answering_display(cfg))])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa_store, target_nodes=["nope"], scenarios=["s1"])

        report = builder.validate()
        assert any(t.reason == "missing_node" and t.node_id == "nope" for t in report.unmatched_targets)

    def test_reports_no_qa_entry(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        # qa_store has the WRONG question
        qa_store = _qa_store([_qa_entry_for("Unrelated?", _answering_display(cfg))])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa_store, target_nodes=["ask"], scenarios=["s1"])

        report = builder.validate()
        assert any(t.reason == "no_qa_entry" and t.scenario_id == "s1" for t in report.unmatched_targets)

    def test_uses_node_override_model_when_set(self):
        """When a ScenarioNode has model_override.answering_model, projection resolves to that model, not config default."""
        from karenina.replay.projection import ScenarioReplayBuilder
        from karenina.schemas.verification.model_identity import ModelIdentity

        override_model = ModelConfig(
            id="override-m",
            model_name="override-m",
            model_provider="anthropic",
        )
        q = Question(question="What is X?", raw_answer="X", keywords=[])
        defn = ScenarioDefinition(
            name="s_override",
            entry_node="ask",
            nodes={
                "ask": ScenarioNode(
                    node_id="ask",
                    question=q,
                    model_override=ModelOverride(answering_model=override_model),
                )
            },
            edges=[ScenarioEdge(source="ask", target=END)],
            outcome_criteria=[],
        )
        bench = Benchmark.create(name="proj-override", version="1.0.0")
        bench.add_scenario(defn)

        cfg = _default_config()
        override_display = ModelIdentity.from_model_config(override_model, role="answering").display_string
        qa_store = _qa_store([_qa_entry_for("What is X?", override_display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa_store, target_nodes=["ask"], scenarios=["s_override"])

        report = builder.validate()
        assert report.matched == 1
        assert report.projected_keys[0].answering_model_id == override_display

    def test_expands_scenarios_none_at_validate_time(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        model_display = _answering_display(cfg)
        qa_store = _qa_store(
            [
                _qa_entry_for("What is X?", model_display),
                _qa_entry_for("What is Y?", model_display),
            ]
        )

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa_store, target_nodes=["ask"], scenarios=None)

        report = builder.validate()
        assert report.matched == 2
        scenario_ids = {k.scenario_id for k in report.projected_keys}
        assert scenario_ids == {"s1", "s2"}


@pytest.mark.unit
class TestValidateDuplicates:
    def test_reports_duplicate_targets_across_projections(self):
        """Two projections staging the same (scenario, node) must report a duplicate."""
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        model_display = _answering_display(cfg)
        qa_a = _qa_store([_qa_entry_for("What is X?", model_display)])
        qa_b = _qa_store([_qa_entry_for("What is X?", model_display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa_a, target_nodes=["ask"], scenarios=["s1"])
        builder.add_qa(qa_b, target_nodes=["ask"], scenarios=["s1"])

        report = builder.validate()
        assert ("s1", "ask") in report.duplicate_targets

    def test_no_duplicate_when_disjoint_scenarios(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        model_display = _answering_display(cfg)
        qa_a = _qa_store([_qa_entry_for("What is X?", model_display)])
        qa_b = _qa_store([_qa_entry_for("What is Y?", model_display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa_a, target_nodes=["ask"], scenarios=["s1"])
        builder.add_qa(qa_b, target_nodes=["ask"], scenarios=["s2"])

        report = builder.validate()
        assert report.duplicate_targets == []


@pytest.mark.unit
class TestValidateOrphans:
    def test_reports_orphan_no_target_scenario(self):
        """An entry whose question does not appear on any declared scenario's targeted nodes is orphan."""
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        model_display = _answering_display(cfg)
        # QA entry for "Unused" question which is not a node question
        qa_store = _qa_store(
            [
                _qa_entry_for("What is X?", model_display),
                _qa_entry_for("Unused question?", model_display),
            ]
        )

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa_store, target_nodes=["ask"], scenarios=["s1"])

        report = builder.validate()
        assert any(
            o.reason == "no_target_scenario" and o.question_id == generate_question_id("Unused question?")
            for o in report.orphan_qa_entries
        )

    def test_reports_orphan_model_id_never_requested(self):
        """An entry whose question matches a node but whose model id never matches is orphan (wrong model)."""
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        wrong_model_display = "anthropic:wrong-model"
        correct_display = _answering_display(cfg)
        # Two entries for same question: one wrong model, one correct.
        qa_store = _qa_store(
            [
                (
                    ReplayKey(
                        question_id=generate_question_id("What is X?"),
                        answering_model_id=wrong_model_display,
                    ),
                    ReplayEntry(raw_trace="wrong model"),
                ),
                (
                    ReplayKey(
                        question_id=generate_question_id("What is X?"),
                        answering_model_id=correct_display,
                    ),
                    ReplayEntry(raw_trace="right model"),
                ),
            ]
        )

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa_store, target_nodes=["ask"], scenarios=["s1"])

        report = builder.validate()
        assert any(
            o.reason == "model_id_never_requested" and o.answering_model_id == wrong_model_display
            for o in report.orphan_qa_entries
        )
        # The correct-model entry is consumed, so it must NOT be orphan.
        assert not any(o.answering_model_id == correct_display for o in report.orphan_qa_entries)

    def test_disjoint_projections_do_not_cross_orphan(self):
        """An entry in store A can only be orphan wrt projection A's declared targets."""
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        qa_a = _qa_store([_qa_entry_for("What is X?", display)])
        qa_b = _qa_store([_qa_entry_for("What is Y?", display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa_a, target_nodes=["ask"], scenarios=["s1"])
        builder.add_qa(qa_b, target_nodes=["ask"], scenarios=["s2"])

        report = builder.validate()
        # Both entries consumed: no orphans.
        assert report.orphan_qa_entries == []

    def test_wildcard_answering_model_qa_entry_is_consumed_not_orphan(self):
        """A QA entry with ``answering_model_id=None`` is consumed by a
        concrete-model probe via the specificity ladder; orphan detection
        must treat it as consumed (not flag it as model_id_never_requested).

        Regression test: earlier pair-based orphan tracking would miss this
        because the consumed set stored concrete (qid, model) pairs while
        the QA entry's key held None. We now track consumed entries by
        object identity.
        """
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()

        wildcard_qa = _qa_store(
            [
                (
                    ReplayKey(
                        question_id=generate_question_id("What is X?"),
                        answering_model_id=None,  # wildcard
                    ),
                    ReplayEntry(raw_trace="wildcard"),
                )
            ]
        )

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(wildcard_qa, target_nodes=["ask"], scenarios=["s1"])

        report = builder.validate()
        assert report.matched == 1, report
        assert report.orphan_qa_entries == [], report.orphan_qa_entries


@pytest.mark.unit
class TestValidatePurity:
    def test_multiple_calls_produce_equal_reports(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        qa = _qa_store([_qa_entry_for("What is X?", display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa, target_nodes=["ask"], scenarios=["s1"])

        a = builder.validate()
        b = builder.validate()
        assert a.projected_keys == b.projected_keys
        assert a.unmatched_targets == b.unmatched_targets
        assert a.orphan_qa_entries == b.orphan_qa_entries
        assert a.duplicate_targets == b.duplicate_targets

    def test_does_not_mutate_staged_state(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        qa = _qa_store([_qa_entry_for("What is X?", display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa, target_nodes=["ask"], scenarios=["s1"])

        before_staged = len(builder._staged)  # noqa: SLF001
        before_qa_entries = len(qa.entries)
        builder.validate()
        assert len(builder._staged) == before_staged  # noqa: SLF001
        assert len(qa.entries) == before_qa_entries

    def test_does_not_mutate_qa_store_miss_policy(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        qa = _qa_store([_qa_entry_for("What is X?", display)])
        qa.miss_policy = "strict"

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa, target_nodes=["ask"], scenarios=["s1"])

        builder.validate()
        assert qa.miss_policy == "strict"
