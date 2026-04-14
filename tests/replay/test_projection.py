"""Unit + integration tests for ScenarioReplayBuilder (R3 projection)."""

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


def _qa_store(entries: list[tuple[ReplayKey, ReplayEntry]] | None = None) -> ReplayStore:
    store = ReplayStore(miss_policy="fall_through")
    for key, entry in entries or []:
        store.register(key, entry)
    return store


def _valid_qa_store() -> ReplayStore:
    """Single QA entry, wildcard replicate."""
    return _qa_store(
        [
            (
                ReplayKey(question_id="q", answering_model_id="openai:gpt-5"),
                ReplayEntry(raw_trace="canned"),
            )
        ]
    )


@pytest.mark.unit
class TestAddQaValidation:
    def test_rejects_none_qa_store(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        builder = ScenarioReplayBuilder(_minimal_benchmark(), config=_default_config())
        with pytest.raises(TypeError, match="qa_store"):
            builder.add_qa(None, target_nodes=["ask"])

    def test_rejects_empty_target_nodes(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        builder = ScenarioReplayBuilder(_minimal_benchmark(), config=_default_config())
        with pytest.raises(ValueError, match="target_nodes"):
            builder.add_qa(_valid_qa_store(), target_nodes=[])

    def test_rejects_empty_scenarios_list(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        builder = ScenarioReplayBuilder(_minimal_benchmark(), config=_default_config())
        with pytest.raises(ValueError, match="scenarios"):
            builder.add_qa(_valid_qa_store(), target_nodes=["ask"], scenarios=[])

    def test_accepts_none_scenarios_as_all(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        builder = ScenarioReplayBuilder(_minimal_benchmark(), config=_default_config())
        out = builder.add_qa(_valid_qa_store(), target_nodes=["ask"], scenarios=None)
        assert out is builder

    def test_rejects_scenario_mode_qa_store(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        scenario_mode = _qa_store(
            [
                (
                    ReplayKey(
                        question_id="q",
                        scenario_id="s1",
                        scenario_node="ask",
                        answering_model_id="openai:gpt-5",
                    ),
                    ReplayEntry(raw_trace="canned"),
                )
            ]
        )
        builder = ScenarioReplayBuilder(_minimal_benchmark(), config=_default_config())
        with pytest.raises(ValueError, match="scenario-mode"):
            builder.add_qa(scenario_mode, target_nodes=["ask"])

    def test_rejects_per_replicate_qa_store(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        per_replicate = _qa_store(
            [
                (
                    ReplayKey(
                        question_id="q",
                        answering_model_id="openai:gpt-5",
                        replicate=1,
                    ),
                    ReplayEntry(raw_trace="canned"),
                )
            ]
        )
        builder = ScenarioReplayBuilder(_minimal_benchmark(), config=_default_config())
        with pytest.raises(ValueError, match="per-replicate"):
            builder.add_qa(per_replicate, target_nodes=["ask"])

    def test_rejects_config_override_with_empty_answering_models(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        builder = ScenarioReplayBuilder(_minimal_benchmark(), config=_default_config())
        # parsing_only=True lets VerificationConfig accept an empty
        # answering_models list, so the builder's own guard is the one
        # under test here.
        bad = VerificationConfig(
            answering_models=[],
            parsing_models=[ModelConfig(id="p", model_name="p", model_provider="anthropic")],
            parsing_only=True,
        )
        with pytest.raises(ValueError, match="answering_models"):
            builder.add_qa(_valid_qa_store(), target_nodes=["ask"], config=bad)

    def test_is_chainable(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        builder = ScenarioReplayBuilder(_minimal_benchmark(), config=_default_config())
        chained = builder.add_qa(_valid_qa_store(), target_nodes=["ask"], scenarios=["s1"])
        assert chained is builder

    def test_dedupes_duplicate_nodes_with_warning(self, caplog):
        from karenina.replay.projection import ScenarioReplayBuilder

        builder = ScenarioReplayBuilder(_minimal_benchmark(), config=_default_config())
        with caplog.at_level("WARNING"):
            builder.add_qa(
                _valid_qa_store(),
                target_nodes=["ask", "ask"],
                scenarios=["s1"],
            )
        assert any("duplicate" in rec.message.lower() for rec in caplog.records)

    def test_dedupes_duplicate_scenarios_with_warning(self, caplog):
        from karenina.replay.projection import ScenarioReplayBuilder

        builder = ScenarioReplayBuilder(_minimal_benchmark(), config=_default_config())
        with caplog.at_level("WARNING"):
            builder.add_qa(
                _valid_qa_store(),
                target_nodes=["ask"],
                scenarios=["s1", "s1"],
            )
        assert any("duplicate" in rec.message.lower() for rec in caplog.records)

    def test_snapshots_config_override(self):
        """Mutating the caller's config after add_qa must not leak into staged state."""
        from karenina.replay.projection import ScenarioReplayBuilder

        override = VerificationConfig(
            answering_models=[
                ModelConfig(id="m1", model_name="m1", model_provider="openai"),
            ],
            parsing_models=[ModelConfig(id="p", model_name="p", model_provider="anthropic")],
        )
        builder = ScenarioReplayBuilder(_minimal_benchmark(), config=_default_config())
        builder.add_qa(_valid_qa_store(), target_nodes=["ask"], config=override)

        # Mutate caller's config after staging
        override.answering_models.clear()

        staged = builder._staged  # noqa: SLF001
        assert len(staged[0].config.answering_models) == 1
        assert staged[0].config.answering_models[0].id == "m1"


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


@pytest.mark.unit
class TestBuild:
    def test_produces_scenario_mode_entries_only(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        qa = _qa_store([_qa_entry_for("What is X?", display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa, target_nodes=["ask"], scenarios=["s1"])

        store = builder.build()
        for key, _entry in store.entries:
            assert key.scenario_id is not None
            assert key.scenario_node is not None

    def test_wildcards_visit_index_and_replicate(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        qa = _qa_store([_qa_entry_for("What is X?", display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa, target_nodes=["ask"], scenarios=["s1"])

        store = builder.build()
        for key, _entry in store.entries:
            assert key.visit_index is None
            assert key.replicate is None

    def test_miss_policy_passed_through_default_strict(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        qa = _qa_store([_qa_entry_for("What is X?", display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)  # default strict
        builder.add_qa(qa, target_nodes=["ask"], scenarios=["s1"])
        assert builder.build().miss_policy == "strict"

    def test_miss_policy_passed_through_fall_through(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        qa = _qa_store([_qa_entry_for("What is X?", display)])

        builder = ScenarioReplayBuilder(bench, config=cfg, miss_policy="fall_through")
        builder.add_qa(qa, target_nodes=["ask"], scenarios=["s1"])
        assert builder.build().miss_policy == "fall_through"

    def test_strict_raises_on_unmatched_targets(self):
        from karenina.exceptions import ProjectionError
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        # WRONG question
        qa = _qa_store([_qa_entry_for("Unrelated?", display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa, target_nodes=["ask"], scenarios=["s1"])
        with pytest.raises(ProjectionError) as exc_info:
            builder.build(strict=True)
        assert exc_info.value.report is not None
        assert any(t.reason == "no_qa_entry" for t in exc_info.value.report.unmatched_targets)

    def test_strict_raises_on_duplicate_targets(self):
        from karenina.exceptions import ProjectionError
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        qa_a = _qa_store([_qa_entry_for("What is X?", display)])
        qa_b = _qa_store([_qa_entry_for("What is X?", display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa_a, target_nodes=["ask"], scenarios=["s1"])
        builder.add_qa(qa_b, target_nodes=["ask"], scenarios=["s1"])

        with pytest.raises(ProjectionError) as exc_info:
            builder.build(strict=True)
        assert ("s1", "ask") in exc_info.value.report.duplicate_targets

    def test_non_strict_warns_and_returns_store(self, caplog):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        # WRONG question
        qa = _qa_store([_qa_entry_for("Unrelated?", display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa, target_nodes=["ask"], scenarios=["s1"])
        with caplog.at_level("WARNING"):
            store = builder.build(strict=False)
        assert store is not None
        assert any(
            "unmatched" in rec.message.lower() or "projection report" in rec.message.lower() for rec in caplog.records
        )

    def test_non_strict_last_projection_wins_on_duplicate(self):
        """When two projections target (s1, ask), the LAST staged wins."""
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)

        first_qa = _qa_store(
            [
                (
                    ReplayKey(
                        question_id=generate_question_id("What is X?"),
                        answering_model_id=display,
                    ),
                    ReplayEntry(raw_trace="FIRST"),
                )
            ]
        )
        last_qa = _qa_store(
            [
                (
                    ReplayKey(
                        question_id=generate_question_id("What is X?"),
                        answering_model_id=display,
                    ),
                    ReplayEntry(raw_trace="LAST"),
                )
            ]
        )

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(first_qa, target_nodes=["ask"], scenarios=["s1"])
        builder.add_qa(last_qa, target_nodes=["ask"], scenarios=["s1"])
        store = builder.build(strict=False)

        hit = store.lookup(
            question_id=generate_question_id("What is X?"),
            scenario_id="s1",
            scenario_node="ask",
            answering_model_id=display,
        )
        assert hit is not None
        assert hit.raw_trace == "LAST"

    def test_empty_builder_returns_empty_store_with_warning(self, caplog):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        builder = ScenarioReplayBuilder(bench, config=cfg)
        with caplog.at_level("WARNING"):
            store = builder.build()
        assert list(store.entries) == []
        assert any("empty builder" in rec.message.lower() for rec in caplog.records)

    def test_build_is_idempotent(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        qa = _qa_store([_qa_entry_for("What is X?", display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa, target_nodes=["ask"], scenarios=["s1"])

        s1 = builder.build()
        s2 = builder.build()
        assert len(s1.entries) == len(s2.entries)
        assert s1 is not s2

    def test_projected_entries_preserve_replay_entry_verbatim(self):
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        qa = ReplayStore(miss_policy="fall_through")
        original_entry = ReplayEntry(
            raw_trace="original",
            trace_messages=[{"role": "assistant", "content": "x"}],
            parsed_answer_fields={"v": 1},
            captured_model_id="openai:gpt-5",
            captured_at="2026-04-14T00:00:00Z",
        )
        qa.register(
            ReplayKey(
                question_id=generate_question_id("What is X?"),
                answering_model_id=display,
            ),
            original_entry,
        )

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa, target_nodes=["ask"], scenarios=["s1"])
        store = builder.build()

        # Exactly one entry matches the projected (scenario, node, model)
        projected_entry = store.entries[0][1]
        assert projected_entry.raw_trace == original_entry.raw_trace
        assert projected_entry.trace_messages == original_entry.trace_messages
        assert projected_entry.parsed_answer_fields == original_entry.parsed_answer_fields
        assert projected_entry.captured_model_id == original_entry.captured_model_id
        assert projected_entry.captured_at == original_entry.captured_at


@pytest.mark.unit
class TestPublicAPIReExports:
    def test_builder_available_from_replay(self):
        from karenina.replay import ScenarioReplayBuilder as A
        from karenina.replay.projection import ScenarioReplayBuilder as B

        assert A is B

    def test_report_types_available_from_replay(self):
        from karenina.replay import OrphanEntry, ProjectionReport, UnmatchedTarget
        from karenina.replay.projection import (
            OrphanEntry as _OE,
        )
        from karenina.replay.projection import (
            ProjectionReport as _PR,
        )
        from karenina.replay.projection import (
            UnmatchedTarget as _UT,
        )

        assert OrphanEntry is _OE
        assert ProjectionReport is _PR
        assert UnmatchedTarget is _UT


@pytest.mark.unit
class TestBuildRoundTrip:
    def test_build_then_save_load_preserves_entries(self, tmp_path):
        from karenina.replay import ReplayStore
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        qa = _qa_store(
            [
                _qa_entry_for("What is X?", display),
                _qa_entry_for("What is Y?", display),
            ]
        )

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa, target_nodes=["ask"], scenarios=None)
        store = builder.build(strict=True)

        path = tmp_path / "projection.json"
        store.save(path)
        reloaded = ReplayStore.load(path)
        assert len(reloaded.entries) == len(store.entries)
        original_keys = {(k.scenario_id, k.scenario_node) for (k, _e) in store.entries}
        reloaded_keys = {(k.scenario_id, k.scenario_node) for (k, _e) in reloaded.entries}
        assert original_keys == reloaded_keys


@pytest.mark.integration
class TestProjectionIntegration:
    def test_projected_entry_serves_every_replicate_via_ladder(self):
        """A single projected wildcard (replicate=None) must serve
        every scenario-replicate under R2's executor."""
        from karenina.replay.projection import ScenarioReplayBuilder

        bench = _minimal_benchmark()
        cfg = _default_config()
        display = _answering_display(cfg)
        qa = _qa_store([_qa_entry_for("What is X?", display)])

        builder = ScenarioReplayBuilder(bench, config=cfg)
        builder.add_qa(qa, target_nodes=["ask"], scenarios=["s1"])
        store = builder.build(strict=True)

        qid = generate_question_id("What is X?")
        hits = [
            store.lookup(
                question_id=qid,
                scenario_id="s1",
                scenario_node="ask",
                answering_model_id=display,
                replicate=r,
            )
            for r in (None, 1, 2, 3, 42)
        ]
        # All 5 probes must hit the same single projected wildcard entry.
        assert all(h is not None for h in hits)
        assert all(h.raw_trace == hits[0].raw_trace for h in hits)

    def test_adversarial_style_two_configs_disjoint_scenarios(self):
        """Two add_qa calls with different configs populate disjoint
        (scenario, node) pairs without cross-contamination."""
        from karenina.replay.projection import ScenarioReplayBuilder

        no_mcp_cfg = VerificationConfig(
            answering_models=[
                ModelConfig(id="ans-no-mcp", model_name="gpt-5", model_provider="openai"),
            ],
            parsing_models=[ModelConfig(id="p", model_name="p", model_provider="anthropic")],
        )
        mcp_cfg = VerificationConfig(
            answering_models=[
                ModelConfig(id="ans-mcp", model_name="gpt-5", model_provider="openai"),
            ],
            parsing_models=[ModelConfig(id="p", model_name="p", model_provider="anthropic")],
        )
        no_mcp_display = _answering_display(no_mcp_cfg)
        mcp_display = _answering_display(mcp_cfg)

        bench = _minimal_benchmark()
        no_mcp_qa = _qa_store([_qa_entry_for("What is X?", no_mcp_display)])
        mcp_qa = _qa_store([_qa_entry_for("What is Y?", mcp_display)])

        # fall_through miss policy so the cross-lookup miss returns None
        # instead of raising, which is the shape of the assertion below.
        builder = ScenarioReplayBuilder(bench, config=no_mcp_cfg, miss_policy="fall_through")
        builder.add_qa(no_mcp_qa, target_nodes=["ask"], scenarios=["s1"])
        builder.add_qa(mcp_qa, target_nodes=["ask"], scenarios=["s2"], config=mcp_cfg)
        store = builder.build(strict=True)

        hit_s1 = store.lookup(
            question_id=generate_question_id("What is X?"),
            scenario_id="s1",
            scenario_node="ask",
            answering_model_id=no_mcp_display,
        )
        hit_s2 = store.lookup(
            question_id=generate_question_id("What is Y?"),
            scenario_id="s2",
            scenario_node="ask",
            answering_model_id=mcp_display,
        )
        assert hit_s1 is not None and hit_s1.raw_trace == "canned:What is X?"
        assert hit_s2 is not None and hit_s2.raw_trace == "canned:What is Y?"

        # And cross-lookups must miss (wrong model or wrong scenario).
        assert (
            store.lookup(
                question_id=generate_question_id("What is X?"),
                scenario_id="s2",
                scenario_node="ask",
                answering_model_id=no_mcp_display,
            )
            is None
        )
