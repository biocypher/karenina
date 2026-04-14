"""Build-phase, round-trip, public re-export, and integration tests for
ScenarioReplayBuilder (R3 projection).

Split from test_projection.py to keep each test module under the
800-line guidance. Covers build() strict/non-strict modes, round-trip
persistence, public API re-exports, and end-to-end integration with
R2's scenario executor.
"""

from __future__ import annotations

import pytest

from karenina.benchmark import Benchmark
from karenina.replay import ReplayEntry, ReplayKey, ReplayStore
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities.question import Question
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode
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
