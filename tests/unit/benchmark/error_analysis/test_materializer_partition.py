"""Tests for the materializer's partitioning + filename rules."""

from __future__ import annotations

import pytest

from karenina.benchmark.error_analysis.materializer import (
    case_filename,
    partition_results,
    sanitize_id,
)
from karenina.schemas.results.verification_result_set import VerificationResultSet
from karenina.schemas.scenario.state import ScenarioExecutionResult, ScenarioState

from .fixtures import make_metadata, make_pass


def _empty_state(current: str = "done") -> ScenarioState:
    return ScenarioState(
        turn=0,
        current_node=current,
        verify_result=None,
        parsed={},
        node_visits={},
        history=[],
        accumulated={},
        node_results={},
    )


@pytest.mark.unit
class TestSanitize:
    def test_accepts_alphanumerics_and_dashes(self):
        assert sanitize_id("q-foo_bar1") == "q-foo_bar1"

    def test_replaces_other_characters(self):
        assert sanitize_id("q.foo/bar baz") == "q_foo_bar_baz"


@pytest.mark.unit
class TestCaseFilename:
    def test_qa_filename_without_replicate(self):
        md = make_metadata(question_id="q-001")
        assert case_filename(metadata=md).endswith("q_q-001.md")

    def test_qa_filename_with_replicate(self):
        md = make_metadata(question_id="q-001", replicate=3)
        assert case_filename(metadata=md).endswith("q_q-001__rep_3.md")

    def test_qa_filename_collision_uses_hash(self):
        md = make_metadata(question_id="q.001")
        used = {"q_q_001.md"}
        name = case_filename(metadata=md, existing=used)
        assert name.startswith("q_q_001__h")
        assert name.endswith(".md")

    def test_scenario_filename_uses_replicate_when_present(self):
        scenario = ScenarioExecutionResult(
            scenario_id="auth.flow",
            status="completed",
            path=["a"],
            turn_count=1,
            history=[],
            turn_results=[make_pass(question_id="q_x")],
            final_state=_empty_state("a"),
            outcome_results={},
            replicate=5,
        )
        name = case_filename(scenario=scenario, monotonic_n=1)
        assert name.endswith("scenario_auth_flow__run_5.md")

    def test_scenario_filename_uses_monotonic_when_replicate_none(self):
        scenario = ScenarioExecutionResult(
            scenario_id="auth",
            status="completed",
            path=["a"],
            turn_count=1,
            history=[],
            turn_results=[make_pass()],
            final_state=_empty_state("a"),
            outcome_results={},
            replicate=None,
        )
        name = case_filename(scenario=scenario, monotonic_n=7)
        assert name.endswith("scenario_auth__run_7.md")


@pytest.mark.unit
class TestCaseFilenameEdge:
    def test_raises_when_neither_metadata_nor_scenario_given(self):
        with pytest.raises(ValueError, match="metadata or scenario"):
            case_filename()

    def test_double_collision_raises_materialization_error(self):
        from karenina.benchmark.error_analysis.exceptions import MaterializationError

        md = make_metadata(question_id="q.001")
        # Collide on both base and hashed names.
        # First call: compute the actual hashed filename, then pre-populate `existing`.
        existing_base_only = {"q_q_001.md"}
        hashed_name = case_filename(metadata=md, existing=existing_base_only)
        # Now include both in `existing` and expect a raise.
        existing_both = {"q_q_001.md", hashed_name}
        with pytest.raises(MaterializationError):
            case_filename(metadata=md, existing=existing_both)

    def test_hash_is_deterministic_for_same_metadata(self):
        md = make_metadata(question_id="q.001")
        existing = {"q_q_001.md"}
        name1 = case_filename(metadata=md, existing=existing)
        name2 = case_filename(metadata=md, existing=existing)
        assert name1 == name2


@pytest.mark.unit
class TestPartition:
    def test_classical_qa_skips_scenario_turns(self):
        qa = make_pass(question_id="q_qa")
        scn_turn = make_pass(question_id="q_turn")
        scn_turn.metadata = make_metadata(
            question_id="q_turn",
            scenario_id="s1",
            scenario_turn=1,
        )
        scenario = ScenarioExecutionResult(
            scenario_id="s1",
            status="completed",
            path=["a"],
            turn_count=1,
            history=[],
            turn_results=[scn_turn],
            final_state=_empty_state("a"),
            outcome_results={},
            replicate=None,
        )
        rs = VerificationResultSet(results=[qa, scn_turn], scenario_results=[scenario])
        qa_list, scn_list = partition_results(rs)
        assert [r.metadata.question_id for r in qa_list] == ["q_qa"]
        assert scn_list == [scenario]

    def test_missing_scenario_aggregate_raises(self):
        qa = make_pass(question_id="q_qa")
        scn_turn = make_pass(question_id="q_turn")
        scn_turn.metadata = make_metadata(
            question_id="q_turn",
            scenario_id="s1",
            scenario_turn=1,
        )
        rs = VerificationResultSet(results=[qa, scn_turn], scenario_results=None)
        from karenina.exceptions import MaterializationError

        with pytest.raises(MaterializationError):
            partition_results(rs)
