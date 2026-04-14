"""Unit + integration tests for ScenarioReplayBuilder (R3 projection)."""

from __future__ import annotations

import pytest

from karenina.replay import ReplayKey


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
