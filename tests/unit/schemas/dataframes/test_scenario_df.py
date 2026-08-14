"""Unit tests for scenario result DataFrame views."""

from __future__ import annotations

import pytest

from karenina.schemas.dataframes.scenario import (
    OUTCOME_COLUMNS,
    SCENARIO_COLUMNS,
    TURN_COLUMNS,
    ScenarioDataFrameBuilder,
)
from karenina.schemas.results import ScenarioResultRecord, ScenarioTurnRecord, VerificationResultSet
from karenina.schemas.scenario.state import (
    ScenarioExecutionResult,
    ScenarioState,
    ScenarioTerminalFailure,
    TurnRecord,
)


def _live_result(*, failed: bool = False) -> ScenarioExecutionResult:
    """Build a representative live scenario execution result."""
    turn = TurnRecord(
        node_id="ask",
        question_text="Question?",
        question_messages=[],
        trace_messages=[],
        raw_response="Answer.",
        parsed_answer=None,
        parsed_fields={"answer": "Answer"},
        verify_result=True,
        verification_result_id="result-1",
    )
    state = ScenarioState(
        turn=1,
        current_node="ask",
        verify_result=True,
        parsed={"answer": "Answer"},
        node_visits={"ask": 1},
        history=[turn],
        accumulated={},
        node_results={},
    )
    return ScenarioExecutionResult(
        scenario_id="scenario-1",
        status="error" if failed else "completed",
        path=["ask"],
        turn_count=1,
        history=[turn],
        turn_results=[],
        final_state=state,
        outcome_results={"initial_correct": True, "attempts": 1, "score": 0.75},
        terminal_failure=(
            ScenarioTerminalFailure(
                node_id="ask",
                category="execution",
                stage="answer",
                reason="failed",
            )
            if failed
            else None
        ),
        replicate=2,
    )


@pytest.mark.unit
class TestScenarioDataFrameBuilder:
    """Cover scenario, turn, and outcome DataFrame construction."""

    def test_builds_scenario_row_with_dynamic_outcomes(self) -> None:
        frame = ScenarioDataFrameBuilder([_live_result()]).build_scenario_dataframe()

        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["scenario_id"] == "scenario-1"
        assert row["scenario_path"] == "ask"
        assert row["outcome_initial_correct"] == True  # noqa: E712
        assert row["outcome_attempts"] == 1
        assert row["outcome_score"] == 0.75
        assert str(frame["replicate"].dtype) == "Int64"

    def test_builds_turn_row(self) -> None:
        frame = ScenarioDataFrameBuilder([_live_result()]).build_turn_dataframe()

        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["scenario_turn"] == 0
        assert row["node_id"] == "ask"
        assert row["parsed_fields"] == {"answer": "Answer"}
        assert row["verification_result_id"] == "result-1"

    def test_builds_typed_outcome_rows(self) -> None:
        frame = ScenarioDataFrameBuilder([_live_result()]).build_outcome_dataframe()
        types = dict(zip(frame["outcome_name"], frame["outcome_type"], strict=True))

        assert types == {
            "attempts": "integer",
            "initial_correct": "boolean",
            "score": "number",
        }

    def test_exports_terminal_failure(self) -> None:
        frame = ScenarioDataFrameBuilder([_live_result(failed=True)]).build_scenario_dataframe()
        row = frame.iloc[0]

        assert row["terminal_failure_node"] == "ask"
        assert row["terminal_failure_category"] == "execution"
        assert row["terminal_failure_stage"] == "answer"
        assert row["terminal_failure_reason"] == "failed"

    def test_accepts_validated_saved_record(self) -> None:
        record = ScenarioResultRecord(
            scenario_id="saved-1",
            status="completed",
            path=["ask", "correction"],
            turn_count=2,
            history=[
                ScenarioTurnRecord(
                    node_id="ask",
                    question_text="Question?",
                    raw_response="No answer.",
                    parsed_fields={},
                    verify_result=False,
                    verification_result_id="saved-result",
                )
            ],
            outcome_results={"self_corrects": False},
        )

        frame = ScenarioDataFrameBuilder([record]).build_scenario_dataframe()
        assert frame.iloc[0]["scenario_path"] == "ask->correction"
        assert frame.iloc[0]["outcome_self_corrects"] == False  # noqa: E712

    def test_empty_frames_have_stable_schemas(self) -> None:
        builder = ScenarioDataFrameBuilder([])

        assert list(builder.build_scenario_dataframe().columns) == SCENARIO_COLUMNS
        assert list(builder.build_turn_dataframe().columns) == TURN_COLUMNS
        assert list(builder.build_outcome_dataframe().columns) == OUTCOME_COLUMNS


@pytest.mark.unit
class TestScenarioResultsView:
    """Cover the VerificationResultSet scenario accessor."""

    def test_accessor_exposes_all_dataframe_views(self) -> None:
        result_set = VerificationResultSet(results=[], scenario_results=[_live_result()])
        scenarios = result_set.get_scenario_results()

        assert len(scenarios.to_dataframe()) == 1
        assert len(scenarios.to_turn_dataframe()) == 1
        assert len(scenarios.to_outcome_dataframe()) == 3
