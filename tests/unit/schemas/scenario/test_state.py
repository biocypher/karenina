"""Tests for scenario state dataclasses."""

import pytest

from karenina.schemas.scenario.state import (
    ScenarioExecutionResult,
    ScenarioState,
    TurnRecord,
)


@pytest.mark.unit
class TestScenarioState:
    def test_initial_state(self):
        state = ScenarioState(
            turn=0,
            current_node="ask",
            verify_result=None,
            parsed={},
            node_visits={},
            history=[],
            accumulated={},
            node_results={},
        )
        assert state.turn == 0
        assert state.current_node == "ask"
        assert state.verify_result is None
        assert state.accumulated == {}
        assert state.node_results == {}

    def test_node_results_auto_populated(self):
        state = ScenarioState(
            turn=1,
            current_node="check",
            verify_result=True,
            parsed={"drug": "venetoclax"},
            node_visits={"ask": 1},
            history=[],
            accumulated={},
            node_results={
                "ask": {
                    "verify_result": True,
                    "parsed": {"drug_name": "venetoclax", "target": "BCL2"},
                    "rubric": {"safety": True, "conciseness": 4},
                },
            },
        )
        assert state.node_results["ask"]["parsed"]["drug_name"] == "venetoclax"
        assert state.node_results["ask"]["verify_result"] is True
        assert state.node_results["ask"]["rubric"]["safety"] is True

    def test_state_is_mutable(self):
        state = ScenarioState(
            turn=0,
            current_node="a",
            verify_result=None,
            parsed={},
            node_visits={},
            history=[],
            accumulated={},
            node_results={},
        )
        state.turn = 1
        state.verify_result = True
        assert state.turn == 1
        assert state.verify_result is True


@pytest.mark.unit
class TestTurnRecord:
    def test_minimal_record(self):
        from karenina.ports.messages import Message

        record = TurnRecord(
            node_id="ask",
            question_text="What is X?",
            question_messages=[Message.user("What is X?")],
            trace_messages=[Message.assistant("It is Y.")],
            raw_response="--- AI Message ---\nIt is Y.",
            parsed_answer=None,
            parsed_fields={},
            verify_result=None,
            verification_result_id=None,
        )
        assert record.node_id == "ask"
        assert record.parsed_answer is None


@pytest.mark.unit
class TestScenarioExecutionResult:
    def test_completed_result(self):
        state = ScenarioState(
            turn=2,
            current_node="done",
            verify_result=True,
            parsed={},
            node_visits={"ask": 1, "done": 1},
            history=[],
            accumulated={},
            node_results={},
        )
        result = ScenarioExecutionResult(
            scenario_id="test",
            status="completed",
            path=["ask", "done"],
            turn_count=2,
            history=[],
            turn_results=[],
            final_state=state,
            outcome_results={"efficiency": True},
        )
        assert result.status == "completed"
        assert result.turn_count == 2
        assert result.outcome_results["efficiency"] is True

    def test_valid_statuses(self):
        for status in ("completed", "limit_reached", "error"):
            state = ScenarioState(
                turn=0,
                current_node="a",
                verify_result=None,
                parsed={},
                node_visits={},
                history=[],
                accumulated={},
                node_results={},
            )
            result = ScenarioExecutionResult(
                scenario_id="t",
                status=status,
                path=[],
                turn_count=0,
                history=[],
                turn_results=[],
                final_state=state,
                outcome_results={},
            )
            assert result.status == status

    def test_replicate_default_none(self):
        state = ScenarioState(
            turn=0,
            current_node="n",
            verify_result=None,
            parsed={},
            node_visits={},
            history=[],
            accumulated={},
            node_results={},
        )
        result = ScenarioExecutionResult(
            scenario_id="s",
            status="completed",
            path=["n"],
            turn_count=1,
            history=[],
            turn_results=[],
            final_state=state,
            outcome_results={},
        )
        assert result.replicate is None

    def test_replicate_set(self):
        state = ScenarioState(
            turn=0,
            current_node="n",
            verify_result=None,
            parsed={},
            node_visits={},
            history=[],
            accumulated={},
            node_results={},
        )
        result = ScenarioExecutionResult(
            scenario_id="s",
            status="completed",
            path=["n"],
            turn_count=1,
            history=[],
            turn_results=[],
            final_state=state,
            outcome_results={},
            replicate=3,
        )
        assert result.replicate == 3
