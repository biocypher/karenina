"""Tests for scenario state dataclasses."""

from unittest.mock import MagicMock

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

    def test_to_replay_store_threads_self_replicate(self, monkeypatch):
        """to_replay_store forwards self.replicate to capture_from_scenario_result by default."""
        from karenina.replay import capture as capture_mod

        captured: dict[str, object] = {}

        def fake_capture(result, *, answering_model_id, **kwargs):
            captured["replicate"] = kwargs.get("replicate")
            captured["answering_model_id"] = answering_model_id
            return MagicMock()  # dummy ReplayStore; we only inspect captured kwargs

        monkeypatch.setattr(capture_mod, "capture_from_scenario_result", fake_capture, raising=True)

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
            replicate=7,
        )
        result.to_replay_store(answering_model_id="m")

        assert captured["replicate"] == 7
        assert captured["answering_model_id"] == "m"

    def test_to_replay_store_explicit_replicate_overrides_self(self, monkeypatch):
        """Caller's explicit replicate kwarg wins over self.replicate."""
        from karenina.replay import capture as capture_mod

        captured: dict[str, object] = {}

        def fake_capture(result, *, answering_model_id, **kwargs):
            captured["replicate"] = kwargs.get("replicate")
            return MagicMock()

        monkeypatch.setattr(capture_mod, "capture_from_scenario_result", fake_capture, raising=True)

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
            replicate=7,
        )
        result.to_replay_store(answering_model_id="m", replicate=42)

        assert captured["replicate"] == 42
