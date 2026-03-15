"""Tests for outcome criterion evaluation."""

from __future__ import annotations

from typing import Any

import pytest

from karenina.scenario.outcome_evaluation import evaluate_outcome
from karenina.schemas.primitives import (
    AllOf,
    AnyOf,
    BooleanMatch,
    ExactMatch,
    NumericExact,
    NumericRange,
)
from karenina.schemas.primitives.scope import (
    AllTurns,
    AnyTurn,
    FirstTurn,
    LastTurn,
    TurnAt,
)
from karenina.schemas.scenario.checks import (
    CountTurns,
    CrossTurnCheck,
    FirstMatchIndex,
    ResultCheck,
    TurnCheck,
)
from karenina.schemas.scenario.state import (
    ScenarioExecutionResult,
    ScenarioState,
    TurnRecord,
)


def _make_turn(
    node_id: str = "ask",
    verify_result: bool | None = True,
    parsed_fields: dict[str, Any] | None = None,
    raw_response: str = "response text",
) -> TurnRecord:
    return TurnRecord(
        node_id=node_id,
        question_text="What?",
        question_messages=[],
        trace_messages=[],
        raw_response=raw_response,
        parsed_answer=None,
        parsed_fields=parsed_fields or {},
        verify_result=verify_result,
        verification_result_id=None,
    )


def _make_result(
    history: list[TurnRecord] | None = None,
    status: str = "completed",
    path: list[str] | None = None,
) -> ScenarioExecutionResult:
    history = history or []
    path = path or [t.node_id for t in history]
    return ScenarioExecutionResult(
        scenario_id="test-scenario",
        status=status,
        path=path,
        turn_count=len(history),
        history=history,
        turn_results=[],
        final_state=ScenarioState(
            turn=len(history),
            current_node=path[-1] if path else "",
            verify_result=None,
            parsed={},
            node_visits={},
            history=history,
            accumulated={},
            node_results={},
        ),
        outcome_results={},
    )


# ---------- TurnCheck ----------


@pytest.mark.unit
class TestTurnCheckEvaluation:
    def test_last_turn_verify_result(self) -> None:
        check = TurnCheck(
            scope=LastTurn(),
            field="verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        )
        result = _make_result(
            history=[
                _make_turn(verify_result=False),
                _make_turn(verify_result=True),
            ]
        )
        assert evaluate_outcome(check, result) is True

    def test_first_turn_node_id(self) -> None:
        check = TurnCheck(
            scope=FirstTurn(),
            field="node_id",
            expected="ask",
            verify_with=ExactMatch(),
        )
        result = _make_result(
            history=[
                _make_turn(node_id="ask"),
                _make_turn(node_id="confirm"),
            ]
        )
        assert evaluate_outcome(check, result) is True

    def test_turn_at_index(self) -> None:
        check = TurnCheck(
            scope=TurnAt(index=1),
            field="verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        )
        result = _make_result(
            history=[
                _make_turn(verify_result=False),
                _make_turn(verify_result=True),
                _make_turn(verify_result=False),
            ]
        )
        assert evaluate_outcome(check, result) is True

    def test_turn_at_negative_index(self) -> None:
        check = TurnCheck(
            scope=TurnAt(index=-1),
            field="verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        )
        result = _make_result(
            history=[
                _make_turn(verify_result=False),
                _make_turn(verify_result=True),
            ]
        )
        assert evaluate_outcome(check, result) is True

    def test_any_turn_quantifier(self) -> None:
        check = TurnCheck(
            scope=AnyTurn(),
            field="verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        )
        result = _make_result(
            history=[
                _make_turn(verify_result=False),
                _make_turn(verify_result=True),
                _make_turn(verify_result=False),
            ]
        )
        assert evaluate_outcome(check, result) is True

    def test_any_turn_no_match(self) -> None:
        check = TurnCheck(
            scope=AnyTurn(),
            field="verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        )
        result = _make_result(
            history=[
                _make_turn(verify_result=False),
                _make_turn(verify_result=False),
            ]
        )
        assert evaluate_outcome(check, result) is False

    def test_any_turn_with_node_filter(self) -> None:
        check = TurnCheck(
            scope=AnyTurn(node_id="confirm"),
            field="verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        )
        result = _make_result(
            history=[
                _make_turn(node_id="ask", verify_result=True),
                _make_turn(node_id="confirm", verify_result=False),
            ]
        )
        # Only "confirm" turns considered; the one confirm turn is False
        assert evaluate_outcome(check, result) is False

    def test_all_turns_quantifier(self) -> None:
        check = TurnCheck(
            scope=AllTurns(),
            field="verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        )
        result = _make_result(
            history=[
                _make_turn(verify_result=True),
                _make_turn(verify_result=True),
            ]
        )
        assert evaluate_outcome(check, result) is True

    def test_all_turns_one_fails(self) -> None:
        check = TurnCheck(
            scope=AllTurns(),
            field="verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        )
        result = _make_result(
            history=[
                _make_turn(verify_result=True),
                _make_turn(verify_result=False),
            ]
        )
        assert evaluate_outcome(check, result) is False

    def test_all_turns_empty_returns_false(self) -> None:
        check = TurnCheck(
            scope=AllTurns(),
            field="verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        )
        result = _make_result(history=[])
        assert evaluate_outcome(check, result) is False

    def test_parsed_field_access(self) -> None:
        check = TurnCheck(
            scope=LastTurn(),
            field="parsed.drug",
            expected="venetoclax",
            verify_with=ExactMatch(),
        )
        result = _make_result(
            history=[
                _make_turn(parsed_fields={"drug": "venetoclax"}),
            ]
        )
        assert evaluate_outcome(check, result) is True

    def test_out_of_range_turn_at_returns_false(self) -> None:
        check = TurnCheck(
            scope=TurnAt(index=99),
            field="verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        )
        result = _make_result(history=[_make_turn()])
        assert evaluate_outcome(check, result) is False


# ---------- ResultCheck ----------


@pytest.mark.unit
class TestResultCheckEvaluation:
    def test_status_check(self) -> None:
        check = ResultCheck(
            field="status",
            expected="completed",
            verify_with=ExactMatch(),
        )
        result = _make_result(status="completed")
        assert evaluate_outcome(check, result) is True

    def test_status_check_fail(self) -> None:
        check = ResultCheck(
            field="status",
            expected="completed",
            verify_with=ExactMatch(),
        )
        result = _make_result(status="error")
        assert evaluate_outcome(check, result) is False

    def test_turn_count_check(self) -> None:
        check = ResultCheck(
            field="turn_count",
            verify_with=NumericRange(min=1, max=3),
        )
        result = _make_result(history=[_make_turn(), _make_turn()])
        assert evaluate_outcome(check, result) is True

    def test_turn_count_exact(self) -> None:
        check = ResultCheck(
            field="turn_count",
            expected=2,
            verify_with=NumericExact(),
        )
        result = _make_result(history=[_make_turn(), _make_turn()])
        assert evaluate_outcome(check, result) is True

    def test_scenario_id_check(self) -> None:
        check = ResultCheck(
            field="scenario_id",
            expected="test-scenario",
            verify_with=ExactMatch(),
        )
        result = _make_result()
        assert evaluate_outcome(check, result) is True


# ---------- CrossTurnCheck ----------


@pytest.mark.unit
class TestCrossTurnCheckEvaluation:
    def test_eq_comparison(self) -> None:
        check = CrossTurnCheck(
            source_turn=FirstTurn(),
            source_field="parsed.drug",
            target_turn=LastTurn(),
            target_field="parsed.drug",
            comparison="eq",
        )
        result = _make_result(
            history=[
                _make_turn(parsed_fields={"drug": "venetoclax"}),
                _make_turn(parsed_fields={"drug": "venetoclax"}),
            ]
        )
        assert evaluate_outcome(check, result) is True

    def test_neq_comparison(self) -> None:
        check = CrossTurnCheck(
            source_turn=FirstTurn(),
            source_field="parsed.drug",
            target_turn=LastTurn(),
            target_field="parsed.drug",
            comparison="neq",
        )
        result = _make_result(
            history=[
                _make_turn(parsed_fields={"drug": "venetoclax"}),
                _make_turn(parsed_fields={"drug": "ibrutinib"}),
            ]
        )
        assert evaluate_outcome(check, result) is True

    def test_gt_comparison(self) -> None:
        check = CrossTurnCheck(
            source_turn=FirstTurn(),
            source_field="parsed.score",
            target_turn=LastTurn(),
            target_field="parsed.score",
            comparison="gt",
        )
        result = _make_result(
            history=[
                _make_turn(parsed_fields={"score": 3}),
                _make_turn(parsed_fields={"score": 5}),
            ]
        )
        # target (5) > source (3) -> True
        assert evaluate_outcome(check, result) is True

    def test_contains_comparison(self) -> None:
        check = CrossTurnCheck(
            source_turn=FirstTurn(),
            source_field="parsed.drug",
            target_turn=LastTurn(),
            target_field="raw_response",
            comparison="contains",
        )
        result = _make_result(
            history=[
                _make_turn(parsed_fields={"drug": "venetoclax"}),
                _make_turn(raw_response="The drug venetoclax is used for CLL."),
            ]
        )
        # target (raw_response) contains source (drug)
        assert evaluate_outcome(check, result) is True

    def test_cross_turn_with_normalize(self) -> None:
        check = CrossTurnCheck(
            source_turn=FirstTurn(),
            source_field="parsed.drug",
            target_turn=LastTurn(),
            target_field="parsed.drug",
            comparison="eq",
            normalize=["lowercase", "strip"],
        )
        result = _make_result(
            history=[
                _make_turn(parsed_fields={"drug": "  Venetoclax  "}),
                _make_turn(parsed_fields={"drug": "venetoclax"}),
            ]
        )
        assert evaluate_outcome(check, result) is True


# ---------- CountTurns ----------


@pytest.mark.unit
class TestCountTurnsEvaluation:
    def test_count_all_turns(self) -> None:
        check = CountTurns()
        result = _make_result(history=[_make_turn(), _make_turn(), _make_turn()])
        assert evaluate_outcome(check, result) == 3

    def test_count_by_node_id(self) -> None:
        check = CountTurns(node_id="ask")
        result = _make_result(
            history=[
                _make_turn(node_id="ask"),
                _make_turn(node_id="confirm"),
                _make_turn(node_id="ask"),
            ]
        )
        assert evaluate_outcome(check, result) == 2

    def test_count_by_verify_result(self) -> None:
        check = CountTurns(verify_result=True)
        result = _make_result(
            history=[
                _make_turn(verify_result=True),
                _make_turn(verify_result=False),
                _make_turn(verify_result=True),
            ]
        )
        assert evaluate_outcome(check, result) == 2

    def test_count_by_node_id_list(self) -> None:
        check = CountTurns(node_id=["ask", "retry"])
        result = _make_result(
            history=[
                _make_turn(node_id="ask"),
                _make_turn(node_id="confirm"),
                _make_turn(node_id="retry"),
            ]
        )
        assert evaluate_outcome(check, result) == 2

    def test_count_combined_filters(self) -> None:
        check = CountTurns(node_id="ask", verify_result=True)
        result = _make_result(
            history=[
                _make_turn(node_id="ask", verify_result=True),
                _make_turn(node_id="ask", verify_result=False),
                _make_turn(node_id="confirm", verify_result=True),
            ]
        )
        assert evaluate_outcome(check, result) == 1


# ---------- FirstMatchIndex ----------


@pytest.mark.unit
class TestFirstMatchIndexEvaluation:
    def test_first_match(self) -> None:
        check = FirstMatchIndex(verify_result=True)
        result = _make_result(
            history=[
                _make_turn(verify_result=False),
                _make_turn(verify_result=True),
                _make_turn(verify_result=True),
            ]
        )
        assert evaluate_outcome(check, result) == 1

    def test_no_match_returns_negative_one(self) -> None:
        check = FirstMatchIndex(verify_result=True)
        result = _make_result(
            history=[
                _make_turn(verify_result=False),
                _make_turn(verify_result=False),
            ]
        )
        assert evaluate_outcome(check, result) == -1

    def test_first_match_by_node_id(self) -> None:
        check = FirstMatchIndex(node_id="confirm")
        result = _make_result(
            history=[
                _make_turn(node_id="ask"),
                _make_turn(node_id="ask"),
                _make_turn(node_id="confirm"),
            ]
        )
        assert evaluate_outcome(check, result) == 2


# ---------- Composition ----------


@pytest.mark.unit
class TestCompositionEvaluation:
    def test_all_of(self) -> None:
        node = AllOf(
            conditions=[
                ResultCheck(field="status", expected="completed", verify_with=ExactMatch()),
                TurnCheck(
                    scope=LastTurn(),
                    field="verify_result",
                    expected=True,
                    verify_with=BooleanMatch(),
                ),
            ]
        )
        result = _make_result(
            status="completed",
            history=[_make_turn(verify_result=True)],
        )
        assert evaluate_outcome(node, result) is True

    def test_all_of_one_fails(self) -> None:
        node = AllOf(
            conditions=[
                ResultCheck(field="status", expected="completed", verify_with=ExactMatch()),
                TurnCheck(
                    scope=LastTurn(),
                    field="verify_result",
                    expected=True,
                    verify_with=BooleanMatch(),
                ),
            ]
        )
        result = _make_result(
            status="error",
            history=[_make_turn(verify_result=True)],
        )
        assert evaluate_outcome(node, result) is False

    def test_any_of(self) -> None:
        node = AnyOf(
            conditions=[
                ResultCheck(field="status", expected="completed", verify_with=ExactMatch()),
                ResultCheck(field="status", expected="limit_reached", verify_with=ExactMatch()),
            ]
        )
        result = _make_result(status="limit_reached")
        assert evaluate_outcome(node, result) is True

    def test_nested_composition(self) -> None:
        node = AllOf(
            conditions=[
                ResultCheck(field="status", expected="completed", verify_with=ExactMatch()),
                AnyOf(
                    conditions=[
                        TurnCheck(
                            scope=LastTurn(),
                            field="verify_result",
                            expected=True,
                            verify_with=BooleanMatch(),
                        ),
                        TurnCheck(
                            scope=FirstTurn(),
                            field="verify_result",
                            expected=True,
                            verify_with=BooleanMatch(),
                        ),
                    ]
                ),
            ]
        )
        result = _make_result(
            status="completed",
            history=[
                _make_turn(verify_result=False),
                _make_turn(verify_result=True),
            ],
        )
        assert evaluate_outcome(node, result) is True
