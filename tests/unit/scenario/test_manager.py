"""Tests for ScenarioManager core turn loop and helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from karenina.scenario.manager import (
    ScenarioManager,
    _evaluate_outcome_criteria,
    _resolve_models,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.state import ScenarioExecutionResult, ScenarioState
from karenina.schemas.scenario.types import (
    ModelOverride,
    ScenarioNode,
    ScenarioOutcomeCriterion,
)

if TYPE_CHECKING:
    from karenina.schemas.entities import Question


def _make_question(text: str = "What?") -> Question:
    from karenina.schemas.entities import Question

    return Question(question=text, raw_answer="Y", answer_template="class Answer: pass")


def _make_model(
    name: str = "claude",
    provider: str = "anthropic",
    model_id: str | None = None,
) -> ModelConfig:
    return ModelConfig(id=model_id or name, model_name=name, model_provider=provider)


# ---------------------------------------------------------------------------
# ScenarioManager instantiation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScenarioManagerInit:
    def test_creates_manager(self):
        manager = ScenarioManager()
        assert manager is not None


# ---------------------------------------------------------------------------
# _resolve_models
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelResolution:
    def test_no_override_uses_base(self):
        q = _make_question()
        node = ScenarioNode(node_id="ask", question=q)
        base_ans = _make_model("claude")
        base_parse = _make_model("claude")
        ans, parse = _resolve_models(node, base_ans, base_parse)
        assert ans.model_name == "claude"
        assert parse.model_name == "claude"

    def test_override_answering(self):
        q = _make_question()
        override = ModelOverride(
            answering_model=_make_model("gpt-4o", "openai"),
        )
        node = ScenarioNode(node_id="ask", question=q, model_override=override)
        base_ans = _make_model("claude")
        base_parse = _make_model("claude")
        ans, parse = _resolve_models(node, base_ans, base_parse)
        assert ans.model_name == "gpt-4o"
        assert parse.model_name == "claude"

    def test_override_parsing(self):
        q = _make_question()
        override = ModelOverride(
            parsing_model=_make_model("gpt-4o-mini", "openai"),
        )
        node = ScenarioNode(node_id="ask", question=q, model_override=override)
        base_ans = _make_model("claude")
        base_parse = _make_model("claude")
        ans, parse = _resolve_models(node, base_ans, base_parse)
        assert ans.model_name == "claude"
        assert parse.model_name == "gpt-4o-mini"

    def test_override_both(self):
        q = _make_question()
        override = ModelOverride(
            answering_model=_make_model("gpt-4o", "openai"),
            parsing_model=_make_model("gpt-4o-mini", "openai"),
        )
        node = ScenarioNode(node_id="ask", question=q, model_override=override)
        base_ans = _make_model("claude")
        base_parse = _make_model("claude")
        ans, parse = _resolve_models(node, base_ans, base_parse)
        assert ans.model_name == "gpt-4o"
        assert parse.model_name == "gpt-4o-mini"

    def test_empty_override_object_uses_base(self):
        q = _make_question()
        override = ModelOverride()  # neither field set
        node = ScenarioNode(node_id="ask", question=q, model_override=override)
        base_ans = _make_model("claude")
        base_parse = _make_model("claude")
        ans, parse = _resolve_models(node, base_ans, base_parse)
        assert ans.model_name == "claude"
        assert parse.model_name == "claude"


# ---------------------------------------------------------------------------
# _evaluate_outcome_criteria
# ---------------------------------------------------------------------------


def _make_execution_result() -> ScenarioExecutionResult:
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
    return ScenarioExecutionResult(
        scenario_id="test",
        status="completed",
        path=["ask", "done"],
        turn_count=2,
        history=[],
        turn_results=[],
        final_state=state,
        outcome_results={},
    )


def _make_scenario_with_criteria(
    criteria: list[ScenarioOutcomeCriterion],
) -> ScenarioDefinition:
    q = _make_question()
    return ScenarioDefinition(
        name="test",
        nodes={"ask": ScenarioNode(node_id="ask", question=q)},
        edges=[],
        entry_node="ask",
        outcome_criteria=criteria,
    )


@pytest.mark.unit
class TestEvaluateOutcomeCriteria:
    def test_callable_criterion_true(self):
        criterion = ScenarioOutcomeCriterion(
            name="fast",
            description="Done quickly?",
            evaluate=lambda r: r.turn_count <= 3,
        )
        scenario = _make_scenario_with_criteria([criterion])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes["fast"] is True

    def test_callable_criterion_false(self):
        criterion = ScenarioOutcomeCriterion(
            name="slow",
            description="Took more than 5 turns?",
            evaluate=lambda r: r.turn_count > 5,
        )
        scenario = _make_scenario_with_criteria([criterion])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes["slow"] is False

    def test_callable_exception_returns_false(self):
        criterion = ScenarioOutcomeCriterion(
            name="broken",
            description="Always fails",
            evaluate=lambda _r: 1 / 0,
        )
        scenario = _make_scenario_with_criteria([criterion])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes["broken"] is False

    def test_declarative_check_dispatches_to_evaluate_outcome(self):
        """Declarative check (criterion.check is not None) dispatches to evaluate_outcome."""
        from karenina.schemas.primitives import ExactMatch
        from karenina.schemas.scenario.checks import ResultCheck

        check_node = ResultCheck(
            field="status",
            expected="completed",
            verify_with=ExactMatch(),
        )
        criterion = ScenarioOutcomeCriterion(
            name="completed_status",
            description="Scenario completed successfully",
            check=check_node,
        )
        scenario = _make_scenario_with_criteria([criterion])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes["completed_status"] is True

    def test_declarative_check_false_result(self):
        """Declarative check returns False when condition is not met."""
        from karenina.schemas.primitives import ExactMatch
        from karenina.schemas.scenario.checks import ResultCheck

        check_node = ResultCheck(
            field="status",
            expected="error",
            verify_with=ExactMatch(),
        )
        criterion = ScenarioOutcomeCriterion(
            name="errored",
            description="Scenario errored",
            check=check_node,
        )
        scenario = _make_scenario_with_criteria([criterion])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes["errored"] is False

    def test_multiple_criteria(self):
        criteria = [
            ScenarioOutcomeCriterion(
                name="fast",
                description="Done quickly?",
                evaluate=lambda r: r.turn_count <= 3,
            ),
            ScenarioOutcomeCriterion(
                name="correct_path",
                description="Correct path?",
                evaluate=lambda r: "ask" in r.path,
            ),
        ]
        scenario = _make_scenario_with_criteria(criteria)
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes["fast"] is True
        assert outcomes["correct_path"] is True

    def test_no_criteria_returns_empty(self):
        scenario = _make_scenario_with_criteria([])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes == {}

    def test_criterion_with_neither_check_nor_evaluate(self):
        """Criterion with neither check nor evaluate logs a warning."""
        criterion = ScenarioOutcomeCriterion(
            name="empty",
            description="No evaluation defined",
        )
        scenario = _make_scenario_with_criteria([criterion])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert "empty" not in outcomes


# ---------------------------------------------------------------------------
# ScenarioManager._report_progress
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReportProgress:
    def test_callback_invoked_with_kwargs(self):
        captured = {}

        def cb(**kwargs):
            captured.update(kwargs)

        ScenarioManager._report_progress(
            cb,
            "my-scenario",
            0,
            "ask",
            True,
            "done",
        )
        assert captured["scenario_id"] == "my-scenario"
        assert captured["scenario_turn"] == 0
        assert captured["scenario_node"] == "ask"
        assert captured["verify_result"] is True
        assert captured["next_node"] == "done"

    def test_none_callback_is_noop(self):
        # Should not raise
        ScenarioManager._report_progress(
            None,
            "my-scenario",
            0,
            "ask",
            True,
            "done",
        )

    def test_callback_exception_is_swallowed(self):
        def bad_cb(**kwargs):
            raise RuntimeError("boom")

        # Should not raise
        ScenarioManager._report_progress(
            bad_cb,
            "my-scenario",
            0,
            "ask",
            True,
            None,
        )
