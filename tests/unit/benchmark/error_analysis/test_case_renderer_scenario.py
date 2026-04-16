"""Tests for scenario case rendering."""

from __future__ import annotations

import pytest
import yaml  # type: ignore[import-untyped]

from karenina.benchmark.error_analysis.case_renderer import render_scenario_case
from karenina.schemas.results.failure import FailureCategory
from karenina.schemas.scenario.state import ScenarioExecutionResult, ScenarioState

from .fixtures import make_failure, make_metadata, make_pass


def _split_frontmatter(body: str) -> tuple[dict, str]:
    assert body.startswith("---\n")
    _, fm, rest = body.split("---\n", 2)
    return yaml.safe_load(fm), rest


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


def _scenario_turn(question_id: str, turn: int, path: list[str], failing: bool = False):
    if failing:
        r = make_failure(
            question_id=question_id,
            category=FailureCategory.CONTENT,
            stage="verify_template",
            reason="wrong answer",
        )
    else:
        r = make_pass(question_id=question_id)
    # Attach scenario metadata.
    r.metadata = make_metadata(
        question_id=question_id,
        scenario_id="auth_flow",
        scenario_turn=turn,
        scenario_node=path[turn - 1],
        scenario_path=path,
        failure=r.metadata.failure,
    )
    return r


@pytest.mark.unit
class TestScenarioRender:
    def test_scenario_pass_frontmatter(self):
        path = ["intro", "clarify", "verify"]
        execution = ScenarioExecutionResult(
            scenario_id="auth_flow",
            status="completed",
            path=path,
            turn_count=3,
            history=[],
            turn_results=[_scenario_turn(f"q_turn_{i + 1}", i + 1, path) for i in range(3)],
            final_state=_empty_state(path[-1]),
            outcome_results={"completed": True},
            replicate=1,
        )
        body = render_scenario_case(
            execution,
            template_sources={"q_turn_1": None, "q_turn_2": None, "q_turn_3": None},
            template_links={},
            artifacts_dir=None,
        )
        fm, rest = _split_frontmatter(body)
        assert fm["outcome"] == "pass"
        assert fm["path"] == path
        assert fm["turn_count"] == 3
        assert fm["failed_turn"] is None
        assert fm["outcome_results"] == {"completed": True}
        assert "## Turn 1" in rest
        assert "## Turn 3" in rest

    def test_scenario_first_failing_turn_marked(self):
        path = ["intro", "act", "verify"]
        turns = [
            _scenario_turn("q_turn_1", 1, path),
            _scenario_turn("q_turn_2", 2, path, failing=True),
            _scenario_turn("q_turn_3", 3, path),
        ]
        execution = ScenarioExecutionResult(
            scenario_id="auth_flow",
            status="completed",
            path=path,
            turn_count=3,
            history=[],
            turn_results=turns,
            final_state=_empty_state(path[-1]),
            outcome_results={"completed": False},
            replicate=None,
        )
        body = render_scenario_case(
            execution,
            template_sources={},
            template_links={},
            artifacts_dir=None,
        )
        fm, rest = _split_frontmatter(body)
        assert fm["outcome"] == "failure"
        assert fm["category"] == "content"
        assert fm["group"] == "content"
        assert fm["failed_turn"] == 2
        assert "# Outcomes" in rest
        assert "completed" in rest
