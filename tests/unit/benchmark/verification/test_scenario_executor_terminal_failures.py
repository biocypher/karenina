"""Tests surfacing scenario terminal failures in ScenarioExecutor errors.

ScenarioManager converts terminal pipeline failures (e.g. an answering
adapter auth error exhausted after retries) into in-band results with
``status='error'`` instead of raising. These tests verify that
``ScenarioExecutor.run_batch`` records such results in its returned
error list so callers branching on ``VerificationResultSet.errors``
see per-scenario failures, not only raised combo exceptions.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from karenina.benchmark.verification.scenario_executor import (
    ScenarioExecutionFailure,
    ScenarioExecutor,
    ScenarioExecutorConfig,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.results.failure import Failure, FailureCategory


def _make_scenario(name: str = "failing_scenario") -> MagicMock:
    """Minimal single-node scenario definition (nodes/edges unneeded: the
    turn pipeline is stubbed, only the definition's name is read)."""
    from karenina.schemas.entities import Question
    from karenina.schemas.scenario.definition import ScenarioDefinition
    from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode

    question = Question(question="what?", raw_answer="y", answer_template="class Answer: pass")
    return ScenarioDefinition(
        name=name,
        nodes={"n1": ScenarioNode(node_id="n1", question=question)},
        edges=[ScenarioEdge(source="n1", target=END)],
        entry_node="n1",
    )


def _make_config() -> MagicMock:
    config = MagicMock()
    config.replay_store = None
    config.replicate_count = 1
    config.request_timeout = None
    config.evaluation_mode = "template_only"
    config.scenario_turn_limit = 5
    config.custom_error_patterns = None
    config.use_full_trace_for_template = False
    config.use_full_trace_for_rubric = False
    config.workspace_output_mode = "none"
    config.workspace_output_dir = None
    return config


def _make_combo(scenario_name: str) -> tuple:
    scenario = _make_scenario(scenario_name)
    return (
        scenario,
        ModelConfig(id="ans", model_name="ans", model_provider="anthropic"),
        ModelConfig(id="parse", model_name="parse", model_provider="anthropic"),
        None,
    )


def _stub_failing_turn(
    monkeypatch: pytest.MonkeyPatch,
    failure: Failure | None,
    scenario_ids: set[str] | None = None,
) -> None:
    """Stub ScenarioManager._run_turn to return the given pipeline failure.

    When ``scenario_ids`` is given, only turns for those scenarios get the
    failure; other scenarios complete without one.
    """
    from karenina.scenario import manager as mgr_mod

    def fake_run_turn(self, **kwargs):
        effective = failure
        if scenario_ids is not None and kwargs.get("scenario_id") not in scenario_ids:
            effective = None
        vr = MagicMock()
        vr.metadata.failure = effective
        vr.metadata.replicate = kwargs.get("replicate")
        vr.metadata.result_id = "rid_stub"
        vr.template.verify_result = False
        vr.rubric = None
        return (vr, [], None, None)

    monkeypatch.setattr(mgr_mod.ScenarioManager, "_run_turn", fake_run_turn, raising=True)


@pytest.mark.unit
class TestTerminalFailureErrorSurfacing:
    """run_batch records in-band status='error' results in the error list."""

    def test_terminal_failure_result_appears_in_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A turn-level adapter failure (status='error' result) is reported
        as an error entry carrying the scenario id, node, and reason."""
        auth_failure = Failure(
            category=FailureCategory.CONNECTION,
            stage="generate_answer",
            reason="AuthenticationError: 401 invalid api key",
        )
        _stub_failing_turn(monkeypatch, auth_failure)

        executor = ScenarioExecutor(parallel=False, config=ScenarioExecutorConfig(enable_cache=False))
        results, errors = executor.run_batch([_make_combo("auth_fail_scenario")], _make_config())

        assert len(results) == 1
        assert results[0].status == "error"
        assert len(errors) == 1
        desc, exc = errors[0]
        assert isinstance(exc, ScenarioExecutionFailure)
        for needle in ("auth_fail_scenario", "n1", "401 invalid api key"):
            assert needle in desc
            assert needle in str(exc)

    def test_terminal_failure_and_raised_combo_both_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An in-band error result and a raised combo exception are both
        present, and completed combos contribute no error entries."""
        timeout_failure = Failure(
            category=FailureCategory.TIMEOUT,
            stage="generate_answer",
            reason="timed out after 120s",
        )
        _stub_failing_turn(monkeypatch, timeout_failure, scenario_ids={"slow_scenario"})

        from karenina.scenario import manager as mgr_mod

        real_run = mgr_mod.ScenarioManager.run

        def run_or_raise(self, scenario, **kwargs):
            if scenario.name == "boom":
                raise RuntimeError("scenario explosion")
            return real_run(self, scenario=scenario, **kwargs)

        monkeypatch.setattr(mgr_mod.ScenarioManager, "run", run_or_raise, raising=True)

        executor = ScenarioExecutor(parallel=False, config=ScenarioExecutorConfig(enable_cache=False))
        results, errors = executor.run_batch(
            [_make_combo("ok_scenario"), _make_combo("boom"), _make_combo("slow_scenario")],
            _make_config(),
        )

        assert [r.status for r in results] == ["completed", "error"]
        assert len(errors) == 2
        descs = [d for d, _ in errors]
        assert sum("slow_scenario" in d for d in descs) == 1
        assert sum("boom" in d for d in descs) == 1
        assert all("ok_scenario" not in d for d in descs)

    def test_completed_scenarios_produce_no_error_entries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Results without status='error' leave the error list empty."""
        _stub_failing_turn(monkeypatch, None)

        executor = ScenarioExecutor(parallel=False, config=ScenarioExecutorConfig(enable_cache=False))
        results, errors = executor.run_batch([_make_combo("fine_scenario")], _make_config())

        assert [r.status for r in results] == ["completed"]
        assert errors == []
