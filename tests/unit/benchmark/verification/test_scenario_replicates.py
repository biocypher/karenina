"""Tests for R2: scenario replicate_count executor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from karenina.scenario.manager import ScenarioManager


@pytest.mark.unit
class TestScenarioManagerReplicateThreading:
    """Verify ScenarioManager.run threads the replicate value correctly."""

    def test_execution_result_carries_replicate(self, monkeypatch):
        """ScenarioManager.run(..., replicate=2) returns a result with replicate=2."""
        from karenina.schemas.scenario.state import ScenarioExecutionResult

        captured: dict[str, object] = {}

        def fake_run(self, scenario, config, base_answering_model, base_parsing_model, **kwargs):
            captured["replicate"] = kwargs.get("replicate")
            final_state = MagicMock()
            return ScenarioExecutionResult(
                scenario_id=scenario.name,
                status="completed",
                path=[],
                turn_count=0,
                history=[],
                turn_results=[],
                final_state=final_state,
                outcome_results={},
                replicate=kwargs.get("replicate"),
            )

        monkeypatch.setattr(ScenarioManager, "run", fake_run, raising=True)
        manager = ScenarioManager()
        scenario = MagicMock(name="scenario_def")
        scenario.name = "foo"
        result = manager.run(
            scenario=scenario,
            config=MagicMock(),
            base_answering_model=MagicMock(),
            base_parsing_model=MagicMock(),
            replicate=2,
        )
        assert result.replicate == 2
        assert captured["replicate"] == 2


@pytest.mark.unit
class TestScenarioExecutorForwardsReplicate:
    """Verify ScenarioExecutor forwards replicate from the 4-tuple to ScenarioManager.run."""

    def test_sequential_forwards_replicate_kwarg(self, monkeypatch):
        from karenina.benchmark.verification import scenario_executor as se
        from karenina.schemas.scenario.state import ScenarioExecutionResult

        captured: list[int | None] = []

        def fake_run(self, *args, **kwargs):
            captured.append(kwargs.get("replicate"))
            return ScenarioExecutionResult(
                scenario_id=kwargs["scenario"].name,
                status="completed",
                path=[],
                turn_count=0,
                history=[],
                turn_results=[],
                final_state=MagicMock(),
                outcome_results={},
                replicate=kwargs.get("replicate"),
            )

        monkeypatch.setattr(se.ScenarioManager, "run", fake_run, raising=True)

        scenario = MagicMock()
        scenario.name = "foo"
        ans = MagicMock()
        ans.model_name = "a"
        parse = MagicMock()
        parse.model_name = "p"

        combos = [
            (scenario, ans, parse, None),
            (scenario, ans, parse, 1),
            (scenario, ans, parse, 2),
        ]
        executor = se.ScenarioExecutor(parallel=False, config=se.ScenarioExecutorConfig(enable_cache=False))
        results, errors = executor.run_batch(combos=combos, config=MagicMock())
        assert not errors
        assert captured == [None, 1, 2]
        assert [r.replicate for r in results] == [None, 1, 2]
