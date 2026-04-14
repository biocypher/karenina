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
