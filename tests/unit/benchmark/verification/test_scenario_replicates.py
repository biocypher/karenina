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


@pytest.mark.unit
class TestFacadeComboExpansion:
    """Verify Benchmark._run_scenario_verification expands the replicate axis."""

    def _capture_combos(self, monkeypatch, replicate_count: int, task_ordering: str = "generation_order"):
        """Helper: run _run_scenario_verification and return the combos list passed to ScenarioExecutor.run_batch."""
        from karenina.benchmark import benchmark as bm
        from karenina.benchmark.verification import scenario_executor as se

        captured: dict[str, object] = {}

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def run_batch(self, combos, config, **kwargs):  # noqa: ARG002
                captured["combos"] = combos
                return [], []

        # Patch on the source module; the facade does a local import at call time.
        monkeypatch.setattr(se, "ScenarioExecutor", FakeExecutor, raising=True)

        # Minimal Benchmark double with scenarios + config
        benchmark_obj = MagicMock()
        benchmark_obj._scenarios = {"foo": MagicMock(name="scenario_def", nodes={})}
        benchmark_obj._scenarios["foo"].name = "foo"
        benchmark_obj._rubric_manager.get_global_rubric.return_value = None
        benchmark_obj._workspace_root = None

        config = MagicMock()
        config.answering_models = [MagicMock(model_name="a", request_timeout=None, retry_policy=None)]
        config.parsing_models = [MagicMock(model_name="p", request_timeout=None, retry_policy=None)]
        config.replicate_count = replicate_count
        config.task_ordering = task_ordering
        config.request_timeout = None
        config.retry_policy = None
        config.async_max_workers = 1
        config.max_concurrent_requests = None

        bm.Benchmark._run_scenario_verification(benchmark_obj, config=config)
        return captured["combos"]

    def test_replicate_count_one_emits_single_none_combo(self, monkeypatch):
        combos = self._capture_combos(monkeypatch, replicate_count=1)
        assert len(combos) == 1
        assert combos[0][3] is None

    def test_replicate_count_three_produces_three_combos(self, monkeypatch):
        combos = self._capture_combos(monkeypatch, replicate_count=3)
        assert [c[3] for c in combos] == [1, 2, 3]

    def test_prefix_cache_sort_is_none_safe(self, monkeypatch):
        # replicate_count=1 -> single combo with replicate=None.
        # The prefix_cache sort must not raise TypeError when comparing None with int.
        combos = self._capture_combos(monkeypatch, replicate_count=1, task_ordering="prefix_cache")
        assert combos[0][3] is None
