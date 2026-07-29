"""End-to-end integration tests for parallel scenario execution.

Verifies that the full parallel scenario flow works correctly, including
the GlobalLLMLimiter lifecycle, cache behavior, and error handling. Mocks
at the ScenarioManager level (the boundary between executor and pipeline
internals).

T13 deliberate flip: the limiter tests here previously pinned the legacy
set_global_llm_semaphore production wiring, which the GlobalLLMLimiter
supersedes. The legacy accessors stay covered by test_global_semaphore.py.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.executor import get_global_llm_limiter
from karenina.benchmark.verification.scenario_executor import (
    ScenarioExecutor,
    ScenarioExecutorConfig,
)


def _make_combo(name: str, ans: str = "model-a", parse: str = "model-p"):
    """Build a mock scenario combo tuple (scenario_def, answering_model, parsing_model, replicate).

    Replicate is None for single-replicate runs (the backward-compat default).
    """
    scenario = MagicMock()
    scenario.name = name
    a = MagicMock()
    a.model_name = ans
    a.id = ans
    p = MagicMock()
    p.model_name = parse
    p.id = parse
    return (scenario, a, p, None)


def _make_result(name: str):
    """Build a mock ScenarioExecutionResult with the given scenario_id."""
    r = MagicMock()
    r.scenario_id = name
    r.turn_results = [MagicMock()]
    return r


@pytest.mark.integration
class TestScenarioParallelE2E:
    """End-to-end tests for ScenarioExecutor parallel and sequential execution."""

    def test_limiter_is_configured_during_execution(self):
        """The GlobalLLMLimiter should be active during run_batch."""
        combos = [_make_combo("s1"), _make_combo("s2")]
        limiter_was_configured = [False]

        def check_limiter(**kwargs):
            if get_global_llm_limiter().capacity == 4:
                limiter_was_configured[0] = True
            return _make_result(kwargs["scenario"].name)

        with patch("karenina.benchmark.verification.scenario_executor.ScenarioManager") as MockManager:
            MockManager.return_value.run.side_effect = check_limiter

            executor = ScenarioExecutor(
                parallel=True,
                config=ScenarioExecutorConfig(max_workers=2, max_concurrent_requests=4),
            )
            config = MagicMock()
            config.request_timeout = None
            results, errors = executor.run_batch(combos, config)

        assert limiter_was_configured[0], "GlobalLLMLimiter should be configured during execution"
        assert len(results) == 2
        assert len(errors) == 0

    def test_limiter_deconfigured_after_execution(self):
        """The GlobalLLMLimiter should be inactive after run_batch completes."""
        combos = [_make_combo("s1")]

        with patch("karenina.benchmark.verification.scenario_executor.ScenarioManager") as MockManager:
            MockManager.return_value.run.return_value = _make_result("s1")

            executor = ScenarioExecutor(
                parallel=False,
                config=ScenarioExecutorConfig(max_concurrent_requests=8),
            )
            config = MagicMock()
            config.request_timeout = None
            executor.run_batch(combos, config)

        assert get_global_llm_limiter().capacity is None

    def test_limiter_deconfigured_on_error(self):
        """The GlobalLLMLimiter should be cleaned up even when a scenario fails."""
        combos = [_make_combo("s1")]

        with patch("karenina.benchmark.verification.scenario_executor.ScenarioManager") as MockManager:
            MockManager.return_value.run.side_effect = RuntimeError("boom")

            executor = ScenarioExecutor(
                parallel=False,
                config=ScenarioExecutorConfig(max_concurrent_requests=4),
            )
            config = MagicMock()
            config.request_timeout = None
            executor.run_batch(combos, config)

        assert get_global_llm_limiter().capacity is None

    def test_parallel_with_multiple_workers(self):
        """Full flow: multiple combos with parallel execution, semaphore, and caching."""
        combos = [_make_combo(f"s{i}") for i in range(4)]
        active = [0]
        max_active = [0]
        lock = threading.Lock()

        def tracked_run(**kwargs):
            with lock:
                active[0] += 1
                max_active[0] = max(max_active[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return _make_result(kwargs["scenario"].name)

        with patch("karenina.benchmark.verification.scenario_executor.ScenarioManager") as MockManager:
            MockManager.return_value.run.side_effect = tracked_run

            executor = ScenarioExecutor(
                parallel=True,
                config=ScenarioExecutorConfig(
                    max_workers=4,
                    max_concurrent_requests=8,
                    enable_cache=True,
                ),
            )
            config = MagicMock()
            config.request_timeout = None
            results, errors = executor.run_batch(combos, config)

        assert len(results) == 4
        assert len(errors) == 0
        assert max_active[0] >= 2
        assert get_global_llm_limiter().capacity is None
