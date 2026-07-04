"""Tests for ScenarioExecutor parallel mode.

Tests verify:
- All combos are executed
- Results preserve original combo ordering even with delayed completion
- Errors are isolated (one failure does not stop others)
- Actual concurrency occurs (max_active >= 2)
- Timeout handling adds a timeout error when batch exceeds deadline
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.scenario_executor import (
    ScenarioExecutor,
    ScenarioExecutorConfig,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_combo(scenario_name: str, model_name: str = "test-model") -> tuple:
    """Create a mock (scenario_def, answering_model, parsing_model, replicate) combo.

    Replicate is None for single-replicate runs (the backward-compat default).
    """
    scenario_def = MagicMock()
    scenario_def.name = scenario_name

    ans_model = MagicMock()
    ans_model.model_name = model_name
    ans_model.id = f"{model_name}-ans"

    parse_model = MagicMock()
    parse_model.model_name = model_name
    parse_model.id = f"{model_name}-parse"

    return (scenario_def, ans_model, parse_model, None)


def _make_exec_result(scenario_id: str = "s1") -> MagicMock:
    """Create a mock ScenarioExecutionResult."""
    result = MagicMock()
    result.scenario_id = scenario_id
    result.status = "completed"
    result.turn_count = 3
    return result


# ============================================================================
# Parallel: runs all combos
# ============================================================================


@pytest.mark.unit
class TestParallelRunsAll:
    """Parallel mode executes all combos and returns results."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_runs_all_combos(self, mock_manager_cls: MagicMock) -> None:
        """All combos complete and results are returned."""
        combos = [
            _make_combo("scenario_a"),
            _make_combo("scenario_b"),
            _make_combo("scenario_c"),
        ]

        results_map = {
            "scenario_a": _make_exec_result("a"),
            "scenario_b": _make_exec_result("b"),
            "scenario_c": _make_exec_result("c"),
        }

        def mock_run(**kwargs):
            return results_map[kwargs["scenario"].name]

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=2, enable_cache=False),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) == 3
        assert len(errors) == 0
        # Results contain all scenario ids (order may differ from input
        # since parallel execution shuffles, but all are present)
        result_ids = {r.scenario_id for r in results}
        assert result_ids == {"a", "b", "c"}

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_empty_combos_returns_empty(self, mock_manager_cls: MagicMock) -> None:  # noqa: ARG002
        """No combos produces no results and no errors."""
        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=2, enable_cache=False),
        )
        results, errors = executor.run_batch([], config)

        assert results == []
        assert errors == []


# ============================================================================
# Parallel: preserves original order
# ============================================================================


@pytest.mark.unit
class TestParallelPreservesOrder:
    """Parallel mode restores original combo order despite async completion."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_order_preserved_with_delayed_completion(self, mock_manager_cls: MagicMock) -> None:
        """Combos that finish out of order are reordered in the result list.

        combo_0 (slow) finishes after combo_1 (fast) and combo_2 (fast),
        but results[0] is still combo_0's result.
        """
        combos = [
            _make_combo("slow"),  # index 0
            _make_combo("fast_1"),  # index 1
            _make_combo("fast_2"),  # index 2
        ]

        result_slow = _make_exec_result("slow")
        result_fast1 = _make_exec_result("fast_1")
        result_fast2 = _make_exec_result("fast_2")

        def mock_run(**kwargs):
            name = kwargs["scenario"].name
            if name == "slow":
                time.sleep(0.15)
                return result_slow
            if name == "fast_1":
                return result_fast1
            return result_fast2

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=3, enable_cache=False),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) == 3
        assert len(errors) == 0
        # Original order restored
        assert results[0].scenario_id == "slow"
        assert results[1].scenario_id == "fast_1"
        assert results[2].scenario_id == "fast_2"


# ============================================================================
# Parallel: error isolation
# ============================================================================


@pytest.mark.unit
class TestParallelErrorIsolation:
    """Parallel mode isolates errors; one failure does not block others."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_single_failure_does_not_stop_others(self, mock_manager_cls: MagicMock) -> None:
        """A failing combo is collected as an error; other combos succeed."""
        combos = [
            _make_combo("ok_1"),
            _make_combo("fail_combo"),
            _make_combo("ok_2"),
        ]

        def mock_run(**kwargs):
            name = kwargs["scenario"].name
            if name == "fail_combo":
                raise RuntimeError("scenario explosion")
            return _make_exec_result(name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=2, enable_cache=False),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) == 2
        result_ids = {r.scenario_id for r in results}
        assert result_ids == {"ok_1", "ok_2"}
        assert len(errors) == 1
        desc, exc = errors[0]
        assert "fail_combo" in desc
        assert isinstance(exc, RuntimeError)

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_all_fail_returns_only_errors(self, mock_manager_cls: MagicMock) -> None:
        """When all combos fail, results is empty and all errors are collected."""
        combos = [_make_combo("s1"), _make_combo("s2")]

        def mock_run(**kwargs):
            raise ValueError(f"fail_{kwargs['scenario'].name}")

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=2, enable_cache=False),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) == 0
        assert len(errors) == 2

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_completes_without_deadlock_on_failure(self, mock_manager_cls: MagicMock) -> None:
        """Failed combos count toward completion; batch finishes without hanging."""
        combos = [_make_combo("ok"), _make_combo("fail"), _make_combo("ok2")]

        def mock_run(**kwargs):
            name = kwargs["scenario"].name
            if name == "fail":
                raise RuntimeError("boom")
            return _make_exec_result(name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=2, enable_cache=False, timeout_seconds=10.0),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) + len(errors) == 3


# ============================================================================
# Parallel: actual concurrency
# ============================================================================


@pytest.mark.unit
class TestParallelConcurrency:
    """Parallel mode achieves actual concurrency with multiple workers."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_concurrent_execution_observed(self, mock_manager_cls: MagicMock) -> None:
        """At least 2 combos execute concurrently (max_active >= 2).

        Each combo sleeps briefly and records the number of active threads
        at peak. With 3+ workers and sleeps, concurrency must occur.
        """
        combos = [_make_combo(f"s{i}") for i in range(4)]

        active_count = [0]
        max_active = [0]
        lock = threading.Lock()

        def mock_run(**kwargs):
            with lock:
                active_count[0] += 1
                if active_count[0] > max_active[0]:
                    max_active[0] = active_count[0]
            time.sleep(0.1)
            with lock:
                active_count[0] -= 1
            return _make_exec_result(kwargs["scenario"].name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=4, enable_cache=False),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) == 4
        assert len(errors) == 0
        assert max_active[0] >= 2, f"Expected concurrent execution, but max_active was {max_active[0]}"


# ============================================================================
# Parallel: timeout handling
# ============================================================================


@pytest.mark.unit
class TestParallelTimeout:
    """Parallel mode handles timeouts by appending a timeout error."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_timeout_adds_error(self, mock_manager_cls: MagicMock) -> None:
        """When combos exceed the timeout, a TimeoutError is appended to errors."""
        combos = [_make_combo("slow_1"), _make_combo("slow_2")]

        def mock_run(**kwargs):
            time.sleep(5.0)  # Much longer than the 1s timeout
            return _make_exec_result(kwargs["scenario"].name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=2,
                enable_cache=False,
                timeout_seconds=1.0,  # Very short timeout
            ),
        )
        results, errors = executor.run_batch(combos, config)

        # At least one error should be a timeout
        timeout_errors = [e for _, e in errors if isinstance(e, TimeoutError)]
        assert len(timeout_errors) >= 1
        assert "timed out" in str(timeout_errors[0])


# ============================================================================
# Parallel: semaphore lifecycle
# ============================================================================


@pytest.mark.unit
class TestParallelSemaphoreLifecycle:
    """Global LLM semaphore is set before workers and cleared after."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_semaphore_active_during_parallel_run(self, mock_manager_cls: MagicMock) -> None:
        """When max_concurrent_requests is set, semaphore is active during worker execution."""
        from karenina.benchmark.verification.executor import get_global_llm_semaphore

        captured: list = []

        def mock_run(**kwargs):
            captured.append(get_global_llm_semaphore())
            return _make_exec_result("s1")

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=1,
                enable_cache=False,
                max_concurrent_requests=5,
            ),
        )
        executor.run_batch([_make_combo("s1")], config)

        assert len(captured) == 1
        assert captured[0] is not None
        # After run_batch, semaphore is cleared
        assert get_global_llm_semaphore() is None

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_semaphore_cleared_after_parallel_run(self, mock_manager_cls: MagicMock) -> None:
        """Semaphore is cleared even when combos fail."""
        from karenina.benchmark.verification.executor import get_global_llm_semaphore

        mock_manager_cls.return_value.run.side_effect = RuntimeError("fail")

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=1,
                enable_cache=False,
                max_concurrent_requests=3,
            ),
        )
        executor.run_batch([_make_combo("s1")], config)

        assert get_global_llm_semaphore() is None


# ============================================================================
# Parallel: portal management
# ============================================================================


@pytest.mark.unit
class TestParallelPortalManagement:
    """Worker threads receive a functional BlockingPortal."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_portal_available_in_workers(self, mock_manager_cls: MagicMock) -> None:
        """Each worker thread has a portal set via set_async_portal."""
        from karenina.benchmark.verification.executor import get_async_portal

        captured_portals: list = []
        lock = threading.Lock()

        def mock_run(**kwargs):
            portal = get_async_portal()
            with lock:
                captured_portals.append(portal)
            return _make_exec_result(kwargs["scenario"].name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=2, enable_cache=False),
        )
        executor.run_batch([_make_combo("s1"), _make_combo("s2")], config)

        assert len(captured_portals) == 2
        assert all(p is not None for p in captured_portals)

        # Portal is cleared after run_batch (on the main thread)
        assert get_async_portal() is None


# ============================================================================
# Parallel: per-worker portals
# ============================================================================


@pytest.mark.unit
class TestScenarioPerWorkerPortals:
    """Each worker thread creates one portal, reused across combos."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    @patch("karenina.benchmark.verification.scenario_executor.start_blocking_portal")
    def test_each_worker_gets_one_portal(
        self,
        mock_start_portal: MagicMock,
        mock_manager_cls: MagicMock,
    ) -> None:
        """With 6 combos and 2 workers, 2 portals are created (one per worker).

        Each worker lazily creates a portal on its first combo and reuses it
        for subsequent combos. This preserves connection pools.
        """
        created_portals: list[MagicMock] = []
        lock = threading.Lock()

        def make_portal(*args, **kwargs):  # noqa: ARG001
            portal = MagicMock()
            with lock:
                created_portals.append(portal)
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=portal)
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        mock_start_portal.side_effect = make_portal

        portal_barrier = threading.Barrier(2, timeout=5.0)
        seen_threads: set[int] = set()
        seen_threads_lock = threading.Lock()

        def mock_run(**kwargs):  # noqa: ARG001
            thread_id = threading.get_ident()
            with seen_threads_lock:
                is_first_combo_for_thread = thread_id not in seen_threads
                seen_threads.add(thread_id)
            if is_first_combo_for_thread:
                portal_barrier.wait()
            return _make_exec_result("s")

        mock_manager_cls.return_value.run.side_effect = mock_run

        combos = [_make_combo(f"s{i}") for i in range(6)]
        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=2, enable_cache=False),
        )
        executor.run_batch(combos, config)

        # One portal per worker (reused across combos)
        assert mock_start_portal.call_count == 2
        assert len(created_portals) == 2
