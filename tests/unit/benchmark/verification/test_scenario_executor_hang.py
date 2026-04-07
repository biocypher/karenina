"""Tests for ScenarioExecutor hang prevention.

These tests verify that the ThreadPoolExecutor-based parallel executor
never hangs due to untracked worker deaths. Each test targets a specific
failure mode that caused hangs with the previous raw-thread design.
"""

import contextlib
import threading
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
    """Create a mock (scenario_def, answering_model, parsing_model) combo."""
    scenario_def = MagicMock()
    scenario_def.name = scenario_name

    ans_model = MagicMock()
    ans_model.model_name = model_name
    ans_model.id = f"{model_name}-ans"

    parse_model = MagicMock()
    parse_model.model_name = model_name
    parse_model.id = f"{model_name}-parse"

    return (scenario_def, ans_model, parse_model)


def _make_exec_result(scenario_id: str = "s1") -> MagicMock:
    """Create a mock ScenarioExecutionResult."""
    result = MagicMock()
    result.scenario_id = scenario_id
    result.status = "completed"
    result.turn_count = 3
    return result


# ============================================================================
# BaseException handling (the original hang bug)
# ============================================================================


@pytest.mark.unit
class TestBaseExceptionNoHang:
    """BaseException from a combo must not cause the batch to hang."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_keyboard_interrupt_no_hang(self, mock_manager_cls: MagicMock) -> None:
        """A combo raising KeyboardInterrupt is collected as an error.

        With the old raw-thread design, KeyboardInterrupt bypassed the
        except Exception handler, leaving the combo untracked and the
        completion event stuck.
        """
        combos = [_make_combo("ok"), _make_combo("interrupt"), _make_combo("ok2")]

        def mock_run(**kwargs):
            if kwargs["scenario"].name == "interrupt":
                raise KeyboardInterrupt()
            return _make_exec_result(kwargs["scenario"].name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=2, enable_cache=False, timeout_seconds=10.0),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) == 2
        assert len(errors) == 1
        assert isinstance(errors[0][1], KeyboardInterrupt)

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_system_exit_no_hang(self, mock_manager_cls: MagicMock) -> None:
        """A combo raising SystemExit is collected as an error without hanging."""
        combos = [_make_combo("ok"), _make_combo("exit")]

        def mock_run(**kwargs):
            if kwargs["scenario"].name == "exit":
                raise SystemExit(1)
            return _make_exec_result(kwargs["scenario"].name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=2, enable_cache=False, timeout_seconds=10.0),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(errors[0][1], SystemExit)

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_all_raise_base_exception(self, mock_manager_cls: MagicMock) -> None:
        """When every combo raises BaseException, batch returns immediately
        with all errors collected (no waiting for timeout).
        """
        combos = [_make_combo("s1"), _make_combo("s2"), _make_combo("s3")]

        mock_manager_cls.return_value.run.side_effect = KeyboardInterrupt()

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=3, enable_cache=False, timeout_seconds=10.0),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) == 0
        assert len(errors) == 3
        assert all(isinstance(e, KeyboardInterrupt) for _, e in errors)

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_mixed_success_and_base_exception(self, mock_manager_cls: MagicMock) -> None:
        """Mix of successful combos and BaseException produces partial results."""
        combos = [
            _make_combo("ok_1"),
            _make_combo("interrupt"),
            _make_combo("ok_2"),
            _make_combo("ok_3"),
        ]

        def mock_run(**kwargs):
            if kwargs["scenario"].name == "interrupt":
                raise KeyboardInterrupt()
            return _make_exec_result(kwargs["scenario"].name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=2, enable_cache=False, timeout_seconds=10.0),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) == 3
        assert len(errors) == 1
        result_ids = {r.scenario_id for r in results}
        assert result_ids == {"ok_1", "ok_2", "ok_3"}


# ============================================================================
# Portal creation failure
# ============================================================================


@pytest.mark.unit
class TestPortalCreationFailure:
    """Portal creation failure in one worker must not block others.

    Under the per-worker portal lifecycle, each worker thread lazily creates
    one BlockingPortal on its first task and reuses it. Portal-creation
    failures are therefore per-worker, not per-combo: the worker whose
    portal creation raises loses whichever combo it was assigned, while
    other workers (with healthy portals) continue running their combos.
    """

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    @patch("karenina.benchmark.verification.scenario_executor.start_blocking_portal")
    def test_portal_failure_collected_as_error(
        self,
        mock_start_portal: MagicMock,
        mock_manager_cls: MagicMock,
    ) -> None:
        """When one worker's portal creation raises, its combo surfaces as
        an error while combos on other workers complete normally.

        A threading.Barrier forces all three workers to reach the portal
        factory simultaneously, guaranteeing each worker calls the factory
        exactly once. Without this synchronization a single worker could
        drain every combo before the others start (mocked work is
        instantaneous), leaving the failure-injection branch unreachable.
        """
        combos = [_make_combo("s1"), _make_combo("s2"), _make_combo("s3")]

        # Force all three workers to create their portal at the same time.
        worker_barrier = threading.Barrier(3, timeout=5.0)
        portal_call_count = [0]
        portal_lock = threading.Lock()
        # Barrier forces the pool to spawn all 3 worker threads before any
        # portal call returns. Without this the pool may reuse a single
        # thread for fast-completing tasks, producing only one portal call
        # total and making the failure-injection counter unreliable.
        portal_barrier = threading.Barrier(3, timeout=5.0)

        def make_portal(*args, **kwargs):  # noqa: ARG001
            # Block until all three workers have reached portal creation,
            # ensuring each worker calls this factory exactly once.
            worker_barrier.wait()
            with portal_lock:
                portal_call_count[0] += 1
                current = portal_call_count[0]
            # Wait for all three workers to reach this point so the counter
            # deterministically covers 1, 2, and 3.
            with contextlib.suppress(threading.BrokenBarrierError):
                portal_barrier.wait()
            if current == 2:
                raise RuntimeError("portal creation failed")
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=MagicMock())
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        mock_start_portal.side_effect = make_portal
        mock_manager_cls.return_value.run.return_value = _make_exec_result("s")

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=3, enable_cache=False, timeout_seconds=10.0),
        )
        results, errors = executor.run_batch(combos, config)

        # Exactly three portal creations (one per worker), the second raising.
        assert portal_call_count[0] == 3
        assert len(results) == 2
        assert len(errors) == 1
        assert isinstance(errors[0][1], RuntimeError)
        assert "portal creation failed" in str(errors[0][1])


# ============================================================================
# Timeout with hanging workers
# ============================================================================


@pytest.mark.unit
class TestTimeoutWithHangingWorkers:
    """Batch timeout returns promptly even when workers are stuck."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_indefinitely_hanging_worker(self, mock_manager_cls: MagicMock) -> None:
        """One combo blocks forever; the other completes. Timeout fires and
        the completed combo's result is returned alongside a timeout error.
        """
        combos = [_make_combo("fast"), _make_combo("hang")]

        block_forever = threading.Event()

        def mock_run(**kwargs):
            if kwargs["scenario"].name == "hang":
                block_forever.wait(timeout=30.0)
                return _make_exec_result("hang")
            return _make_exec_result("fast")

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=2,
                enable_cache=False,
                timeout_seconds=1.0,
            ),
        )
        results, errors = executor.run_batch(combos, config)

        # Unblock the hanging worker so the test thread can exit cleanly
        block_forever.set()

        # The fast combo should have completed
        assert len(results) >= 1
        fast_results = [r for r in results if r.scenario_id == "fast"]
        assert len(fast_results) == 1

        # A timeout error should be present
        timeout_errors = [(d, e) for d, e in errors if isinstance(e, TimeoutError)]
        assert len(timeout_errors) >= 1
        assert "timed out" in str(timeout_errors[0][1])
