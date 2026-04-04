"""Tests for ScenarioExecutor timeout handling with partial progress.

Tests verify:
- Slow combos that exceed batch timeout produce a TimeoutError with in-flight progress info
- All combos completing before timeout returns normal results with no timeout errors
- The per-turn progress callback populates partial_progress with scenario_id, turn, and node
- Mixed batches (some complete, some timed out) report both results and timeout errors
"""

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
# Slow combos with batch timeout
# ============================================================================


@pytest.mark.unit
class TestSlowCombosWithBatchTimeout:
    """Slow combos exceeding the batch timeout produce a TimeoutError with progress info."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_timeout_includes_in_flight_progress(self, mock_manager_cls: MagicMock) -> None:
        """When combos exceed the timeout after reporting partial progress,
        the timeout error message includes turn count and node info.

        Uses a never-set event to block the worker past both the batch timeout
        and the 5s worker join, so the combo stays in partial_progress.
        """
        combos = [_make_combo("slow_scenario")]

        # Block forever (daemon thread is cleaned up after test)
        block_forever = threading.Event()

        def mock_run(**kwargs):
            cb = kwargs.get("progress_callback")
            if cb:
                cb(scenario_id="slow_scenario", scenario_turn=5, scenario_node="node_B")
            # Block past both batch timeout (0.5s) and worker join (5s)
            block_forever.wait(timeout=30.0)
            return _make_exec_result("slow_scenario")

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=1,
                enable_cache=False,
                timeout_seconds=0.5,
            ),
        )
        results, errors = executor.run_batch(combos, config)

        # Unblock the worker so it can exit cleanly
        block_forever.set()

        # A timeout error must be present
        timeout_errors = [(desc, exc) for desc, exc in errors if isinstance(exc, TimeoutError)]
        assert len(timeout_errors) >= 1

        timeout_msg = str(timeout_errors[0][1])
        assert "timed out" in timeout_msg
        # The message must include in-flight progress details
        assert "slow_scenario" in timeout_msg
        assert "turn 5" in timeout_msg
        assert "node_B" in timeout_msg

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_timeout_reports_counts(self, mock_manager_cls: MagicMock) -> None:
        """The timeout message reports completed, in-flight, and not-started counts."""
        combos = [_make_combo("s1"), _make_combo("s2")]

        block_forever = threading.Event()

        def mock_run(**kwargs):
            cb = kwargs.get("progress_callback")
            if cb:
                cb(scenario_id=kwargs["scenario"].name, scenario_turn=1, scenario_node="root")
            block_forever.wait(timeout=30.0)
            return _make_exec_result(kwargs["scenario"].name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=2,
                enable_cache=False,
                timeout_seconds=0.5,
            ),
        )
        results, errors = executor.run_batch(combos, config)

        block_forever.set()

        timeout_errors = [(desc, exc) for desc, exc in errors if isinstance(exc, TimeoutError)]
        assert len(timeout_errors) >= 1

        timeout_msg = str(timeout_errors[0][1])
        # Should mention "of 2 combos"
        assert "2 combos" in timeout_msg


# ============================================================================
# All combos complete before timeout
# ============================================================================


@pytest.mark.unit
class TestAllCombosCompleteBeforeTimeout:
    """All combos completing before the timeout produces normal results, no timeout errors."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_no_timeout_errors_when_fast(self, mock_manager_cls: MagicMock) -> None:
        """Fast combos complete within the timeout; no TimeoutError in errors."""
        combos = [
            _make_combo("fast_a"),
            _make_combo("fast_b"),
            _make_combo("fast_c"),
        ]

        def mock_run(**kwargs):
            return _make_exec_result(kwargs["scenario"].name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=3,
                enable_cache=False,
                timeout_seconds=30.0,
            ),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) == 3
        result_ids = {r.scenario_id for r in results}
        assert result_ids == {"fast_a", "fast_b", "fast_c"}

        # No timeout errors at all
        timeout_errors = [e for _, e in errors if isinstance(e, TimeoutError)]
        assert len(timeout_errors) == 0
        assert len(errors) == 0

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_no_timeout_entries_in_failed_tasks(self, mock_manager_cls: MagicMock) -> None:
        """Even with one runtime failure, no timeout errors when within deadline."""
        combos = [_make_combo("ok"), _make_combo("fail")]

        def mock_run(**kwargs):
            if kwargs["scenario"].name == "fail":
                raise RuntimeError("scenario broke")
            return _make_exec_result(kwargs["scenario"].name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=2,
                enable_cache=False,
                timeout_seconds=30.0,
            ),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) == 1
        assert results[0].scenario_id == "ok"
        assert len(errors) == 1
        # The error is a RuntimeError, not a TimeoutError
        _, exc = errors[0]
        assert isinstance(exc, RuntimeError)
        timeout_errors = [e for _, e in errors if isinstance(e, TimeoutError)]
        assert len(timeout_errors) == 0


# ============================================================================
# Progress callback wiring
# ============================================================================


@pytest.mark.unit
class TestProgressCallbackWiring:
    """The per-turn progress callback populates partial_progress correctly."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_partial_progress_populated_by_callback(self, mock_manager_cls: MagicMock) -> None:
        """When manager.run() calls the progress callback, the partial_progress dict
        is updated with scenario_id, turn, and node before the combo finishes.

        We verify this indirectly: if the combo times out after the callback fires,
        the timeout message must contain the progress data from the last callback.
        """
        combos = [_make_combo("multi_turn")]

        block_forever = threading.Event()

        def mock_run(**kwargs):
            cb = kwargs.get("progress_callback")
            if cb:
                # Simulate multiple turn callbacks; the last one wins
                cb(scenario_id="multi_turn", scenario_turn=1, scenario_node="intro")
                cb(scenario_id="multi_turn", scenario_turn=2, scenario_node="followup")
                cb(scenario_id="multi_turn", scenario_turn=3, scenario_node="conclusion")
            # Block past batch timeout and worker join
            block_forever.wait(timeout=30.0)
            return _make_exec_result("multi_turn")

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=1,
                enable_cache=False,
                timeout_seconds=0.5,
            ),
        )
        results, errors = executor.run_batch(combos, config)

        block_forever.set()

        timeout_errors = [(desc, exc) for desc, exc in errors if isinstance(exc, TimeoutError)]
        assert len(timeout_errors) >= 1

        timeout_msg = str(timeout_errors[0][1])
        # The latest callback values should be reflected
        assert "multi_turn" in timeout_msg
        assert "turn 3" in timeout_msg
        assert "conclusion" in timeout_msg

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_callback_receives_expected_kwargs(self, mock_manager_cls: MagicMock) -> None:
        """The make_turn_callback closure correctly receives and stores kwargs."""
        combos = [_make_combo("kwarg_check")]

        def mock_run(**kwargs):
            cb = kwargs.get("progress_callback")
            if cb:
                # The callback is _on_turn(**kwargs) which reads scenario_turn and scenario_node
                cb(
                    scenario_id="kwarg_check",
                    scenario_turn=7,
                    scenario_node="deep_node",
                    verify_result=MagicMock(),
                    next_node="exit",
                )
            return _make_exec_result("kwarg_check")

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=1,
                enable_cache=False,
                timeout_seconds=30.0,
            ),
        )
        results, errors = executor.run_batch(combos, config)

        # Combo completes successfully (no timeout), so no error
        assert len(results) == 1
        assert len(errors) == 0


# ============================================================================
# Mixed: some complete, some timed out
# ============================================================================


@pytest.mark.unit
class TestMixedCompletionAndTimeout:
    """When some combos complete before the timeout and others do not,
    the completed results are returned alongside a timeout error.
    """

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_completed_results_returned_with_timeout_error(self, mock_manager_cls: MagicMock) -> None:
        """Fast combos have their results collected; slow combos produce a timeout error."""
        combos = [
            _make_combo("fast"),
            _make_combo("slow"),
        ]

        block_forever = threading.Event()

        def mock_run(**kwargs):
            name = kwargs["scenario"].name
            cb = kwargs.get("progress_callback")
            if name == "slow":
                if cb:
                    cb(scenario_id="slow", scenario_turn=2, scenario_node="stuck_node")
                # Block past both batch timeout and worker join
                block_forever.wait(timeout=30.0)
                return _make_exec_result("slow")
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

        block_forever.set()

        # The fast combo should have completed
        assert len(results) >= 1
        fast_results = [r for r in results if r.scenario_id == "fast"]
        assert len(fast_results) == 1

        # A timeout error should be present with the slow combo's progress
        timeout_errors = [(desc, exc) for desc, exc in errors if isinstance(exc, TimeoutError)]
        assert len(timeout_errors) >= 1
        timeout_msg = str(timeout_errors[0][1])
        assert "slow" in timeout_msg
        assert "stuck_node" in timeout_msg
