"""Tests for ScenarioExecutor sequential mode.

Tests verify:
- All combos are executed in order
- Results preserve original combo ordering
- Errors are isolated (one failure does not stop others)
- Progress callback is invoked before and after each combo
- Answer cache is created when enabled
- Global LLM semaphore lifecycle
"""

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
# Sequential: runs all combos
# ============================================================================


@pytest.mark.unit
class TestSequentialRunsAll:
    """Sequential mode executes all combos and returns results."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_runs_all_combos_in_order(self, mock_manager_cls: MagicMock) -> None:
        """All combos are executed, and results preserve the original order."""
        combos = [
            _make_combo("scenario_a"),
            _make_combo("scenario_b"),
            _make_combo("scenario_c"),
        ]

        result_a = _make_exec_result("a")
        result_b = _make_exec_result("b")
        result_c = _make_exec_result("c")

        call_order: list[str] = []

        def mock_run(**kwargs):
            name = kwargs["scenario"].name
            call_order.append(name)
            return {"scenario_a": result_a, "scenario_b": result_b, "scenario_c": result_c}[name]

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=False,
            config=ScenarioExecutorConfig(enable_cache=False),
        )
        results, errors = executor.run_batch(combos, config)

        assert call_order == ["scenario_a", "scenario_b", "scenario_c"]
        assert len(results) == 3
        assert results[0].scenario_id == "a"
        assert results[1].scenario_id == "b"
        assert results[2].scenario_id == "c"
        assert len(errors) == 0

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_empty_combos_returns_empty(self, mock_manager_cls: MagicMock) -> None:
        """No combos produces no results and no errors."""
        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=False,
            config=ScenarioExecutorConfig(enable_cache=False),
        )
        results, errors = executor.run_batch([], config)

        assert results == []
        assert errors == []
        mock_manager_cls.return_value.run.assert_not_called()

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_passes_all_kwargs_to_manager(self, mock_manager_cls: MagicMock) -> None:
        """run_batch forwards config, run_name, global_rubric, and answer_cache to manager.run()."""
        combo = _make_combo("s1")
        mock_manager_cls.return_value.run.return_value = _make_exec_result("s1")

        config = MagicMock()
        rubric = MagicMock()
        executor = ScenarioExecutor(
            parallel=False,
            config=ScenarioExecutorConfig(enable_cache=True),
        )
        executor.run_batch([combo], config, global_rubric=rubric, run_name="my_run")

        call_kwargs = mock_manager_cls.return_value.run.call_args.kwargs
        assert call_kwargs["config"] is config
        assert call_kwargs["run_name"] == "my_run"
        assert call_kwargs["global_rubric"] is rubric
        assert call_kwargs["answer_cache"] is not None  # cache enabled


# ============================================================================
# Sequential: error isolation
# ============================================================================


@pytest.mark.unit
class TestSequentialErrorIsolation:
    """Sequential mode isolates errors and continues processing."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_single_failure_does_not_stop_others(self, mock_manager_cls: MagicMock) -> None:
        """A failing combo is recorded as an error; subsequent combos still execute."""
        combos = [
            _make_combo("s1"),
            _make_combo("s2_fail"),
            _make_combo("s3"),
        ]

        def mock_run(**kwargs):
            name = kwargs["scenario"].name
            if name == "s2_fail":
                raise RuntimeError("scenario explosion")
            return _make_exec_result(name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=False,
            config=ScenarioExecutorConfig(enable_cache=False),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) == 2
        assert results[0].scenario_id == "s1"
        assert results[1].scenario_id == "s3"
        assert len(errors) == 1
        desc, exc = errors[0]
        assert "s2_fail" in desc
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
            parallel=False,
            config=ScenarioExecutorConfig(enable_cache=False),
        )
        results, errors = executor.run_batch(combos, config)

        assert len(results) == 0
        assert len(errors) == 2


# ============================================================================
# Sequential: progress callback
# ============================================================================


@pytest.mark.unit
class TestSequentialProgressCallback:
    """Sequential mode invokes progress callback before and after each combo."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_callback_called_before_and_after(self, mock_manager_cls: MagicMock) -> None:
        """Callback receives (idx, total, None) before and (idx, total, result) after each combo."""
        combos = [_make_combo("s1"), _make_combo("s2")]
        result_s1 = _make_exec_result("s1")
        result_s2 = _make_exec_result("s2")

        def mock_run(**kwargs):
            return {"s1": result_s1, "s2": result_s2}[kwargs["scenario"].name]

        mock_manager_cls.return_value.run.side_effect = mock_run

        callback_calls: list[tuple] = []

        def progress_cb(completed: int, total: int, result: object) -> None:
            callback_calls.append((completed, total, result))

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=False,
            config=ScenarioExecutorConfig(enable_cache=False),
        )
        executor.run_batch(combos, config, progress_callback=progress_cb)

        # 2 combos: before + after for each = 4 calls
        assert len(callback_calls) == 4
        # Before s1
        assert callback_calls[0] == (1, 2, None)
        # After s1
        assert callback_calls[1] == (1, 2, result_s1)
        # Before s2
        assert callback_calls[2] == (2, 2, None)
        # After s2
        assert callback_calls[3] == (2, 2, result_s2)

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_callback_on_failure_only_before(self, mock_manager_cls: MagicMock) -> None:
        """For a failing combo, callback is called before but not after (with result)."""
        combos = [_make_combo("fail_combo")]

        def mock_run(**kwargs):
            raise RuntimeError("boom")

        mock_manager_cls.return_value.run.side_effect = mock_run

        callback_calls: list[tuple] = []

        def progress_cb(completed: int, total: int, result: object) -> None:
            callback_calls.append((completed, total, result))

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=False,
            config=ScenarioExecutorConfig(enable_cache=False),
        )
        executor.run_batch(combos, config, progress_callback=progress_cb)

        # Only the "before" call (with None), no "after" call
        assert len(callback_calls) == 1
        assert callback_calls[0] == (1, 1, None)


# ============================================================================
# Sequential: global limiter lifecycle
# ============================================================================


@pytest.mark.unit
class TestSequentialSemaphoreLifecycle:
    """GlobalLLMLimiter is configured before execution and dropped after.

    T13 deliberate flip: these tests previously pinned the legacy
    set_global_llm_semaphore production wiring, which the GlobalLLMLimiter
    supersedes. The legacy accessors stay covered by
    test_global_semaphore.py.
    """

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_limiter_configured_when_set(self, mock_manager_cls: MagicMock) -> None:
        """When max_concurrent_requests is set, the limiter is active during run."""
        from karenina.benchmark.verification.executor import get_global_llm_limiter

        captured: list = []

        def mock_run(**kwargs):
            captured.append(get_global_llm_limiter().capacity)
            return _make_exec_result("s1")

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=False,
            config=ScenarioExecutorConfig(enable_cache=False, max_concurrent_requests=5),
        )
        executor.run_batch([_make_combo("s1")], config)

        assert captured == [5]
        # After run_batch, the limiter is deconfigured
        assert get_global_llm_limiter().capacity is None

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_limiter_unconfigured_when_not_set(self, mock_manager_cls: MagicMock) -> None:
        """When max_concurrent_requests is None, the limiter stays uncapped."""
        from karenina.benchmark.verification.executor import get_global_llm_limiter

        captured: list = []

        def mock_run(**kwargs):
            captured.append(get_global_llm_limiter().capacity)
            return _make_exec_result("s1")

        mock_manager_cls.return_value.run.side_effect = mock_run

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=False,
            config=ScenarioExecutorConfig(enable_cache=False, max_concurrent_requests=None),
        )
        executor.run_batch([_make_combo("s1")], config)

        assert captured[0] is None

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_limiter_deconfigured_on_error(self, mock_manager_cls: MagicMock) -> None:
        """The limiter is deconfigured even when all combos fail."""
        from karenina.benchmark.verification.executor import get_global_llm_limiter

        mock_manager_cls.return_value.run.side_effect = RuntimeError("boom")

        config = MagicMock()
        executor = ScenarioExecutor(
            parallel=False,
            config=ScenarioExecutorConfig(enable_cache=False, max_concurrent_requests=3),
        )
        executor.run_batch([_make_combo("s1")], config)

        assert get_global_llm_limiter().capacity is None


# ============================================================================
# ScenarioExecutorConfig defaults
# ============================================================================


@pytest.mark.unit
class TestScenarioExecutorConfig:
    """ScenarioExecutorConfig has correct defaults."""

    def test_default_values(self) -> None:
        """Default config matches documented values."""
        cfg = ScenarioExecutorConfig()
        assert cfg.max_workers == 2  # DEFAULT_ASYNC_MAX_WORKERS
        assert cfg.max_concurrent_requests is None
        assert cfg.enable_cache is True
        assert cfg.timeout_seconds is None

    def test_custom_values(self) -> None:
        """Custom values override defaults."""
        cfg = ScenarioExecutorConfig(
            max_workers=8,
            max_concurrent_requests=10,
            enable_cache=False,
            timeout_seconds=300.0,
        )
        assert cfg.max_workers == 8
        assert cfg.max_concurrent_requests == 10
        assert cfg.enable_cache is False
        assert cfg.timeout_seconds == 300.0
