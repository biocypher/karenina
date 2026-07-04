"""Regression tests for issue 189: thread pool shutdown race drops in-flight combos.

When a parallel batch hits ``timeout_seconds``, the old implementation called
``pool.shutdown(wait=False, cancel_futures=True)`` and immediately tore down
the per-worker portals. Any combo whose Future was still in the ``RUNNING``
state at that moment was silently dropped: the worker eventually set a result
on the Future, but nobody ever harvested it into ``results_by_index``.

These tests reproduce that race deterministically with fake combos that
finish a short, fixed delay **after** the batch timeout fires. They assert
that every submitted combo is accounted for in either the returned results
or a per-combo error entry.

See ``issues/189-thread-pool-shutdown-race-during-final-verification.md``.
"""

from __future__ import annotations

import time
from typing import Any
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


def _make_exec_result(scenario_id: str) -> MagicMock:
    """Create a mock ScenarioExecutionResult with the expected scenario_id."""
    result = MagicMock()
    result.scenario_id = scenario_id
    result.status = "completed"
    result.turn_count = 1
    return result


def _is_batch_aggregate_timeout(desc: str, exc: BaseException) -> bool:
    """Detect the batch-level aggregate TimeoutError entry.

    ``ScenarioExecutor._run_parallel`` appends a single aggregate TimeoutError
    to ``failed_tasks`` on timeout for operator visibility. That entry is not
    per-combo; it represents the batch as a whole. Per-combo entries reference
    a specific ``combo_desc`` string containing the scenario name.
    """
    return isinstance(exc, TimeoutError) and "Parallel scenario batch timed out" in str(exc)


def _combo_is_accounted_for(
    combo_name: str,
    results: list[Any],
    errors: list[tuple[str, BaseException]],
) -> bool:
    """Return True iff the combo is in results or has its own per-combo error entry."""
    result_names = {r.scenario_id for r in results}
    if combo_name in result_names:
        return True
    for desc, exc in errors:
        if _is_batch_aggregate_timeout(desc, exc):
            continue
        if combo_name in desc:
            return True
    return False


# ============================================================================
# Silent-loss repro (issue 189 primary symptom)
# ============================================================================


@pytest.mark.unit
class TestInFlightCombosNotSilentlyDropped:
    """Combos still running when a batch timeout fires must not be silently lost.

    These tests reproduce the race described in issue 189. They are expected
    to fail against the pre-fix implementation (which drops in-flight combos)
    and pass against the fixed implementation (which either harvests them
    after ``pool.shutdown(wait=True)`` returns or emits a per-combo error).
    """

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_slow_combos_finishing_after_timeout_are_recovered(self, mock_manager_cls: MagicMock) -> None:
        """Four combos: two fast, two slow. The slow ones finish shortly after
        the batch timeout fires. All four must appear in results or errors.
        """
        combos = [_make_combo(f"c{i}") for i in range(4)]
        fast_names = {"c0", "c1"}

        timeout_s = 0.3
        slow_delay = 0.8  # slow combos wake up 0.5s past the batch timeout

        def mock_run(**kwargs: Any) -> MagicMock:
            name = kwargs["scenario"].name
            if name not in fast_names:
                time.sleep(slow_delay)
            return _make_exec_result(name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=4,
                enable_cache=False,
                timeout_seconds=timeout_s,
            ),
        )
        results, errors = executor.run_batch(combos, MagicMock())

        # Every submitted combo must be accounted for. The fast ones through
        # ``results``; the slow ones either through ``results`` (post-shutdown
        # sweep harvested them) or through a per-combo entry in ``errors``
        # (they were cancelled). The batch-level aggregate TimeoutError does
        # not count: it represents the batch, not any specific combo.
        for combo_name in ("c0", "c1", "c2", "c3"):
            assert _combo_is_accounted_for(combo_name, results, errors), (
                f"Combo {combo_name} was silently dropped. "
                f"results={[r.scenario_id for r in results]}, "
                f"errors={[(d, type(e).__name__, str(e)[:80]) for d, e in errors]}"
            )

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_slow_combo_failing_after_timeout_is_recorded_per_combo(self, mock_manager_cls: MagicMock) -> None:
        """A slow combo that raises after the batch timeout must appear as a
        per-combo entry in ``failed_tasks``, not be silently dropped.
        """
        combos = [_make_combo(f"c{i}") for i in range(4)]
        fast_names = {"c0", "c1"}

        timeout_s = 0.3
        slow_delay = 0.8

        def mock_run(**kwargs: Any) -> MagicMock:
            name = kwargs["scenario"].name
            if name not in fast_names:
                time.sleep(slow_delay)
                raise RuntimeError(f"scenario {name} broke late")
            return _make_exec_result(name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=4,
                enable_cache=False,
                timeout_seconds=timeout_s,
            ),
        )
        results, errors = executor.run_batch(combos, MagicMock())

        # Fast combos land in results.
        result_names = {r.scenario_id for r in results}
        assert "c0" in result_names
        assert "c1" in result_names

        # Slow combos must each appear as a per-combo error entry (RuntimeError),
        # not be silently dropped.
        for combo_name in ("c2", "c3"):
            matching = [(desc, exc) for desc, exc in errors if combo_name in desc and isinstance(exc, RuntimeError)]
            assert matching, (
                f"Combo {combo_name} failed after timeout but produced no per-combo "
                f"error entry. errors={[(d, type(e).__name__, str(e)[:80]) for d, e in errors]}"
            )

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_all_combos_slow_and_late_finishing_are_all_recovered(self, mock_manager_cls: MagicMock) -> None:
        """When every combo is still in-flight at the timeout moment, the fix
        must recover all of them via the post-shutdown sweep. This is the
        worst case for the race: zero fast anchors in ``results_by_index``
        before the timeout fires.
        """
        combos = [_make_combo(f"late{i}") for i in range(3)]

        timeout_s = 0.3
        slow_delay = 0.6  # all combos wake up 0.3s after the timeout

        def mock_run(**kwargs: Any) -> MagicMock:
            time.sleep(slow_delay)
            return _make_exec_result(kwargs["scenario"].name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=3,
                enable_cache=False,
                timeout_seconds=timeout_s,
            ),
        )
        results, errors = executor.run_batch(combos, MagicMock())

        for combo_name in ("late0", "late1", "late2"):
            assert _combo_is_accounted_for(combo_name, results, errors), (
                f"Combo {combo_name} was silently dropped. "
                f"results={[r.scenario_id for r in results]}, "
                f"errors={[(d, type(e).__name__, str(e)[:80]) for d, e in errors]}"
            )


# ============================================================================
# Happy path: no regression in the non-timeout case
# ============================================================================


@pytest.mark.unit
class TestNoRegressionOnHappyPath:
    """The silent-loss fix must not change behavior when the batch does not time out."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_fast_combos_without_timeout_produce_clean_results(self, mock_manager_cls: MagicMock) -> None:
        """Fast combos with a generous timeout must produce clean results and
        an empty errors list. No synthetic timeout entries should appear.
        """
        combos = [_make_combo(f"fast{i}") for i in range(4)]

        def mock_run(**kwargs: Any) -> MagicMock:
            return _make_exec_result(kwargs["scenario"].name)

        mock_manager_cls.return_value.run.side_effect = mock_run

        executor = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(
                max_workers=4,
                enable_cache=False,
                timeout_seconds=30.0,
            ),
        )
        results, errors = executor.run_batch(combos, MagicMock())

        assert len(results) == 4
        assert {r.scenario_id for r in results} == {"fast0", "fast1", "fast2", "fast3"}
        assert errors == []
