"""Scenario execution with parallel/sequential support.

This module provides the ScenarioExecutor class for running scenario verification
combos either sequentially or in parallel using ThreadPoolExecutor with asyncio
BlockingPortal for proper async event loop management.

Peer to VerificationExecutor, adapted for multi-turn scenario execution
where each "task" is an entire scenario graph traversal rather than a single
question verification.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anyio.from_thread import start_blocking_portal

if TYPE_CHECKING:
    from karenina.schemas.entities import Rubric
    from karenina.schemas.verification import VerificationConfig

from karenina.scenario.manager import ScenarioManager
from karenina.schemas.scenario.state import ScenarioExecutionResult
from karenina.schemas.verification.config import DEFAULT_ASYNC_MAX_WORKERS
from karenina.utils.answer_cache import AnswerTraceCache

from .async_lifecycle import (
    PRE_TEARDOWN_ACLOSE_TIMEOUT,
    aclose_portal_adapters,
    get_global_llm_limiter,
)
from .portal_pool import ExecutionTuning, run_in_portal_pool, sequential_portal

logger = logging.getLogger(__name__)

# Type alias for a scenario combo: (scenario_def, answering_model, parsing_model, replicate).
# The fourth element is None for single-replicate runs, mirroring the QA convention.
ScenarioCombo = tuple[Any, Any, Any, int | None]

# Type alias for progress callback: (completed, total, result_or_none)
ProgressCallback = Callable[[int, int, ScenarioExecutionResult | None], None]


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ScenarioExecutorConfig:
    """Configuration for scenario execution.

    Attributes:
        max_workers: Maximum number of parallel worker threads.
        max_concurrent_requests: GlobalLLMLimiter capacity entered for the
            duration of the batch (process-wide cap on concurrent LLM
            request setups). None leaves the limiter unconfigured.
        enable_cache: Whether to enable node-level answer caching.
        timeout_seconds: Maximum wall-clock seconds for a parallel batch. Set
            to None (the default) to disable the batch-level timeout; the
            executor then runs until all combos finish. Set to a positive float
            to enforce a ceiling.
    """

    max_workers: int = DEFAULT_ASYNC_MAX_WORKERS
    max_concurrent_requests: int | None = None
    enable_cache: bool = True
    timeout_seconds: float | None = None

    def to_tuning(self) -> ExecutionTuning:
        """Derive the shared :class:`ExecutionTuning` for the portal pool.

        ``enable_cache`` stays scenario-executor-local (it shapes the worker
        callable, not the pool).
        """
        return ExecutionTuning(
            max_workers=self.max_workers,
            timeout_seconds=self.timeout_seconds,
            max_concurrent_requests=self.max_concurrent_requests,
        )


# ============================================================================
# Executor
# ============================================================================


class ScenarioExecutor:
    """Executes scenario verification combos with parallel/sequential support.

    Each "task" is a full scenario graph traversal via ScenarioManager.run().
    Supports both sequential (single thread, single portal) and parallel
    (thread pool with shared portal) execution modes.

    Example:
        >>> executor = ScenarioExecutor(parallel=True, config=ScenarioExecutorConfig(max_workers=4))
        >>> results, errors = executor.run_batch(combos, config, global_rubric=rubric)
    """

    def __init__(
        self,
        parallel: bool = True,
        config: ScenarioExecutorConfig | None = None,
    ) -> None:
        """Initialize the scenario executor.

        Args:
            parallel: Whether to run combos in parallel (default: True).
            config: Optional execution configuration.
        """
        self.parallel = parallel
        self.config = config or ScenarioExecutorConfig()

    def run_batch(
        self,
        combos: list[ScenarioCombo],
        config: VerificationConfig,
        global_rubric: Rubric | None = None,
        run_name: str | None = None,
        progress_callback: ProgressCallback | None = None,
        workspace_root: Path | None = None,
    ) -> tuple[list[ScenarioExecutionResult], list[tuple[str, BaseException]]]:
        """Execute scenario combos and return results with errors.

        Args:
            combos: List of (scenario_def, answering_model, parsing_model, replicate)
                tuples. The fourth element is None for single-replicate runs.
            config: Verification configuration (non-model settings).
            global_rubric: Optional global rubric applied per-turn.
            run_name: Optional run name for tracking.
            progress_callback: Optional callback(completed, total, result_or_none)
                called before (with None) and after (with result) each combo.
            workspace_root: Optional workspace root directory. Plumbed into
                each ScenarioManager.run(...) call so GenerateAnswer's
                workspace resolution works when agentic_parsing is enabled.

        Returns:
            Tuple of (results_list, errors_list). Results preserve the original
            combo ordering. Errors are (description, exception) tuples.
        """
        if self.parallel:
            results, errors = self._run_parallel(
                combos, config, global_rubric, run_name, progress_callback, workspace_root
            )
        else:
            results, errors = self._run_sequential(
                combos, config, global_rubric, run_name, progress_callback, workspace_root
            )

        if not errors and config.workspace_output_mode != "none" and config.workspace_output_dir is not None:
            from karenina.benchmark.verification.workspace_capture import compact_manifest

            try:
                compact_manifest(config.workspace_output_dir)
            except Exception:  # noqa: BLE001
                logger.warning("Workspace capture manifest compaction raised; continuing", exc_info=True)
        return results, errors

    @staticmethod
    def _describe_combo(combo: ScenarioCombo) -> str:
        """Build the human-readable combo label used in error entries and logs."""
        scenario_def, ans_model, parse_model, replicate = combo
        return f"Scenario '{scenario_def.name}' with {ans_model.model_name}/{parse_model.model_name}" + (
            f" rep={replicate}" if replicate is not None else ""
        )

    @staticmethod
    def _make_turn_callback(
        idx: int,
        scenario_name: str,
        partial_progress: dict[int, dict[str, Any]],
        progress_lock: threading.Lock,
    ) -> Callable[..., None]:
        """Create a per-turn callback that records combo progress.

        The callback writes into ``partial_progress`` under
        ``progress_lock``, the same lock that serializes batch progress
        dispatch, so all progress state mutation stays single-locked.
        """

        def _on_turn(**kwargs: Any) -> None:
            turn_num = kwargs.get("scenario_turn", 0)
            node_id = kwargs.get("scenario_node", "")
            with progress_lock:
                partial_progress[idx] = {
                    "scenario_id": scenario_name,
                    "turn": turn_num,
                    "node": node_id,
                }

        return _on_turn

    def _run_sequential(
        self,
        combos: list[ScenarioCombo],
        config: VerificationConfig,
        global_rubric: Rubric | None,
        run_name: str | None,
        progress_callback: ProgressCallback | None,
        workspace_root: Path | None = None,
    ) -> tuple[list[ScenarioExecutionResult], list[tuple[str, BaseException]]]:
        """Execute combos one at a time with a shared BlockingPortal.

        On failure, logs the error and continues processing remaining combos.
        The portal lifecycle (shared event loop plus pre-teardown adapter
        aclose) comes from :func:`sequential_portal`; the factory and sweep
        closures resolve ``start_blocking_portal`` /
        ``aclose_portal_adapters`` / ``PRE_TEARDOWN_ACLOSE_TIMEOUT`` from
        this module's globals at call time so existing monkeypatch targets
        keep working. The per-turn progress callback is wired here too
        (mirroring parallel mode), so sequential runs record turn/node
        progress per combo.

        Args:
            combos: Scenario combos to execute.
            config: Verification configuration.
            global_rubric: Optional global rubric.
            run_name: Optional run name.
            progress_callback: Optional progress callback.

        Returns:
            Tuple of (results, errors).
        """
        answer_cache = AnswerTraceCache() if self.config.enable_cache else None

        results: list[ScenarioExecutionResult] = []
        errors: list[tuple[str, BaseException]] = []
        total = len(combos)
        partial_progress: dict[int, dict[str, Any]] = {}
        progress_lock = threading.Lock()

        # Ref-counted enable of the process-wide LLM concurrency cap for
        # the duration of the batch. The adapters' async leaves borrow
        # permits, and with capacity None this is a no-op enable.
        with (
            get_global_llm_limiter().configure(self.config.max_concurrent_requests),
            sequential_portal(
                portal_factory=lambda: start_blocking_portal(backend="asyncio"),
                pre_teardown=lambda portal: aclose_portal_adapters(portal, timeout=PRE_TEARDOWN_ACLOSE_TIMEOUT),
            ),
        ):
            for idx, combo in enumerate(combos, 1):
                scenario_def, ans_model, parse_model, replicate = combo
                combo_desc = self._describe_combo(combo)

                # Progress callback before starting (result=None)
                if progress_callback:
                    progress_callback(idx, total, None)

                try:
                    manager = ScenarioManager()
                    exec_result = manager.run(
                        scenario=scenario_def,
                        config=config,
                        base_answering_model=ans_model,
                        base_parsing_model=parse_model,
                        run_name=run_name,
                        global_rubric=global_rubric,
                        answer_cache=answer_cache,
                        progress_callback=self._make_turn_callback(
                            idx - 1, scenario_def.name, partial_progress, progress_lock
                        ),
                        workspace_root=workspace_root,
                        replicate=replicate,
                    )
                    results.append(exec_result)
                except Exception as e:
                    logger.error("Sequential scenario failed: %s: %s", combo_desc, e)
                    errors.append((combo_desc, e))
                    continue

                # Progress callback after completion (with result)
                if progress_callback:
                    progress_callback(idx, total, exec_result)

        return results, errors

    def _run_parallel(
        self,
        combos: list[ScenarioCombo],
        config: VerificationConfig,
        global_rubric: Rubric | None,
        run_name: str | None,
        progress_callback: ProgressCallback | None,
        workspace_root: Path | None = None,
    ) -> tuple[list[ScenarioExecutionResult], list[tuple[str, BaseException]]]:
        """Execute combos in parallel via the shared portal pool.

        The pooled skeleton (per-worker portals, timeout sweep,
        post-shutdown drain, drop detection, pre-teardown aclose, progress
        serialization) lives in :func:`run_in_portal_pool`. This method
        keeps the scenario specifics: ``ScenarioManager.run`` dispatch, the
        per-turn progress callback, the limiter gate, and the aggregate
        timeout diagnostic with in-flight/recovered accounting. It never
        raises for per-combo failures or batch timeout; everything surfaces
        through the returned ``(results, errors)`` tuple.

        Args:
            combos: Scenario combos to execute.
            config: Verification configuration.
            global_rubric: Optional global rubric.
            run_name: Optional run name.
            progress_callback: Optional progress callback.

        Returns:
            Tuple of (results, errors) with results in original combo order.
        """
        logger.info("Parallel scenario execution: %d combos with %d workers", len(combos), self.config.max_workers)

        answer_cache = AnswerTraceCache() if self.config.enable_cache else None

        total = len(combos)
        partial_progress: dict[int, dict[str, Any]] = {}
        progress_lock = threading.Lock()

        def execute_combo(idx: int, combo: ScenarioCombo) -> ScenarioExecutionResult:
            """Execute a single scenario combo using the worker's portal."""
            scenario_def, ans_model, parse_model, replicate = combo
            manager = ScenarioManager()
            return manager.run(
                scenario=scenario_def,
                config=config,
                base_answering_model=ans_model,
                base_parsing_model=parse_model,
                run_name=run_name,
                global_rubric=global_rubric,
                answer_cache=answer_cache,
                progress_callback=self._make_turn_callback(idx, scenario_def.name, partial_progress, progress_lock),
                workspace_root=workspace_root,
                replicate=replicate,
            )

        outcome = run_in_portal_pool(
            combos,
            execute_combo,
            tuning=self.config.to_tuning(),
            on_progress=progress_callback,
            describe=lambda _idx, combo: self._describe_combo(combo),
            # Ref-counted enable of the process-wide LLM concurrency cap for
            # the duration of the batch (see _run_sequential). The pool
            # enters and exits the gate around the run; configure's
            # ref-counting gives save-and-restore semantics.
            gate=get_global_llm_limiter().configure(self.config.max_concurrent_requests),
            # Closures over this module's globals so monkeypatches on
            # scenario_executor.start_blocking_portal /
            # scenario_executor.aclose_portal_adapters /
            # scenario_executor.PRE_TEARDOWN_ACLOSE_TIMEOUT keep working.
            portal_factory=lambda: start_blocking_portal(backend="asyncio"),
            pre_teardown=lambda portal: aclose_portal_adapters(portal, timeout=PRE_TEARDOWN_ACLOSE_TIMEOUT),
            progress_lock=progress_lock,
            item_noun="Combo",
        )

        # Restore original order
        results: list[ScenarioExecutionResult] = [
            outcome.results_by_index[idx] for idx in sorted(outcome.results_by_index.keys())
        ]
        failed_tasks: list[tuple[str, BaseException]] = list(outcome.failed_items)

        if outcome.timed_out:
            # Log partial progress for combos that were in-flight at the
            # moment the timeout fired. Some of these may have been recovered
            # by the post-shutdown drain and now appear in the results; the
            # diagnostic message still reports them so operators can see
            # what was running at the critical moment.
            in_flight_at_timeout: set[int] = outcome.diagnostics["in_flight_at_timeout"]
            completed_at_timeout: int = outcome.diagnostics["completed_at_timeout"]
            in_flight_info: list[str] = []
            for idx in sorted(in_flight_at_timeout):
                if idx in partial_progress:
                    info = partial_progress[idx]
                    in_flight_info.append(
                        f"  combo {idx}: {info['scenario_id']} reached turn {info['turn']} at node {info['node']}"
                    )

            in_flight_count = len(in_flight_at_timeout)
            not_started = total - completed_at_timeout - in_flight_count
            # The pool only sets timed_out when a finite timeout was passed,
            # so timeout_seconds is guaranteed non-None here.
            timeout_label = (
                f"{self.config.timeout_seconds:.0f}s" if self.config.timeout_seconds is not None else "unknown timeout"
            )
            timeout_msg = (
                f"Parallel scenario batch timed out after {timeout_label} "
                f"({completed_at_timeout} completed, {in_flight_count} in-flight, "
                f"{not_started} not started of {total} combos)"
            )
            if in_flight_info:
                timeout_msg += "\nIn-flight combo progress:\n" + "\n".join(in_flight_info)

            # Report how many in-flight combos were recovered vs lost after
            # the drain finished.
            recovered_from_drain: int = outcome.diagnostics["recovered_from_drain"]
            if recovered_from_drain:
                timeout_msg += (
                    f"\nRecovered {recovered_from_drain} of {in_flight_count} in-flight "
                    f"combos during post-shutdown drain."
                )

            logger.error("%s", timeout_msg)
            failed_tasks.append((timeout_msg, TimeoutError(timeout_msg)))

        return results, failed_tasks
