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
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anyio.from_thread import start_blocking_portal

if TYPE_CHECKING:
    from anyio.from_thread import BlockingPortal

    from karenina.schemas.entities import Rubric
    from karenina.schemas.verification import VerificationConfig

from karenina.scenario.manager import ScenarioManager
from karenina.schemas.scenario.state import ScenarioExecutionResult
from karenina.schemas.verification.config import DEFAULT_ASYNC_MAX_WORKERS
from karenina.utils.answer_cache import AnswerTraceCache

from .executor import get_async_portal, set_async_portal, set_global_llm_semaphore

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
        max_concurrent_requests: Global LLM semaphore permits. None disables the semaphore.
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
        sem = (
            threading.Semaphore(self.config.max_concurrent_requests)
            if self.config.max_concurrent_requests is not None
            else None
        )

        results: list[ScenarioExecutionResult] = []
        errors: list[tuple[str, BaseException]] = []
        total = len(combos)

        set_global_llm_semaphore(sem)
        try:
            with start_blocking_portal(backend="asyncio") as portal:
                set_async_portal(portal)
                try:
                    for idx, (scenario_def, ans_model, parse_model, replicate) in enumerate(combos, 1):
                        combo_desc = (
                            f"Scenario '{scenario_def.name}' with "
                            f"{ans_model.model_name}/{parse_model.model_name}"
                            + (f" rep={replicate}" if replicate is not None else "")
                        )

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
                finally:
                    # Drop per-portal adapter tracking so a sequential run
                    # does not leak entries into the module-global map across
                    # batches. Sequential mode was never affected by the
                    # parallel teardown ordering bug, so there is no need to
                    # pre-close adapters here (cleanup_resources still handles
                    # it on the shared portal's live loop).
                    from karenina.adapters.registry import clear_portal_adapter_refs

                    clear_portal_adapter_refs(portal)
                    set_async_portal(None)
        finally:
            set_global_llm_semaphore(None)

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
        """Execute combos in parallel using ThreadPoolExecutor.

        Each combo is submitted as a separate Future with its own BlockingPortal.
        Results are collected via as_completed() with a timeout. Every submission
        produces a Future, so no combo can be silently lost.

        Args:
            combos: Scenario combos to execute.
            config: Verification configuration.
            global_rubric: Optional global rubric.
            run_name: Optional run name.
            progress_callback: Optional progress callback.

        Returns:
            Tuple of (results, errors) with results in original combo order.
        """
        max_workers = self.config.max_workers
        logger.info("Parallel scenario execution: %d combos with %d workers", len(combos), max_workers)

        answer_cache = AnswerTraceCache() if self.config.enable_cache else None
        sem = (
            threading.Semaphore(self.config.max_concurrent_requests)
            if self.config.max_concurrent_requests is not None
            else None
        )

        total = len(combos)
        partial_progress: dict[int, dict[str, Any]] = {}
        progress_lock = threading.Lock()
        completed_count = [0]

        def make_turn_callback(idx: int, scenario_name: str) -> Callable[..., None]:
            """Create a per-turn callback that tracks progress."""

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

        # Per-worker portal management: each worker thread creates one portal
        # that is reused across all combos on that thread. This preserves httpx
        # connection pools and avoids "Event loop is closed" errors from rapid
        # portal churn.
        #
        # Each entry is (context_manager, portal). The portal reference is
        # kept separately so the finally block can look up adapters tracked
        # against that portal in the registry and call their aclose() on the
        # portal's own event loop BEFORE the portal is torn down. See
        # VerificationExecutor._run_parallel for the parallel twin.
        _portal_resources: list[tuple[Any, BlockingPortal]] = []
        _portal_init_lock = threading.Lock()

        def _ensure_worker_portal() -> None:
            """Lazily create a BlockingPortal for this worker thread."""
            if get_async_portal() is not None:
                return
            cm = start_blocking_portal(backend="asyncio")
            portal = cm.__enter__()
            set_async_portal(portal)
            with _portal_init_lock:
                _portal_resources.append((cm, portal))

        def execute_combo(idx: int, combo: ScenarioCombo) -> tuple[int, ScenarioExecutionResult]:
            """Execute a single scenario combo using the worker's portal."""
            _ensure_worker_portal()
            scenario_def, ans_model, parse_model, replicate = combo
            manager = ScenarioManager()
            result = manager.run(
                scenario=scenario_def,
                config=config,
                base_answering_model=ans_model,
                base_parsing_model=parse_model,
                run_name=run_name,
                global_rubric=global_rubric,
                answer_cache=answer_cache,
                progress_callback=make_turn_callback(idx, scenario_def.name),
                workspace_root=workspace_root,
                replicate=replicate,
            )
            return (idx, result)

        results_by_index: dict[int, ScenarioExecutionResult] = {}
        failed_tasks: list[tuple[str, BaseException]] = []

        set_global_llm_semaphore(sem)
        try:
            pool = ThreadPoolExecutor(max_workers=max_workers)
            try:
                # Submit all combos
                future_to_meta: dict[Future[tuple[int, ScenarioExecutionResult]], tuple[int, str]] = {}
                for idx, (scenario_def, ans_model, parse_model, replicate) in enumerate(combos):
                    combo_desc = (
                        f"Scenario '{scenario_def.name}' with "
                        f"{ans_model.model_name}/{parse_model.model_name}"
                        + (f" rep={replicate}" if replicate is not None else "")
                    )
                    future = pool.submit(execute_combo, idx, (scenario_def, ans_model, parse_model, replicate))
                    future_to_meta[future] = (idx, combo_desc)

                # Collect results as they complete
                collected: set[Future[tuple[int, ScenarioExecutionResult]]] = set()
                timed_out = False
                # Snapshot of indices still in-flight at the moment the batch
                # timeout fires, taken before the post-shutdown drain runs.
                # Used for diagnostic messages so operators can see what was
                # running at the timeout even if those combos are later
                # recovered by pool.shutdown(wait=True).
                in_flight_at_timeout: set[int] = set()
                completed_at_timeout = 0
                try:
                    for future in as_completed(future_to_meta, timeout=self.config.timeout_seconds):
                        collected.add(future)
                        idx, combo_desc = future_to_meta[future]
                        try:
                            result_idx, exec_result = future.result()
                            results_by_index[result_idx] = exec_result

                            if progress_callback:
                                with progress_lock:
                                    completed_count[0] += 1
                                    progress_callback(completed_count[0], total, exec_result)
                        except BaseException as e:
                            logger.error("Parallel scenario failed: %s: %s", combo_desc, e)
                            failed_tasks.append((combo_desc, e))
                except TimeoutError:
                    timed_out = True
                    # Sweep futures that completed between last yield and timeout
                    for future, (_idx, combo_desc) in future_to_meta.items():
                        if future.done() and future not in collected:
                            collected.add(future)
                            try:
                                result_idx, exec_result = future.result()
                                results_by_index[result_idx] = exec_result
                            except BaseException as e:
                                failed_tasks.append((combo_desc, e))
                    # Snapshot indices that were in-flight (not in
                    # results_by_index) at this moment, before the post-shutdown
                    # drain runs. The drain may recover some of these.
                    completed_at_timeout = len(results_by_index)
                    in_flight_at_timeout = {idx for idx in range(total) if idx not in results_by_index}
            finally:
                # Always wait for pool workers to finish before tearing down
                # worker portals. With wait=False, in-flight workers could
                # outlive their portals and crash on callback dispatch with
                # "cannot schedule new futures after shutdown", or set Future
                # results that are never harvested (silent drop).
                pool.shutdown(wait=True, cancel_futures=timed_out)

                # Post-shutdown sweep: after wait=True returns, every future
                # is in a terminal state. Harvest any futures that finished
                # (successfully or with an error) while the pool was draining.
                if timed_out:
                    for future, (_idx, combo_desc) in future_to_meta.items():
                        if future in collected:
                            continue
                        if future.cancelled():
                            failed_tasks.append(
                                (
                                    combo_desc,
                                    TimeoutError(f"Combo cancelled before start: {combo_desc}"),
                                )
                            )
                            collected.add(future)
                            continue
                        try:
                            result_idx, exec_result = future.result()
                            results_by_index[result_idx] = exec_result
                        except BaseException as e:
                            failed_tasks.append((combo_desc, e))
                        collected.add(future)

                # Drop-detection invariant. After pool.shutdown(wait=True)
                # every future must be in a terminal state, so the sweeps
                # above should have covered everything. If this fires, the
                # shutdown race has reopened and combos were being lost.
                uncollected = [(future, meta) for future, meta in future_to_meta.items() if future not in collected]
                if uncollected:
                    logger.error(
                        "Parallel executor dropped %d combos after pool.shutdown(wait=True). "
                        "Emitting synthetic failure entries.",
                        len(uncollected),
                    )
                    for future, (_idx, combo_desc) in uncollected:
                        failed_tasks.append(
                            (
                                combo_desc,
                                TimeoutError(f"Combo left uncollected by parallel executor: {combo_desc}"),
                            )
                        )
                        collected.add(future)

                # Pre-teardown aclose: close adapter-owned httpx clients on
                # the portal loop that opened them, BEFORE the portal is torn
                # down. Without this, the downstream cleanup_resources() call
                # runs on a fresh loop and httpx raises "Event loop is closed"
                # because its transports are pinned to the dead portal loop.
                # Bounded timeout mirrors langchain/parser.py:297-312 to avoid
                # wedging the finally block on a stuck aclose.
                from karenina.adapters.registry import (
                    clear_portal_adapter_refs,
                    snapshot_adapters_for_portal,
                )

                from .executor import PRE_TEARDOWN_ACLOSE_TIMEOUT

                for _cm, portal in _portal_resources:
                    for adapter in snapshot_adapters_for_portal(portal):
                        if not hasattr(adapter, "aclose"):
                            continue
                        future = portal.start_task_soon(adapter.aclose)
                        try:
                            future.result(timeout=PRE_TEARDOWN_ACLOSE_TIMEOUT)
                        except TimeoutError:
                            # Cancel the abandoned coroutine so the portal's
                            # loop does not block its context manager's
                            # __exit__ waiting for it to finish.
                            future.cancel()
                            logger.warning(
                                "Pre-teardown aclose timed out on %s (>%ss); proceeding with portal teardown",
                                type(adapter).__name__,
                                PRE_TEARDOWN_ACLOSE_TIMEOUT,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Pre-teardown aclose failed on %s: %s",
                                type(adapter).__name__,
                                exc,
                            )
                    clear_portal_adapter_refs(portal)

                # Clean up worker portals (event loops) only after workers
                # have fully exited, so no worker can still be using a portal
                # when its context manager exits.
                for cm, _portal in _portal_resources:
                    try:
                        cm.__exit__(None, None, None)
                    except Exception:
                        logger.debug("Portal cleanup error", exc_info=True)
        finally:
            set_global_llm_semaphore(None)

        # Restore original order
        results: list[ScenarioExecutionResult] = []
        for idx in sorted(results_by_index.keys()):
            results.append(results_by_index[idx])

        if timed_out:
            # Log partial progress for combos that were in-flight at the
            # moment the timeout fired. Some of these may have been recovered
            # by the post-shutdown drain and now appear in results_by_index;
            # the diagnostic message still reports them so operators can see
            # what was running at the critical moment.
            in_flight_info: list[str] = []
            for idx in sorted(in_flight_at_timeout):
                if idx in partial_progress:
                    info = partial_progress[idx]
                    in_flight_info.append(
                        f"  combo {idx}: {info['scenario_id']} reached turn {info['turn']} at node {info['node']}"
                    )

            in_flight_count = len(in_flight_at_timeout)
            not_started = total - completed_at_timeout - in_flight_count
            # as_completed only raises TimeoutError when a finite timeout was
            # passed, so timeout_seconds is guaranteed non-None here.
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
            recovered_from_drain = sum(1 for idx in in_flight_at_timeout if idx in results_by_index)
            if recovered_from_drain:
                timeout_msg += (
                    f"\nRecovered {recovered_from_drain} of {in_flight_count} in-flight "
                    f"combos during post-shutdown drain."
                )

            logger.error("%s", timeout_msg)
            failed_tasks.append((timeout_msg, TimeoutError(timeout_msg)))

        return results, failed_tasks
