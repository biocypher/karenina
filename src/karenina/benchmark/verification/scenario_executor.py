"""Scenario execution with parallel/sequential support.

This module provides the ScenarioExecutor class for running scenario verification
combos either sequentially or in parallel using a thread pool with asyncio
BlockingPortal for proper async event loop management.

Peer to VerificationExecutor, adapted for multi-turn scenario execution
where each "task" is an entire scenario graph traversal rather than a single
question verification.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
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

from .executor import set_async_portal, set_global_llm_semaphore

logger = logging.getLogger(__name__)

# Type alias for a scenario combo: (scenario_def, answering_model, parsing_model)
ScenarioCombo = tuple[Any, Any, Any]

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
        timeout_seconds: Maximum wall-clock seconds for a parallel batch (higher default
            than VerificationExecutor since scenarios are multi-turn).
    """

    max_workers: int = DEFAULT_ASYNC_MAX_WORKERS
    max_concurrent_requests: int | None = None
    enable_cache: bool = True
    timeout_seconds: float = 1200.0


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
    ) -> tuple[list[ScenarioExecutionResult], list[tuple[str, BaseException]]]:
        """Execute scenario combos and return results with errors.

        Args:
            combos: List of (scenario_def, answering_model, parsing_model) tuples.
            config: Verification configuration (non-model settings).
            global_rubric: Optional global rubric applied per-turn.
            run_name: Optional run name for tracking.
            progress_callback: Optional callback(completed, total, result_or_none)
                called before (with None) and after (with result) each combo.

        Returns:
            Tuple of (results_list, errors_list). Results preserve the original
            combo ordering. Errors are (description, exception) tuples.
        """
        if self.parallel:
            return self._run_parallel(combos, config, global_rubric, run_name, progress_callback)
        return self._run_sequential(combos, config, global_rubric, run_name, progress_callback)

    def _run_sequential(
        self,
        combos: list[ScenarioCombo],
        config: VerificationConfig,
        global_rubric: Rubric | None,
        run_name: str | None,
        progress_callback: ProgressCallback | None,
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
                    for idx, (scenario_def, ans_model, parse_model) in enumerate(combos, 1):
                        combo_desc = (
                            f"Scenario '{scenario_def.name}' with {ans_model.model_name}/{parse_model.model_name}"
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
    ) -> tuple[list[ScenarioExecutionResult], list[tuple[str, BaseException]]]:
        """Execute combos in parallel with thread pool and shared BlockingPortal.

        Uses threading.Thread workers with a shared work queue, mirroring the
        pattern in VerificationExecutor._run_parallel.

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

        # Create work queue with (original_index, combo)
        work_queue: queue.Queue[tuple[int, ScenarioCombo] | None] = queue.Queue()
        for idx, combo in enumerate(combos):
            work_queue.put((idx, combo))

        # Thread-safe storage for results (indexed by original position)
        results_lock = threading.Lock()
        results_by_index: dict[int, ScenarioExecutionResult] = {}
        # Partial progress for in-flight combos (turn tracking via callback)
        partial_progress: dict[int, dict[str, Any]] = {}

        # Thread-safe error tracking
        failed_tasks: list[tuple[str, BaseException]] = []

        # Thread-safe progress tracking
        progress_lock = threading.Lock()
        completed_count = [0]

        # Completion tracking
        tasks_completed_event = threading.Event()

        def worker(portal: BlockingPortal) -> None:
            """Worker that processes scenario combos from the shared queue."""
            set_async_portal(portal)
            try:
                while True:
                    try:
                        item = work_queue.get(timeout=1.0)
                    except queue.Empty:
                        with results_lock:
                            if len(results_by_index) + len(failed_tasks) >= total:
                                tasks_completed_event.set()
                        continue

                    if item is None:
                        work_queue.task_done()
                        break

                    original_index, (scenario_def, ans_model, parse_model) = item
                    combo_desc = f"Scenario '{scenario_def.name}' with {ans_model.model_name}/{parse_model.model_name}"

                    # Progress callback before starting (result=None)
                    if progress_callback:
                        with progress_lock:
                            progress_callback(completed_count[0] + 1, total, None)

                    try:
                        manager = ScenarioManager()

                        def make_turn_callback(idx: int, scenario_name: str) -> Callable[..., None]:
                            """Create a per-turn callback that tracks progress."""

                            def _on_turn(**kwargs: Any) -> None:
                                turn_num = kwargs.get("scenario_turn", 0)
                                node_id = kwargs.get("scenario_node", "")
                                with results_lock:
                                    partial_progress[idx] = {
                                        "scenario_id": scenario_name,
                                        "turn": turn_num,
                                        "node": node_id,
                                    }

                            return _on_turn

                        exec_result = manager.run(
                            scenario=scenario_def,
                            config=config,
                            base_answering_model=ans_model,
                            base_parsing_model=parse_model,
                            run_name=run_name,
                            global_rubric=global_rubric,
                            answer_cache=answer_cache,
                            progress_callback=make_turn_callback(original_index, scenario_def.name),
                        )

                        with results_lock:
                            results_by_index[original_index] = exec_result
                            if len(results_by_index) + len(failed_tasks) >= total:
                                tasks_completed_event.set()

                        # Progress callback after completion (with result)
                        if progress_callback:
                            with progress_lock:
                                completed_count[0] += 1
                                progress_callback(completed_count[0], total, exec_result)

                    except Exception as e:
                        logger.error("Parallel scenario failed: %s: %s", combo_desc, e)
                        with results_lock:
                            failed_tasks.append((combo_desc, e))
                            if len(results_by_index) + len(failed_tasks) >= total:
                                tasks_completed_event.set()

                    finally:
                        work_queue.task_done()

            finally:
                set_async_portal(None)

        # Set global semaphore before spawning workers
        set_global_llm_semaphore(sem)
        try:
            with start_blocking_portal(backend="asyncio") as portal:
                workers = []
                for _ in range(max_workers):
                    t = threading.Thread(target=worker, args=(portal,), daemon=True)
                    t.start()
                    workers.append(t)

                # Wait for all combos to complete, with timeout
                completed = tasks_completed_event.wait(timeout=self.config.timeout_seconds)

                # Send shutdown signal to workers
                for _ in range(max_workers):
                    work_queue.put(None)

                # Wait for workers to finish
                for t in workers:
                    t.join(timeout=5.0)
        finally:
            set_global_llm_semaphore(None)

        # Restore original order
        results: list[ScenarioExecutionResult] = []
        for idx in sorted(results_by_index.keys()):
            results.append(results_by_index[idx])

        if not completed:
            # Log partial progress for in-flight combos
            in_flight_info: list[str] = []
            for idx in range(total):
                if idx not in results_by_index and idx in partial_progress:
                    info = partial_progress[idx]
                    in_flight_info.append(
                        f"  combo {idx}: {info['scenario_id']} reached turn {info['turn']} at node {info['node']}"
                    )

            completed_count_final = len(results_by_index)
            partial_count = len(in_flight_info)
            timeout_msg = (
                f"Parallel scenario batch timed out after {self.config.timeout_seconds:.0f}s "
                f"({completed_count_final} completed, {partial_count} in-flight, "
                f"{total - completed_count_final - partial_count} not started of {total} combos)"
            )
            if in_flight_info:
                timeout_msg += "\nIn-flight combo progress:\n" + "\n".join(in_flight_info)
            logger.error("%s", timeout_msg)
            failed_tasks.append((timeout_msg, TimeoutError(timeout_msg)))

        return results, failed_tasks
