"""Verification task execution with parallel/sequential support.

This module provides the VerificationExecutor class for running verification
tasks either sequentially or in parallel using ThreadPoolExecutor with asyncio
BlockingPortal for proper async event loop management.

The thread-local portal storage allows worker threads to share async execution
context without passing the portal through all function signatures. The portal
storage, the global LLM semaphore, and the pre-teardown aclose bound live in
the async_lifecycle leaf module and are re-exported here for back-compat.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anyio.from_thread import start_blocking_portal

if TYPE_CHECKING:
    from collections.abc import Iterable

from karenina.schemas.results.failure import FailureCategory
from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.config import DEFAULT_ASYNC_MAX_WORKERS
from karenina.utils.answer_cache import AnswerTraceCache

# Async lifecycle primitives moved to the async_lifecycle leaf module.
# Re-exported here (with explicit aliases) so that
# karenina.benchmark.verification.executor.<name> remains a valid import path
# and monkeypatch target for existing callers and tests.
from .async_lifecycle import (
    _SENTINEL as _SENTINEL,  # noqa: PLC0414
)
from .async_lifecycle import (
    PRE_TEARDOWN_ACLOSE_TIMEOUT as PRE_TEARDOWN_ACLOSE_TIMEOUT,  # noqa: PLC0414
)
from .async_lifecycle import (
    GlobalLLMLimiter as GlobalLLMLimiter,  # noqa: PLC0414
)
from .async_lifecycle import aclose_portal_adapters
from .async_lifecycle import (
    get_async_portal as get_async_portal,  # noqa: PLC0414
)
from .async_lifecycle import (
    get_global_llm_limiter as get_global_llm_limiter,  # noqa: PLC0414
)
from .async_lifecycle import (
    get_global_llm_semaphore as get_global_llm_semaphore,  # noqa: PLC0414
)
from .async_lifecycle import (
    set_async_portal as set_async_portal,  # noqa: PLC0414
)
from .async_lifecycle import (
    set_global_llm_semaphore as set_global_llm_semaphore,  # noqa: PLC0414
)
from .portal_pool import ExecutionTuning, run_in_portal_pool, sequential_portal

logger = logging.getLogger(__name__)

# Failure categories whose answerer trace is generally not worth hydrating into
# the workspace cache. Timeout rows are a special case below: when the adapter
# captured a non-empty partial trace, a partial-trace-scoring resume can use it
# as parser input and should not regenerate the answerer.
_INFRA_FAILURE_CATEGORIES: frozenset[FailureCategory] = frozenset(
    {
        FailureCategory.CONNECTION,
        FailureCategory.RATE_LIMIT,
        FailureCategory.TIMEOUT,
        FailureCategory.SERVER_ERROR,
        FailureCategory.UNEXPECTED_ERROR,
    }
)


def _is_parser_stage(stage: str | None) -> bool:
    return stage in {"parse_template", "ParseTemplate", "AgenticParseTemplate"}


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ExecutorConfig:
    """Configuration for verification execution.

    Attributes:
        max_workers: Maximum number of parallel workers (default: 2)
        enable_cache: Whether to enable answer caching (default: True)
        retry_wait_seconds: Seconds to wait for IN_PROGRESS cache entries (default: 5.0)
        timeout_seconds: Maximum wall-clock seconds for a parallel batch.
            Set to None (the default) to disable the batch-level timeout; the
            executor then runs until all tasks finish. Set to a positive float
            to enforce a ceiling.
        max_requeue_count: Maximum times a task can be requeued before forcing a
            fresh generation (default: 5). When exceeded, the cache entry is reset
            and the task restarts with retry_count=0.
        answerer_concurrency_limits: Optional dict mapping answerer
            ``ModelConfig.id`` to the max number of concurrent tasks allowed on
            that answerer. Already normalized (the public int-or-dict form is
            reduced to a plain dict by ``_normalize_answerer_limits``). ``None``
            (the default) disables caps; every task runs unthrottled.
    """

    max_workers: int = DEFAULT_ASYNC_MAX_WORKERS
    enable_cache: bool = True
    retry_wait_seconds: float = 5.0
    timeout_seconds: float | None = None
    max_requeue_count: int = 5
    answerer_concurrency_limits: dict[str, int] | None = None

    def to_tuning(self) -> ExecutionTuning:
        """Derive the shared :class:`ExecutionTuning` for the portal pool.

        ``enable_cache`` stays QA-executor-local (it shapes the worker
        callable, not the pool).
        """
        return ExecutionTuning(
            max_workers=self.max_workers,
            timeout_seconds=self.timeout_seconds,
            retry_wait_seconds=self.retry_wait_seconds,
            max_requeue_count=self.max_requeue_count,
            answerer_concurrency_limits=self.answerer_concurrency_limits,
        )


# ============================================================================
# Executor
# ============================================================================


class VerificationExecutor:
    """Executes verification tasks with parallel/sequential support.

    This class encapsulates the execution logic for running verification tasks,
    supporting both sequential and parallel execution modes with proper async
    event loop management via AnyIO BlockingPortal.

    Example:
        >>> executor = VerificationExecutor(parallel=True, config=ExecutorConfig(max_workers=4))
        >>> results = executor.run_batch(tasks, progress_callback=my_callback)
    """

    def __init__(self, parallel: bool = True, config: ExecutorConfig | None = None):
        """Initialize the verification executor.

        Args:
            parallel: Whether to run tasks in parallel (default: True)
            config: Optional execution configuration
        """
        self.parallel = parallel
        self.config = config or ExecutorConfig()
        self._endpoint_semaphores: dict[str, threading.Semaphore] = {}
        self._endpoint_semaphores_lock = threading.Lock()
        # Answerer ids already logged as running without a configured cap,
        # so the debug line fires once per id instead of once per task.
        self._uncapped_answerer_ids_logged: set[str] = set()

    def _get_endpoint_semaphore(self, ans_id: str, limit: int) -> threading.Semaphore:
        """Return the semaphore for ``ans_id``, creating it on first access.

        Thread-safe: the mutation of ``self._endpoint_semaphores`` is guarded
        by ``self._endpoint_semaphores_lock`` so that two workers racing for
        the same unseen key share a single ``threading.Semaphore`` instance.
        """
        existing = self._endpoint_semaphores.get(ans_id)
        if existing is not None:
            return existing
        with self._endpoint_semaphores_lock:
            existing = self._endpoint_semaphores.get(ans_id)
            if existing is None:
                existing = threading.Semaphore(limit)
                self._endpoint_semaphores[ans_id] = existing
            return existing

    def _resolve_endpoint_semaphore(self, task: dict[str, Any]) -> threading.Semaphore | None:
        """Resolve the per-answerer cap semaphore for ``task``, if any.

        Returns ``None`` when caps are disabled or when the task's answerer
        has no configured cap. The no-cap-for-this-id case logs once per
        answerer id at DEBUG so operators can see that a configured limits
        dict silently lets some answerers run unthrottled (typically an id
        typo or a model added after the limits were written).
        """
        limits = self.config.answerer_concurrency_limits
        if not limits:
            return None

        ans_id: str | None = getattr(task["answering_model"], "id", None)
        limit = limits.get(ans_id) if ans_id else None
        if limit is None or ans_id is None:
            label = ans_id if ans_id is not None else "<missing id>"
            with self._endpoint_semaphores_lock:
                should_log = label not in self._uncapped_answerer_ids_logged
                if should_log:
                    self._uncapped_answerer_ids_logged.add(label)
            if should_log:
                logger.debug(
                    "answerer_concurrency_limits is configured but answerer id %s has no cap, running unthrottled",
                    label,
                )
            return None

        return self._get_endpoint_semaphore(ans_id, limit)

    def _execute_task_with_cap(
        self,
        task: dict[str, Any],
        answer_cache: AnswerTraceCache | None,
        cache_status: str | None,
        cached_answer_data: dict[str, Any] | None,
    ) -> tuple[str, VerificationResult]:
        """Run ``execute_task`` with optional per-answerer concurrency capping.

        Imports ``execute_task`` lazily to avoid circular imports. The
        parallel cache-requeue loop manages the same semaphore manually (it
        must release the permit while waiting on another worker's
        IN_PROGRESS entry). This helper serves the simpler call sites
        (sequential, cache disabled, and the force-fresh fallback).
        """
        from .batch_runner import execute_task

        sem = self._resolve_endpoint_semaphore(task)
        if sem is None:
            return execute_task(task, answer_cache, cache_status, cached_answer_data)
        with sem:
            return execute_task(task, answer_cache, cache_status, cached_answer_data)

    @staticmethod
    def _hydrate_cache_from_results(
        answer_cache: AnswerTraceCache | None,
        results: Iterable[VerificationResult] | None,
    ) -> None:
        """Pre-populate the workspace cache with answerer traces from prior results.

        On sink-resume the executor instantiates a fresh cache. If the
        resumed config adds new parser variants for already-completed
        ``(question_id, answerer, replicate)`` triples, those new parser
        tasks are not in ``skip_triples`` (different parse_key), enter the
        queue, miss the empty cache, and regenerate the answerer at
        non-zero temperature. This helper closes that gap by writing a
        ``HIT`` entry into the cache for every prior triple whose answerer
        trace is real.

        Args:
            answer_cache: The workspace cache to populate. ``None`` is a
                no-op (caching is disabled).
            results: Iterable of prior :class:`VerificationResult` rows
                (typically from a ``ProgressiveFileSink`` resume buffer).
                ``None`` is a no-op.

        Raises:
            ValueError: If a hydrated cache key fails the replicate-axis
                invariant. This indicates a bug in
                :func:`generate_answer_cache_key` and should never fire in
                practice.
        """
        if answer_cache is None or results is None:
            return

        from types import SimpleNamespace

        from karenina.schemas.verification.model_identity import ModelIdentity

        from .utils.cache_helpers import (
            extract_answer_data_from_result,
            generate_answer_cache_key,
        )

        populated = 0
        skipped = 0
        dupes = 0

        for result in results:
            template = result.template
            if template is None or template.raw_llm_response is None:
                skipped += 1
                continue

            failure = result.metadata.failure
            if failure is not None:
                is_timeout_partial_trace = failure.category == FailureCategory.TIMEOUT and bool(
                    (template.raw_llm_response or "").strip()
                )
                if (
                    failure.category in _INFRA_FAILURE_CATEGORIES
                    and not _is_parser_stage(failure.stage)
                    and not is_timeout_partial_trace
                ):
                    skipped += 1
                    continue
                if failure.stage == "TraceValidationAutoFail":
                    skipped += 1
                    continue
                if failure.reason.startswith("Empty or whitespace-only trace"):
                    skipped += 1
                    continue

            answering_identity = result.metadata.answering
            if not isinstance(answering_identity, ModelIdentity):
                # Defensive: metadata.answering is typed as ModelIdentity,
                # but tolerate dict-shaped inputs by validating into one.
                answering_identity = ModelIdentity.model_validate(answering_identity)

            answerer_id = answering_identity.canonical_key
            replicate = result.metadata.replicate

            task_shaped = {
                "question_id": result.metadata.question_id,
                "answering_model": SimpleNamespace(id=answerer_id),
                "replicate": replicate,
            }
            cache_key = generate_answer_cache_key(task_shaped)

            # Replicate-axis invariant: if the source row is replicated, the
            # cache key must carry the per-replicate suffix so multiple
            # replicates of the same (qid, answerer) become independent
            # entries. A failure here would mean the key generator and the
            # task-dict shape have drifted apart.
            if replicate is not None and f"_rep{replicate}" not in cache_key:
                raise ValueError(
                    f"Hydration produced cache key {cache_key!r} without "
                    f"_rep{replicate} suffix; replicate-axis invariant violated."
                )

            status, _ = answer_cache.get_or_reserve(cache_key)
            if status == "HIT":
                # Duplicate parser-row for the same answerer triple. The
                # first write already populated the entry; nothing to do.
                dupes += 1
                continue
            if status == "IN_PROGRESS":
                # Should never happen during single-threaded hydration, but
                # if some other path reserved this key, leave it alone.
                dupes += 1
                continue

            answer_data = extract_answer_data_from_result(result)
            answer_cache.complete(cache_key, answer_data, error=None)
            populated += 1

        logger.info(
            "Hydrated %d answerer traces from prior sink results (skipped %d failures, %d duplicates)",
            populated,
            skipped,
            dupes,
        )

    def run_batch(
        self,
        tasks: list[dict[str, Any]],
        progress_callback: Callable[[int, int, VerificationResult | None], None] | None = None,
        prior_results: Iterable[VerificationResult] | None = None,
    ) -> dict[str, VerificationResult]:
        """Execute verification tasks.

        Args:
            tasks: List of task dictionaries with verification parameters
            progress_callback: Optional callback(current, total, result | None) for progress updates.
                             Called before starting each task (with preview result) and
                             after completion (with actual result).
            prior_results: Optional iterable of completed
                :class:`VerificationResult` rows from a sink-resume.
                When ``enable_cache`` is true these are written into the
                workspace cache via :meth:`_hydrate_cache_from_results`
                so new parser variants share the prior answerer trace.

        Returns:
            Dictionary mapping result keys to verification results
        """
        if self.parallel:
            return self._run_parallel(tasks, progress_callback, prior_results)
        else:
            return self._run_sequential(tasks, progress_callback, prior_results)

    def _run_sequential(
        self,
        tasks: list[dict[str, Any]],
        progress_callback: Callable[[int, int, VerificationResult | None], None] | None,
        prior_results: Iterable[VerificationResult] | None = None,
    ) -> dict[str, VerificationResult]:
        """Execute tasks one at a time on a shared BlockingPortal.

        On failure, logs the error, records it, and continues processing the
        remaining tasks. If any tasks failed, raises VerificationBatchError
        with partial results after the loop completes. The portal lifecycle
        (shared event loop plus pre-teardown adapter aclose) comes from
        :func:`sequential_portal`; the factory and sweep closures resolve
        ``start_blocking_portal`` / ``aclose_portal_adapters`` /
        ``PRE_TEARDOWN_ACLOSE_TIMEOUT`` from this module's globals at call
        time so existing monkeypatch targets keep working.

        Args:
            tasks: List of task dictionaries
            progress_callback: Optional progress callback
            prior_results: Optional iterable of completed
                :class:`VerificationResult` rows used to hydrate the
                workspace cache before execution starts.

        Returns:
            Dictionary mapping result keys to verification results

        Raises:
            VerificationBatchError: If one or more tasks failed during execution.
        """
        from karenina.exceptions import VerificationBatchError

        from .utils.cache_helpers import log_cache_stats
        from .utils.task_helpers import create_preview_result

        answer_cache = AnswerTraceCache() if self.config.enable_cache else None
        if answer_cache is not None and prior_results is not None:
            self._hydrate_cache_from_results(answer_cache, prior_results)
        results: dict[str, VerificationResult] = {}
        errors: list[tuple[str, BaseException]] = []
        total = len(tasks)

        with sequential_portal(
            portal_factory=lambda: start_blocking_portal(backend="asyncio"),
            pre_teardown=lambda portal: aclose_portal_adapters(portal, timeout=PRE_TEARDOWN_ACLOSE_TIMEOUT),
        ):
            for idx, task in enumerate(tasks, 1):
                # Call progress callback BEFORE starting task (with preview result)
                if progress_callback:
                    preview_result = create_preview_result(task)
                    progress_callback(idx, total, preview_result)

                try:
                    # Execute the task with answer cache
                    result_key, result = self._execute_task_with_cap(task, answer_cache, None, None)
                    results[result_key] = result
                except Exception as e:
                    logger.error("Sequential task %s failed: %s", task["question_id"], e)
                    errors.append((task["question_id"], e))
                    continue

                # Call progress callback AFTER completion (with actual result)
                if progress_callback:
                    progress_callback(idx, total, result)

        # Log cache statistics
        if answer_cache:
            log_cache_stats(answer_cache, mode="sequential")

        if errors:
            raise VerificationBatchError(
                f"{len(errors)} of {total} verification tasks failed",
                partial_results=results,
                errors=errors,
            )

        return results

    def _run_parallel(
        self,
        tasks: list[dict[str, Any]],
        progress_callback: Callable[[int, int, VerificationResult | None], None] | None,
        prior_results: Iterable[VerificationResult] | None = None,
    ) -> dict[str, VerificationResult]:
        """Execute tasks in parallel via the shared portal pool.

        The pooled skeleton (per-worker portals, timeout sweep,
        post-shutdown drain, drop detection, pre-teardown aclose, progress
        serialization) lives in :func:`run_in_portal_pool`. This method
        keeps the QA specifics: the cache requeue loop, per-answerer
        concurrency caps, cache hydration, the preview callback, and the
        ``VerificationBatchError`` contract.

        Args:
            tasks: List of task dictionaries
            progress_callback: Optional progress callback
            prior_results: Optional iterable of completed
                :class:`VerificationResult` rows used to hydrate the
                workspace cache before workers spin up.

        Returns:
            Dictionary mapping result keys to verification results (in original task order)

        Raises:
            VerificationBatchError: If the batch times out or any tasks fail.
        """
        from karenina.exceptions import VerificationBatchError

        from .utils.cache_helpers import generate_answer_cache_key, log_cache_stats
        from .utils.task_helpers import create_preview_result

        logger.info("Parallel execution: %d tasks with %d workers", len(tasks), self.config.max_workers)

        answer_cache = AnswerTraceCache() if self.config.enable_cache else None
        if answer_cache is not None and prior_results is not None:
            self._hydrate_cache_from_results(answer_cache, prior_results)
        total = len(tasks)
        max_requeue = self.config.max_requeue_count
        retry_wait = self.config.retry_wait_seconds

        def execute_task_with_retry(_idx: int, task: dict[str, Any]) -> tuple[str, VerificationResult]:
            """Execute a single verification task with cache retry.

            The per-answerer cap permit wraps the cache reservation AND the
            execution: a surplus worker blocked on the cap must not reserve
            an IN_PROGRESS cache entry while merely waiting for a permit.
            The wait-for-another-worker branch (someone else owns the
            IN_PROGRESS reservation) deliberately does NOT hold the permit
            while sleeping, so a capped answerer's permits stay available to
            workers that can actually make progress.
            """
            from .batch_runner import execute_task

            if not answer_cache:
                return self._execute_task_with_cap(task, answer_cache, None, None)

            sem = self._resolve_endpoint_semaphore(task)
            cache_key = generate_answer_cache_key(task)
            for attempt in range(max_requeue + 1):
                if sem is not None:
                    sem.acquire()
                try:
                    status, cached_answer_data = answer_cache.get_or_reserve(cache_key)
                    if status != "IN_PROGRESS":
                        # MISS or HIT: reserve-and-execute under the permit.
                        return execute_task(task, answer_cache, status, cached_answer_data)
                finally:
                    if sem is not None:
                        sem.release()

                # IN_PROGRESS: another worker owns the reservation. Wait
                # for it WITHOUT holding the permit.
                if attempt == 0:
                    logger.debug(
                        "Task %s in progress (retry=%d), retrying immediately",
                        cache_key,
                        attempt,
                    )
                    continue

                completed = answer_cache.wait_for_completion(cache_key, timeout=retry_wait)
                if not completed:
                    logger.debug(
                        "Task %s still in progress after %ss, retrying",
                        cache_key,
                        retry_wait,
                    )

            # Exceeded max requeue: force reset, generate fresh
            logger.warning(
                "Task %s exceeded max requeue count (%d), generating fresh",
                cache_key,
                max_requeue,
            )
            answer_cache.force_reset(cache_key)
            return self._execute_task_with_cap(task, answer_cache, None, None)

        # Progress hooks. The preview fires from the worker thread before
        # the task runs; the completion hook fires from the collector. Both
        # are serialized under the pool's single progress lock.
        on_item_start: Callable[[int, int, dict[str, Any]], None] | None = None
        on_progress: Callable[[int, int, tuple[str, VerificationResult]], None] | None = None
        if progress_callback is not None:
            caller_progress = progress_callback

            def _preview(current: int, total_: int, task: dict[str, Any]) -> None:
                caller_progress(current, total_, create_preview_result(task))

            def _completed(current: int, total_: int, value: tuple[str, VerificationResult]) -> None:
                caller_progress(current, total_, value[1])

            on_item_start = _preview
            on_progress = _completed

        outcome = run_in_portal_pool(
            tasks,
            execute_task_with_retry,
            tuning=self.config.to_tuning(),
            on_progress=on_progress,
            on_item_start=on_item_start,
            describe=lambda _idx, task: str(task["question_id"]),
            # Closures over this module's globals so monkeypatches on
            # executor.start_blocking_portal / executor.aclose_portal_adapters
            # / executor.PRE_TEARDOWN_ACLOSE_TIMEOUT keep working.
            portal_factory=lambda: start_blocking_portal(backend="asyncio"),
            pre_teardown=lambda portal: aclose_portal_adapters(portal, timeout=PRE_TEARDOWN_ACLOSE_TIMEOUT),
            item_noun="Task",
        )

        # Restore original order and convert to dictionary
        results: dict[str, VerificationResult] = {}
        for idx in sorted(outcome.results_by_index.keys()):
            result_key, verification_result = outcome.results_by_index[idx]
            results[result_key] = verification_result

        # Log cache statistics
        if answer_cache:
            log_cache_stats(answer_cache, mode="parallel mode")

        # Handle timeout. The pool only sets timed_out when a finite
        # timeout was passed, so timeout_seconds is guaranteed non-None here.
        if outcome.timed_out:
            timeout_label = (
                f"{self.config.timeout_seconds:.0f} seconds"
                if self.config.timeout_seconds is not None
                else "unknown timeout"
            )
            raise VerificationBatchError(
                f"Parallel batch timed out after {timeout_label} ({len(results)} of {total} tasks completed)",
                partial_results=results,
                errors=outcome.failed_items,
            )

        # Handle partial failures
        if outcome.failed_items:
            raise VerificationBatchError(
                f"{len(outcome.failed_items)} of {total} verification tasks failed",
                partial_results=results,
                errors=outcome.failed_items,
            )

        return results


def get_default_max_workers() -> int:
    """Get the default max workers from environment variable or default.

    Returns:
        Max workers value from KARENINA_ASYNC_MAX_WORKERS env var or 2
    """
    return int(os.getenv("KARENINA_ASYNC_MAX_WORKERS", str(DEFAULT_ASYNC_MAX_WORKERS)))
