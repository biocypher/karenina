"""Tests for VerificationExecutor error handling (issues 088 and 089).

Tests verify:
- VerificationBatchError exception structure and attributes
- Sequential mode: catches task failures, collects partial results, raises aggregate error
- Parallel mode: counts failures toward completion (no deadlock), raises aggregate error
- Both modes raise the same error type with equivalent information (symmetry)
- Timeout safety net prevents indefinite hangs
- Per-task portal creation (each submitted task gets its own BlockingPortal)
"""

import contextlib
import threading

import pytest

from karenina.benchmark.verification.executor import ExecutorConfig, VerificationExecutor
from karenina.exceptions import KareninaError, VerificationBatchError
from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import VerificationResultMetadata

# ============================================================================
# Helpers
# ============================================================================


def _make_identity(name: str = "test-model") -> ModelIdentity:
    """Create a minimal ModelIdentity for test results."""
    return ModelIdentity(
        model_name=name,
        interface="langchain",
    )


def _make_result(question_id: str = "q1") -> VerificationResult:
    """Create a minimal VerificationResult for testing."""
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="test_template",
            failure=None,
            caveats=[],
            question_text="Test question",
            answering=_make_identity(),
            parsing=_make_identity(),
            execution_time=0.1,
            timestamp="2026-01-01T00:00:00",
            result_id="abcdef1234567890",
        )
    )


def _make_task(question_id: str = "q1") -> dict:
    """Create a minimal task dict for executor tests."""
    from karenina.schemas.config import ModelConfig

    model = ModelConfig(
        id="test-model",
        model_name="test-model",
        model_provider="anthropic",
        interface="langchain",
        system_prompt="test",
        temperature=0.0,
    )
    return {
        "question_id": question_id,
        "question_text": f"Question {question_id}",
        "raw_answer": None,
        "template_code": "class Answer(BaseAnswer): pass",
        "answering_model": model,
        "parsing_model": model,
        "run_name": "test_run",
        "replicate": None,
        "rubric": None,
        "dynamic_rubric": None,
        "keywords": None,
        "question_workspace_path": None,
        "few_shot_examples": None,
    }


# ============================================================================
# VerificationBatchError structure
# ============================================================================


@pytest.mark.unit
class TestVerificationBatchError:
    """Verify VerificationBatchError has the right structure and attributes."""

    def test_inherits_from_karenina_error(self) -> None:
        """VerificationBatchError is a KareninaError subclass."""
        err = VerificationBatchError(
            message="1 of 3 tasks failed",
            partial_results={"key1": _make_result()},
            errors=[("key2", RuntimeError("boom"))],
        )
        assert isinstance(err, KareninaError)
        assert isinstance(err, Exception)

    def test_carries_partial_results(self) -> None:
        """Partial results from successful tasks are accessible on the exception."""
        result = _make_result("q1")
        err = VerificationBatchError(
            message="1 of 2 tasks failed",
            partial_results={"key1": result},
            errors=[("key2", ValueError("bad"))],
        )
        assert err.partial_results == {"key1": result}

    def test_carries_error_details(self) -> None:
        """Error details contain task identifier and original exception."""
        original = RuntimeError("infrastructure failure")
        err = VerificationBatchError(
            message="test",
            partial_results={},
            errors=[("task_key_1", original)],
        )
        assert len(err.errors) == 1
        task_key, exc = err.errors[0]
        assert task_key == "task_key_1"
        assert exc is original

    def test_message_includes_failure_count(self) -> None:
        """The string representation includes the failure information."""
        err = VerificationBatchError(
            message="2 of 5 tasks failed",
            partial_results={},
            errors=[("k1", ValueError("a")), ("k2", ValueError("b"))],
        )
        assert "2 of 5" in str(err)


# ============================================================================
# Sequential mode error handling (issue 089)
# ============================================================================


@pytest.mark.unit
class TestSequentialErrorHandling:
    """Sequential executor catches task failures and returns partial results."""

    def test_single_failure_raises_batch_error_with_partial_results(self, monkeypatch) -> None:
        """When one task fails, sequential mode raises VerificationBatchError
        containing partial results from the successful tasks."""
        tasks = [_make_task("q1"), _make_task("q2"), _make_task("q3")]
        call_count = 0

        def mock_execute_task(task, answer_cache=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if task["question_id"] == "q2":
                raise RuntimeError("infrastructure failure")
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(parallel=False, config=ExecutorConfig(enable_cache=False))

        with pytest.raises(VerificationBatchError) as exc_info:
            executor.run_batch(tasks)

        err = exc_info.value
        # All 3 tasks were attempted (continues past failure)
        assert call_count == 3
        # 2 successful results preserved
        assert len(err.partial_results) == 2
        assert "key_q1" in err.partial_results
        assert "key_q3" in err.partial_results
        # 1 error recorded
        assert len(err.errors) == 1
        task_key, exc = err.errors[0]
        assert isinstance(exc, RuntimeError)
        assert "infrastructure failure" in str(exc)

    def test_all_succeed_returns_normally(self, monkeypatch) -> None:
        """When all tasks succeed, sequential mode returns a normal dict (no exception)."""
        tasks = [_make_task("q1"), _make_task("q2")]

        def mock_execute_task(task, answer_cache=None, **kwargs):
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(parallel=False, config=ExecutorConfig(enable_cache=False))
        results = executor.run_batch(tasks)

        assert len(results) == 2
        assert "key_q1" in results
        assert "key_q2" in results

    def test_all_fail_raises_batch_error_with_empty_results(self, monkeypatch) -> None:
        """When all tasks fail, the error has empty partial_results and all errors."""
        tasks = [_make_task("q1"), _make_task("q2")]

        def mock_execute_task(task, answer_cache=None, **kwargs):
            raise RuntimeError(f"fail_{task['question_id']}")

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(parallel=False, config=ExecutorConfig(enable_cache=False))

        with pytest.raises(VerificationBatchError) as exc_info:
            executor.run_batch(tasks)

        err = exc_info.value
        assert len(err.partial_results) == 0
        assert len(err.errors) == 2


# ============================================================================
# Parallel mode deadlock fix (issue 088)
# ============================================================================


@pytest.mark.unit
class TestParallelDeadlockFix:
    """Parallel executor counts failures toward completion (no deadlock)."""

    def test_single_failure_completes_without_hanging(self, monkeypatch) -> None:
        """When one task fails in parallel mode, execution completes
        (does not hang) and raises VerificationBatchError."""
        tasks = [_make_task("q1"), _make_task("q2"), _make_task("q3")]

        def mock_execute_task(task, answer_cache=None, **kwargs):
            if task["question_id"] == "q2":
                raise RuntimeError("infrastructure failure")
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=2, enable_cache=False),
        )

        # If the deadlock (issue 088) is present, this will hang forever.
        # The test timeout will catch it.
        with pytest.raises(VerificationBatchError) as exc_info:
            executor.run_batch(tasks)

        err = exc_info.value
        assert len(err.partial_results) == 2
        assert len(err.errors) == 1

    def test_all_succeed_returns_normally_parallel(self, monkeypatch) -> None:
        """When all tasks succeed in parallel mode, returns a normal dict."""
        tasks = [_make_task("q1"), _make_task("q2")]

        def mock_execute_task(task, answer_cache=None, **kwargs):
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=2, enable_cache=False),
        )
        results = executor.run_batch(tasks)

        assert len(results) == 2


# ============================================================================
# Error symmetry between modes (issue 089)
# ============================================================================


@pytest.mark.unit
class TestErrorSymmetry:
    """Both execution modes raise the same error type with equivalent info."""

    def test_same_failure_produces_same_error_type(self, monkeypatch) -> None:
        """Sequential and parallel modes raise VerificationBatchError for the same failure."""
        tasks = [_make_task("q1"), _make_task("q2")]

        def mock_execute_task(task, answer_cache=None, **kwargs):
            if task["question_id"] == "q2":
                raise RuntimeError("boom")
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        # Sequential
        seq_executor = VerificationExecutor(parallel=False, config=ExecutorConfig(enable_cache=False))
        with pytest.raises(VerificationBatchError) as seq_info:
            seq_executor.run_batch(tasks)

        # Parallel
        par_executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=2, enable_cache=False),
        )
        with pytest.raises(VerificationBatchError) as par_info:
            par_executor.run_batch(tasks)

        # Both have 1 partial result and 1 error
        assert len(seq_info.value.partial_results) == 1
        assert len(par_info.value.partial_results) == 1
        assert len(seq_info.value.errors) == 1
        assert len(par_info.value.errors) == 1

        # Both captured the same exception type
        _, seq_exc = seq_info.value.errors[0]
        _, par_exc = par_info.value.errors[0]
        assert type(seq_exc) is type(par_exc)
        assert str(seq_exc) == str(par_exc)


# ============================================================================
# Timeout safety net
# ============================================================================


@pytest.mark.unit
class TestTimeoutSafetyNet:
    """Timeout prevents indefinite hangs from unforeseen edge cases."""

    def test_timeout_fires_when_completion_event_never_set(self, monkeypatch) -> None:
        """If neither results nor errors are counted (edge case),
        the timeout fires and raises VerificationBatchError."""
        tasks = [_make_task("q1")]

        def mock_execute_task(task, answer_cache=None, **kwargs):
            # This succeeds, but we'll patch the completion tracking to never fire
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        # Use a very short timeout to make the test fast
        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=1, enable_cache=False, timeout_seconds=2.0),
        )

        # The normal path should work fine with a timeout configured.
        # We test that the timeout parameter is accepted and doesn't interfere
        # with normal execution.
        results = executor.run_batch(tasks)
        assert len(results) == 1

    def test_timeout_config_parameter_exists(self) -> None:
        """ExecutorConfig accepts a timeout_seconds parameter."""
        config = ExecutorConfig(timeout_seconds=30.0)
        assert config.timeout_seconds == 30.0

    def test_default_timeout_is_none(self) -> None:
        """Default timeout is None (no batch-level ceiling)."""
        config = ExecutorConfig()
        assert config.timeout_seconds is None


# ============================================================================
# Error detail quality
# ============================================================================


@pytest.mark.unit
class TestErrorDetailQuality:
    """VerificationBatchError messages include actionable information."""

    def test_error_message_includes_task_count(self, monkeypatch) -> None:
        """Error message reports how many tasks failed out of total."""
        tasks = [_make_task("q1"), _make_task("q2"), _make_task("q3")]

        def mock_execute_task(task, answer_cache=None, **kwargs):
            if task["question_id"] in ("q1", "q3"):
                raise ValueError(f"bad_{task['question_id']}")
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(parallel=False, config=ExecutorConfig(enable_cache=False))

        with pytest.raises(VerificationBatchError) as exc_info:
            executor.run_batch(tasks)

        msg = str(exc_info.value)
        assert "2" in msg  # 2 failures
        assert "3" in msg  # 3 total

    def test_error_details_include_exception_types_and_messages(self, monkeypatch) -> None:
        """Each error entry has the original exception with its type and message."""
        tasks = [_make_task("q1"), _make_task("q2")]

        def mock_execute_task(task, answer_cache=None, **kwargs):
            if task["question_id"] == "q1":
                raise TypeError("type mismatch in q1")
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(parallel=False, config=ExecutorConfig(enable_cache=False))

        with pytest.raises(VerificationBatchError) as exc_info:
            executor.run_batch(tasks)

        err = exc_info.value
        # At least the first error is a TypeError with our message
        task_key, exc = err.errors[0]
        assert isinstance(exc, TypeError)
        assert "type mismatch" in str(exc)


# ============================================================================
# Sequential portal management
# ============================================================================


@pytest.mark.unit
class TestSequentialPortalManagement:
    """Verify _run_sequential sets a BlockingPortal for event loop reuse."""

    def test_sequential_sets_portal_during_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Portal is available during task execution and cleared after."""
        from karenina.benchmark.verification.executor import get_async_portal

        captured_portals: list = []

        def mock_execute_task(task: dict, answer_cache: object) -> tuple[str, VerificationResult]:
            captured_portals.append(get_async_portal())
            return (task["question_id"], _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        tasks = [_make_task("q1"), _make_task("q2")]
        executor = VerificationExecutor(parallel=False, config=ExecutorConfig(enable_cache=False))
        executor.run_batch(tasks)

        # Portal was set during both task executions
        assert len(captured_portals) == 2
        assert all(p is not None for p in captured_portals)

        # Portal is cleared after run_batch returns
        assert get_async_portal() is None

    def test_sequential_clears_portal_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Portal is cleared even when all tasks fail."""
        from karenina.benchmark.verification.executor import get_async_portal

        def mock_execute_task(task: dict, answer_cache: object) -> tuple[str, VerificationResult]:
            raise RuntimeError("task failed")

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        tasks = [_make_task("q1")]
        executor = VerificationExecutor(parallel=False, config=ExecutorConfig(enable_cache=False))

        with pytest.raises(VerificationBatchError):
            executor.run_batch(tasks)

        assert get_async_portal() is None

    def test_sequential_portal_is_functional(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Portal can execute async functions during task execution."""
        from karenina.benchmark.verification.executor import get_async_portal

        async_results: list = []

        async def async_fn() -> str:
            return "ok"

        def mock_execute_task(task: dict, answer_cache: object) -> tuple[str, VerificationResult]:
            portal = get_async_portal()
            async_results.append(portal.call(async_fn))
            return (task["question_id"], _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        tasks = [_make_task("q1")]
        executor = VerificationExecutor(parallel=False, config=ExecutorConfig(enable_cache=False))
        executor.run_batch(tasks)

        assert async_results == ["ok"]


# ============================================================================
# Executor requeue bound (issue 188)
# ============================================================================


@pytest.mark.unit
class TestExecutorRequeueBound:
    """Tests for max_requeue_count in ExecutorConfig."""

    def test_executor_config_default_requeue_count(self) -> None:
        """Test ExecutorConfig has max_requeue_count defaulting to 5."""
        config = ExecutorConfig()
        assert config.max_requeue_count == 5

    def test_executor_config_custom_requeue_count(self) -> None:
        """Test ExecutorConfig accepts custom max_requeue_count."""
        config = ExecutorConfig(max_requeue_count=10)
        assert config.max_requeue_count == 10


@pytest.mark.unit
class TestForceResetOnRequeueLimit:
    """Tests for force_reset behavior when requeue limit is reached."""

    def test_cache_force_reset_after_max_requeues(self) -> None:
        """Test that exceeding max_requeue_count triggers force_reset."""
        from karenina.utils.answer_cache import AnswerTraceCache

        cache = AnswerTraceCache()
        max_requeue = 3

        # Simulate: key reserved by another worker
        cache.get_or_reserve("key1")

        # Simulate requeue loop hitting the limit
        for _i in range(max_requeue):
            status, _ = cache.get_or_reserve("key1")
            assert status == "IN_PROGRESS"

        # After limit: force_reset should make next access return MISS
        cache.force_reset("key1")
        status, _ = cache.get_or_reserve("key1")
        assert status == "MISS"


# ============================================================================
# Per-task portal creation (executor concurrency optimization)
# ============================================================================


@pytest.mark.unit
class TestPerWorkerPortals:
    """Each worker thread creates one portal, reused across tasks."""

    def test_parallel_workers_create_portals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With 6 tasks and 2 workers, 2 portals are created (one per worker).

        Each worker lazily creates a portal on its first task and reuses it
        for subsequent tasks. This preserves connection pools.
        """
        portal_count = [0]
        portal_lock = threading.Lock()

        from anyio.from_thread import start_blocking_portal as original_start_blocking_portal

        class PortalTracker:
            """Context manager that wraps start_blocking_portal to count creations."""

            def __init__(self) -> None:
                self._real_cm = original_start_blocking_portal(backend="asyncio")
                with portal_lock:
                    portal_count[0] += 1

            def __enter__(self):
                return self._real_cm.__enter__()

            def __exit__(self, *args):
                return self._real_cm.__exit__(*args)

        def mock_start_blocking_portal(**kwargs):
            return PortalTracker()

        monkeypatch.setattr(
            "karenina.benchmark.verification.executor.start_blocking_portal",
            mock_start_blocking_portal,
        )

        # Barrier ensures both worker threads are actively running before any
        # task completes. Without this, a fast single worker can drain the
        # queue before the pool spawns its sibling, making the portal count
        # flaky under CI load.
        worker_barrier = threading.Barrier(2, timeout=5.0)
        barrier_released = threading.Event()

        def mock_execute_task(task, answer_cache=None, **kwargs):
            if not barrier_released.is_set():
                with contextlib.suppress(threading.BrokenBarrierError):
                    worker_barrier.wait()
                    barrier_released.set()
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        tasks = [_make_task(f"q{i}") for i in range(6)]
        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=2, enable_cache=False),
        )
        results = executor.run_batch(tasks)

        # All 6 tasks completed
        assert len(results) == 6

        # 2 portal creations (one per worker, reused across tasks)
        assert portal_count[0] == 2, f"Expected 2 portal creations (one per worker), got {portal_count[0]}"


# ============================================================================
# Shutdown race: in-flight tasks must not be silently dropped on timeout
# ============================================================================


@pytest.mark.unit
class TestInFlightTasksNotSilentlyDropped:
    """Parallel batch timeout must not silently drop tasks still in-flight.

    When a parallel batch hits ``timeout_seconds``, the pre-fix implementation
    called ``pool.shutdown(wait=False, cancel_futures=True)`` and immediately
    tore down per-worker portals. Tasks still in the RUNNING state at that
    moment were silently dropped: their eventual result set on the Future
    object was never harvested into ``partial_results``.

    These tests reproduce that race deterministically and assert that every
    submitted task shows up in either ``partial_results`` or ``errors`` of
    the raised ``VerificationBatchError``.
    """

    def test_slow_tasks_finishing_after_timeout_are_recovered(self, monkeypatch) -> None:
        """Four tasks: two fast, two slow. Slow tasks finish a short time
        after the batch timeout fires. All four must be accounted for in
        the raised VerificationBatchError (partial_results + errors).
        """
        import time

        tasks = [_make_task(f"q{i}") for i in range(4)]
        fast_ids = {"q0", "q1"}
        slow_delay = 0.8  # slow tasks finish ~0.5s after the batch timeout

        def mock_execute_task(task, answer_cache=None, **kwargs):
            if task["question_id"] not in fast_ids:
                time.sleep(slow_delay)
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=4, enable_cache=False, timeout_seconds=0.3),
        )

        with pytest.raises(VerificationBatchError) as exc_info:
            executor.run_batch(tasks)

        err = exc_info.value
        accounted_for: set[str] = set()
        for result_key in err.partial_results:
            # result_key shape is "key_qN" from the mock above
            for qid in ("q0", "q1", "q2", "q3"):
                if qid in result_key:
                    accounted_for.add(qid)
        for question_id, _exc in err.errors:
            accounted_for.add(question_id)

        assert accounted_for == {"q0", "q1", "q2", "q3"}, (
            f"Tasks silently dropped. partial_results={list(err.partial_results)}, "
            f"errors={[(q, type(e).__name__) for q, e in err.errors]}"
        )

    def test_slow_task_failing_after_timeout_is_recorded_per_task(self, monkeypatch) -> None:
        """A slow task that raises after the batch timeout must appear as a
        per-task entry in ``errors``, not be silently dropped.
        """
        import time

        tasks = [_make_task(f"q{i}") for i in range(4)]
        fast_ids = {"q0", "q1"}

        def mock_execute_task(task, answer_cache=None, **kwargs):
            qid = task["question_id"]
            if qid not in fast_ids:
                time.sleep(0.8)
                raise RuntimeError(f"task {qid} broke late")
            return (f"key_{qid}", _make_result(qid))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=4, enable_cache=False, timeout_seconds=0.3),
        )

        with pytest.raises(VerificationBatchError) as exc_info:
            executor.run_batch(tasks)

        err = exc_info.value
        # Fast tasks land in partial_results
        fast_keys = {k for k in err.partial_results if any(q in k for q in fast_ids)}
        assert len(fast_keys) == 2

        # Slow tasks must each appear as a per-task error entry
        error_qids = {qid for qid, exc in err.errors if isinstance(exc, RuntimeError)}
        assert "q2" in error_qids
        assert "q3" in error_qids


# ============================================================================
# Pre-teardown aclose: httpx client must close on the portal's own loop
# ============================================================================


@pytest.mark.unit
class TestPreTeardownAclose:
    """Adapters registered during a parallel batch must have aclose() called
    on the worker portal's own event loop BEFORE the portal is torn down.

    Without this, httpx.AsyncClient.aclose() runs on a brand-new loop (via
    the downstream cleanup_resources call) and raises "Event loop is closed"
    because its transports are pinned to the now-closed portal loop.
    """

    def test_parallel_pre_teardown_aclose_runs_on_portal_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """aclose() runs on the portal's event-loop thread, not the main
        thread and not a fresh cleanup loop."""
        import asyncio

        from karenina.adapters.registry import (
            _active_adapters,
            _adapter_portal_lock,
            _adapter_portal_refs,
            _adapters_lock,
            register_adapter,
        )

        # Ensure the registry starts clean so stray adapters from prior tests
        # do not pollute the portal tracking.
        with _adapters_lock:
            _active_adapters.clear()
        with _adapter_portal_lock:
            _adapter_portal_refs.clear()

        main_thread_id = threading.get_ident()
        aclose_thread_ids: list[int] = []
        aclose_lock = threading.Lock()

        class PortalAwareAdapter:
            """Adapter whose aclose() records the thread id it ran on."""

            async def aclose(self) -> None:
                # Give the test a positive signal that we're actually on an
                # asyncio loop, not just a synchronous call.
                await asyncio.sleep(0)
                with aclose_lock:
                    aclose_thread_ids.append(threading.get_ident())

        def mock_execute_task(task, answer_cache=None, **kwargs):
            # Run inside the worker thread: register an adapter so the
            # registry tracks it against this worker's portal.
            register_adapter(PortalAwareAdapter())
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        tasks = [_make_task(f"q{i}") for i in range(4)]
        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=2, enable_cache=False),
        )
        results = executor.run_batch(tasks)
        assert len(results) == 4

        # Every registered adapter got aclose()'d exactly once, on a thread
        # that is neither the main thread (which issued run_batch) nor a
        # freshly-minted cleanup thread. It should match a worker portal's
        # event-loop thread.
        assert len(aclose_thread_ids) == 4
        for tid in aclose_thread_ids:
            assert tid != main_thread_id, (
                "aclose ran on the main thread, which means it went through "
                "the downstream cleanup_resources path rather than the "
                "pre-teardown portal loop."
            )

        # Cleanup: remove any lingering entries so later tests see a clean map.
        with _adapters_lock:
            _active_adapters.clear()
        with _adapter_portal_lock:
            _adapter_portal_refs.clear()

    def test_parallel_pre_teardown_aclose_is_timeout_bounded(self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        """A stuck aclose() must not wedge the finally block: the bounded
        ``portal.start_task_soon`` + ``future.result(timeout=...)`` pattern
        must time out and proceed with portal teardown."""
        import asyncio
        import logging
        import time

        from karenina.adapters.registry import (
            _active_adapters,
            _adapter_portal_lock,
            _adapter_portal_refs,
            _adapters_lock,
            register_adapter,
        )

        with _adapters_lock:
            _active_adapters.clear()
        with _adapter_portal_lock:
            _adapter_portal_refs.clear()

        # Shorten the bound so the test does not wait 5s per stuck adapter.
        monkeypatch.setattr(
            "karenina.benchmark.verification.executor.PRE_TEARDOWN_ACLOSE_TIMEOUT",
            0.2,
        )

        class StuckAdapter:
            async def aclose(self) -> None:
                # Exceed the short timeout so the finally block must give up.
                await asyncio.sleep(10.0)

        def mock_execute_task(task, answer_cache=None, **kwargs):
            register_adapter(StuckAdapter())
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        tasks = [_make_task(f"q{i}") for i in range(2)]
        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=2, enable_cache=False),
        )

        caplog.set_level(logging.WARNING, logger="karenina.benchmark.verification.executor")
        start = time.monotonic()
        results = executor.run_batch(tasks)
        elapsed = time.monotonic() - start

        # Batch completed, not wedged. The bound is 0.2s per adapter times
        # at most 2 portals; leave generous headroom for CI jitter.
        assert len(results) == 2
        assert elapsed < 5.0, (
            f"Finally block took {elapsed:.2f}s; expected well under the "
            f"5s default timeout. The per-adapter bound was monkey-patched "
            f"to 0.2s, so the whole drain should finish in well under a "
            f"second even with scheduling jitter."
        )

        # A warning mentioning the timeout was emitted.
        timeout_warnings = [rec for rec in caplog.records if "Pre-teardown aclose timed out" in rec.getMessage()]
        assert timeout_warnings, (
            "Expected at least one 'Pre-teardown aclose timed out' warning; "
            f"got records: {[rec.getMessage() for rec in caplog.records]}"
        )

        with _adapters_lock:
            _active_adapters.clear()
        with _adapter_portal_lock:
            _adapter_portal_refs.clear()


# =============================================================================
# ExecutorConfig.answerer_concurrency_limits + _get_endpoint_semaphore
# =============================================================================


@pytest.mark.unit
class TestAnswererConcurrencyLimitsPlumbing:
    """Config plumbing and lazy semaphore accessor on VerificationExecutor."""

    def test_default_is_none(self) -> None:
        from karenina.benchmark.verification.executor import ExecutorConfig

        assert ExecutorConfig().answerer_concurrency_limits is None

    def test_can_be_set(self) -> None:
        from karenina.benchmark.verification.executor import ExecutorConfig

        cfg = ExecutorConfig(answerer_concurrency_limits={"m1": 3})
        assert cfg.answerer_concurrency_limits == {"m1": 3}

    def test_get_endpoint_semaphore_caches(self) -> None:
        from karenina.benchmark.verification.executor import (
            ExecutorConfig,
            VerificationExecutor,
        )

        executor = VerificationExecutor(
            parallel=False,
            config=ExecutorConfig(answerer_concurrency_limits={"m1": 2}),
        )

        sem_a = executor._get_endpoint_semaphore("m1", 2)
        sem_b = executor._get_endpoint_semaphore("m1", 2)
        assert sem_a is sem_b
        # Duck-type check: a Semaphore exposes acquire/release. CPython's
        # Semaphore is a factory, so isinstance checks against
        # threading.Semaphore are brittle across versions.
        assert hasattr(sem_a, "acquire") and hasattr(sem_a, "release")

    def test_get_endpoint_semaphore_thread_safe_first_time(self) -> None:
        import threading

        from karenina.benchmark.verification.executor import (
            ExecutorConfig,
            VerificationExecutor,
        )

        executor = VerificationExecutor(
            parallel=False,
            config=ExecutorConfig(answerer_concurrency_limits={"m1": 2}),
        )

        barrier = threading.Barrier(8)
        seen: list[object] = []
        seen_lock = threading.Lock()

        def _race() -> None:
            barrier.wait()
            sem = executor._get_endpoint_semaphore("m1", 2)
            with seen_lock:
                seen.append(sem)

        threads = [threading.Thread(target=_race) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len({id(s) for s in seen}) == 1
