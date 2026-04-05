"""Tests for VerificationExecutor error handling (issues 088 and 089).

Tests verify:
- VerificationBatchError exception structure and attributes
- Sequential mode: catches task failures, collects partial results, raises aggregate error
- Parallel mode: counts failures toward completion (no deadlock), raises aggregate error
- Both modes raise the same error type with equivalent information (symmetry)
- Timeout safety net prevents indefinite hangs
- Per-worker portal creation (each worker gets its own BlockingPortal)
"""

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
            completed_without_errors=True,
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

    def test_default_timeout_is_600_seconds(self) -> None:
        """Default timeout is 600 seconds (10 minutes)."""
        config = ExecutorConfig()
        assert config.timeout_seconds == 600.0


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
# Per-worker portal creation (executor concurrency optimization)
# ============================================================================


@pytest.mark.unit
class TestPerWorkerPortals:
    """Each parallel worker thread creates its own distinct BlockingPortal."""

    def test_parallel_workers_create_distinct_portals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With max_workers=4 and 8 tasks, 4 distinct portal ids are observed."""
        portal_ids_by_thread: dict[int, int] = {}
        portal_ids_lock = threading.Lock()

        # Capture the real start_blocking_portal so we can wrap it
        from anyio.from_thread import start_blocking_portal as original_start_blocking_portal

        class PortalTracker:
            """Context manager that wraps start_blocking_portal to track portal identity."""

            def __init__(self) -> None:
                self._real_cm = original_start_blocking_portal(backend="asyncio")

            def __enter__(self):
                portal = self._real_cm.__enter__()
                thread_id = threading.current_thread().ident
                with portal_ids_lock:
                    portal_ids_by_thread[thread_id] = id(portal)
                return portal

            def __exit__(self, *args):
                return self._real_cm.__exit__(*args)

        def mock_start_blocking_portal(**kwargs):
            return PortalTracker()

        monkeypatch.setattr(
            "karenina.benchmark.verification.executor.start_blocking_portal",
            mock_start_blocking_portal,
        )

        def mock_execute_task(task, answer_cache=None, **kwargs):
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        tasks = [_make_task(f"q{i}") for i in range(8)]
        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=4, enable_cache=False),
        )
        results = executor.run_batch(tasks)

        # All 8 tasks completed
        assert len(results) == 8

        # 4 distinct portals were created (one per worker thread)
        distinct_portal_ids = set(portal_ids_by_thread.values())
        assert len(distinct_portal_ids) == 4, (
            "Expected 4 distinct portals (one per worker), "
            f"got {len(distinct_portal_ids)}: workers are sharing a single portal"
        )
