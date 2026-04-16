"""Tests for VerificationExecutor hang prevention.

These tests verify that the ThreadPoolExecutor-based parallel executor
never hangs due to untracked worker deaths. Each test targets a specific
failure mode that caused hangs with the previous raw-thread design.
"""

import contextlib
import threading

import pytest

from karenina.benchmark.verification.executor import ExecutorConfig, VerificationExecutor
from karenina.exceptions import VerificationBatchError
from karenina.schemas.config import ModelConfig
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
# BaseException handling
# ============================================================================


@pytest.mark.unit
class TestVerificationBaseExceptionNoHang:
    """BaseException from a task must not cause the batch to hang."""

    def test_keyboard_interrupt_no_hang(self, monkeypatch) -> None:
        """A task raising KeyboardInterrupt is collected as an error.

        With the old raw-thread design, KeyboardInterrupt bypassed the
        except Exception handler, leaving the task untracked.
        """
        tasks = [_make_task("q1"), _make_task("q2"), _make_task("q3")]

        def mock_execute_task(task, answer_cache=None, **kwargs):
            if task["question_id"] == "q2":
                raise KeyboardInterrupt()
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=2, enable_cache=False, timeout_seconds=10.0),
        )

        with pytest.raises(VerificationBatchError) as exc_info:
            executor.run_batch(tasks)

        err = exc_info.value
        assert len(err.partial_results) == 2
        assert len(err.errors) == 1
        assert isinstance(err.errors[0][1], KeyboardInterrupt)

    def test_system_exit_no_hang(self, monkeypatch) -> None:
        """A task raising SystemExit is collected as an error without hanging."""
        tasks = [_make_task("q1"), _make_task("q2")]

        def mock_execute_task(task, answer_cache=None, **kwargs):
            if task["question_id"] == "q2":
                raise SystemExit(1)
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=2, enable_cache=False, timeout_seconds=10.0),
        )

        with pytest.raises(VerificationBatchError) as exc_info:
            executor.run_batch(tasks)

        err = exc_info.value
        assert len(err.partial_results) == 1
        assert len(err.errors) == 1
        assert isinstance(err.errors[0][1], SystemExit)

    def test_all_raise_base_exception(self, monkeypatch) -> None:
        """When every task raises BaseException, batch returns immediately
        with all errors collected.
        """
        tasks = [_make_task("q1"), _make_task("q2")]

        def mock_execute_task(task, answer_cache=None, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=2, enable_cache=False, timeout_seconds=10.0),
        )

        with pytest.raises(VerificationBatchError) as exc_info:
            executor.run_batch(tasks)

        err = exc_info.value
        assert len(err.partial_results) == 0
        assert len(err.errors) == 2
        assert all(isinstance(e, KeyboardInterrupt) for _, e in err.errors)

    def test_mixed_success_and_base_exception(self, monkeypatch) -> None:
        """Mix of successful tasks and BaseException produces partial results."""
        tasks = [_make_task("q1"), _make_task("q2"), _make_task("q3"), _make_task("q4")]

        def mock_execute_task(task, answer_cache=None, **kwargs):
            if task["question_id"] == "q2":
                raise KeyboardInterrupt()
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=2, enable_cache=False, timeout_seconds=10.0),
        )

        with pytest.raises(VerificationBatchError) as exc_info:
            executor.run_batch(tasks)

        err = exc_info.value
        assert len(err.partial_results) == 3
        assert len(err.errors) == 1


# ============================================================================
# Portal creation failure
# ============================================================================


@pytest.mark.unit
class TestVerificationPortalCreationFailure:
    """Portal creation failure for one task must not block others."""

    def test_portal_failure_collected_as_error(self, monkeypatch) -> None:
        """When start_blocking_portal raises for one task, the error is
        captured by the Future and the other tasks complete normally.
        """
        tasks = [_make_task("q1"), _make_task("q2"), _make_task("q3")]

        portal_call_count = [0]
        portal_lock = threading.Lock()

        from anyio.from_thread import start_blocking_portal as original_start_blocking_portal

        # Barrier forces all 3 worker threads to reach portal creation before
        # any one of them proceeds. Without this, a fast worker could handle
        # every task alone before sibling workers spawn, so only one portal
        # attempt ever happens and the expected second-attempt failure never
        # fires. The barrier is released after the third arrival so downstream
        # operations (portal teardown, retries) are not blocked.
        portal_barrier = threading.Barrier(3, timeout=5.0)

        class FailingPortalTracker:
            """Wraps start_blocking_portal; the second call raises RuntimeError."""

            def __init__(self) -> None:
                with contextlib.suppress(threading.BrokenBarrierError):
                    portal_barrier.wait()

                with portal_lock:
                    portal_call_count[0] += 1
                    self._should_fail = portal_call_count[0] == 2

                if self._should_fail:
                    raise RuntimeError("portal creation failed")

                self._real_cm = original_start_blocking_portal(backend="asyncio")

            def __enter__(self):
                return self._real_cm.__enter__()

            def __exit__(self, *args):
                return self._real_cm.__exit__(*args)

        def mock_start_blocking_portal(**kwargs):
            return FailingPortalTracker()

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

        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(max_workers=3, enable_cache=False, timeout_seconds=10.0),
        )

        with pytest.raises(VerificationBatchError) as exc_info:
            executor.run_batch(tasks)

        err = exc_info.value
        assert len(err.partial_results) == 2
        assert len(err.errors) == 1
        assert isinstance(err.errors[0][1], RuntimeError)
        assert "portal creation failed" in str(err.errors[0][1])


# ============================================================================
# Timeout with hanging workers
# ============================================================================


@pytest.mark.unit
class TestVerificationTimeoutWithHangingWorkers:
    """Batch timeout returns promptly even when workers are stuck."""

    def test_indefinitely_hanging_worker(self, monkeypatch) -> None:
        """One task blocks forever; the other completes. Timeout fires and
        the completed task's result is in partial_results.
        """
        tasks = [_make_task("q1"), _make_task("q2")]

        block_forever = threading.Event()

        def mock_execute_task(task, answer_cache=None, **kwargs):
            if task["question_id"] == "q2":
                block_forever.wait(timeout=30.0)
                return (f"key_{task['question_id']}", _make_result(task["question_id"]))
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(
                max_workers=2,
                enable_cache=False,
                timeout_seconds=1.0,
            ),
        )

        with pytest.raises(VerificationBatchError) as exc_info:
            executor.run_batch(tasks)

        # Unblock the hanging worker
        block_forever.set()

        err = exc_info.value
        assert len(err.partial_results) >= 1
        assert "key_q1" in err.partial_results
        assert "timed out" in str(err)


# ============================================================================
# Cache retry completes without hang
# ============================================================================


@pytest.mark.unit
class TestCacheRetryCompletes:
    """Cache retry with internal loop completes without hanging."""

    def test_cache_retry_both_tasks_complete(self, monkeypatch) -> None:
        """Two tasks share the same cache key. The first generates the answer,
        the second retries and gets the cached result. Both complete.
        """
        model = ModelConfig(
            id="shared-model",
            model_name="shared-model",
            model_provider="anthropic",
            interface="langchain",
            system_prompt="test",
            temperature=0.0,
        )
        # Same question + model = same cache key
        task1 = {
            **_make_task("q1"),
            "answering_model": model,
            "parsing_model": model,
        }
        task2 = {
            **_make_task("q1"),  # Same question_id
            "answering_model": model,
            "parsing_model": model,
        }
        tasks = [task1, task2]

        call_count = [0]
        call_lock = threading.Lock()

        def mock_execute_task(task, answer_cache=None, **kwargs):
            with call_lock:
                call_count[0] += 1
            return (f"key_{task['question_id']}_{call_count[0]}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(
                max_workers=2,
                enable_cache=True,
                retry_wait_seconds=0.5,
                timeout_seconds=10.0,
            ),
        )

        results = executor.run_batch(tasks)

        # Both tasks should complete (no hang)
        assert len(results) == 2
