"""Tests for per-answerer cap acquisition placement in the parallel executor (T16a).

The per-answerer endpoint cap must wrap the cache-reservation loop, not just
``execute_task``: a surplus worker blocked on the cap must not reserve an
IN_PROGRESS cache entry while merely waiting for a permit. Conversely, the
wait-for-another-worker branch (someone else owns the IN_PROGRESS
reservation) must not hold the permit while sleeping. Also covers the
one-time debug log for answerer ids without a configured cap.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from karenina.benchmark.verification.executor import ExecutorConfig, VerificationExecutor
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import VerificationResultMetadata
from karenina.utils.answer_cache import AnswerTraceCache

# ============================================================================
# Helpers
# ============================================================================


def _make_result(question_id: str = "q1") -> VerificationResult:
    identity = ModelIdentity(model_name="test-model", interface="langchain")
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="test_template",
            failure=None,
            caveats=[],
            question_text="Test question",
            answering=identity,
            parsing=identity,
            execution_time=0.1,
            timestamp="2026-01-01T00:00:00",
            result_id="abcdef1234567890",
        )
    )


def _make_task(question_id: str, ans_id: str = "capped") -> dict:
    model = ModelConfig(
        id=ans_id,
        model_name=ans_id,
        model_provider="openai",
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


def _passthrough_execute_task(task, *_args, **_kwargs):
    """execute_task stub returning a canned result for the task."""
    return (f"key_{task['question_id']}", _make_result(task["question_id"]))


class RecordingAnswerCache(AnswerTraceCache):
    """AnswerTraceCache that records every key passed to get_or_reserve."""

    def __init__(self) -> None:
        super().__init__()
        self.reserve_calls: list[str] = []
        self._record_lock = threading.Lock()

    def get_or_reserve(self, key: str) -> tuple[str, dict | None]:
        with self._record_lock:
            self.reserve_calls.append(key)
        return super().get_or_reserve(key)

    def distinct_keys(self) -> set[str]:
        with self._record_lock:
            return set(self.reserve_calls)


# ============================================================================
# Permit wraps the reservation loop
# ============================================================================


@pytest.mark.unit
class TestCapWrapsCacheReservation:
    """A worker blocked on the cap must not reserve IN_PROGRESS first."""

    def test_blocked_surplus_worker_does_not_reserve(self, monkeypatch) -> None:
        """Two tasks share an answerer capped at 1. While the permit holder
        is blocked inside execute_task, the other worker waits on the permit
        BEFORE touching the cache, so exactly one cache key is reserved.
        With the pre-T16 placement (cap acquired after the reservation),
        both keys would already be reserved during the blocked window.
        """
        cache = RecordingAnswerCache()
        monkeypatch.setattr(
            "karenina.benchmark.verification.executor.AnswerTraceCache",
            lambda: cache,
        )

        started = threading.Event()
        release = threading.Event()

        def mock_execute_task(task, answer_cache=None, cache_status=None, cached_answer_data=None):
            started.set()
            release.wait(timeout=10.0)
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(
                max_workers=2,
                enable_cache=True,
                answerer_concurrency_limits={"capped": 1},
                timeout_seconds=30.0,
            ),
        )
        tasks = [_make_task("q1"), _make_task("q2")]

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(executor.run_batch, tasks)
            try:
                assert started.wait(timeout=5.0), "permit holder never started executing"
                # Give the surplus worker ample time to (incorrectly) reach
                # the cache before blocking on the permit.
                time.sleep(0.5)
                blocked_window_keys = cache.distinct_keys()
                assert len(blocked_window_keys) == 1, (
                    f"surplus worker reserved while waiting for a permit: {blocked_window_keys}"
                )
            finally:
                release.set()
            results = future.result(timeout=15.0)

        assert len(results) == 2
        # After the permit was handed over, the second key was reserved too.
        assert len(cache.distinct_keys()) == 2

    def test_in_progress_waiter_does_not_hold_permit(self, monkeypatch) -> None:
        """Three workers, cap 2, two tasks sharing one cache key plus one
        independent task. The worker that loses the shared-key race waits on
        the other's IN_PROGRESS entry; while it sleeps it must NOT hold a
        permit, so the independent task can take the second permit and run
        even though the shared-key holder is still blocked.
        """
        started_q1 = threading.Event()
        started_q2 = threading.Event()
        release = threading.Event()
        key_counter = [0]
        counter_lock = threading.Lock()

        def mock_execute_task(task, answer_cache=None, cache_status=None, cached_answer_data=None):
            qid = task["question_id"]
            if qid == "q1":
                started_q1.set()
                release.wait(timeout=10.0)
            else:
                started_q2.set()
            with counter_lock:
                key_counter[0] += 1
                unique = key_counter[0]
            return (f"key_{qid}_{unique}", _make_result(qid))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(
            parallel=True,
            config=ExecutorConfig(
                max_workers=3,
                enable_cache=True,
                retry_wait_seconds=0.2,
                max_requeue_count=50,
                answerer_concurrency_limits={"capped": 2},
                timeout_seconds=30.0,
            ),
        )
        # q1 twice (same cache key) plus q2 (own key), all on the capped answerer.
        tasks = [_make_task("q1"), _make_task("q1"), _make_task("q2")]

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(executor.run_batch, tasks)
            try:
                assert started_q1.wait(timeout=5.0), "shared-key holder never started"
                # q2 must be able to run while q1 blocks and the duplicate
                # q1 worker waits on IN_PROGRESS. If the waiter held its
                # permit while sleeping, both permits would be consumed and
                # q2 could not start until release.
                assert started_q2.wait(timeout=5.0), "independent task starved while a waiter idled on the cap"
                assert not release.is_set()
            finally:
                release.set()
            results = future.result(timeout=15.0)

        assert len(results) == 3


# ============================================================================
# One-time debug log for uncapped answerer ids
# ============================================================================


@pytest.mark.unit
class TestUncappedAnswererDebugLog:
    """Unknown answerer ids log once at DEBUG instead of silently no-opping."""

    def _executor(self, limits: dict[str, int] | None) -> VerificationExecutor:
        return VerificationExecutor(
            parallel=False,
            config=ExecutorConfig(answerer_concurrency_limits=limits),
        )

    def test_unknown_id_logs_once(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            _passthrough_execute_task,
        )
        executor = self._executor({"known": 1})
        task = _make_task("q1", ans_id="mystery")

        with caplog.at_level("DEBUG", logger="karenina.benchmark.verification.executor"):
            executor._execute_task_with_cap(task, None, None, None)
            executor._execute_task_with_cap(task, None, None, None)

        records = [r for r in caplog.records if "mystery" in r.getMessage() and "no cap" in r.getMessage()]
        assert len(records) == 1
        assert records[0].levelname == "DEBUG"

    def test_known_id_does_not_log(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            _passthrough_execute_task,
        )
        executor = self._executor({"capped": 2})
        task = _make_task("q1", ans_id="capped")

        with caplog.at_level("DEBUG", logger="karenina.benchmark.verification.executor"):
            executor._execute_task_with_cap(task, None, None, None)

        assert not [r for r in caplog.records if "no cap" in r.getMessage()]

    def test_no_limits_configured_does_not_log(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            _passthrough_execute_task,
        )
        executor = self._executor(None)
        task = _make_task("q1", ans_id="anything")

        with caplog.at_level("DEBUG", logger="karenina.benchmark.verification.executor"):
            executor._execute_task_with_cap(task, None, None, None)

        assert not [r for r in caplog.records if "no cap" in r.getMessage()]
