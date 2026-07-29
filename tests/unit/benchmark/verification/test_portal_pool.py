"""Contract tests for the shared portal pool helper (T14).

These tests pin the behavior extracted from the two executor twins
(``VerificationExecutor._run_parallel`` and ``ScenarioExecutor._run_parallel``)
BEFORE the migration, so the helper cannot drift from the twins' semantics:

- Every submitted index ends in ``results_by_index`` or ``failed_items``
  (drop-detection invariant, including the synthetic TimeoutError entries).
- The wall-clock timeout sweep uses ``pool.shutdown(wait=True,
  cancel_futures=True)`` and the post-shutdown drain recovers futures that
  finished while the pool was draining.
- The bounded pre-teardown aclose hook runs on every worker portal BEFORE
  the portal context manager exits.
- Progress dispatch (item-start previews and completions) is serialized
  under one lock.
- The ``gate`` hook enters the limiter configuration for the duration of the
  pooled run with save-and-restore semantics (nesting cannot lift an outer
  cap).
- The sequential lifecycle helper mirrors the parallel pre/post lifecycle.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from anyio.from_thread import start_blocking_portal

from karenina.benchmark.verification.async_lifecycle import (
    get_async_portal,
    get_global_llm_limiter,
    set_async_portal,
)
from karenina.benchmark.verification.portal_pool import (
    ExecutionTuning,
    PortalPoolOutcome,
    run_in_portal_pool,
    sequential_portal,
)

# ============================================================================
# Helpers
# ============================================================================


def _describe(idx: int, item: Any) -> str:
    return f"item-{item}"


def _assert_all_accounted(outcome: PortalPoolOutcome, items: list[Any]) -> None:
    """Drop-detection invariant: every submitted index is in results or failures."""
    accounted = set(outcome.results_by_index)
    failed_descs = [desc for desc, _exc in outcome.failed_items]
    for idx, item in enumerate(items):
        if idx in accounted:
            continue
        assert any(_describe(idx, item) in desc for desc in failed_descs), (
            f"Index {idx} (item {item}) silently dropped. "
            f"results={sorted(outcome.results_by_index)}, failed={failed_descs}"
        )


@pytest.fixture(autouse=True)
def _clean_portal_state():
    """Keep main-thread portal state clean around each test."""
    set_async_portal(None)
    yield
    set_async_portal(None)


# ============================================================================
# Basic completion and failure accounting
# ============================================================================


@pytest.mark.unit
class TestPortalPoolBasics:
    """Results land by submission index; failures carry describe() labels."""

    def test_all_items_complete(self) -> None:
        items = ["a", "b", "c"]
        outcome = run_in_portal_pool(
            items,
            lambda _idx, item: f"r-{item}",
            tuning=ExecutionTuning(max_workers=2),
            describe=_describe,
        )
        assert outcome.results_by_index == {0: "r-a", 1: "r-b", 2: "r-c"}
        assert outcome.failed_items == []
        assert outcome.timed_out is False

    def test_empty_items(self) -> None:
        outcome = run_in_portal_pool(
            [],
            lambda _idx, item: item,
            tuning=ExecutionTuning(max_workers=2, timeout_seconds=5.0),
            describe=_describe,
        )
        assert outcome.results_by_index == {}
        assert outcome.failed_items == []
        assert outcome.timed_out is False

    def test_failure_recorded_with_describe_label(self) -> None:
        items = ["ok1", "boom", "ok2"]

        def worker(idx: int, item: str) -> str:
            if item == "boom":
                raise RuntimeError("worker exploded")
            return f"r-{item}"

        outcome = run_in_portal_pool(
            items,
            worker,
            tuning=ExecutionTuning(max_workers=2),
            describe=_describe,
        )
        assert outcome.results_by_index == {0: "r-ok1", 2: "r-ok2"}
        assert len(outcome.failed_items) == 1
        desc, exc = outcome.failed_items[0]
        assert desc == "item-boom"
        assert isinstance(exc, RuntimeError)
        assert outcome.timed_out is False
        _assert_all_accounted(outcome, items)

    def test_base_exception_collected_without_hang(self) -> None:
        """BaseException (KeyboardInterrupt) is harvested like the twins do."""
        items = ["ok", "interrupt"]

        def worker(idx: int, item: str) -> str:
            if item == "interrupt":
                raise KeyboardInterrupt()
            return f"r-{item}"

        outcome = run_in_portal_pool(
            items,
            worker,
            tuning=ExecutionTuning(max_workers=2, timeout_seconds=10.0),
            describe=_describe,
        )
        assert outcome.results_by_index == {0: "r-ok"}
        assert len(outcome.failed_items) == 1
        assert isinstance(outcome.failed_items[0][1], KeyboardInterrupt)
        _assert_all_accounted(outcome, items)


# ============================================================================
# Timeout sweep, cancel_futures, post-shutdown recovery, diagnostics
# ============================================================================


@pytest.mark.unit
class TestPortalPoolTimeout:
    """The timeout sweep cancels queued items and drains in-flight ones."""

    def test_cancelled_items_get_synthetic_timeout_failures(self) -> None:
        """One worker, three items: the running item finishes after the
        timeout (recovered by the post-shutdown drain), the two queued items
        are cancelled and surface as synthetic TimeoutError entries.
        """
        items = ["slow", "queued1", "queued2"]

        def worker(idx: int, item: str) -> str:
            if item == "slow":
                time.sleep(0.8)
            return f"r-{item}"

        outcome = run_in_portal_pool(
            items,
            worker,
            tuning=ExecutionTuning(max_workers=1, timeout_seconds=0.3),
            describe=_describe,
            item_noun="Task",
        )

        assert outcome.timed_out is True
        # The in-flight item was recovered by the post-shutdown drain.
        assert outcome.results_by_index == {0: "r-slow"}
        # The queued items were cancelled (cancel_futures=True) and got
        # synthetic TimeoutError entries with the noun-shaped message.
        assert len(outcome.failed_items) == 2
        for desc, exc in outcome.failed_items:
            assert isinstance(exc, TimeoutError)
            assert "Task cancelled before start" in str(exc)
            assert desc in ("item-queued1", "item-queued2")
        _assert_all_accounted(outcome, items)

    def test_late_failure_recorded_per_item(self) -> None:
        """An in-flight item that raises after the timeout appears as a
        per-item error entry, not a silent drop (issue 189 semantics).
        """
        items = ["fast", "late_fail"]

        def worker(idx: int, item: str) -> str:
            if item == "late_fail":
                time.sleep(0.8)
                raise RuntimeError("broke late")
            return f"r-{item}"

        outcome = run_in_portal_pool(
            items,
            worker,
            tuning=ExecutionTuning(max_workers=2, timeout_seconds=0.3),
            describe=_describe,
        )

        assert outcome.timed_out is True
        assert outcome.results_by_index == {0: "r-fast"}
        matching = [(d, e) for d, e in outcome.failed_items if d == "item-late_fail" and isinstance(e, RuntimeError)]
        assert matching, f"late failure dropped: {outcome.failed_items}"
        _assert_all_accounted(outcome, items)

    def test_all_in_flight_recovered_by_drain(self) -> None:
        """Worst case for the shutdown race: zero completions before the
        timeout. All items must be recovered by the post-shutdown drain.
        """
        items = ["a", "b", "c"]

        def worker(idx: int, item: str) -> str:
            time.sleep(0.6)
            return f"r-{item}"

        outcome = run_in_portal_pool(
            items,
            worker,
            tuning=ExecutionTuning(max_workers=3, timeout_seconds=0.3),
            describe=_describe,
        )

        assert outcome.timed_out is True
        assert outcome.results_by_index == {0: "r-a", 1: "r-b", 2: "r-c"}
        assert outcome.failed_items == []

    def test_timeout_diagnostics(self) -> None:
        """Diagnostics carry the snapshot taken when the timeout fired,
        mirroring the scenario twin's in_flight/completed/recovered fields.
        """
        items = ["fast", "slow"]

        def worker(idx: int, item: str) -> str:
            if item == "slow":
                time.sleep(0.8)
            return f"r-{item}"

        outcome = run_in_portal_pool(
            items,
            worker,
            tuning=ExecutionTuning(max_workers=2, timeout_seconds=0.3),
            describe=_describe,
        )

        assert outcome.timed_out is True
        assert outcome.diagnostics["completed_at_timeout"] == 1
        assert outcome.diagnostics["in_flight_at_timeout"] == {1}
        # The slow item finished during the drain and was recovered.
        assert outcome.diagnostics["recovered_from_drain"] == 1
        assert 1 in outcome.results_by_index

    def test_no_timeout_leaves_diagnostics_empty_shape(self) -> None:
        outcome = run_in_portal_pool(
            ["a"],
            lambda _idx, item: item,
            tuning=ExecutionTuning(max_workers=1, timeout_seconds=30.0),
            describe=_describe,
        )
        assert outcome.timed_out is False
        assert outcome.diagnostics["completed_at_timeout"] == 0
        assert outcome.diagnostics["in_flight_at_timeout"] == set()
        assert outcome.diagnostics["recovered_from_drain"] == 0


# ============================================================================
# Drop-detection synthetic failure (defensive branch)
# ============================================================================


@pytest.mark.unit
class TestDropDetection:
    """If a future somehow escapes both sweeps, a synthetic failure is emitted."""

    def test_uncollected_futures_get_synthetic_entries(self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        """Force as_completed to yield nothing so every future is left
        uncollected after pool.shutdown(wait=True). The drop-detection
        invariant must emit one synthetic TimeoutError per item instead of
        silently dropping them.
        """
        import logging

        from karenina.benchmark.verification import portal_pool as pp

        def _yield_nothing(_fs: Any, timeout: float | None = None) -> Any:  # noqa: ARG001
            return iter(())

        monkeypatch.setattr(pp, "as_completed", _yield_nothing)

        caplog.set_level(logging.ERROR, logger="karenina.benchmark.verification.portal_pool")

        items = ["x", "y"]
        outcome = run_in_portal_pool(
            items,
            lambda _idx, item: f"r-{item}",
            tuning=ExecutionTuning(max_workers=2),
            describe=_describe,
            item_noun="Combo",
        )

        assert outcome.results_by_index == {}
        assert len(outcome.failed_items) == 2
        for desc, exc in outcome.failed_items:
            assert isinstance(exc, TimeoutError)
            assert "Combo left uncollected" in str(exc)
            assert desc in ("item-x", "item-y")
        assert any("dropped" in rec.getMessage() for rec in caplog.records)


# ============================================================================
# Portal lifecycle: per-worker portals, factory failure, aclose before teardown
# ============================================================================


@pytest.mark.unit
class TestPortalLifecycle:
    """Worker portal creation, reuse, failure handling, and pre-teardown aclose."""

    def test_one_portal_per_worker_reused_across_items(self) -> None:
        created: list[Any] = []
        lock = threading.Lock()

        def factory():
            cm = start_blocking_portal(backend="asyncio")
            with lock:
                created.append(cm)
            return cm

        outcome = run_in_portal_pool(
            ["a", "b", "c"],
            lambda _idx, item: f"r-{item}",
            tuning=ExecutionTuning(max_workers=1),
            describe=_describe,
            portal_factory=factory,
        )
        assert len(outcome.results_by_index) == 3
        assert len(created) == 1, "single worker must create exactly one portal"

    def test_worker_sees_portal_via_thread_local(self) -> None:
        seen: list[Any] = []
        lock = threading.Lock()

        def worker(idx: int, item: str) -> str:
            with lock:
                seen.append(get_async_portal())
            return item

        run_in_portal_pool(
            ["a", "b"],
            worker,
            tuning=ExecutionTuning(max_workers=2),
            describe=_describe,
        )
        assert len(seen) == 2
        assert all(p is not None for p in seen)
        # Main thread never gains a portal from the pooled run.
        assert get_async_portal() is None

    def test_portal_factory_failure_recorded_as_item_failure(self) -> None:
        """A portal factory failure loses only the current item; the worker
        retries the factory on its next item (matching the twins' lazy
        per-task _ensure_worker_portal pattern).
        """
        calls = [0]
        lock = threading.Lock()

        def flaky_factory():
            with lock:
                calls[0] += 1
                if calls[0] == 1:
                    raise RuntimeError("portal creation failed")
            return start_blocking_portal(backend="asyncio")

        outcome = run_in_portal_pool(
            ["a", "b"],
            lambda _idx, item: f"r-{item}",
            tuning=ExecutionTuning(max_workers=1),
            describe=_describe,
            portal_factory=flaky_factory,
        )
        assert len(outcome.failed_items) == 1
        desc, exc = outcome.failed_items[0]
        assert desc == "item-a"
        assert isinstance(exc, RuntimeError)
        assert "portal creation failed" in str(exc)
        assert outcome.results_by_index == {1: "r-b"}

    def test_pre_teardown_runs_on_live_portal_before_exit(self) -> None:
        """The pre_teardown hook fires once per created portal while the
        portal's loop is still alive (BEFORE the context manager exits).
        """
        swept: list[Any] = []

        def pre_teardown(portal: Any) -> None:
            # The portal loop must still be running at sweep time.
            assert getattr(portal, "_event_loop_thread_id", None) is not None, (
                "pre_teardown ran after the portal was torn down"
            )
            swept.append(portal)

        outcome = run_in_portal_pool(
            ["a", "b", "c", "d"],
            lambda _idx, item: f"r-{item}",
            tuning=ExecutionTuning(max_workers=2),
            describe=_describe,
            pre_teardown=pre_teardown,
        )
        assert len(outcome.results_by_index) == 4
        assert 1 <= len(swept) <= 2
        assert len(set(map(id, swept))) == len(swept), "sweep ran twice on one portal"

    def test_every_created_portal_cm_is_exited(self) -> None:
        """Every portal context manager opened by a worker is exited after
        the run (no leaked portal event loops).
        """
        exits: list[int] = []
        created: list[int] = []
        lock = threading.Lock()

        class RecordingPortalCM:
            def __init__(self) -> None:
                self._cm = start_blocking_portal(backend="asyncio")
                with lock:
                    created.append(id(self))

            def __enter__(self) -> Any:
                return self._cm.__enter__()

            def __exit__(self, *args: Any) -> Any:
                with lock:
                    exits.append(id(self))
                return self._cm.__exit__(*args)

        outcome = run_in_portal_pool(
            ["a", "b", "c", "d"],
            lambda _idx, item: f"r-{item}",
            tuning=ExecutionTuning(max_workers=2),
            describe=_describe,
            portal_factory=RecordingPortalCM,
        )
        assert len(outcome.results_by_index) == 4
        assert created, "no portals were created"
        assert sorted(exits) == sorted(created), f"portal context managers leaked: created={created}, exited={exits}"

    def test_pre_teardown_runs_even_on_timeout(self) -> None:
        swept: list[Any] = []

        def worker(idx: int, item: str) -> str:
            time.sleep(0.5)
            return item

        run_in_portal_pool(
            ["a"],
            worker,
            tuning=ExecutionTuning(max_workers=1, timeout_seconds=0.2),
            describe=_describe,
            pre_teardown=lambda portal: swept.append(portal),
        )
        assert len(swept) == 1


# ============================================================================
# Progress dispatch under one lock
# ============================================================================


@pytest.mark.unit
class TestProgressSerialization:
    """Item-start previews and completion callbacks share one lock."""

    def test_progress_callbacks_never_interleave(self) -> None:
        """A slow on_progress must never be re-entered concurrently, and
        on_item_start must not interleave with it (both run under the lock).
        """
        inside = [False]
        violations = [0]
        guard = threading.Lock()

        def _enter() -> None:
            with guard:
                if inside[0]:
                    violations[0] += 1
                inside[0] = True

        def _exit() -> None:
            with guard:
                inside[0] = False

        def on_item_start(current: int, total: int, item: Any) -> None:
            _enter()
            time.sleep(0.02)
            _exit()

        def on_progress(current: int, total: int, result: Any) -> None:
            _enter()
            time.sleep(0.02)
            _exit()

        run_in_portal_pool(
            list(range(8)),
            lambda _idx, item: item,
            tuning=ExecutionTuning(max_workers=4),
            describe=_describe,
            on_item_start=on_item_start,
            on_progress=on_progress,
        )
        assert violations[0] == 0, f"{violations[0]} concurrent progress dispatches observed"

    def test_progress_counts_match_twin_semantics(self) -> None:
        """on_item_start sees completed+1 as a preview index and on_progress
        sees the post-increment completed count with the result.

        Even with a single worker the full interleaving is NOT
        deterministic: the worker thread fires the next item's preview while
        the collector thread is still dispatching the previous completion
        (the twins had the same race, since completed_count is incremented
        by the collector and read by the worker). What IS exact and pinned
        here: the done sequence is done(1, r-a) then done(2, r-b), starts
        arrive in item order, and every start's current equals the number
        of done events dispatched before it plus one (the twins'
        completed+1 preview convention, event by event).
        """
        events: list[tuple[str, int, int, Any]] = []
        lock = threading.Lock()

        def on_item_start(current: int, total: int, item: Any) -> None:
            with lock:
                events.append(("start", current, total, item))

        def on_progress(current: int, total: int, result: Any) -> None:
            with lock:
                events.append(("done", current, total, result))

        run_in_portal_pool(
            ["a", "b"],
            lambda _idx, item: f"r-{item}",
            tuning=ExecutionTuning(max_workers=1),
            describe=_describe,
            on_item_start=on_item_start,
            on_progress=on_progress,
        )

        # Done dispatch is exact: post-increment counts, in completion order.
        done_events = [e for e in events if e[0] == "done"]
        assert done_events == [("done", 1, 2, "r-a"), ("done", 2, 2, "r-b")]
        # Starts arrive in item order with the shared total.
        start_events = [e for e in events if e[0] == "start"]
        assert [e[3] for e in start_events] == ["a", "b"]
        assert all(e[2] == 2 for e in events)
        # The completed+1 convention, pinned event by event: each preview's
        # current is the number of completions dispatched before it plus one.
        dones_so_far = 0
        for kind, current, _total, _payload in events:
            if kind == "start":
                assert current == dones_so_far + 1, (
                    f"preview current={current} but {dones_so_far} completions had been dispatched"
                )
            else:
                dones_so_far += 1

    def test_drain_recovered_results_do_not_fire_on_progress(self) -> None:
        """Matching the twins: results harvested by the timeout sweeps do not
        dispatch the completion callback.
        """
        completions: list[Any] = []

        def worker(idx: int, item: str) -> str:
            time.sleep(0.5)
            return item

        outcome = run_in_portal_pool(
            ["a"],
            worker,
            tuning=ExecutionTuning(max_workers=1, timeout_seconds=0.2),
            describe=_describe,
            on_progress=lambda _c, _t, r: completions.append(r),
        )
        assert outcome.timed_out is True
        assert 0 in outcome.results_by_index
        assert completions == []


# ============================================================================
# Gate hook (limiter configure save-and-restore)
# ============================================================================


@pytest.mark.unit
class TestGateHook:
    """The gate context manager wraps the pooled run."""

    def test_gate_configures_limiter_for_duration(self) -> None:
        limiter = get_global_llm_limiter()
        seen: list[int | None] = []
        lock = threading.Lock()

        def worker(idx: int, item: str) -> str:
            with lock:
                seen.append(limiter.capacity)
            return item

        run_in_portal_pool(
            ["a", "b"],
            worker,
            tuning=ExecutionTuning(max_workers=2),
            describe=_describe,
            gate=limiter.configure(3),
        )
        assert seen == [3, 3]
        assert limiter.capacity is None

    def test_nested_gate_cannot_lift_outer_cap(self) -> None:
        """Save-and-restore semantics: an outer configure(2) survives a
        pooled run whose gate asks for a larger cap, and is restored after.
        """
        limiter = get_global_llm_limiter()
        seen: list[int | None] = []

        with limiter.configure(2):
            run_in_portal_pool(
                ["a"],
                lambda _idx, _item: seen.append(limiter.capacity),
                tuning=ExecutionTuning(max_workers=1),
                describe=_describe,
                gate=limiter.configure(5),
            )
            assert limiter.capacity == 2
        assert seen == [2]
        assert limiter.capacity is None

    def test_gate_exits_on_failure(self) -> None:
        limiter = get_global_llm_limiter()

        def worker(idx: int, item: str) -> str:
            raise RuntimeError("boom")

        outcome = run_in_portal_pool(
            ["a"],
            worker,
            tuning=ExecutionTuning(max_workers=1),
            describe=_describe,
            gate=limiter.configure(4),
        )
        assert len(outcome.failed_items) == 1
        assert limiter.capacity is None


# ============================================================================
# Sequential lifecycle parity
# ============================================================================


@pytest.mark.unit
class TestSequentialPortalParity:
    """sequential_portal shares the pre/post lifecycle with the pool."""

    def test_portal_set_during_and_cleared_after(self) -> None:
        observed: list[Any] = []
        with sequential_portal() as portal:
            assert portal is not None
            observed.append(get_async_portal())
        assert observed == [portal]
        assert get_async_portal() is None

    def test_pre_teardown_called_on_live_portal(self) -> None:
        swept: list[Any] = []

        def pre_teardown(portal: Any) -> None:
            assert getattr(portal, "_event_loop_thread_id", None) is not None
            swept.append(portal)

        with sequential_portal(pre_teardown=pre_teardown) as portal:
            pass
        assert swept == [portal]

    def test_pre_teardown_and_clear_on_exception(self) -> None:
        swept: list[Any] = []
        with pytest.raises(RuntimeError, match="boom"), sequential_portal(pre_teardown=swept.append):
            raise RuntimeError("boom")
        assert len(swept) == 1
        assert get_async_portal() is None

    def test_custom_portal_factory_used(self) -> None:
        calls = [0]

        def factory():
            calls[0] += 1
            return start_blocking_portal(backend="asyncio")

        with sequential_portal(portal_factory=factory):
            pass
        assert calls[0] == 1


# ============================================================================
# ExecutionTuning shape
# ============================================================================


@pytest.mark.unit
class TestExecutionTuning:
    def test_defaults(self) -> None:
        t = ExecutionTuning()
        assert t.max_workers == 2  # DEFAULT_ASYNC_MAX_WORKERS
        assert t.timeout_seconds is None
        assert t.retry_wait_seconds == 5.0
        assert t.max_requeue_count == 5
        assert t.answerer_concurrency_limits is None
        assert t.max_concurrent_requests is None

    def test_executor_config_derives_tuning(self) -> None:
        from karenina.benchmark.verification.executor import ExecutorConfig

        cfg = ExecutorConfig(
            max_workers=7,
            enable_cache=False,
            retry_wait_seconds=1.5,
            timeout_seconds=42.0,
            max_requeue_count=9,
            answerer_concurrency_limits={"m1": 3},
        )
        t = cfg.to_tuning()
        assert isinstance(t, ExecutionTuning)
        assert t.max_workers == 7
        assert t.retry_wait_seconds == 1.5
        assert t.timeout_seconds == 42.0
        assert t.max_requeue_count == 9
        assert t.answerer_concurrency_limits == {"m1": 3}

    def test_scenario_config_derives_tuning(self) -> None:
        from karenina.benchmark.verification.scenario_executor import ScenarioExecutorConfig

        cfg = ScenarioExecutorConfig(
            max_workers=4,
            max_concurrent_requests=6,
            enable_cache=False,
            timeout_seconds=12.0,
        )
        t = cfg.to_tuning()
        assert isinstance(t, ExecutionTuning)
        assert t.max_workers == 4
        assert t.max_concurrent_requests == 6
        assert t.timeout_seconds == 12.0
