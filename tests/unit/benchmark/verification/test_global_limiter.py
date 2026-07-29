"""Tests for the GlobalLLMLimiter (T13, design decision D2).

The limiter is the process-wide enforcement mechanism for
``VerificationConfig.max_concurrent_requests``: a ref-counted,
cross-event-loop cap on concurrent LLM request setups, borrowed at the
adapter async leaves. The legacy module-global threading.Semaphore
(``get/set_global_llm_semaphore`` and ``with_llm_semaphore``) stays
functional but is no longer set by production code. Its tests live in
test_global_semaphore.py / test_parallel_base_semaphore.py and must keep
passing unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import pytest
from anyio.from_thread import start_blocking_portal

from karenina.benchmark.verification import async_lifecycle, executor
from karenina.benchmark.verification.async_lifecycle import (
    GlobalLLMLimiter,
    gate_stream_establishment,
    get_global_llm_limiter,
)
from karenina.exceptions import GlobalLimiterTimeoutError


class _InFlight:
    """Thread-safe in-flight counter usable across loops and threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = 0
        self.max_observed = 0
        self.total = 0

    def enter(self) -> None:
        with self._lock:
            self._current += 1
            self.total += 1
            self.max_observed = max(self.max_observed, self._current)

    def exit(self) -> None:
        with self._lock:
            self._current -= 1


@pytest.mark.unit
class TestBorrowUnconfigured:
    """borrow() is a zero-cost no-op while no configure is active."""

    @pytest.mark.asyncio
    async def test_borrow_noop_when_unconfigured(self) -> None:
        limiter = GlobalLLMLimiter()
        assert limiter.capacity is None
        async with limiter.borrow():
            pass
        # No semaphore was ever created.
        assert limiter._semaphore is None

    @pytest.mark.asyncio
    async def test_borrow_noop_with_none_configure(self) -> None:
        """configure(None) enables the ref-count but sets no capacity."""
        limiter = GlobalLLMLimiter()
        with limiter.configure(None):
            assert limiter.capacity is None
            async with limiter.borrow():
                pass
            assert limiter._semaphore is None


@pytest.mark.unit
class TestCapEnforcedCrossLoop:
    """The cap holds across multiple event loops on multiple threads."""

    def test_cap_enforced_across_threads_with_separate_loops(self) -> None:
        """Six borrows on two separate loops never exceed cap=2.

        This proves the core is not loop-affine: each thread runs its own
        asyncio loop (the same shape as karenina's one-portal-per-worker
        execution), and the shared cap still binds across both.
        """
        limiter = GlobalLLMLimiter()
        counter = _InFlight()
        errors: list[BaseException] = []

        async def borrower() -> None:
            async with limiter.borrow():
                counter.enter()
                try:
                    await asyncio.sleep(0.1)
                finally:
                    counter.exit()

        def thread_main() -> None:
            async def run_three() -> None:
                await asyncio.gather(*(borrower() for _ in range(3)))

            try:
                asyncio.run(run_three())
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with limiter.configure(2):
            threads = [threading.Thread(target=thread_main) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        assert not errors
        assert counter.total == 6
        assert counter.max_observed <= 2

    def test_cap_one_sync_invoke_via_portals_does_not_deadlock(self) -> None:
        """cap=1 with sync invoke through real BlockingPortals completes.

        Mirrors the production dispatch: sync invoke() reaches the wire
        only via portal.call(ainvoke), so exactly one borrow happens per
        logical request and cap=1 serializes without deadlocking.
        """
        limiter = GlobalLLMLimiter()
        counter = _InFlight()
        errors: list[BaseException] = []

        class _StubAdapter:
            async def ainvoke(self, payload: str) -> str:
                async with limiter.borrow():
                    counter.enter()
                    try:
                        await asyncio.sleep(0.05)
                    finally:
                        counter.exit()
                    return f"ok:{payload}"

            def invoke(self, payload: str) -> str:
                portal = async_lifecycle.get_async_portal()
                assert portal is not None
                return portal.call(self.ainvoke, payload)

        adapter = _StubAdapter()

        def worker(idx: int) -> None:
            try:
                with start_blocking_portal(backend="asyncio") as portal:
                    async_lifecycle.set_async_portal(portal)
                    try:
                        for j in range(2):
                            assert adapter.invoke(f"{idx}-{j}") == f"ok:{idx}-{j}"
                    finally:
                        async_lifecycle.set_async_portal(None)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with limiter.configure(1):
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
            for t in threads:
                assert not t.is_alive(), "deadlock: worker did not finish at cap=1"

        assert not errors
        assert counter.total == 6
        assert counter.max_observed == 1


@pytest.mark.unit
class TestConfigureRefCount:
    """configure() is ref-counted and never resizes mid-flight."""

    def test_nested_same_capacity_shares(self) -> None:
        limiter = GlobalLLMLimiter()
        with limiter.configure(4):
            assert limiter.capacity == 4
            with limiter.configure(4):
                assert limiter.capacity == 4
            # Inner exit keeps the outer cap.
            assert limiter.capacity == 4
        assert limiter.capacity is None

    def test_nested_none_keeps_outer_capacity(self) -> None:
        limiter = GlobalLLMLimiter()
        with limiter.configure(3):
            with limiter.configure(None):
                assert limiter.capacity == 3
            assert limiter.capacity == 3
        assert limiter.capacity is None

    def test_differing_capacity_warns_and_keeps_first(self, caplog: pytest.LogCaptureFixture) -> None:
        limiter = GlobalLLMLimiter()
        caplog.set_level(logging.WARNING, logger="karenina.benchmark.verification.async_lifecycle")
        with limiter.configure(2):
            with limiter.configure(5):
                assert limiter.capacity == 2
            assert limiter.capacity == 2
        assert limiter.capacity is None
        warning = next(rec for rec in caplog.records if "already configured" in rec.message)
        assert "2" in warning.message
        assert "5" in warning.message

    def test_first_non_none_sets_capacity_after_none_enable(self) -> None:
        limiter = GlobalLLMLimiter()
        with limiter.configure(None):
            assert limiter.capacity is None
            with limiter.configure(6):
                assert limiter.capacity == 6
        assert limiter.capacity is None


@pytest.mark.unit
class TestBorrowCancellationAndTimeout:
    """Cancellation and acquire timeout never leak permits."""

    @pytest.mark.asyncio
    async def test_cancellation_during_wait_does_not_leak(self) -> None:
        # Opt-in acquire timeout keeps the executor thread of the
        # cancelled waiter bounded if the handshake ever regressed.
        limiter = GlobalLLMLimiter(acquire_timeout=5.0)
        with limiter.configure(1):
            entered = asyncio.Event()
            release = asyncio.Event()

            async def holder() -> None:
                async with limiter.borrow():
                    entered.set()
                    await release.wait()

            holder_task = asyncio.create_task(holder())
            await entered.wait()

            async def waiter() -> None:
                async with limiter.borrow():
                    pass

            waiter_task = asyncio.create_task(waiter())
            # Let the waiter reach the blocking acquire in the executor.
            await asyncio.sleep(0.2)
            waiter_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter_task

            # The cancelled waiter's acquire thread may still grab the
            # permit when the holder releases, and the handshake must give
            # it back. Capacity recovers: a fresh borrow succeeds promptly.
            release.set()
            await holder_task
            async with asyncio.timeout(5):
                async with limiter.borrow():
                    pass

    @pytest.mark.asyncio
    async def test_opt_in_acquire_timeout_raises_domain_error_without_leak(self) -> None:
        """The acquire timeout is opt-in via the constructor (default unbounded)."""
        limiter = GlobalLLMLimiter(acquire_timeout=0.2)
        with limiter.configure(1):
            entered = asyncio.Event()
            release = asyncio.Event()

            async def holder() -> None:
                async with limiter.borrow():
                    entered.set()
                    await release.wait()

            holder_task = asyncio.create_task(holder())
            await entered.wait()

            with pytest.raises(GlobalLimiterTimeoutError):
                async with limiter.borrow():
                    pass

            release.set()
            await holder_task
            # The timed-out borrow did not consume a permit.
            async with asyncio.timeout(5):
                async with limiter.borrow():
                    pass

    @pytest.mark.asyncio
    async def test_default_acquire_is_unbounded(self) -> None:
        """The default limiter blocks until a permit frees (legacy semantics).

        No GlobalLimiterTimeoutError can fire without the opt-in
        constructor timeout: a healthy backlogged batch waits as long as
        it needs to.
        """
        limiter = GlobalLLMLimiter()
        assert limiter._acquire_timeout is None
        with limiter.configure(1):
            entered = asyncio.Event()
            release = asyncio.Event()

            async def holder() -> None:
                async with limiter.borrow():
                    entered.set()
                    await release.wait()

            holder_task = asyncio.create_task(holder())
            await entered.wait()

            async def delayed_release() -> None:
                await asyncio.sleep(0.4)
                release.set()

            releaser = asyncio.create_task(delayed_release())
            # Waits through saturation without raising, then proceeds.
            async with limiter.borrow():
                pass
            await asyncio.gather(holder_task, releaser)

    @pytest.mark.asyncio
    async def test_exception_in_body_releases_permit(self) -> None:
        limiter = GlobalLLMLimiter()
        with limiter.configure(1):
            with pytest.raises(RuntimeError, match="boom"):
                async with limiter.borrow():
                    raise RuntimeError("boom")
            async with asyncio.timeout(5):
                async with limiter.borrow():
                    pass


@pytest.mark.unit
class TestGateStreamEstablishment:
    """gate_stream_establishment holds the permit for the first item only."""

    @pytest.mark.asyncio
    async def test_permit_released_after_first_item(self) -> None:
        limiter = get_global_llm_limiter()

        async def source() -> Any:
            yield "first"
            yield "second"

        with limiter.configure(1):
            gated = gate_stream_establishment(source())
            first = await gated.__anext__()
            assert first == "first"
            # Permit was released before yielding: a fresh borrow succeeds
            # while the stream is still open.
            async with asyncio.timeout(5):
                async with limiter.borrow():
                    pass
            assert await gated.__anext__() == "second"
            with pytest.raises(StopAsyncIteration):
                await gated.__anext__()

    @pytest.mark.asyncio
    async def test_empty_source_does_not_leak(self) -> None:
        limiter = get_global_llm_limiter()

        async def source() -> Any:
            for item in ():  # pragma: no cover
                yield item

        with limiter.configure(1):
            items = [item async for item in gate_stream_establishment(source())]
            assert items == []
            async with asyncio.timeout(5):
                async with limiter.borrow():
                    pass

    @pytest.mark.asyncio
    async def test_passthrough_when_unconfigured(self) -> None:
        async def source() -> Any:
            yield 1
            yield 2

        items = [item async for item in gate_stream_establishment(source())]
        assert items == [1, 2]


@pytest.mark.unit
class TestSingletonAndReExports:
    """Singleton identity and the executor module re-export."""

    def test_get_global_llm_limiter_is_singleton(self) -> None:
        assert get_global_llm_limiter() is get_global_llm_limiter()

    def test_executor_reexports_leaf_symbols(self) -> None:
        assert executor.get_global_llm_limiter is async_lifecycle.get_global_llm_limiter
        assert executor.GlobalLLMLimiter is async_lifecycle.GlobalLLMLimiter


@pytest.mark.unit
class TestExecutorWiring:
    """QA and scenario paths enter configure(max_concurrent_requests)."""

    def test_scenario_sequential_configures_limiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        from karenina.benchmark.verification.scenario_executor import (
            ScenarioExecutor,
            ScenarioExecutorConfig,
        )

        limiter = get_global_llm_limiter()
        captured: list[int | None] = []

        def mock_run(**kwargs: Any) -> Any:
            captured.append(limiter.capacity)
            result = MagicMock()
            result.scenario_id = "s1"
            return result

        manager_cls = MagicMock()
        manager_cls.return_value.run.side_effect = mock_run
        monkeypatch.setattr(
            "karenina.benchmark.verification.scenario_executor.ScenarioManager",
            manager_cls,
        )

        scenario_def = MagicMock()
        scenario_def.name = "s1"
        ans = MagicMock()
        ans.model_name = "m-a"
        parse = MagicMock()
        parse.model_name = "m-p"
        config = MagicMock()
        config.workspace_output_mode = "none"
        config.workspace_output_dir = None

        sx = ScenarioExecutor(
            parallel=False,
            config=ScenarioExecutorConfig(enable_cache=False, max_concurrent_requests=5),
        )
        results, errors = sx.run_batch([(scenario_def, ans, parse, None)], config)

        assert not errors
        assert captured == [5]
        assert limiter.capacity is None

    def test_scenario_parallel_configures_limiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        from karenina.benchmark.verification.scenario_executor import (
            ScenarioExecutor,
            ScenarioExecutorConfig,
        )

        limiter = get_global_llm_limiter()
        captured: list[int | None] = []

        def mock_run(**kwargs: Any) -> Any:
            captured.append(limiter.capacity)
            result = MagicMock()
            result.scenario_id = "s1"
            return result

        manager_cls = MagicMock()
        manager_cls.return_value.run.side_effect = mock_run
        monkeypatch.setattr(
            "karenina.benchmark.verification.scenario_executor.ScenarioManager",
            manager_cls,
        )

        scenario_def = MagicMock()
        scenario_def.name = "s1"
        ans = MagicMock()
        ans.model_name = "m-a"
        parse = MagicMock()
        parse.model_name = "m-p"
        config = MagicMock()
        config.workspace_output_mode = "none"
        config.workspace_output_dir = None

        sx = ScenarioExecutor(
            parallel=True,
            config=ScenarioExecutorConfig(max_workers=1, enable_cache=False, max_concurrent_requests=7),
        )
        results, errors = sx.run_batch([(scenario_def, ans, parse, None)], config)

        assert not errors
        assert captured == [7]
        assert limiter.capacity is None

    def test_qa_batch_runner_configures_limiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """run_verification_batch wraps the executor run in configure(n)."""
        from datetime import datetime

        import karenina.benchmark.verification.batch_runner as batch_runner_module
        import karenina.benchmark.verification.executor as executor_module
        from karenina.benchmark.verification.batch_runner import run_verification_batch
        from karenina.schemas.config import ModelConfig
        from karenina.schemas.verification import FinishedTemplate, VerificationConfig

        limiter = get_global_llm_limiter()
        captured: list[int | None] = []

        class _CapturingExecutor:
            def __init__(self, parallel: bool = True, config: Any = None) -> None:  # noqa: ARG002
                self.parallel = parallel

            def run_batch(self, tasks: Any, progress_callback: Any, prior_results: Any = None) -> dict[str, Any]:  # noqa: ARG002
                captured.append(limiter.capacity)
                return {}

        class _FakeExecutorConfig:
            def __init__(self, **_kwargs: Any) -> None:
                pass

        monkeypatch.setattr(executor_module, "VerificationExecutor", _CapturingExecutor)
        monkeypatch.setattr(executor_module, "ExecutorConfig", _FakeExecutorConfig)
        monkeypatch.setenv("AUTOSAVE_DATABASE", "false")
        monkeypatch.setattr(batch_runner_module, "cleanup_resources", lambda: None)

        model = ModelConfig(
            id="m",
            model_name="m",
            interface="openai_endpoint",
            endpoint_base_url="http://localhost:1",
            endpoint_api_key="EMPTY",
        )
        config = VerificationConfig(
            answering_models=[model],
            parsing_models=[model],
            max_concurrent_requests=3,
        )
        template = FinishedTemplate(
            question_id="q1",
            question_text="Question?",
            question_preview="Question?",
            template_code="class Answer: pass",
            last_modified=datetime.utcnow().isoformat(),
        )

        run_verification_batch(templates=[template], config=config)

        assert captured == [3]
        assert limiter.capacity is None

    @pytest.mark.asyncio
    async def test_borrow_waits_when_saturated(self) -> None:
        """A saturated limiter makes borrow wait until a permit frees up."""
        limiter = GlobalLLMLimiter()
        with limiter.configure(1):
            release = asyncio.Event()
            entered = asyncio.Event()

            async def holder() -> None:
                async with limiter.borrow():
                    entered.set()
                    await release.wait()

            holder_task = asyncio.create_task(holder())
            await entered.wait()

            start = time.monotonic()

            async def delayed_release() -> None:
                await asyncio.sleep(0.3)
                release.set()

            releaser = asyncio.create_task(delayed_release())
            async with limiter.borrow():
                waited = time.monotonic() - start
            await asyncio.gather(holder_task, releaser)
            assert waited >= 0.25
