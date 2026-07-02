"""Tests for the shared pre-teardown aclose sweep (async_lifecycle.aclose_portal_adapters).

Covers:
- Both sequential executor paths invoke the sweep before portal teardown.
- The sweep is bounded: a wedged adapter aclose gets cancelled at the bound
  and the sweep never raises out of the caller's finally block.
"""

from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import MagicMock, patch

import pytest
from anyio.from_thread import start_blocking_portal

from karenina.benchmark.verification.async_lifecycle import (
    aclose_portal_adapters,
    set_async_portal,
)
from karenina.benchmark.verification.executor import ExecutorConfig, VerificationExecutor
from karenina.benchmark.verification.scenario_executor import (
    ScenarioExecutor,
    ScenarioExecutorConfig,
)
from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import VerificationResultMetadata

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


def _make_task(question_id: str = "q1") -> dict:
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


def _make_combo(scenario_name: str = "s1") -> tuple:
    scenario_def = MagicMock()
    scenario_def.name = scenario_name
    ans_model = MagicMock()
    ans_model.model_name = "test-model"
    parse_model = MagicMock()
    parse_model.model_name = "test-model"
    return (scenario_def, ans_model, parse_model, None)


@pytest.fixture(autouse=True)
def _clean_portal_state():
    """Keep thread-local portal state clean around each test."""
    set_async_portal(None)
    yield
    set_async_portal(None)


# ============================================================================
# Sequential paths invoke the sweep
# ============================================================================


@pytest.mark.unit
class TestSequentialPathsInvokeSweep:
    """Both _run_sequential paths must run the pre-teardown aclose sweep."""

    def test_qa_sequential_invokes_sweep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sweep_calls: list[tuple] = []

        def _recording_sweep(portal, timeout):
            sweep_calls.append((portal, timeout))

        monkeypatch.setattr(
            "karenina.benchmark.verification.executor.aclose_portal_adapters",
            _recording_sweep,
        )

        def mock_execute_task(task, answer_cache=None, cache_status=None, cached_answer_data=None):
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        executor = VerificationExecutor(parallel=False, config=ExecutorConfig(enable_cache=False))
        results = executor.run_batch([_make_task("q1"), _make_task("q2")])

        assert len(results) == 2
        assert len(sweep_calls) == 1, "QA sequential path must invoke the sweep exactly once"
        portal, timeout = sweep_calls[0]
        # The shared sequential portal is passed in with the configured bound.
        assert portal is not None
        assert timeout == pytest.approx(5.0)

    def test_qa_sequential_invokes_sweep_even_on_task_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from karenina.exceptions import VerificationBatchError

        sweep_calls: list[tuple] = []
        monkeypatch.setattr(
            "karenina.benchmark.verification.executor.aclose_portal_adapters",
            lambda portal, timeout: sweep_calls.append((portal, timeout)),
        )

        def _boom(task, answer_cache=None, cache_status=None, cached_answer_data=None):
            raise RuntimeError("task failed")

        monkeypatch.setattr("karenina.benchmark.verification.batch_runner.execute_task", _boom)

        executor = VerificationExecutor(parallel=False, config=ExecutorConfig(enable_cache=False))
        with pytest.raises(VerificationBatchError):
            executor.run_batch([_make_task("q1")])

        assert len(sweep_calls) == 1, "Sweep must run from the finally block even when tasks fail"

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_scenario_sequential_invokes_sweep(
        self, mock_manager_cls: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sweep_calls: list[tuple] = []
        monkeypatch.setattr(
            "karenina.benchmark.verification.scenario_executor.aclose_portal_adapters",
            lambda portal, timeout: sweep_calls.append((portal, timeout)),
        )

        exec_result = MagicMock()
        mock_manager_cls.return_value.run.return_value = exec_result

        config = MagicMock()
        config.workspace_output_mode = "none"

        executor = ScenarioExecutor(parallel=False, config=ScenarioExecutorConfig(enable_cache=False))
        results, errors = executor.run_batch([_make_combo("s1"), _make_combo("s2")], config)

        assert len(results) == 2
        assert errors == []
        assert len(sweep_calls) == 1, "Scenario sequential path must invoke the sweep exactly once"
        portal, timeout = sweep_calls[0]
        assert portal is not None
        assert timeout == pytest.approx(5.0)


# ============================================================================
# Sweep is bounded and never raises
# ============================================================================


@pytest.mark.unit
class TestSweepBoundedAndNonRaising:
    """aclose_portal_adapters must be bounded and must never raise."""

    def test_wedged_aclose_cancelled_at_bound(self, caplog: pytest.LogCaptureFixture) -> None:
        from karenina.adapters.registry import (
            _adapter_portal_lock,
            _adapter_portal_refs,
            register_adapter,
            snapshot_adapters_for_portal,
        )

        with _adapter_portal_lock:
            _adapter_portal_refs.clear()

        class StuckAdapter:
            async def aclose(self) -> None:
                await asyncio.sleep(30.0)

        caplog.set_level(logging.WARNING, logger="karenina.benchmark.verification.async_lifecycle")

        with start_blocking_portal(backend="asyncio") as portal:
            set_async_portal(portal)
            adapter = StuckAdapter()
            register_adapter(adapter)

            start = time.monotonic()
            # Must not raise even though aclose is wedged far beyond the bound.
            aclose_portal_adapters(portal, timeout=0.2)
            elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"Sweep took {elapsed:.2f}s; the 0.2s bound was not honored"
        timeout_warnings = [rec for rec in caplog.records if "Pre-teardown aclose timed out" in rec.getMessage()]
        assert timeout_warnings, "Expected a timeout warning for the wedged aclose"
        # Tracking for this portal was cleared despite the timeout.
        assert snapshot_adapters_for_portal(portal) == []

    def test_dispatch_failure_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        from karenina.adapters.registry import (
            _adapter_portal_lock,
            _adapter_portal_refs,
            register_adapter,
            snapshot_adapters_for_portal,
        )

        with _adapter_portal_lock:
            _adapter_portal_refs.clear()

        class SomeAdapter:
            async def aclose(self) -> None:  # pragma: no cover - never scheduled
                return None

        fake_portal = MagicMock()
        fake_portal._event_loop_thread_id = 12345
        fake_portal.start_task_soon.side_effect = RuntimeError("portal is shutting down")

        set_async_portal(fake_portal)
        register_adapter(SomeAdapter())

        caplog.set_level(logging.WARNING, logger="karenina.benchmark.verification.async_lifecycle")
        # Must not raise even though dispatch itself fails.
        aclose_portal_adapters(fake_portal, timeout=0.1)

        failure_warnings = [rec for rec in caplog.records if "Pre-teardown aclose failed" in rec.getMessage()]
        assert failure_warnings, "Expected a failure warning when dispatch raises"
        assert snapshot_adapters_for_portal(fake_portal) == []

    def test_adapter_without_aclose_is_skipped(self) -> None:
        from karenina.adapters.registry import (
            _adapter_portal_lock,
            _adapter_portal_refs,
            register_adapter,
        )

        with _adapter_portal_lock:
            _adapter_portal_refs.clear()

        class NoAcloseAdapter:
            pass

        fake_portal = MagicMock()
        fake_portal._event_loop_thread_id = 12345

        set_async_portal(fake_portal)
        register_adapter(NoAcloseAdapter())

        aclose_portal_adapters(fake_portal, timeout=0.1)
        fake_portal.start_task_soon.assert_not_called()
