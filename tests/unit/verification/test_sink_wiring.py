"""Unit tests for sink wiring in :func:`run_verification_batch`.

These tests stub :class:`VerificationExecutor` to avoid spinning up any
live LLM call while still exercising the real ``run_verification_batch``
code path: sink-lifecycle events, skip_triples merging on resume, and the
partial-results capture of :class:`VerificationBatchError`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from karenina.benchmark.verification import batch_runner as batch_runner_module
from karenina.benchmark.verification.batch_runner import run_verification_batch
from karenina.benchmark.verification.sinks import (
    InMemorySink,
    ProgressiveFileSink,
)
from karenina.exceptions import VerificationBatchError
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import (
    FinishedTemplate,
    VerificationConfig,
    VerificationResult,
    VerificationResultMetadata,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.utils.progressive_save import TaskIdentifier

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _model(model_name: str = "qwen3.5-a3b") -> ModelConfig:
    return ModelConfig(
        id=model_name,
        model_name=model_name,
        interface="openai_endpoint",
        endpoint_base_url="http://codon-gpu-001:8002",
        endpoint_api_key="EMPTY",
        temperature=0.0,
    )


def _config(answering: list[ModelConfig] | None = None, parsing: list[ModelConfig] | None = None) -> VerificationConfig:
    return VerificationConfig(
        answering_models=answering or [_model("qwen3.5-a3b")],
        parsing_models=parsing or [_model("qwen3.5-a3b")],
    )


def _template(question_id: str, code: str = "class Answer: pass") -> FinishedTemplate:
    return FinishedTemplate(
        question_id=question_id,
        question_text=f"Question {question_id}?",
        question_preview=f"Question {question_id}?",
        template_code=code,
        last_modified=datetime.utcnow().isoformat(),
    )


def _result_for_task(task: dict[str, Any]) -> VerificationResult:
    answering = ModelIdentity.from_model_config(task["answering_model"], role="answering")
    parsing = ModelIdentity.from_model_config(task["parsing_model"], role="parsing")
    timestamp = datetime.utcnow().isoformat()
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=task["question_id"],
        answering=answering,
        parsing=parsing,
        timestamp=timestamp,
        replicate=task.get("replicate"),
    )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=task["question_id"],
            template_id="template_hash",
            question_text=task["question_text"],
            answering=answering,
            parsing=parsing,
            execution_time=0.01,
            timestamp=timestamp,
            replicate=task.get("replicate"),
            result_id=result_id,
            run_name=task.get("run_name") or "test_run",
        ),
    )


class _SuccessExecutor:
    """Returns a canned success result for each task."""

    def __init__(self, parallel: bool = True, config: Any = None) -> None:  # noqa: ARG002
        self.parallel = parallel

    def run_batch(self, tasks: list[dict[str, Any]], progress_callback: Any) -> dict[str, VerificationResult]:
        results: dict[str, VerificationResult] = {}
        for idx, task in enumerate(tasks, 1):
            result = _result_for_task(task)
            key = TaskIdentifier.from_result(result).to_key()
            results[key] = result
            if progress_callback:
                progress_callback(idx, len(tasks), result)
        return results


class _FailingExecutor:
    """Succeeds for the first N tasks, raises with partial results for the rest."""

    def __init__(self, succeed_first: int) -> None:
        self.succeed_first = succeed_first
        self.parallel = True

    def run_batch(self, tasks: list[dict[str, Any]], progress_callback: Any) -> dict[str, VerificationResult]:
        partial: dict[str, VerificationResult] = {}
        errors = []
        for idx, task in enumerate(tasks):
            if idx < self.succeed_first:
                result = _result_for_task(task)
                partial[TaskIdentifier.from_result(result).to_key()] = result
                if progress_callback:
                    progress_callback(idx + 1, len(tasks), result)
            else:
                errors.append((task["question_id"], RuntimeError("boom")))
        raise VerificationBatchError(
            f"{len(errors)} of {len(tasks)} verification tasks failed",
            partial_results=partial,
            errors=errors,
        )


def _install_executor(monkeypatch: pytest.MonkeyPatch, fake_cls: type[Any]) -> None:
    """Replace VerificationExecutor + its ExecutorConfig for this test."""

    class _FakeExecutorConfig:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    import karenina.benchmark.verification.executor as executor_module

    monkeypatch.setattr(executor_module, "VerificationExecutor", fake_cls)
    monkeypatch.setattr(executor_module, "ExecutorConfig", _FakeExecutorConfig)
    # Disable auto-save so we don't touch any DB.
    monkeypatch.setenv("AUTOSAVE_DATABASE", "false")
    # Also disable cleanup_resources (it would try to drain async portals).
    monkeypatch.setattr(batch_runner_module, "cleanup_resources", lambda: None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSinkLifecycle:
    def test_successful_run_invokes_start_result_finalize(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_executor(monkeypatch, _SuccessExecutor)
        sink = InMemorySink()
        templates = [_template("q1"), _template("q2")]

        result_set = run_verification_batch(
            templates=templates,
            config=_config(),
            run_name="r",
            sink=sink,
        )

        assert sink.started is True
        assert len(sink.manifest) == 2
        assert len(sink.results) == 2
        assert sink.finalized_all_complete is True
        assert len(result_set) == 2

    def test_partial_failure_returns_partial_and_finalizes_partial(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_executor(monkeypatch, lambda parallel=True, config=None: _FailingExecutor(succeed_first=1))  # noqa: ARG005
        sink = InMemorySink()
        templates = [_template("q1"), _template("q2")]

        result_set = run_verification_batch(
            templates=templates,
            config=_config(),
            run_name="r",
            sink=sink,
        )

        assert len(result_set) == 1
        assert sink.finalized_all_complete is False

    def test_partial_failure_without_sink_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_executor(monkeypatch, lambda parallel=True, config=None: _FailingExecutor(succeed_first=0))  # noqa: ARG005
        templates = [_template("q1")]

        with pytest.raises(VerificationBatchError):
            run_verification_batch(templates=templates, config=_config(), run_name="r")


@pytest.mark.unit
class TestResumeSkipsCompletedTriples:
    def test_sink_completed_triples_are_merged_into_skip_triples(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_executor(monkeypatch, _SuccessExecutor)

        # First run: complete 1 of 2 templates, bail out via partial failure.
        sink = ProgressiveFileSink(
            output_path=tmp_path / "out.json",
            config=_config(),
            benchmark_path="bench.jsonld",
        )

        # Fake executor: succeed for q1, fail for q2.
        def _mixed_executor(parallel: bool = True, config: Any = None) -> Any:  # noqa: ARG001
            return _MixedExecutor()

        class _MixedExecutor:
            parallel = True

            def run_batch(self, tasks: list[dict[str, Any]], progress_callback: Any) -> dict[str, VerificationResult]:
                partial: dict[str, VerificationResult] = {}
                errors: list[tuple[str, BaseException]] = []
                for idx, task in enumerate(tasks):
                    if task["question_id"] == "q1":
                        result = _result_for_task(task)
                        partial[TaskIdentifier.from_result(result).to_key()] = result
                        if progress_callback:
                            progress_callback(idx + 1, len(tasks), result)
                    else:
                        errors.append((task["question_id"], RuntimeError("boom")))
                raise VerificationBatchError(
                    "partial",
                    partial_results=partial,
                    errors=errors,
                )

        import karenina.benchmark.verification.executor as executor_module

        monkeypatch.setattr(executor_module, "VerificationExecutor", _mixed_executor)

        run_verification_batch(
            templates=[_template("q1"), _template("q2")],
            config=_config(),
            run_name="r",
            sink=sink,
        )
        assert sink.state_path.exists()
        assert sink.completed_count == 1

        # Resume: restart with a fresh sink that has q1 completed. The
        # executor must only see q2 in its task queue.
        resume_sink = ProgressiveFileSink.load_for_resume(sink.state_path)

        seen_question_ids: list[str] = []

        class _AssertExecutor:
            parallel = True

            def run_batch(self, tasks: list[dict[str, Any]], progress_callback: Any) -> dict[str, VerificationResult]:
                seen_question_ids.extend(t["question_id"] for t in tasks)
                results: dict[str, VerificationResult] = {}
                for idx, task in enumerate(tasks, 1):
                    result = _result_for_task(task)
                    results[TaskIdentifier.from_result(result).to_key()] = result
                    if progress_callback:
                        progress_callback(idx, len(tasks), result)
                return results

        monkeypatch.setattr(
            executor_module,
            "VerificationExecutor",
            lambda parallel=True, config=None: _AssertExecutor(),  # noqa: ARG005
        )

        result_set = run_verification_batch(
            templates=[_template("q1"), _template("q2")],
            config=_config(),
            run_name="r",
            sink=resume_sink,
        )

        # Only q2 should have been executed in the resume run.
        assert seen_question_ids == ["q2"]
        # Final set aggregates the re-executed results; the sink keeps the
        # original q1 completion in memory.
        assert resume_sink.completed_count == 2
        assert len(result_set) == 1  # Only what this run's executor returned.
        # .state should be gone now that all tasks are covered.
        assert not sink.state_path.exists()
