"""Integration tests for progressive-save + resume via the Python API.

Mocks :class:`VerificationExecutor` so the pipeline runs without any live
LLM, but exercises the real :meth:`Benchmark.run_verification` and
:meth:`Benchmark.resume_verification` paths end-to-end, including sink
wiring, ``skip_triples`` propagation, and sidecar lifecycle.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from karenina.benchmark import Benchmark
from karenina.benchmark.verification import batch_runner as batch_runner_module
from karenina.benchmark.verification.sinks import (
    CompositeSink,
    InMemorySink,
    ProgressiveFileSink,
)
from karenina.exceptions import VerificationBatchError
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import (
    VerificationConfig,
    VerificationResult,
    VerificationResultMetadata,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.utils.progressive_save import TaskIdentifier

TEMPLATE_CODE = (
    "from karenina.schemas.entities.answer import BaseAnswer\n"
    "class Answer(BaseAnswer):\n"
    "    value: str = 'ok'\n"
    "    def model_post_init(self, __context):\n"
    "        self.correct = {'value': 'ok'}\n"
    "    def verify(self):\n"
    "        return self.value == self.correct['value']\n"
)


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
            run_name=task.get("run_name") or "resume_test",
        ),
    )


def _install_executor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_question_ids: set[str] | None = None,
    observed_ids: list[str] | None = None,
) -> None:
    """Replace VerificationExecutor with a stub that records which task ids it saw."""

    fail_question_ids = fail_question_ids or set()

    class _Stub:
        parallel = True

        def __init__(self, parallel: bool = True, config: Any = None) -> None:  # noqa: ARG002
            pass

        def run_batch(self, tasks: list[dict[str, Any]], progress_callback: Any) -> dict[str, VerificationResult]:
            partial: dict[str, VerificationResult] = {}
            errors: list[tuple[str, BaseException]] = []
            for idx, task in enumerate(tasks):
                if observed_ids is not None:
                    observed_ids.append(task["question_id"])
                if task["question_id"] in fail_question_ids:
                    errors.append((task["question_id"], RuntimeError("mocked failure")))
                    continue
                result = _result_for_task(task)
                partial[TaskIdentifier.from_result(result).to_key()] = result
                if progress_callback:
                    progress_callback(idx + 1, len(tasks), result)
            if errors:
                raise VerificationBatchError(
                    f"{len(errors)} tasks failed",
                    partial_results=partial,
                    errors=errors,
                )
            return partial

    class _FakeExecutorConfig:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    import karenina.benchmark.verification.executor as executor_module

    monkeypatch.setattr(executor_module, "VerificationExecutor", _Stub)
    monkeypatch.setattr(executor_module, "ExecutorConfig", _FakeExecutorConfig)
    monkeypatch.setenv("AUTOSAVE_DATABASE", "false")
    monkeypatch.setattr(batch_runner_module, "cleanup_resources", lambda: None)


def _make_benchmark(tmp_path: Path, question_ids: list[str]) -> Benchmark:
    bench = Benchmark.create(name="test_resume", description="resume test", version="1.0.0")
    for qid in question_ids:
        bench.add_question(
            question=f"Q {qid}?",
            raw_answer="ok",
            answer_template=TEMPLATE_CODE,
            question_id=qid,
            finished=True,
        )
    return bench


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestProgressiveFileSinkE2E:
    def test_fresh_run_finalizes_and_deletes_sidecars(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _install_executor(monkeypatch)
        bench = _make_benchmark(tmp_path, ["q1", "q2", "q3"])
        output = tmp_path / "results.json"

        sink = ProgressiveFileSink(output_path=output, config=_config(), benchmark_path=str(tmp_path / "bench.jsonld"))
        result_set = bench.run_verification(config=_config(), run_name="fresh", sink=sink)

        assert len(result_set) == 3
        assert output.exists()
        assert not sink.state_path.exists()
        assert not sink.jsonl_path.exists()

    def test_partial_failure_keeps_state_and_resume_finishes_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        observed: list[str] = []
        _install_executor(monkeypatch, fail_question_ids={"q3"}, observed_ids=observed)

        bench = _make_benchmark(tmp_path, ["q1", "q2", "q3"])
        output = tmp_path / "results.json"
        sink = ProgressiveFileSink(output_path=output, config=_config(), benchmark_path=str(tmp_path / "bench.jsonld"))
        bench.run_verification(config=_config(), run_name="part", sink=sink)

        # Failure path: sidecars retained, partial set written.
        assert sink.state_path.exists()
        assert sink.jsonl_path.exists()
        assert not output.exists()
        assert observed == ["q1", "q2", "q3"]  # first attempt tried all

        # Second attempt: stub no longer fails q3.
        observed.clear()
        _install_executor(monkeypatch, observed_ids=observed)
        result_set = bench.resume_verification(state_path=sink.state_path, run_name="resume")

        # Only q3 should be re-executed thanks to skip_triples.
        assert observed == ["q3"]
        assert not sink.state_path.exists()
        assert not sink.jsonl_path.exists()
        assert output.exists()
        # Resume pass only returns rows run in this pass (1), but the final
        # export on disk contains all 3 from the sink's buffer.
        assert len(result_set) == 1


@pytest.mark.integration
class TestCompositeSinkE2E:
    def test_file_and_memory_sinks_both_observe_all_results(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_executor(monkeypatch)
        bench = _make_benchmark(tmp_path, ["q1", "q2"])
        output = tmp_path / "combined.json"

        mem = InMemorySink()
        file_sink = ProgressiveFileSink(
            output_path=output, config=_config(), benchmark_path=str(tmp_path / "bench.jsonld")
        )
        composite = CompositeSink([mem, file_sink])

        result_set = bench.run_verification(config=_config(), run_name="combo", sink=composite)

        assert len(result_set) == 2
        assert len(mem.results) == 2
        assert output.exists()
        assert mem.finalized_all_complete is True
