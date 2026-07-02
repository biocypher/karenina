"""Multi-threaded hammer tests for sink internal locking (T17).

Each sink is driven from N threads calling ``on_result`` concurrently
WITHOUT any external lock (the executors' progress-lock serialization is
deliberately absent here). Correctness must come from the sinks' own
internal locks: no lost rows, no duplicated rows, a line-parseable JSONL
sidecar, and a consistent ``.state`` manifest.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import pytest

from karenina.benchmark.verification.sinks import (
    AgenticProgressiveFileSink,
    CompositeSink,
    InMemorySink,
    ProgressiveFileSink,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import (
    VerificationConfig,
    VerificationResult,
    VerificationResultMetadata,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import VerificationResultTemplate
from karenina.utils.progressive_save import TaskIdentifier

N_THREADS = 8
ROWS_PER_THREAD = 12
TOTAL_ROWS = N_THREADS * ROWS_PER_THREAD

# ---------------------------------------------------------------------------
# Helpers (mirroring tests/unit/verification/test_sinks.py fixtures)
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


def _config() -> VerificationConfig:
    return VerificationConfig(
        answering_models=[_model()],
        parsing_models=[_model()],
    )


def _result(question_id: str) -> VerificationResult:
    answering = ModelIdentity(interface="openai_endpoint", model_name="qwen3.5-a3b")
    parsing = ModelIdentity(interface="openai_endpoint", model_name="qwen3.5-a3b")
    timestamp = datetime.utcnow().isoformat()
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=question_id,
        answering=answering,
        parsing=parsing,
        timestamp=timestamp,
        replicate=None,
    )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="template_hash",
            question_text="What is 2+2?",
            answering=answering,
            parsing=parsing,
            execution_time=0.5,
            timestamp=timestamp,
            replicate=None,
            result_id=result_id,
            run_name="test_run",
        ),
        template=VerificationResultTemplate(raw_llm_response="The answer is 4."),
    )


def _make_rows() -> list[list[VerificationResult]]:
    """One distinct row batch per thread (all question ids unique)."""
    return [[_result(f"q_t{t}_i{i}") for i in range(ROWS_PER_THREAD)] for t in range(N_THREADS)]


def _hammer(sink, rows_per_thread: list[list[VerificationResult]]) -> None:
    """Drive sink.on_result from N threads with a start barrier, no external lock."""
    barrier = threading.Barrier(N_THREADS)
    failures: list[BaseException] = []
    failures_lock = threading.Lock()

    def _worker(rows: list[VerificationResult]) -> None:
        try:
            barrier.wait(timeout=10.0)
            for row in rows:
                sink.on_result(row)
        except BaseException as e:  # noqa: BLE001
            with failures_lock:
                failures.append(e)

    threads = [threading.Thread(target=_worker, args=(rows,)) for rows in rows_per_thread]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)
    assert not any(thread.is_alive() for thread in threads), "hammer thread wedged"
    assert not failures, f"worker raised: {failures}"


def _manifest_for(rows_per_thread: list[list[VerificationResult]]) -> list[str]:
    return [TaskIdentifier.from_result(row).to_key() for rows in rows_per_thread for row in rows]


# ---------------------------------------------------------------------------
# ProgressiveFileSink
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProgressiveFileSinkThreadSafety:
    def test_concurrent_on_result_no_lost_or_duplicated_rows(self, tmp_path: Path) -> None:
        rows_per_thread = _make_rows()
        sink = ProgressiveFileSink(tmp_path / "results.json", _config(), "bench.jsonld")
        sink.on_start(_manifest_for(rows_per_thread), _config())

        _hammer(sink, rows_per_thread)

        # JSONL parses line by line: exactly one well-formed line per row.
        lines = sink.jsonl_path.read_text(encoding="utf-8").splitlines()
        non_empty = [line for line in lines if line.strip()]
        assert len(non_empty) == TOTAL_ROWS
        seen_qids = [json.loads(line)["metadata"]["question_id"] for line in non_empty]
        assert len(set(seen_qids)) == TOTAL_ROWS

        # In-memory bookkeeping: no lost or duplicated rows.
        assert sink.completed_count == TOTAL_ROWS
        assert len(sink.get_all_results()) == TOTAL_ROWS
        assert len(sink.completed_triples()) == TOTAL_ROWS

        # .state manifest consistent with the JSONL.
        state = json.loads(sink.state_path.read_text(encoding="utf-8"))
        assert state["completed_count"] == TOTAL_ROWS
        assert len(state["completed_task_ids"]) == TOTAL_ROWS
        assert state["total_tasks"] == TOTAL_ROWS
        assert set(state["completed_task_ids"]) == set(state["task_manifest"])

    def test_concurrent_on_result_then_finalize(self, tmp_path: Path) -> None:
        """Finalize after a concurrent run writes the full export."""
        rows_per_thread = _make_rows()
        sink = ProgressiveFileSink(tmp_path / "results.json", _config(), "bench.jsonld")
        sink.on_start(_manifest_for(rows_per_thread), _config())

        _hammer(sink, rows_per_thread)
        sink.on_finalize(all_complete=True)

        export = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert len(export["results"]) == TOTAL_ROWS
        assert not sink.state_path.exists()
        assert not sink.jsonl_path.exists()


@pytest.mark.unit
class TestAgenticProgressiveFileSinkThreadSafety:
    def test_concurrent_on_result_rows_and_trace_sidecars(self, tmp_path: Path) -> None:
        rows_per_thread = _make_rows()
        sink = AgenticProgressiveFileSink(
            tmp_path / "results.json",
            _config(),
            "bench.jsonld",
            trace_output_dir=tmp_path / "traces",
        )
        sink.on_start(_manifest_for(rows_per_thread), _config())

        _hammer(sink, rows_per_thread)

        non_empty = [line for line in sink.jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(non_empty) == TOTAL_ROWS
        assert sink.completed_count == TOTAL_ROWS
        # One trace dir per question, each with the answering trace.
        trace_dirs = list((tmp_path / "traces").iterdir())
        assert len(trace_dirs) == TOTAL_ROWS
        assert all((d / "answering_trace.txt").exists() for d in trace_dirs)


# ---------------------------------------------------------------------------
# InMemorySink
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInMemorySinkThreadSafety:
    def test_concurrent_on_result_no_lost_or_duplicated_rows(self) -> None:
        rows_per_thread = _make_rows()
        sink = InMemorySink()
        sink.on_start(_manifest_for(rows_per_thread), _config())

        _hammer(sink, rows_per_thread)

        assert len(sink.results) == TOTAL_ROWS
        qids = [row.metadata.question_id for row in sink.results]
        assert len(set(qids)) == TOTAL_ROWS
        assert len(sink.completed_triples()) == TOTAL_ROWS


# ---------------------------------------------------------------------------
# CompositeSink
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompositeSinkThreadSafety:
    def test_concurrent_fan_out_reaches_all_children_exactly_once(self, tmp_path: Path) -> None:
        rows_per_thread = _make_rows()
        memory = InMemorySink()
        progressive = ProgressiveFileSink(tmp_path / "results.json", _config(), "bench.jsonld")
        composite = CompositeSink([memory, progressive])
        composite.on_start(_manifest_for(rows_per_thread), _config())

        _hammer(composite, rows_per_thread)

        assert len(memory.results) == TOTAL_ROWS
        assert len({row.metadata.question_id for row in memory.results}) == TOTAL_ROWS
        assert progressive.completed_count == TOTAL_ROWS
        non_empty = [line for line in progressive.jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(non_empty) == TOTAL_ROWS

    def test_child_exception_releases_lock_and_propagates(self) -> None:
        """A raising child must not corrupt the composite: the exception
        propagates (per-child isolation is deferred), the RLock is released,
        and subsequent events still fan out.
        """

        class ExplodingSink(InMemorySink):
            def on_result(self, result: VerificationResult) -> None:
                if result.metadata.question_id == "boom":
                    raise RuntimeError("child sink exploded")
                super().on_result(result)

        healthy = InMemorySink()
        composite = CompositeSink([ExplodingSink(), healthy])
        composite.on_start([], _config())

        with pytest.raises(RuntimeError, match="child sink exploded"):
            composite.on_result(_result("boom"))

        # The composite stays usable: the lock was released and later
        # events reach every child.
        composite.on_result(_result("after"))
        assert [row.metadata.question_id for row in healthy.results] == ["after"]
        composite.on_finalize(all_complete=False)
        assert healthy.finalized_all_complete is False
