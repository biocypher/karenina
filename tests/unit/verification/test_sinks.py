"""Unit tests for ResultSink, ProgressiveFileSink, CompositeSink, InMemorySink."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from karenina.benchmark.verification.sinks import (
    STATE_FORMAT_VERSION,
    AgenticProgressiveFileSink,
    CompositeSink,
    InMemorySink,
    ProgressiveFileSink,
    ResultSink,
    is_completed_result,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    VerificationConfig,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.utils.progressive_save import TaskIdentifier


def _model(model_name: str = "qwen3.5-a3b", tools: list[str] | None = None) -> ModelConfig:
    return ModelConfig(
        id=model_name,
        model_name=model_name,
        interface="openai_endpoint",
        endpoint_base_url="http://codon-gpu-001:8002",
        endpoint_api_key="EMPTY",
        temperature=0.0,
        mcp_urls_dict=dict.fromkeys(tools or [], "http://example/mcp") or None,
    )


def _config(answering: list[ModelConfig] | None = None, parsing: list[ModelConfig] | None = None) -> VerificationConfig:
    return VerificationConfig(
        answering_models=answering or [_model("qwen3.5-a3b")],
        parsing_models=parsing or [_model("qwen3.5-a3b")],
    )


def _result(
    question_id: str,
    answering_name: str = "qwen3.5-a3b",
    parsing_name: str = "qwen3.5-a3b",
    replicate: int | None = None,
    *,
    completed: bool = True,
    failure: Failure | None = None,
    raw_llm_response: str = "",
    investigation_trace: str | None = None,
) -> VerificationResult:
    answering = ModelIdentity(interface="openai_endpoint", model_name=answering_name)
    parsing = ModelIdentity(interface="openai_endpoint", model_name=parsing_name)
    timestamp = datetime.utcnow().isoformat() if completed else ""
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=question_id,
        answering=answering,
        parsing=parsing,
        timestamp=timestamp or "preview",
        replicate=replicate,
    )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="template_hash",
            question_text="What is 2+2?",
            answering=answering,
            parsing=parsing,
            execution_time=0.5 if completed else 0.0,
            timestamp=timestamp,
            replicate=replicate,
            result_id=result_id,
            run_name="test_run",
            failure=failure,
        ),
        template=VerificationResultTemplate(
            raw_llm_response=raw_llm_response,
            investigation_trace=investigation_trace,
        ),
    )


def _manifest_from_results(results: list[VerificationResult]) -> list[str]:
    return [TaskIdentifier.from_result(r).to_key() for r in results]


# ---------------------------------------------------------------------------
# is_completed_result
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsCompletedResult:
    def test_none_is_not_completed(self):
        assert is_completed_result(None) is False

    def test_empty_timestamp_is_preview(self):
        result = _result("q1", completed=False)
        assert is_completed_result(result) is False

    def test_populated_timestamp_is_completed(self):
        result = _result("q1", completed=True)
        assert is_completed_result(result) is True


# ---------------------------------------------------------------------------
# InMemorySink
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInMemorySink:
    def test_satisfies_result_sink_protocol(self):
        sink = InMemorySink()
        assert isinstance(sink, ResultSink)

    def test_on_start_records_manifest(self):
        sink = InMemorySink()
        sink.on_start(["a", "b"], _config())
        assert sink.started is True
        assert sink.manifest == ["a", "b"]

    def test_on_result_filters_previews(self):
        sink = InMemorySink()
        sink.on_result(_result("q1", completed=False))
        sink.on_result(_result("q1", completed=True))
        assert len(sink.results) == 1

    def test_completed_triples_from_results(self):
        sink = InMemorySink()
        r = _result("q1", replicate=2)
        sink.on_result(r)
        triples = sink.completed_triples()
        assert len(triples) == 1
        triple = next(iter(triples))
        assert triple[0] == "q1"
        assert triple[3] == 2

    def test_on_finalize_records_terminal_state(self):
        sink = InMemorySink()
        sink.on_finalize(all_complete=True)
        assert sink.finalized_all_complete is True


# ---------------------------------------------------------------------------
# ProgressiveFileSink
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProgressiveFileSink:
    def _fresh(self, tmp_path: Path) -> ProgressiveFileSink:
        output = tmp_path / "results.json"
        return ProgressiveFileSink(output_path=output, config=_config(), benchmark_path="bench.jsonld")

    def test_satisfies_result_sink_protocol(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        assert isinstance(sink, ResultSink)

    def test_on_start_creates_state_and_jsonl(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        results = [_result("q1"), _result("q2")]
        sink.on_start(_manifest_from_results(results), sink.config)

        assert sink.state_path.exists()
        assert sink.jsonl_path.exists()

        state = json.loads(sink.state_path.read_text())
        assert state["format_version"] == STATE_FORMAT_VERSION
        assert state["total_tasks"] == 2
        assert state["completed_count"] == 0
        assert state["failed_count"] == 0
        assert state["config_hash"]  # non-empty
        # Initial JSONL is empty.
        assert sink.jsonl_path.read_text() == ""

    def test_on_result_appends_to_jsonl_and_updates_state(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        r1 = _result("q1")
        r2 = _result("q2")
        sink.on_start(_manifest_from_results([r1, r2]), sink.config)

        sink.on_result(r1)
        sink.on_result(r2)

        lines = sink.jsonl_path.read_text().splitlines()
        assert len(lines) == 2
        state = json.loads(sink.state_path.read_text())
        assert state["completed_count"] == 2
        assert len(state["completed_task_ids"]) == 2

    def test_on_result_ignores_preview(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        r = _result("q1")
        sink.on_start([TaskIdentifier.from_result(r).to_key()], sink.config)
        sink.on_result(_result("q1", completed=False))

        assert sink.jsonl_path.read_text() == ""
        state = json.loads(sink.state_path.read_text())
        assert state["completed_count"] == 0

    def test_on_result_append_is_unconditional_but_completed_is_idempotent(self, tmp_path: Path):
        """JSONL always appends; ``_completed`` tracks combo-level state.

        Scenario runs emit one ``on_result`` per turn, all sharing the same
        combo-level task key. We must append every turn to the JSONL but
        mark the combo complete only once. QA runs never emit true
        duplicates (one result per task), so the same invariant holds there
        trivially.
        """
        sink = self._fresh(tmp_path)
        r = _result("q1")
        sink.on_start([TaskIdentifier.from_result(r).to_key()], sink.config)
        sink.on_result(r)
        sink.on_result(r)

        assert len(sink.jsonl_path.read_text().splitlines()) == 2
        assert sink.completed_count == 1

    def test_completed_triples_round_trip(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        r = _result("q1", replicate=3)
        sink.on_start([TaskIdentifier.from_result(r).to_key()], sink.config)
        sink.on_result(r)

        triples = sink.completed_triples()
        assert len(triples) == 1
        q_id, ans_key, parse_key, rep = next(iter(triples))
        assert q_id == "q1"
        assert rep == 3
        assert ans_key.startswith("openai_endpoint:")
        assert parse_key.startswith("openai_endpoint:")

    def test_finalize_all_complete_writes_export_and_deletes_sidecars(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        r = _result("q1")
        sink.on_start([TaskIdentifier.from_result(r).to_key()], sink.config)
        sink.on_result(r)

        sink.on_finalize(all_complete=True)

        assert sink.output_path.exists()
        assert not sink.state_path.exists()
        assert not sink.jsonl_path.exists()

    def test_finalize_partial_keeps_sidecars(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        r = _result("q1")
        sink.on_start([TaskIdentifier.from_result(r).to_key()], sink.config)
        sink.on_result(r)

        sink.on_finalize(all_complete=False)

        assert not sink.output_path.exists()
        assert sink.state_path.exists()
        assert sink.jsonl_path.exists()

    def test_load_for_resume_restores_completed_triples(self, tmp_path: Path):
        r1 = _result("q1")
        r2 = _result("q2")
        manifest = _manifest_from_results([r1, r2])

        first = ProgressiveFileSink(
            output_path=tmp_path / "results.json",
            config=_config(),
            benchmark_path="bench.jsonld",
        )
        first.on_start(manifest, first.config)
        first.on_result(r1)
        first.on_finalize(all_complete=False)

        second = ProgressiveFileSink.load_for_resume(first.state_path)
        assert second.total_tasks == 2
        assert second.completed_count == 1
        assert len(second.completed_triples()) == 1
        assert len(second.get_all_results()) == 1

    def test_load_for_resume_rejects_wrong_format_version(self, tmp_path: Path):
        state = tmp_path / "results.json.state"
        state.write_text(json.dumps({"format_version": "0.1", "output_path": str(tmp_path / "results.json")}))
        with pytest.raises(ValueError, match="Incompatible state format"):
            ProgressiveFileSink.load_for_resume(state)

    def test_load_for_resume_missing_state(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ProgressiveFileSink.load_for_resume(tmp_path / "nope.state")

    def test_load_for_resume_tolerates_corrupt_trailing_jsonl_line(self, tmp_path: Path):
        r = _result("q1")
        first = ProgressiveFileSink(
            output_path=tmp_path / "results.json",
            config=_config(),
            benchmark_path="bench.jsonld",
        )
        first.on_start([TaskIdentifier.from_result(r).to_key()], first.config)
        first.on_result(r)
        first.on_finalize(all_complete=False)

        with open(first.jsonl_path, "a") as f:
            f.write("{not-json, partial\n")

        second = ProgressiveFileSink.load_for_resume(first.state_path)
        assert second.completed_count == 1

    def test_mark_failed_recorded_in_state(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        sink.on_start(["k1", "k2"], sink.config)
        sink.mark_failed("k1")
        state = json.loads(sink.state_path.read_text())
        assert state["failed_task_ids"] == ["k1"]

    def test_on_start_twice_keeps_stored_manifest(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        sink.on_start(["k1", "k2"], sink.config)
        sink.on_start(["k2"], sink.config)
        assert sink.total_tasks == 2


@pytest.mark.unit
class TestAgenticProgressiveFileSink:
    def _fresh(self, tmp_path: Path, *, keep_sidecars: bool = False) -> AgenticProgressiveFileSink:
        return AgenticProgressiveFileSink(
            output_path=tmp_path / "results.json",
            config=_config(),
            benchmark_path="bench.jsonld",
            trace_output_dir=tmp_path / "traces",
            keep_progress_sidecars=keep_sidecars,
        )

    def test_writes_readable_trace_sidecars_on_result(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        result = _result(
            "q1",
            raw_llm_response="answer trace",
            investigation_trace="investigation trace",
        )
        sink.on_start([TaskIdentifier.from_result(result).to_key()], sink.config)

        sink.on_result(result)

        assert (tmp_path / "traces" / "q1" / "answering_trace.txt").read_text() == "answer trace"
        assert (tmp_path / "traces" / "q1" / "investigation_trace.txt").read_text() == "investigation trace"

    def test_parser_failure_is_not_a_completed_triple_but_remains_hydratable(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        failure = Failure(
            category=FailureCategory.PARSING,
            stage="AgenticParseTemplate",
            reason="structured extraction failed",
        )
        result = _result("q1", failure=failure, raw_llm_response="usable answer trace")
        sink.on_start([TaskIdentifier.from_result(result).to_key()], sink.config)

        sink.on_result(result)

        assert sink.completed_triples() == set()
        assert list(sink.iter_results())[0].template.raw_llm_response == "usable answer trace"

    def test_success_after_parser_failure_becomes_terminal_and_export_dedupes(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        parser_failure = Failure(
            category=FailureCategory.PARSING,
            stage="AgenticParseTemplate",
            reason="structured extraction failed",
        )
        failed = _result("q1", failure=parser_failure, raw_llm_response="old trace")
        passed = _result("q1", raw_llm_response="new trace")
        manifest_key = TaskIdentifier.from_result(failed).to_key()
        sink.on_start([manifest_key], sink.config)

        sink.on_result(failed)
        sink.on_result(passed)
        sink.on_finalize(all_complete=True)

        assert len(sink.completed_triples()) == 1
        export = json.loads((tmp_path / "results.json").read_text())
        assert len(export["results"]) == 1
        assert export["results"][0]["template"]["raw_llm_response"] == "new trace"
        assert not sink.state_path.exists()
        assert not sink.jsonl_path.exists()

    def test_parser_failure_keeps_resume_sidecars_even_when_batch_completed(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        failure = Failure(
            category=FailureCategory.PARSING,
            stage="parse_template",
            reason="parse failed",
        )
        result = _result("q1", failure=failure, raw_llm_response="answer trace")
        sink.on_start([TaskIdentifier.from_result(result).to_key()], sink.config)
        sink.on_result(result)

        sink.on_finalize(all_complete=True)

        assert sink.output_path.exists()
        assert sink.state_path.exists()
        assert sink.jsonl_path.exists()
        export = json.loads(sink.output_path.read_text())
        assert export["metadata"]["job_summary"]["is_complete"] is False

    def test_batch_partial_export_is_marked_incomplete_without_parser_failures(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        result = _result("q1", raw_llm_response="answer trace")
        sink.on_start([TaskIdentifier.from_result(result).to_key(), "pending"], sink.config)
        sink.on_result(result)

        sink.on_finalize(all_complete=False)

        export = json.loads(sink.output_path.read_text())
        assert export["metadata"]["job_summary"]["is_complete"] is False
        assert sink.state_path.exists()

    def test_keep_sidecars_preserves_state_after_success(self, tmp_path: Path):
        sink = self._fresh(tmp_path, keep_sidecars=True)
        result = _result("q1", raw_llm_response="answer trace")
        sink.on_start([TaskIdentifier.from_result(result).to_key()], sink.config)
        sink.on_result(result)

        sink.on_finalize(all_complete=True)

        assert sink.output_path.exists()
        assert sink.state_path.exists()
        assert sink.jsonl_path.exists()

    def test_final_export_summary_counts_content_failures_as_failed(self, tmp_path: Path):
        sink = self._fresh(tmp_path, keep_sidecars=True)
        passed = _result("q1", raw_llm_response="answer trace")
        failed = _result(
            "q2",
            failure=Failure(
                category=FailureCategory.CONTENT,
                stage="verify_template",
                reason="verify_template returned False",
            ),
            raw_llm_response="answer trace",
        )
        sink.on_start(_manifest_from_results([passed, failed]), sink.config)
        sink.on_result(passed)
        sink.on_result(failed)

        sink.on_finalize(all_complete=True)

        export = json.loads(sink.output_path.read_text())
        summary = export["metadata"]["job_summary"]
        assert summary["successful_count"] == 1
        assert summary["failed_count"] == 1
        assert summary["is_complete"] is True

    def test_final_export_summary_uses_latest_row_when_retry_succeeds(self, tmp_path: Path):
        sink = self._fresh(tmp_path, keep_sidecars=True)
        failure = Failure(
            category=FailureCategory.PARSING,
            stage="AgenticParseTemplate",
            reason="structured extraction failed",
        )
        failed = _result("q1", failure=failure, raw_llm_response="old trace")
        passed = _result("q1", raw_llm_response="new trace")
        sink.on_start([TaskIdentifier.from_result(failed).to_key()], sink.config)
        sink.on_result(failed)
        sink.on_result(passed)

        sink.on_finalize(all_complete=True)

        export = json.loads(sink.output_path.read_text())
        summary = export["metadata"]["job_summary"]
        assert summary["successful_count"] == 1
        assert summary["failed_count"] == 0


# ---------------------------------------------------------------------------
# CompositeSink
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompositeSink:
    def test_satisfies_result_sink_protocol(self):
        composite = CompositeSink([InMemorySink(), InMemorySink()])
        assert isinstance(composite, ResultSink)

    def test_fans_out_on_start(self):
        a, b = InMemorySink(), InMemorySink()
        composite = CompositeSink([a, b])
        composite.on_start(["k1"], _config())
        assert a.started is True
        assert b.started is True

    def test_fans_out_on_result(self):
        a, b = InMemorySink(), InMemorySink()
        composite = CompositeSink([a, b])
        r = _result("q1")
        composite.on_result(r)
        assert len(a.results) == 1
        assert len(b.results) == 1

    def test_fans_out_on_finalize(self):
        a, b = InMemorySink(), InMemorySink()
        composite = CompositeSink([a, b])
        composite.on_finalize(all_complete=True)
        assert a.finalized_all_complete is True
        assert b.finalized_all_complete is True

    def test_completed_triples_union(self):
        a, b = InMemorySink(), InMemorySink()
        a.on_result(_result("q1"))
        b.on_result(_result("q2"))
        composite = CompositeSink([a, b])
        triples = composite.completed_triples()
        assert {t[0] for t in triples} == {"q1", "q2"}

    def test_composite_with_progressive_file_sink(self, tmp_path: Path):
        mem = InMemorySink()
        file_sink = ProgressiveFileSink(
            output_path=tmp_path / "results.json",
            config=_config(),
            benchmark_path="bench.jsonld",
        )
        composite = CompositeSink([mem, file_sink])

        r = _result("q1")
        composite.on_start([TaskIdentifier.from_result(r).to_key()], _config())
        composite.on_result(r)
        composite.on_finalize(all_complete=True)

        assert len(mem.results) == 1
        assert (tmp_path / "results.json").exists()


class TestSeedPriorResults:
    """Coverage for seed_prior_results on each sink implementation."""

    def test_progressive_sink_seed_populates_state(self, tmp_path: Path):
        from karenina.schemas.results import VerificationResultSet

        prior = VerificationResultSet(results=[_result("q1"), _result("q2")])
        sink = ProgressiveFileSink(
            output_path=tmp_path / "results.json",
            config=_config(),
            benchmark_path="bench.jsonld",
        )
        sink.seed_prior_results(prior)

        assert sink.completed_count == 2
        assert len(sink.get_all_results()) == 2
        assert sink.jsonl_path.exists()
        assert sink.state_path.exists()
        state = json.loads(sink.state_path.read_text())
        assert state["completed_count"] == 2
        assert len(state["task_manifest"]) == 2

    def test_progressive_sink_seed_skips_preview_rows(self, tmp_path: Path):
        from karenina.schemas.results import VerificationResultSet

        prior = VerificationResultSet(
            results=[_result("q1"), _result("q2", completed=False)],
        )
        sink = ProgressiveFileSink(
            output_path=tmp_path / "results.json",
            config=_config(),
            benchmark_path="bench.jsonld",
        )
        sink.seed_prior_results(prior)

        assert sink.completed_count == 1

    def test_progressive_sink_seed_is_idempotent(self, tmp_path: Path):
        from karenina.schemas.results import VerificationResultSet

        prior = VerificationResultSet(results=[_result("q1")])
        sink = ProgressiveFileSink(
            output_path=tmp_path / "results.json",
            config=_config(),
            benchmark_path="bench.jsonld",
        )
        sink.seed_prior_results(prior)
        sink.seed_prior_results(prior)

        assert sink.completed_count == 1
        assert len(sink.get_all_results()) == 1
        jsonl_lines = [line for line in sink.jsonl_path.read_text().splitlines() if line.strip()]
        assert len(jsonl_lines) == 1

    def test_progressive_sink_on_start_unions_with_seeded_manifest(self, tmp_path: Path):
        from karenina.schemas.results import VerificationResultSet

        prior_r = _result("q1")
        new_r = _result("q2")
        sink = ProgressiveFileSink(
            output_path=tmp_path / "results.json",
            config=_config(),
            benchmark_path="bench.jsonld",
        )
        sink.seed_prior_results(VerificationResultSet(results=[prior_r]))

        prior_key = TaskIdentifier.from_result(prior_r).to_key()
        new_key = TaskIdentifier.from_result(new_r).to_key()
        sink.on_start([new_key], _config())

        state = json.loads(sink.state_path.read_text())
        assert set(state["task_manifest"]) == {prior_key, new_key}

    def test_progressive_sink_final_export_contains_seeded_rows(self, tmp_path: Path):
        from karenina.schemas.results import VerificationResultSet

        prior = VerificationResultSet(results=[_result("q1"), _result("q2")])
        sink = ProgressiveFileSink(
            output_path=tmp_path / "results.json",
            config=_config(),
            benchmark_path="bench.jsonld",
        )
        sink.seed_prior_results(prior)

        new_r = _result("q3")
        sink.on_start([TaskIdentifier.from_result(new_r).to_key()], _config())
        sink.on_result(new_r)
        sink.on_finalize(all_complete=True)

        export = json.loads((tmp_path / "results.json").read_text())
        question_ids = {entry["metadata"]["question_id"] for entry in export["results"]}
        assert question_ids == {"q1", "q2", "q3"}

    def test_in_memory_sink_seed_populates_results_and_manifest(self):
        from karenina.schemas.results import VerificationResultSet

        sink = InMemorySink()
        prior = VerificationResultSet(results=[_result("q1"), _result("q2")])
        sink.seed_prior_results(prior)

        assert len(sink.results) == 2
        assert len(sink.manifest) == 2

    def test_in_memory_sink_seed_is_idempotent(self):
        from karenina.schemas.results import VerificationResultSet

        sink = InMemorySink()
        prior = VerificationResultSet(results=[_result("q1")])
        sink.seed_prior_results(prior)
        sink.seed_prior_results(prior)

        assert len(sink.results) == 1
        assert len(sink.manifest) == 1

    def test_in_memory_sink_on_start_unions_with_seeded(self):
        from karenina.schemas.results import VerificationResultSet

        sink = InMemorySink()
        prior_r = _result("q1")
        new_r = _result("q2")
        sink.seed_prior_results(VerificationResultSet(results=[prior_r]))

        prior_key = TaskIdentifier.from_result(prior_r).to_key()
        new_key = TaskIdentifier.from_result(new_r).to_key()
        sink.on_start([new_key], _config())

        assert set(sink.manifest) == {prior_key, new_key}

    def test_composite_sink_forwards_seed(self):
        from karenina.schemas.results import VerificationResultSet

        a, b = InMemorySink(), InMemorySink()
        composite = CompositeSink([a, b])
        prior = VerificationResultSet(results=[_result("q1")])
        composite.seed_prior_results(prior)

        assert len(a.results) == 1
        assert len(b.results) == 1

    def test_progressive_sink_satisfies_protocol(self, tmp_path: Path):
        sink = ProgressiveFileSink(
            output_path=tmp_path / "results.json",
            config=_config(),
            benchmark_path="bench.jsonld",
        )
        assert isinstance(sink, ResultSink)
