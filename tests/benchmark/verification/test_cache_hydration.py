"""Tests for AnswerTraceCache hydration on sink-resume.

Covers the bug where ``run_verification`` resumed via
``ProgressiveFileSink.load_for_resume()`` would not pre-populate the
workspace cache, causing new parser variants for already-completed
``(qid, answerer, replicate)`` triples to regenerate the answerer at
non-zero temperature.

The fix lives in
:meth:`karenina.benchmark.verification.executor.VerificationExecutor._hydrate_cache_from_results`
and is wired through :func:`run_verification_batch` in
``batch_runner.py``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from karenina.benchmark.verification.executor import VerificationExecutor
from karenina.benchmark.verification.sinks import (
    CompositeSink,
    InMemorySink,
    ProgressiveFileSink,
)
from karenina.benchmark.verification.utils.cache_helpers import (
    generate_answer_cache_key,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    VerificationConfig,
    VerificationResult,
    VerificationResultMetadata,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultTemplate,
)
from karenina.utils.answer_cache import AnswerTraceCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model(model_name: str = "qwen3.5-a3b") -> ModelConfig:
    return ModelConfig(
        id=model_name,
        model_name=model_name,
        interface="openai_endpoint",
        endpoint_base_url="http://example.invalid",
        endpoint_api_key="EMPTY",
        temperature=0.0,
    )


def _config(answering: ModelConfig | None = None, parsing: ModelConfig | None = None) -> VerificationConfig:
    return VerificationConfig(
        answering_models=[answering or _model("qwen-answerer")],
        parsing_models=[parsing or _model("qwen-parser-A")],
    )


def _make_result(
    *,
    question_id: str = "q1",
    answering_name: str = "qwen-answerer",
    parsing_name: str = "qwen-parser-A",
    replicate: int | None = None,
    raw_llm_response: str = "--- AI Message ---\nThe answer is 4.",
    failure: Failure | None = None,
    template_present: bool = True,
) -> VerificationResult:
    answering = ModelIdentity(interface="openai_endpoint", model_name=answering_name)
    parsing = ModelIdentity(interface="openai_endpoint", model_name=parsing_name)
    timestamp = datetime.utcnow().isoformat()
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=question_id,
        answering=answering,
        parsing=parsing,
        timestamp=timestamp,
        replicate=replicate,
    )
    template: VerificationResultTemplate | None
    if template_present:
        template = VerificationResultTemplate(
            raw_llm_response=raw_llm_response,
            trace_messages=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": raw_llm_response},
            ],
        )
    else:
        template = None
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="0" * 32,
            question_text="What is 2+2?",
            answering=answering,
            parsing=parsing,
            execution_time=0.5,
            timestamp=timestamp,
            replicate=replicate,
            result_id=result_id,
            run_name="test_run",
            failure=failure,
        ),
        template=template,
    )


def _expected_cache_key(result: VerificationResult) -> str:
    """Build the cache key the executor should produce for ``result``."""
    return generate_answer_cache_key(
        {
            "question_id": result.metadata.question_id,
            "answering_model": SimpleNamespace(id=result.metadata.answering.canonical_key),
            "replicate": result.metadata.replicate,
        }
    )


def _persist_and_resume(tmp_path: Path, results: list[VerificationResult]) -> ProgressiveFileSink:
    """Create a sink, persist ``results``, then return a resumed sink."""
    output_path = tmp_path / "results.json"
    sink = ProgressiveFileSink(
        output_path=output_path,
        config=_config(),
        benchmark_path=str(tmp_path / "benchmark.json"),
    )
    manifest: list[str] = []
    sink.on_start(manifest, _config())
    for row in results:
        sink.on_result(row)
    # Simulate a partial completion so the sidecars are kept on disk.
    sink.on_finalize(all_complete=False)

    return ProgressiveFileSink.load_for_resume(sink.state_path)


# ---------------------------------------------------------------------------
# Hydration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCacheHydration:
    def test_hydration_populates_cache_from_sidecar_results(self, tmp_path: Path) -> None:
        """Four rows, same (qid, answerer, replicate), four parsers.

        After hydration the single answerer cache key should resolve to a
        HIT carrying the prior raw_llm_response.
        """
        rows = [
            _make_result(
                question_id="q1",
                parsing_name=f"parser-{letter}",
                raw_llm_response="--- AI Message ---\nshared answer",
            )
            for letter in ("A", "B", "C", "D")
        ]
        sink = _persist_and_resume(tmp_path, rows)
        cache = AnswerTraceCache()

        VerificationExecutor._hydrate_cache_from_results(cache, sink.iter_results())

        cache_key = _expected_cache_key(rows[0])
        status, data = cache.get_or_reserve(cache_key)
        assert status == "HIT"
        assert data is not None
        assert data["raw_llm_response"] == "--- AI Message ---\nshared answer"
        # Three of the four rows hit the deduplication branch on hydration.
        stats = cache.get_stats()
        assert stats["hits"] >= 1

    def test_hydration_skips_infra_failures(self, tmp_path: Path) -> None:
        """Rows with infra-class failure metadata or empty traces are skipped.

        Healthy rows still hydrate. The skipped rows' answerer keys remain
        misses on lookup.
        """
        healthy = _make_result(
            question_id="qh",
            answering_name="ans-healthy",
            raw_llm_response="--- AI Message ---\ngood",
        )
        connection_failure = _make_result(
            question_id="qc",
            answering_name="ans-connection",
            failure=Failure(
                category=FailureCategory.CONNECTION,
                stage="GenerateAnswer",
                reason="Connection reset by peer",
            ),
        )
        rate_limit_failure = _make_result(
            question_id="qr",
            answering_name="ans-rate",
            failure=Failure(
                category=FailureCategory.RATE_LIMIT,
                stage="GenerateAnswer",
                reason="429 Too Many Requests",
            ),
        )
        timeout_failure = _make_result(
            question_id="qt",
            answering_name="ans-timeout",
            failure=Failure(
                category=FailureCategory.TIMEOUT,
                stage="GenerateAnswer",
                reason="Request timed out",
            ),
        )
        server_error_failure = _make_result(
            question_id="qs",
            answering_name="ans-server",
            failure=Failure(
                category=FailureCategory.SERVER_ERROR,
                stage="GenerateAnswer",
                reason="500 Internal Server Error",
            ),
        )
        unexpected_failure = _make_result(
            question_id="qu",
            answering_name="ans-unexpected",
            failure=Failure(
                category=FailureCategory.UNEXPECTED_ERROR,
                stage="GenerateAnswer",
                reason="Unexpected adapter crash",
            ),
        )
        trace_validation_failure = _make_result(
            question_id="qv",
            answering_name="ans-tracevalid",
            failure=Failure(
                category=FailureCategory.TRACE_VALIDATION,
                stage="TraceValidationAutoFail",
                reason="Trace did not end with AI message",
            ),
        )
        empty_trace_failure = _make_result(
            question_id="qe",
            answering_name="ans-empty",
            failure=Failure(
                category=FailureCategory.CONTENT,
                stage="ParseTemplate",
                reason="Empty or whitespace-only trace from answerer",
            ),
        )

        rows = [
            healthy,
            connection_failure,
            rate_limit_failure,
            timeout_failure,
            server_error_failure,
            unexpected_failure,
            trace_validation_failure,
            empty_trace_failure,
        ]
        sink = _persist_and_resume(tmp_path, rows)
        cache = AnswerTraceCache()

        VerificationExecutor._hydrate_cache_from_results(cache, sink.iter_results())

        # Healthy row hits.
        healthy_key = _expected_cache_key(healthy)
        status, data = cache.get_or_reserve(healthy_key)
        assert status == "HIT"
        assert data is not None
        assert data["raw_llm_response"] == "--- AI Message ---\ngood"

        # Each failure row's key must be a MISS (i.e., never hydrated).
        # Use a fresh cache so the prior get_or_reserve does not pollute counts.
        for skipped_row in (
            connection_failure,
            rate_limit_failure,
            timeout_failure,
            server_error_failure,
            unexpected_failure,
            trace_validation_failure,
            empty_trace_failure,
        ):
            fresh_cache = AnswerTraceCache()
            VerificationExecutor._hydrate_cache_from_results(fresh_cache, sink.iter_results())
            key = _expected_cache_key(skipped_row)
            status, _data = fresh_cache.get_or_reserve(key)
            assert status == "MISS", f"Expected MISS for {skipped_row.metadata.question_id}, got {status}"

    def test_hydration_skips_template_with_no_raw_response(self, tmp_path: Path) -> None:
        """A row with template=None or empty raw_llm_response is skipped.

        We treat "no real trace" as a hard reason not to populate the
        cache, even when the failure metadata is absent.
        """
        no_template = _make_result(question_id="q_noTemp", template_present=False)
        sink = _persist_and_resume(tmp_path, [no_template])
        cache = AnswerTraceCache()

        VerificationExecutor._hydrate_cache_from_results(cache, sink.iter_results())

        key = _expected_cache_key(no_template)
        status, _data = cache.get_or_reserve(key)
        assert status == "MISS"

    def test_hydration_keys_per_replicate(self, tmp_path: Path) -> None:
        """Replicate=1 and replicate=2 hydrate to independent cache entries.

        A new task running at replicate=2 must hit the rep=2 trace, not
        the rep=1 trace.
        """
        rep1 = _make_result(
            question_id="q_rep",
            replicate=1,
            raw_llm_response="--- AI Message ---\nrep1 text",
        )
        rep2 = _make_result(
            question_id="q_rep",
            replicate=2,
            raw_llm_response="--- AI Message ---\nrep2 text",
        )
        sink = _persist_and_resume(tmp_path, [rep1, rep2])
        cache = AnswerTraceCache()

        VerificationExecutor._hydrate_cache_from_results(cache, sink.iter_results())

        key1 = _expected_cache_key(rep1)
        key2 = _expected_cache_key(rep2)
        assert key1 != key2
        assert "_rep1" in key1
        assert "_rep2" in key2

        status1, data1 = cache.get_or_reserve(key1)
        status2, data2 = cache.get_or_reserve(key2)
        assert status1 == "HIT"
        assert status2 == "HIT"
        assert data1 is not None and data2 is not None
        assert data1["raw_llm_response"] == "--- AI Message ---\nrep1 text"
        assert data2["raw_llm_response"] == "--- AI Message ---\nrep2 text"

    def test_resume_with_new_parser_reuses_hydrated_answer(self, tmp_path: Path) -> None:
        """End-to-end shape: a new parser task hits the hydrated cache.

        The original bug was that the new parser task entered the queue,
        called ``answer_cache.get_or_reserve(key)`` on the empty workspace
        cache, and got MISS, triggering fresh answerer generation. With
        hydration in place, the same call returns HIT carrying the prior
        raw_llm_response.
        """
        prior_row = _make_result(
            question_id="q_extend",
            answering_name="ans-shared",
            parsing_name="parser-A",
            raw_llm_response="--- AI Message ---\nprior answer",
        )
        sink = _persist_and_resume(tmp_path, [prior_row])
        cache = AnswerTraceCache()

        VerificationExecutor._hydrate_cache_from_results(cache, sink.iter_results())

        # Simulate the new parser variant landing on the queue: same
        # (qid, answerer, replicate) but parser=B. The cache key is
        # parser-free by design, so a HIT here proves no fresh answerer
        # generation will fire.
        new_parser_task: dict[str, Any] = {
            "question_id": prior_row.metadata.question_id,
            "answering_model": SimpleNamespace(
                id=prior_row.metadata.answering.canonical_key,
            ),
            "replicate": prior_row.metadata.replicate,
        }
        cache_key = generate_answer_cache_key(new_parser_task)
        status, data = cache.get_or_reserve(cache_key)
        assert status == "HIT"
        assert data is not None
        # Byte-match against the sidecar row's raw_llm_response.
        assert data["raw_llm_response"] == prior_row.template.raw_llm_response

    def test_hydration_disabled_when_enable_cache_false(self, tmp_path: Path) -> None:
        """When the executor's cache is None, hydration is a no-op.

        ``ExecutorConfig.enable_cache=False`` produces ``answer_cache=None``
        in the executor; the helper must accept that path without raising.
        """
        prior_row = _make_result(question_id="q_disabled")
        sink = _persist_and_resume(tmp_path, [prior_row])

        # No cache constructed: pass None and confirm the helper does not
        # touch any state. We assert it does not raise; there is no cache
        # to inspect.
        VerificationExecutor._hydrate_cache_from_results(None, sink.iter_results())

        # Sanity: a fresh disabled-cache scenario also no-ops with an
        # empty results iterable.
        VerificationExecutor._hydrate_cache_from_results(None, None)


# ---------------------------------------------------------------------------
# Sink iter_results coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIterResults:
    def test_progressive_file_sink_iter_results(self, tmp_path: Path) -> None:
        rows = [_make_result(question_id="qa"), _make_result(question_id="qb")]
        sink = _persist_and_resume(tmp_path, rows)
        seen = list(sink.iter_results())
        assert len(seen) == 2
        assert {r.metadata.question_id for r in seen} == {"qa", "qb"}

    def test_in_memory_sink_iter_results(self) -> None:
        sink = InMemorySink()
        sink.on_start([], _config())
        first = _make_result(question_id="qa")
        second = _make_result(question_id="qb")
        sink.on_result(first)
        sink.on_result(second)
        seen = list(sink.iter_results())
        assert [r.metadata.question_id for r in seen] == ["qa", "qb"]

    def test_composite_sink_iter_results_fans_out(self, tmp_path: Path) -> None:
        progressive = _persist_and_resume(tmp_path, [_make_result(question_id="qp")])
        memory = InMemorySink()
        memory.on_start([], _config())
        memory.on_result(_make_result(question_id="qm"))

        composite = CompositeSink([progressive, memory])
        seen_ids = [r.metadata.question_id for r in composite.iter_results()]
        # Order: progressive sink first, then in-memory.
        assert seen_ids == ["qp", "qm"]
