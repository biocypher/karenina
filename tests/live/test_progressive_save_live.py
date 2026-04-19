"""Live acceptance tests for the progressive-save / resume overhaul.

These tests hit real vLLM endpoints on codon-gpu-001 and exercise the
full verification pipeline. They are opt-in: set ``KARENINA_LIVE_TESTS=1``
and keep the endpoints reachable at their default ports. A small subset
of the ``paper_examples/QA`` benchmark (3 questions) is used to keep
runtime bounded.

The three scenarios together cover the key correctness gates for the
overhaul:

1. **Fresh run, clean finalize**: single model, sink writes final JSON
   and deletes sidecars.
2. **Triple-level resume skip**: drop a subset of triples from a prior
   result set via ``skip_triples`` (mirrors what ProgressiveFileSink does
   on resume) and assert only the delta runs live.
3. **Multi-model fan-out resume skip**: two answering models
   (qwen3.5-a3b + qwen3.6-a3b) exercise cross-model triple accounting.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from karenina.benchmark import Benchmark
from karenina.benchmark.verification.sinks import ProgressiveFileSink
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig
from karenina.schemas.verification.model_identity import ModelIdentity

_LIVE_OPT_IN = os.getenv("KARENINA_LIVE_TESTS") == "1"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _LIVE_OPT_IN, reason="set KARENINA_LIVE_TESTS=1 to run"),
]


DEFAULT_QA_BENCHMARK = Path(
    os.getenv(
        "KARENINA_LIVE_QA_BENCHMARK",
        "/Users/carli/Projects/karenina-salvage/paper_examples/QA/qa_benchmark.jsonld",
    )
)

# Endpoints on codon-gpu-001. See memory/reference_gpu_server.md.
VLLM_QWEN35 = os.getenv("VLLM_QWEN35_URL", "http://codon-gpu-001:8002")
VLLM_QWEN36 = os.getenv("VLLM_QWEN36_URL", "http://codon-gpu-001:8103")

OPENAI_EXTRA_KWARGS: dict = {
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    "max_retries": 0,
}


def _model(model_name: str, base_url: str) -> ModelConfig:
    return ModelConfig(
        id=model_name,
        model_name=model_name,
        interface="openai_endpoint",
        endpoint_base_url=base_url,
        endpoint_api_key="EMPTY",
        temperature=0.0,
        request_timeout=180.0,
        extra_kwargs=OPENAI_EXTRA_KWARGS,
    )


def _config(answering: list[ModelConfig], parsing: list[ModelConfig] | None = None) -> VerificationConfig:
    return VerificationConfig(
        answering_models=answering,
        parsing_models=parsing or [answering[0]],
        replicate_count=1,
        evaluation_mode="template_only",
        async_enabled=True,
        async_max_workers=3,
    )


@pytest.fixture(scope="module")
def qa_subset() -> Benchmark:
    """Load 3 QA benchmark questions into a fresh in-memory Benchmark."""
    if not DEFAULT_QA_BENCHMARK.exists():
        pytest.skip(f"QA benchmark not found at {DEFAULT_QA_BENCHMARK}")
    full = Benchmark.load(DEFAULT_QA_BENCHMARK)
    all_templates = full.get_finished_templates()
    subset_ids = {t.question_id for t in all_templates[:3]}

    subset = Benchmark.create(name="live-progsave-subset", description="live test subset", version="1.0.0")
    for template in all_templates:
        if template.question_id not in subset_ids:
            continue
        q_data = full._questions_cache[template.question_id]
        subset.add_question(
            question=q_data["question"],
            raw_answer=q_data.get("raw_answer", ""),
            answer_template=q_data["answer_template"],
            question_id=template.question_id,
            finished=True,
        )
    return subset


# ---------------------------------------------------------------------------
# Scenario 1: fresh run, clean finalize
# ---------------------------------------------------------------------------


def test_live_fresh_run_clean_finalize(qa_subset: Benchmark, tmp_path: Path) -> None:
    output = tmp_path / "live_fresh.json"
    config = _config([_model("qwen3.5-a3b", VLLM_QWEN35)])
    sink = ProgressiveFileSink(
        output_path=output,
        config=config,
        benchmark_path=str(DEFAULT_QA_BENCHMARK),
    )

    result_set = qa_subset.run_verification(config=config, run_name="live-fresh", sink=sink)

    assert len(result_set) == 3
    assert output.exists()
    assert not sink.state_path.exists()
    assert not sink.jsonl_path.exists()


# ---------------------------------------------------------------------------
# Scenario 2: triple-level resume skip (single model)
# ---------------------------------------------------------------------------


def test_live_resume_skips_completed_triples(qa_subset: Benchmark, tmp_path: Path) -> None:
    """Run all 3 triples once, then simulate resume and assert nothing runs.

    Proves the happy-path integration: completed_triples flow from the
    sink into ``skip_triples``, ``generate_task_queue`` drops them, and
    the executor is handed an empty queue.
    """
    output = tmp_path / "live_resume.json"
    config = _config([_model("qwen3.5-a3b", VLLM_QWEN35)])

    sink = ProgressiveFileSink(output_path=output, config=config, benchmark_path=str(DEFAULT_QA_BENCHMARK))
    first_set = qa_subset.run_verification(config=config, run_name="live-resume-1", sink=sink)
    assert not sink.state_path.exists(), "fresh run should finalize"
    assert len(first_set) == 3

    triples: set = {
        (
            r.metadata.question_id,
            r.metadata.answering.canonical_key,
            r.metadata.parsing.canonical_key,
            r.metadata.replicate,
        )
        for r in first_set
    }
    assert len(triples) == 3

    skip_config = config.model_copy(update={"skip_triples": frozenset(triples)})
    second_set = qa_subset.run_verification(config=skip_config, run_name="live-resume-2")

    # Every triple was already done; the second run must produce zero new rows.
    assert len(second_set) == 0


# ---------------------------------------------------------------------------
# Scenario 3: multi-model fan-out resume skip
# ---------------------------------------------------------------------------


def test_live_multi_model_fanout_resume(qa_subset: Benchmark, tmp_path: Path) -> None:
    """3 questions × 2 answering models = 6 triples. Complete 3 triples on
    qwen3.5 first, then add qwen3.6 as a second answerer and assert only
    the 3 qwen3.6 triples run live (triple-level skip across models).
    """
    qwen35 = _model("qwen3.5-a3b", VLLM_QWEN35)
    qwen36 = _model("qwen3.6-a3b", VLLM_QWEN36)

    first_config = _config([qwen35])
    first_set = qa_subset.run_verification(config=first_config, run_name="live-fanout-1")
    assert len(first_set) == 3

    qwen35_key = ModelIdentity.from_model_config(qwen35, role="answering").canonical_key
    parsing_key = ModelIdentity.from_model_config(qwen35, role="parsing").canonical_key
    skip_triples: set = {(r.metadata.question_id, qwen35_key, parsing_key, r.metadata.replicate) for r in first_set}

    second_config = _config([qwen35, qwen36]).model_copy(update={"skip_triples": frozenset(skip_triples)})
    second_set = qa_subset.run_verification(config=second_config, run_name="live-fanout-2")

    # Only qwen3.6 triples should have run live.
    assert len(second_set) == 3
    for r in second_set:
        assert r.metadata.answering.model_name == "qwen3.6-a3b"
