"""Unit tests for Benchmark.extend_rubric and extend_rubric_run."""

from __future__ import annotations

from typing import Any

import pytest

from karenina.benchmark import Benchmark
from karenina.benchmark.verification.extension import extend_rubric_run
from karenina.replay import ReplayStore
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import (
    LLMRubricTrait,
    MetricRubricTrait,
    RegexRubricTrait,
    Rubric,
)
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)

ANSWERING_MODEL = ModelConfig(
    id="primary",
    model_name="primary-model",
    interface="openai_endpoint",
    endpoint_base_url="http://example.invalid:8000",
    endpoint_api_key="EMPTY",
    temperature=0.0,
)
JUDGE_A = ModelConfig(
    id="judge-a",
    model_name="judge-a",
    interface="openai_endpoint",
    endpoint_base_url="http://example.invalid:8000",
    endpoint_api_key="EMPTY",
    temperature=0.0,
)
OTHER_ANSWERING_MODEL = ModelConfig(
    id="other",
    model_name="other-model",
    interface="openai_endpoint",
    endpoint_base_url="http://example.invalid:8000",
    endpoint_api_key="EMPTY",
    temperature=0.0,
)


def _make_result(
    *,
    question_id: str,
    run_name: str,
    answering: ModelConfig = ANSWERING_MODEL,
    parsing: ModelConfig = JUDGE_A,
    replicate: int | None = None,
    raw_response: str = "answer-text",
    rubric: VerificationResultRubric | None = None,
) -> VerificationResult:
    ans_identity = ModelIdentity.from_model_config(answering, role="answering")
    parse_identity = ModelIdentity.from_model_config(parsing, role="parsing")
    timestamp = "2026-04-19T00:00:00Z"
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=question_id,
        answering=ans_identity,
        parsing=parse_identity,
        timestamp=timestamp,
        replicate=replicate,
    )
    metadata = VerificationResultMetadata(
        question_id=question_id,
        template_id="no_template",
        question_text=f"Q for {question_id}",
        answering=ans_identity,
        parsing=parse_identity,
        execution_time=1.0,
        timestamp=timestamp,
        result_id=result_id,
        run_name=run_name,
        replicate=replicate,
        evaluation_mode="template_only",
    )
    template = VerificationResultTemplate(
        raw_llm_response=raw_response,
        parsed_llm_response={"value": 42},
        verify_result=True,
    )
    return VerificationResult(
        metadata=metadata,
        template=template,
        rubric=rubric or VerificationResultRubric(),
    )


def _make_prior_set(
    *,
    run_name: str,
    question_ids: list[str],
    replicates: list[int | None] | None = None,
    rubric_factory: Any | None = None,
) -> VerificationResultSet:
    replicates = replicates if replicates is not None else [None]
    rows: list[VerificationResult] = []
    for qid in question_ids:
        for rep in replicates:
            rows.append(
                _make_result(
                    question_id=qid,
                    run_name=run_name,
                    replicate=rep,
                    raw_response=f"raw-{qid}-{rep}",
                    rubric=rubric_factory(qid, rep) if rubric_factory else None,
                )
            )
    return VerificationResultSet(results=rows)


def _make_bench() -> Benchmark:
    bench = Benchmark.create(name="unit-test-extend-rubric", version="0.0.1")
    # Attach a couple of questions so per-question rubric operations work.
    bench.add_question("Q for q1", raw_answer="a1", question_id="q1")
    bench.add_question("Q for q2", raw_answer="a2", question_id="q2")
    return bench


def _rubric_payload_for(benchmark: Benchmark, question_id: str) -> VerificationResultRubric:
    """Build a stub rubric result matching the benchmark's rubric for the question."""
    merged = benchmark._rubric_manager.get_merged_rubric_for_question(question_id)
    payload = VerificationResultRubric(rubric_evaluation_performed=True, rubric_evaluation_strategy="batch")
    if merged is None:
        return payload
    if merged.llm_traits:
        payload.llm_trait_scores = {t.name: 4 for t in merged.llm_traits}
    if merged.regex_traits:
        payload.regex_trait_scores = {t.name: True for t in merged.regex_traits}
    if merged.callable_traits:
        payload.callable_trait_scores = {t.name: True for t in merged.callable_traits}
    if merged.agentic_traits:
        payload.agentic_trait_scores = {t.name: "fake-agentic" for t in merged.agentic_traits}
    return payload


class _FakeRubricRecorder:
    """Monkeypatch target for Benchmark.run_verification in rubric-only mode.

    Produces one row per (qid, answering, parsing, replicate) triple,
    with rubric fields derived from the benchmark's attached rubric.
    Template fields are left empty (mirrors how rubric_only mode would
    behave on replay).
    """

    def __init__(self, benchmark: Benchmark) -> None:
        self.bench = benchmark
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        config: VerificationConfig,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        async_enabled: bool | None = None,
        progress_callback: Any | None = None,  # noqa: ARG002
    ) -> VerificationResultSet:
        self.calls.append(
            {
                "config": config,
                "question_ids": list(question_ids or []),
                "run_name": run_name,
                "async_enabled": async_enabled,
            }
        )
        effective_qids = list(question_ids or [])
        replicate_slots: list[int | None] = (
            [None] if config.replicate_count <= 1 else list(range(1, config.replicate_count + 1))
        )
        rows: list[VerificationResult] = []
        for qid in effective_qids:
            for ans_model in config.answering_models:
                for parse_model in config.parsing_models:
                    for rep in replicate_slots:
                        rows.append(
                            _make_result(
                                question_id=qid,
                                run_name=run_name or "unnamed",
                                answering=ans_model,
                                parsing=parse_model,
                                replicate=rep,
                                raw_response=f"raw-{qid}-{ans_model.id}-{rep}",
                                rubric=_rubric_payload_for(self.bench, qid),
                            )
                        )
        return VerificationResultSet(results=rows)


@pytest.fixture
def bench_and_recorder(monkeypatch: pytest.MonkeyPatch) -> tuple[Benchmark, _FakeRubricRecorder]:
    bench = _make_bench()
    recorder = _FakeRubricRecorder(bench)
    monkeypatch.setattr(Benchmark, "run_verification", recorder)
    return bench, recorder


@pytest.mark.unit
class TestExtendRubricEnrichment:
    def test_prior_rows_enriched_in_place(self, bench_and_recorder: tuple[Benchmark, _FakeRubricRecorder]) -> None:
        bench, recorder = bench_and_recorder
        prior = _make_prior_set(run_name="first", question_ids=["q1", "q2"])

        bench.set_global_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="clarity", description="clear?", kind="score", min_score=1, max_score=5)
                ],
                regex_traits=[RegexRubricTrait(name="cite", description="cites?", pattern=r"\bref\b")],
            )
        )

        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL],
            parsing_models=[JUDGE_A],
            evaluation_mode="template_only",
        )
        merged = bench.extend_rubric(prior, config, store=False)

        assert len(merged.results) == 2, "row count preserved"

        for r in merged.results:
            assert r.rubric.llm_trait_scores == {"clarity": 4}
            assert r.rubric.regex_trait_scores == {"cite": True}
            # Template fields left untouched.
            assert r.template.verify_result is True
            assert r.template.raw_llm_response.startswith("raw-")

        # The replay store was attached and evaluation_mode rewritten.
        call = recorder.calls[-1]
        assert call["config"].replay_store is not None
        assert call["config"].evaluation_mode == "rubric_only"


@pytest.mark.unit
class TestExtendRubricUnion:
    def test_new_traits_unioned_with_prior(self, bench_and_recorder: tuple[Benchmark, _FakeRubricRecorder]) -> None:
        bench, _ = bench_and_recorder

        def prior_rubric(qid: str, rep: int | None) -> VerificationResultRubric:  # noqa: ARG001
            return VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores={"coherence": 4},
            )

        prior = _make_prior_set(run_name="union", question_ids=["q1"], rubric_factory=prior_rubric)
        bench.set_global_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="clarity", description="clear?", kind="score", min_score=1, max_score=5)
                ]
            )
        )

        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL],
            parsing_models=[JUDGE_A],
        )
        merged = bench.extend_rubric(prior, config, store=False)

        assert len(merged.results) == 1
        scores = merged.results[0].rubric.llm_trait_scores
        assert scores == {"coherence": 4, "clarity": 4}


@pytest.mark.unit
class TestExtendRubricCollision:
    def test_same_trait_name_collision_raises(self, bench_and_recorder: tuple[Benchmark, _FakeRubricRecorder]) -> None:
        bench, _ = bench_and_recorder

        def prior_rubric(qid: str, rep: int | None) -> VerificationResultRubric:  # noqa: ARG001
            return VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores={"clarity": 3},
            )

        prior = _make_prior_set(run_name="collide", question_ids=["q1"], rubric_factory=prior_rubric)
        bench.set_global_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="clarity", description="clear?", kind="score", min_score=1, max_score=5)
                ]
            )
        )

        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL],
            parsing_models=[JUDGE_A],
        )
        with pytest.raises(ValueError, match="collision in llm_trait_scores"):
            bench.extend_rubric(prior, config, store=False)


@pytest.mark.unit
class TestExtendRubricPerQuestion:
    def test_question_specific_trait_applies_only_to_that_question(
        self, bench_and_recorder: tuple[Benchmark, _FakeRubricRecorder]
    ) -> None:
        bench, _ = bench_and_recorder
        prior = _make_prior_set(run_name="per-q", question_ids=["q1", "q2"])

        bench.set_global_rubric(
            Rubric(llm_traits=[LLMRubricTrait(name="trait_g", description="g", kind="score", min_score=1, max_score=5)])
        )
        bench.set_question_rubric(
            "q1",
            Rubric(
                llm_traits=[LLMRubricTrait(name="trait_q1", description="q1", kind="score", min_score=1, max_score=5)]
            ),
        )

        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL],
            parsing_models=[JUDGE_A],
        )
        merged = bench.extend_rubric(prior, config, store=False)

        by_q = {r.metadata.question_id: r for r in merged.results}
        assert set(by_q["q1"].rubric.llm_trait_scores or {}) == {"trait_g", "trait_q1"}
        assert set(by_q["q2"].rubric.llm_trait_scores or {}) == {"trait_g"}


@pytest.mark.unit
class TestExtendRubricValidation:
    def _make_rubric_bench(self) -> Benchmark:
        bench = _make_bench()
        bench.set_global_rubric(
            Rubric(llm_traits=[LLMRubricTrait(name="trait_g", description="g", kind="score", min_score=1, max_score=5)])
        )
        return bench

    def test_empty_prior_raises(self) -> None:
        bench = self._make_rubric_bench()
        config = VerificationConfig(answering_models=[ANSWERING_MODEL], parsing_models=[JUDGE_A])
        with pytest.raises(ValueError, match="at least one VerificationResult"):
            extend_rubric_run(bench, VerificationResultSet(results=[]), config)

    def test_replay_store_already_set_raises(self) -> None:
        bench = self._make_rubric_bench()
        prior = _make_prior_set(run_name="r", question_ids=["q1"])
        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL],
            parsing_models=[JUDGE_A],
            replay_store=ReplayStore(),
        )
        with pytest.raises(ValueError, match="replay_store must be None"):
            extend_rubric_run(bench, prior, config)

    def test_unsupported_evaluation_mode_raises(self) -> None:
        bench = self._make_rubric_bench()
        prior = _make_prior_set(run_name="r", question_ids=["q1"])
        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL],
            parsing_models=[JUDGE_A],
            evaluation_mode="template_and_rubric",
        )
        with pytest.raises(ValueError, match="evaluation_mode must be"):
            extend_rubric_run(bench, prior, config)

    def test_missing_rubric_raises(self) -> None:
        bench = _make_bench()  # no rubric
        prior = _make_prior_set(run_name="r", question_ids=["q1"])
        config = VerificationConfig(answering_models=[ANSWERING_MODEL], parsing_models=[JUDGE_A])
        with pytest.raises(ValueError, match="benchmark has no rubric attached"):
            extend_rubric_run(bench, prior, config)

    def test_metric_trait_rejected(self) -> None:
        bench = _make_bench()
        bench.set_global_rubric(
            Rubric(
                metric_traits=[
                    MetricRubricTrait(
                        name="coverage",
                        description="coverage metric",
                        metrics=["precision"],
                        tp_instructions=["must mention X"],
                    )
                ]
            )
        )
        prior = _make_prior_set(run_name="r", question_ids=["q1"])
        config = VerificationConfig(answering_models=[ANSWERING_MODEL], parsing_models=[JUDGE_A])
        with pytest.raises(ValueError, match="Metric traits are not supported"):
            extend_rubric_run(bench, prior, config)

    def test_parsing_model_mismatch_raises(self) -> None:
        bench = self._make_rubric_bench()
        prior = _make_prior_set(run_name="r", question_ids=["q1"])

        other_judge = ModelConfig(
            id="judge-z",
            model_name="judge-z",
            interface="openai_endpoint",
            endpoint_base_url="http://example.invalid:8000",
            endpoint_api_key="EMPTY",
            temperature=0.0,
        )
        config = VerificationConfig(answering_models=[ANSWERING_MODEL], parsing_models=[other_judge])
        with pytest.raises(ValueError, match="parsing_models does not match"):
            extend_rubric_run(bench, prior, config)

    def test_answering_model_mismatch_raises(self) -> None:
        bench = self._make_rubric_bench()
        prior = _make_prior_set(run_name="r", question_ids=["q1"])
        config = VerificationConfig(answering_models=[OTHER_ANSWERING_MODEL], parsing_models=[JUDGE_A])
        with pytest.raises(ValueError, match="answering_models does not cover"):
            extend_rubric_run(bench, prior, config)

    def test_replicate_count_mismatch_raises(self) -> None:
        bench = self._make_rubric_bench()
        prior = _make_prior_set(run_name="r", question_ids=["q1", "q2"], replicates=[1, 2])
        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL],
            parsing_models=[JUDGE_A],
            replicate_count=3,
        )
        with pytest.raises(ValueError, match="does not match the replicate fan-out"):
            extend_rubric_run(bench, prior, config)


@pytest.mark.unit
class TestExtendRubricStoreFlag:
    def test_store_true_populates_results_manager(
        self, bench_and_recorder: tuple[Benchmark, _FakeRubricRecorder]
    ) -> None:
        bench, _ = bench_and_recorder
        prior = _make_prior_set(run_name="stored", question_ids=["q1", "q2"])
        bench.set_global_rubric(
            Rubric(llm_traits=[LLMRubricTrait(name="ok", description="ok", kind="score", min_score=1, max_score=5)])
        )
        config = VerificationConfig(answering_models=[ANSWERING_MODEL], parsing_models=[JUDGE_A])
        bench.extend_rubric(prior, config, store=True)

        fetched = bench.get_verification_results(run_name="stored")
        assert len(fetched) == 2
