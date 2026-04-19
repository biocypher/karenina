"""Unit tests for Benchmark.extend_judgment and extend_verification_run."""

from __future__ import annotations

from typing import Any

import pytest

from karenina.benchmark import Benchmark
from karenina.benchmark.verification.extension import extend_verification_run
from karenina.schemas.config import ModelConfig
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
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
JUDGE_B = ModelConfig(
    id="judge-b",
    model_name="judge-b",
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
    verify_result: bool = True,
    raw_response: str = "answer-text",
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
        verify_result=verify_result,
    )
    return VerificationResult(metadata=metadata, template=template)


def _make_prior_set(
    *,
    run_name: str,
    question_ids: list[str],
    replicates: list[int | None] | None = None,
) -> VerificationResultSet:
    replicates = replicates if replicates is not None else [None]
    rows = [
        _make_result(
            question_id=qid,
            run_name=run_name,
            replicate=rep,
            raw_response=f"raw-{qid}-{rep}",
        )
        for qid in question_ids
        for rep in replicates
    ]
    return VerificationResultSet(results=rows)


def _make_bench() -> Benchmark:
    return Benchmark.create(name="unit-test-extend", version="0.0.1")


class _FakeVerifyRecorder:
    """Monkeypatch target for Benchmark.run_verification.

    Records the call and returns a synthetic VerificationResultSet with one
    row per (question_id, replicate) pair derived from the observed config.
    """

    def __init__(self, *, judge: ModelConfig = JUDGE_B) -> None:
        self.calls: list[dict[str, Any]] = []
        self.judge = judge

    def __call__(
        self,
        config: VerificationConfig,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        async_enabled: bool | None = None,
        progress_callback: Any | None = None,  # noqa: ARG002 (kept to mirror real signature)
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

        skip = config.skip_triples or frozenset()
        rows: list[VerificationResult] = []
        for qid in effective_qids:
            for ans_model in config.answering_models:
                ans_key = ModelIdentity.from_model_config(ans_model, role="answering").canonical_key
                for parse_model in config.parsing_models:
                    parse_key = ModelIdentity.from_model_config(parse_model, role="parsing").canonical_key
                    for rep in replicate_slots:
                        if (qid, ans_key, parse_key, rep) in skip:
                            continue
                        rows.append(
                            _make_result(
                                question_id=qid,
                                run_name=run_name or "unnamed",
                                answering=ans_model,
                                parsing=parse_model,
                                replicate=rep,
                                raw_response=f"raw-{qid}-{ans_model.id}-{rep}",
                                verify_result=True,
                            )
                        )
        return VerificationResultSet(results=rows)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _FakeVerifyRecorder:
    recorder = _FakeVerifyRecorder(judge=JUDGE_B)
    monkeypatch.setattr(Benchmark, "run_verification", recorder)
    return recorder


@pytest.mark.unit
class TestExtendJudgmentHappyPath:
    def test_single_replicate_merges_and_stamps(self, recorder: _FakeVerifyRecorder) -> None:
        bench = _make_bench()
        prior = _make_prior_set(run_name="first", question_ids=["q1", "q2", "q3"])

        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL],
            parsing_models=[JUDGE_B],
            evaluation_mode="template_only",
        )

        merged = bench.extend_judgment(prior, config, store=False)

        assert len(merged.results) == 6, "3 prior + 3 new = 6"
        run_names = {r.metadata.run_name for r in merged.results}
        assert run_names == {"first"}

        parsing_keys = {r.metadata.parsing.canonical_key for r in merged.results}
        assert len(parsing_keys) == 2, "two distinct parsing identities (j1 and j2)"

        call = recorder.calls[-1]
        assert call["config"].replay_store is not None, "replay store must be attached"
        assert call["run_name"] == "first"
        assert sorted(call["question_ids"]) == ["q1", "q2", "q3"]

    def test_multiple_replicates(self, recorder: _FakeVerifyRecorder) -> None:  # noqa: ARG002
        bench = _make_bench()
        prior = _make_prior_set(
            run_name="rep-run",
            question_ids=["q1", "q2"],
            replicates=[1, 2, 3],
        )

        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL],
            parsing_models=[JUDGE_B],
            evaluation_mode="template_only",
            replicate_count=3,
        )

        merged = bench.extend_judgment(prior, config, store=False)

        assert len(merged.results) == 2 * 2 * 3, "2 questions x 2 judges x 3 replicates"
        pairs: dict[tuple[str, int | None], int] = {}
        for r in merged.results:
            key = (r.metadata.question_id, r.metadata.replicate)
            pairs[key] = pairs.get(key, 0) + 1
        for key, count in pairs.items():
            assert count == 2, f"expected 2 parsers per (qid, replicate) got {count} for {key}"


@pytest.mark.unit
class TestExtendJudgmentMultipleAnsweringModels:
    def test_two_answerers_one_prior_judge_extended_with_one_new_judge(self, recorder: _FakeVerifyRecorder) -> None:
        bench = _make_bench()
        # Prior run: 2 questions x 2 answerers x 1 judge => 4 rows.
        prior_rows = [
            _make_result(
                question_id=qid,
                run_name="multi-ans",
                answering=ans,
                parsing=JUDGE_A,
                raw_response=f"raw-{qid}-{ans.id}",
            )
            for qid in ("q1", "q2")
            for ans in (ANSWERING_MODEL, OTHER_ANSWERING_MODEL)
        ]
        prior = VerificationResultSet(results=prior_rows)

        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL, OTHER_ANSWERING_MODEL],
            parsing_models=[JUDGE_B],
            evaluation_mode="template_only",
        )

        merged = bench.extend_judgment(prior, config, store=False)

        assert len(merged.results) == 2 * 2 * 2, "2 questions x 2 answerers x 2 judges"

        # Every (question, answering) pair should have exactly 2 parsers.
        pairs: dict[tuple[str, str], set[str]] = {}
        for r in merged.results:
            key = (r.metadata.question_id, r.metadata.answering.canonical_key)
            pairs.setdefault(key, set()).add(r.metadata.parsing.canonical_key)
        assert len(pairs) == 4
        for key, parsers in pairs.items():
            assert len(parsers) == 2, f"expected both judges for {key}, got {parsers}"

        # Replay store was attached to the extension call.
        call = recorder.calls[-1]
        assert call["config"].replay_store is not None
        # The captured store must carry one entry per (question, answering) pair.
        assert len(call["config"].replay_store.entries) == 4


@pytest.mark.unit
class TestExtendJudgmentAddAnsweringModel:
    def test_new_answerer_rows_added_prior_rows_kept(self, recorder: _FakeVerifyRecorder) -> None:
        bench = _make_bench()
        prior = _make_prior_set(run_name="add-ans", question_ids=["q1", "q2"])

        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL, OTHER_ANSWERING_MODEL],
            parsing_models=[JUDGE_A],
            evaluation_mode="template_only",
        )

        merged = bench.extend_judgment(prior, config, store=False)

        assert len(merged.results) == 4, "2 prior (A1 x JA) + 2 new (A2 x JA) = 4"

        # Every (question, answering) pair exists exactly once.
        pairs: set[tuple[str, str]] = {
            (r.metadata.question_id, r.metadata.answering.canonical_key) for r in merged.results
        }
        assert len(pairs) == 4
        assert {p[0] for p in pairs} == {"q1", "q2"}

        # Recorder only saw the new answerer (skip filter removed A1xJA tasks).
        saw_ans_keys = {
            ModelIdentity.from_model_config(m, role="answering").canonical_key
            for m in recorder.calls[-1]["config"].answering_models
        }
        assert len(saw_ans_keys) == 2
        skip = recorder.calls[-1]["config"].skip_triples
        assert skip is not None and len(skip) == 2


@pytest.mark.unit
class TestExtendJudgmentAddReplicate:
    def test_added_replicate_runs_live_prior_replicates_kept(self, recorder: _FakeVerifyRecorder) -> None:  # noqa: ARG002
        bench = _make_bench()
        prior = _make_prior_set(
            run_name="add-rep",
            question_ids=["q1", "q2"],
            replicates=[1, 2, 3],
        )

        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL],
            parsing_models=[JUDGE_A],
            evaluation_mode="template_only",
            replicate_count=4,
        )

        merged = bench.extend_judgment(prior, config, store=False)

        assert len(merged.results) == 8, "6 prior (reps 1..3) + 2 new (rep 4) = 8"

        rep_counts: dict[int | None, int] = {}
        for r in merged.results:
            rep_counts[r.metadata.replicate] = rep_counts.get(r.metadata.replicate, 0) + 1
        assert rep_counts == {1: 2, 2: 2, 3: 2, 4: 2}


@pytest.mark.unit
class TestExtendJudgmentAddAllAxes:
    def test_full_symmetric_matrix(self, recorder: _FakeVerifyRecorder) -> None:
        bench = _make_bench()
        prior = _make_prior_set(
            run_name="all-axes",
            question_ids=["q1", "q2"],
            replicates=[1, 2, 3],
        )

        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL, OTHER_ANSWERING_MODEL],
            parsing_models=[JUDGE_A, JUDGE_B],
            evaluation_mode="template_only",
            replicate_count=4,
        )

        merged = bench.extend_judgment(prior, config, store=False)

        # Full joint matrix: 2 questions x 2 answerers x 2 judges x 4 replicates = 32
        assert len(merged.results) == 32

        triples = {
            (
                r.metadata.question_id,
                r.metadata.answering.canonical_key,
                r.metadata.parsing.canonical_key,
                r.metadata.replicate,
            )
            for r in merged.results
        }
        assert len(triples) == 32, "every (q, ans, parse, rep) triple is unique"

        skip = recorder.calls[-1]["config"].skip_triples
        assert skip is not None and len(skip) == 6
        # The 6 skipped triples are exactly the prior rows.
        assert triples.issuperset(skip)


@pytest.mark.unit
class TestExtendJudgmentValidation:
    def test_empty_prior_raises(self) -> None:
        bench = _make_bench()
        config = VerificationConfig(answering_models=[ANSWERING_MODEL], parsing_models=[JUDGE_B])
        with pytest.raises(ValueError, match="at least one VerificationResult"):
            extend_verification_run(bench, VerificationResultSet(results=[]), config)

    def test_answering_mismatch_raises(self) -> None:
        bench = _make_bench()
        prior = _make_prior_set(run_name="r", question_ids=["q1"])
        config = VerificationConfig(answering_models=[OTHER_ANSWERING_MODEL], parsing_models=[JUDGE_B])
        with pytest.raises(ValueError, match="answering_models does not cover"):
            extend_verification_run(bench, prior, config)

    def test_replicate_count_reduction_raises(self) -> None:
        bench = _make_bench()
        prior = _make_prior_set(run_name="r", question_ids=["q1", "q2"], replicates=[1, 2])
        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL],
            parsing_models=[JUDGE_B],
            replicate_count=1,
        )
        with pytest.raises(ValueError, match="Replicate reduction is not supported"):
            extend_verification_run(bench, prior, config)

    def test_replay_store_already_set_raises(self) -> None:
        from karenina.replay import ReplayStore

        bench = _make_bench()
        prior = _make_prior_set(run_name="r", question_ids=["q1"])
        config = VerificationConfig(
            answering_models=[ANSWERING_MODEL],
            parsing_models=[JUDGE_B],
            replay_store=ReplayStore(),
        )
        with pytest.raises(ValueError, match="replay_store must be None"):
            extend_verification_run(bench, prior, config)

    def test_inconsistent_run_names_raises(self) -> None:
        bench = _make_bench()
        prior_a = _make_prior_set(run_name="one", question_ids=["q1"])
        prior_b = _make_prior_set(run_name="two", question_ids=["q2"])
        merged_prior = VerificationResultSet(results=list(prior_a.results) + list(prior_b.results))
        config = VerificationConfig(answering_models=[ANSWERING_MODEL], parsing_models=[JUDGE_B])
        with pytest.raises(ValueError, match="inconsistent run_names"):
            extend_verification_run(bench, merged_prior, config)


@pytest.mark.unit
class TestExtendJudgmentRunNameOverride:
    def test_override_stamps_all_rows(self, recorder: _FakeVerifyRecorder) -> None:  # noqa: ARG002
        bench = _make_bench()
        prior = _make_prior_set(run_name="original", question_ids=["q1"])
        config = VerificationConfig(answering_models=[ANSWERING_MODEL], parsing_models=[JUDGE_B])
        merged = bench.extend_judgment(prior, config, run_name="custom", store=False)

        run_names = {r.metadata.run_name for r in merged.results}
        assert run_names == {"custom"}


@pytest.mark.unit
class TestExtendJudgmentStoreFlag:
    def test_store_true_populates_results_manager(self, recorder: _FakeVerifyRecorder) -> None:  # noqa: ARG002
        bench = _make_bench()
        prior = _make_prior_set(run_name="stored", question_ids=["q1", "q2"])
        config = VerificationConfig(answering_models=[ANSWERING_MODEL], parsing_models=[JUDGE_B])
        bench.extend_judgment(prior, config, store=True)

        fetched = bench.get_verification_results(run_name="stored")
        assert len(fetched) == 4

    def test_store_false_leaves_manager_empty(self, recorder: _FakeVerifyRecorder) -> None:  # noqa: ARG002
        bench = _make_bench()
        prior = _make_prior_set(run_name="not-stored", question_ids=["q1"])
        config = VerificationConfig(answering_models=[ANSWERING_MODEL], parsing_models=[JUDGE_B])
        bench.extend_judgment(prior, config, store=False)

        assert "not-stored" not in bench._results_manager._in_memory_results
