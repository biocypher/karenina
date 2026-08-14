"""Tests for post-hoc rubric evaluation over stored results."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from karenina.benchmark.verification.async_lifecycle import get_async_portal
from karenina.benchmark.verification.post_hoc import (
    RowContext,
    evaluate_rubric_on_results,
    evaluate_rubric_on_texts,
)
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import RegexRubricTrait, Rubric
from karenina.schemas.results import ResultRowKey
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity


def _result(
    question_id: str = "q1",
    parsing_model_name: str = "judge",
    raw_llm_response: str = "",
) -> VerificationResult:
    answering = ModelIdentity(interface="openai_endpoint", model_name="answerer")
    parsing = ModelIdentity(interface="openai_endpoint", model_name=parsing_model_name)
    timestamp = datetime.now(UTC).isoformat()
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="template_hash",
            question_text="What is 2+2?",
            answering=answering,
            parsing=parsing,
            execution_time=0.5,
            timestamp=timestamp,
            replicate=1,
            result_id=VerificationResultMetadata.compute_result_id(
                question_id=question_id,
                answering=answering,
                parsing=parsing,
                timestamp=timestamp,
                replicate=1,
            ),
            run_name="test_run",
        ),
        template=VerificationResultTemplate(raw_llm_response=raw_llm_response),
    )


def _mock_verification_result() -> VerificationResult:
    answering = ModelIdentity(interface="taskeval", model_name="user-provided")
    parsing = ModelIdentity(interface="langchain", model_name="judge-model")
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q1",
            template_id="template_hash",
            question_text="What is X?",
            answering=answering,
            parsing=parsing,
            execution_time=0.1,
            timestamp="2026-01-01T00:00:00+00:00",
            result_id="abcd1234abcd1234",
        )
    )


def _parsing_model() -> ModelConfig:
    return ModelConfig(
        id="judge",
        model_provider="openai",
        model_name="judge-model",
        interface="openai_endpoint",
        endpoint_base_url="http://localhost:9999",
        endpoint_api_key="test-key",
    )


def _regex_rubric() -> Rubric:
    return Rubric(
        regex_traits=[
            RegexRubricTrait(name="MentionsTool", description="d", pattern=r"Tool Message"),
            RegexRubricTrait(name="IsEmpty", description="d", pattern=r"\A\s*\Z"),
        ]
    )


@pytest.mark.unit
class TestEvaluateRubricOnResults:
    def test_regex_traits_score_each_row(self) -> None:
        rows = [
            _result(question_id="q1", raw_llm_response="--- Tool Message (call_id: x) ---\nok"),
            _result(question_id="q2", raw_llm_response="plain answer"),
        ]
        judgments = list(evaluate_rubric_on_results(rows, _regex_rubric(), _parsing_model(), max_workers=1))
        by_question = {judgment.key.question_id: judgment for judgment in judgments}
        assert by_question["q1"].scores["MentionsTool"] is True
        assert by_question["q2"].scores["MentionsTool"] is False
        assert all(judgment.error is None for judgment in judgments)

    def test_empty_trace_still_yields_a_judgment(self) -> None:
        judgments = list(
            evaluate_rubric_on_results(
                [_result(raw_llm_response="")],
                _regex_rubric(),
                _parsing_model(),
                max_workers=1,
            )
        )
        assert len(judgments) == 1
        assert judgments[0].scores["IsEmpty"] is True

    def test_collapse_parser_siblings_dedupes_and_records_siblings(self) -> None:
        rows = [
            _result(parsing_model_name="judge-a", raw_llm_response="same"),
            _result(parsing_model_name="judge-b", raw_llm_response="same"),
        ]
        judgments = list(evaluate_rubric_on_results(rows, _regex_rubric(), _parsing_model(), max_workers=1))
        assert len(judgments) == 1
        assert len(judgments[0].sibling_keys) == 2
        assert judgments[0].representative_result_id == rows[0].metadata.result_id
        assert judgments[0].sibling_result_ids == tuple(row.metadata.result_id for row in rows)

    def test_custom_sibling_identity_preserves_distinct_traces(self) -> None:
        rows = [
            _result(raw_llm_response="first"),
            _result(raw_llm_response="second"),
        ]
        judgments = list(
            evaluate_rubric_on_results(
                rows,
                _regex_rubric(),
                _parsing_model(),
                sibling_identity=lambda result: result.template.raw_llm_response,
                max_workers=1,
            )
        )
        assert len(judgments) == 2

    def test_rubric_factory_builds_a_row_specific_rubric(self) -> None:
        rows = [
            _result(question_id="q1", raw_llm_response="q1"),
            _result(question_id="q2", raw_llm_response="q2"),
        ]

        def factory(result: VerificationResult, _rubric: Rubric) -> Rubric:
            return Rubric(
                regex_traits=[
                    RegexRubricTrait(
                        name="MatchesQuestion",
                        description="d",
                        pattern=result.metadata.question_id,
                    )
                ]
            )

        judgments = list(
            evaluate_rubric_on_results(
                rows,
                _regex_rubric(),
                _parsing_model(),
                rubric_factory=factory,
                max_workers=1,
            )
        )
        assert all(judgment.scores["MatchesQuestion"] is True for judgment in judgments)

    def test_disabling_collapse_keeps_parser_siblings(self) -> None:
        rows = [
            _result(parsing_model_name="judge-a", raw_llm_response="same"),
            _result(parsing_model_name="judge-b", raw_llm_response="same"),
        ]
        judgments = list(
            evaluate_rubric_on_results(
                rows,
                _regex_rubric(),
                _parsing_model(),
                collapse_parser_siblings=False,
                max_workers=1,
            )
        )
        assert len(judgments) == 2

    def test_row_filter_skips_rows(self) -> None:
        rows = [
            _result(question_id="keep", raw_llm_response="x"),
            _result(question_id="drop", raw_llm_response="x"),
        ]
        judgments = list(
            evaluate_rubric_on_results(
                rows,
                _regex_rubric(),
                _parsing_model(),
                row_filter=lambda result: result.metadata.question_id == "keep",
                max_workers=1,
            )
        )
        assert [judgment.key.question_id for judgment in judgments] == ["keep"]

    def test_texts_core_judges_prebuilt_items(self) -> None:
        key = ResultRowKey(
            question_id="q1",
            answering_key="a",
            parsing_key="p",
            replicate=1,
            run_name=None,
        )
        judgments = list(
            evaluate_rubric_on_texts(
                [(key, "--- Tool Message (call_id: x) ---\nok", RowContext())],
                _regex_rubric(),
                _parsing_model(),
                max_workers=1,
            )
        )
        assert judgments[0].key == key
        assert judgments[0].scores["MentionsTool"] is True

    def test_row_context_forwards_ground_truth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_run(**kwargs: Any) -> VerificationResult:
            captured.update(kwargs)
            return _mock_verification_result()

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            fake_run,
        )
        list(
            evaluate_rubric_on_results(
                [_result(raw_llm_response="text")],
                _regex_rubric(),
                _parsing_model(),
                row_context=lambda _result: RowContext(
                    question="What is X?",
                    ground_truth="the reference",
                ),
                max_workers=1,
            )
        )
        assert captured["question_text"] == "What is X?"
        assert captured["raw_answer"] == "the reference"

    def test_evaluation_error_is_returned_on_judgment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail_run(**_kwargs: Any) -> VerificationResult:
            raise RuntimeError("judge unavailable")

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            fail_run,
        )
        judgment = next(
            evaluate_rubric_on_results(
                [_result(raw_llm_response="text")],
                _regex_rubric(),
                _parsing_model(),
                max_workers=1,
            )
        )
        assert judgment.scores == {}
        assert judgment.error == "judge unavailable"

    def test_trait_failure_without_failed_question_is_returned_as_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rubric_result = SimpleNamespace(
            rubric=SimpleNamespace(
                get_all_trait_scores=lambda: {"BrokenJudge": None},
                get_llm_trait_labels=lambda: {},
            )
        )
        outcome = SimpleNamespace(
            global_eval=SimpleNamespace(
                failed_questions={},
                verification_results={"q1": [rubric_result]},
            )
        )
        monkeypatch.setattr(
            "karenina.benchmark.verification.post_hoc.TaskEval.evaluate",
            lambda *_args, **_kwargs: outcome,
        )

        judgment = next(
            evaluate_rubric_on_results(
                [_result(raw_llm_response="text")],
                _regex_rubric(),
                _parsing_model(),
                max_workers=1,
            )
        )

        assert judgment.scores == {}
        assert judgment.error == "TaskEval produced no usable rubric scores"

    def test_parallel_rows_share_one_async_portal(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Post-hoc workers keep model clients on one live event loop."""
        portals: list[object] = []
        rubric_result = SimpleNamespace(
            rubric=SimpleNamespace(
                get_all_trait_scores=lambda: {"MentionsTool": True},
                get_llm_trait_labels=lambda: {},
            )
        )
        outcome = SimpleNamespace(
            global_eval=SimpleNamespace(
                failed_questions={},
                verification_results={"q1": [rubric_result]},
            )
        )

        def evaluate(*_args: Any, **_kwargs: Any) -> Any:
            portal = get_async_portal()
            assert portal is not None
            portals.append(portal)
            return outcome

        monkeypatch.setattr(
            "karenina.benchmark.verification.post_hoc.TaskEval.evaluate",
            evaluate,
        )
        judgments = list(
            evaluate_rubric_on_results(
                [
                    _result(question_id="q1", raw_llm_response="first"),
                    _result(question_id="q2", raw_llm_response="second"),
                ],
                _regex_rubric(),
                _parsing_model(),
                collapse_parser_siblings=False,
                max_workers=2,
            )
        )

        assert len(judgments) == 2
        assert len(portals) == 2
        assert portals[0] is portals[1]
