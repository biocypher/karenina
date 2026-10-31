"""Re-judging must preserve answers while producing an independent verdict."""

from unittest.mock import Mock

import pytest

from karenina.benchmark import Benchmark
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.parser import ParsePortResult, ParserPort
from karenina.schemas.config import ModelConfig
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.results.failure import FailureCategory
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultTemplate,
)

TEMPLATE = """
from pydantic import Field
from karenina.schemas.entities import BaseAnswer

class Answer(BaseAnswer):
    value: int = Field(description="The numeric answer")

    def model_post_init(self, __context):
        self.correct = {"value": 42}

    def verify(self) -> bool:
        return self.value == self.correct["value"]
"""


def _model(name: str) -> ModelConfig:
    return ModelConfig(
        id=name,
        model_name=name,
        interface="openai_endpoint",
        endpoint_base_url="http://example.invalid:8000",
        endpoint_api_key="EMPTY",
    )


def _prior_row(
    question_id: str,
    answering: ModelConfig,
    judge: ModelConfig,
    verdict: bool,
    trace: str,
) -> VerificationResult:
    answering_identity = ModelIdentity.from_model_config(answering, role="answering")
    parsing_identity = ModelIdentity.from_model_config(judge, role="parsing")
    timestamp = "2026-10-30T00:00:00Z"
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="prior-template",
            question_text="What is six times seven?",
            answering=answering_identity,
            parsing=parsing_identity,
            execution_time=1.0,
            timestamp=timestamp,
            result_id=VerificationResultMetadata.compute_result_id(
                question_id=question_id,
                answering=answering_identity,
                parsing=parsing_identity,
                timestamp=timestamp,
            ),
            run_name="rejudging",
            evaluation_mode="template_only",
        ),
        template=VerificationResultTemplate(
            raw_llm_response=trace,
            parsed_llm_response={"value": 42 if verdict else 0},
            verify_result=verdict,
        ),
    )


@pytest.mark.integration
@pytest.mark.pipeline
@pytest.mark.parametrize("prior_verdict", [True, False], ids=["prior-pass", "prior-fail"])
@pytest.mark.parametrize(
    ("outcome", "expected_verdict", "expected_failure"),
    [
        ("pass", True, None),
        ("content_failure", False, FailureCategory.CONTENT),
        ("abstention", False, FailureCategory.ABSTENTION),
        ("insufficient", False, FailureCategory.SUFFICIENCY),
        ("parse_failure", None, FailureCategory.PARSING),
        ("blank_answer", None, FailureCategory.PARSING),
        ("invalid_agent_trace", False, FailureCategory.TRACE_VALIDATION),
    ],
)
def test_new_judge_verdict_is_independent_of_prior_judge(
    monkeypatch: pytest.MonkeyPatch,
    prior_verdict: bool,
    outcome: str,
    expected_verdict: bool | None,
    expected_failure: FailureCategory | None,
) -> None:
    """Exercise the public extension API, replay, guards, parser, and finalizer."""
    answering, old_judge, new_judge = (_model(name) for name in ("answerer", "old-judge", "new-judge"))
    trace = "The answer is 42."
    if outcome == "blank_answer":
        trace = ""
    elif outcome == "invalid_agent_trace":
        answering = answering.model_copy(update={"mcp_urls_dict": {"search": "http://example.invalid/mcp"}})
        trace = "--- AI Message ---\n\nTool Calls:\n  search(...)\n\n--- Tool Message ---\n\n42"

    bench = Benchmark.create(name="independent-judge", version="0.0.1")
    question_id = bench.add_question(
        question="What is six times seven?",
        raw_answer="42",
        answer_template=TEMPLATE,
        finished=True,
    )
    prior = VerificationResultSet(results=[_prior_row(question_id, answering, old_judge, prior_verdict, trace)])
    prior_snapshot = prior.model_dump(mode="json")

    # Replace only the model boundaries; extension and all pipeline stages run.
    answering_llm = Mock(side_effect=AssertionError("re-judging must not generate a new answer"))
    answering_agent = Mock(side_effect=AssertionError("re-judging must not run an answering agent"))
    generate_path = "karenina.benchmark.verification.stages.pipeline.generate_answer"
    monkeypatch.setattr(f"{generate_path}.get_llm", answering_llm)
    monkeypatch.setattr(f"{generate_path}.get_agent", answering_agent)

    parser = Mock(spec=ParserPort)
    parser.capabilities = PortCapabilities()
    if outcome == "parse_failure":
        parser.parse_to_pydantic.side_effect = ValueError("new judge could not extract the answer")
    else:
        parser.parse_to_pydantic.side_effect = lambda _messages, schema: ParsePortResult(
            parsed=schema(value=0 if outcome == "content_failure" else 42)
        )
    parser_factory = Mock(return_value=parser)
    evaluator_path = "karenina.benchmark.verification.evaluators.template.evaluator"
    monkeypatch.setattr(f"{evaluator_path}.get_parser", parser_factory)
    monkeypatch.setattr(f"{evaluator_path}.get_llm", Mock())

    stage_path = "karenina.benchmark.verification.stages.pipeline"
    monkeypatch.setattr(
        f"{stage_path}.abstention_check.detect_abstention",
        Mock(return_value=(outcome == "abstention", True, "new judge abstention assessment", {})),
    )
    monkeypatch.setattr(
        f"{stage_path}.sufficiency_check.detect_sufficiency",
        Mock(return_value=(outcome != "insufficient", True, "new judge sufficiency assessment", {})),
    )

    merged = bench.extend_template(
        prior,
        VerificationConfig(
            answering_models=[answering],
            parsing_models=[old_judge, new_judge],
            evaluation_mode="template_only",
            abstention_enabled=True,
            sufficiency_enabled=True,
        ),
        async_enabled=False,
        store=False,
    )

    assert not merged.errors
    assert len(merged.results) == 2
    old_key = ModelIdentity.from_model_config(old_judge, role="parsing").canonical_key
    new_key = ModelIdentity.from_model_config(new_judge, role="parsing").canonical_key
    rows = {row.metadata.parsing.canonical_key: row for row in merged.results}
    assert set(rows) == {old_key, new_key}
    assert rows[old_key].model_dump(mode="json") == prior_snapshot["results"][0]
    assert prior.model_dump(mode="json") == prior_snapshot

    result = rows[new_key]
    assert result.template is not None
    assert result.template.raw_llm_response == trace
    assert result.template.verify_result is expected_verdict
    if expected_failure is None:
        assert result.metadata.failure is None
    else:
        assert result.metadata.failure is not None
        assert result.metadata.failure.category is expected_failure

    answering_llm.assert_not_called()
    answering_agent.assert_not_called()
    if outcome in {"pass", "content_failure", "parse_failure"}:
        parser_factory.assert_called_once()
        actual_judge = parser_factory.call_args.args[0]
        assert ModelIdentity.from_model_config(actual_judge, role="parsing").canonical_key == new_key
        parser.parse_to_pydantic.assert_called_once()
    else:
        parser.parse_to_pydantic.assert_not_called()
