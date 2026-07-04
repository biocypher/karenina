"""Fixture builders that stamp out VerificationResult instances.

These are hand-constructed results used by materializer, renderer, and
indexer unit tests. The materializer is a pure renderer; capturing its
inputs from a live pipeline run adds no fidelity and cannot reach
failure categories like RATE_LIMIT or TRACE_VALIDATION.
"""

from __future__ import annotations

from typing import Any

from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result import VerificationResult
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)


def _model_identity(display: str = "anthropic/claude-opus-4-6") -> ModelIdentity:
    interface, name = display.split("/", 1)
    return ModelIdentity(interface=interface, model_name=name)


def make_metadata(
    *,
    question_id: str = "q_test",
    template_id: str = "template_test",
    replicate: int | None = None,
    failure: Failure | None = None,
    scenario_id: str | None = None,
    scenario_turn: int | None = None,
    scenario_path: list[str] | None = None,
    scenario_node: str | None = None,
    answering_display: str = "anthropic/claude-opus-4-6",
    parsing_display: str = "anthropic/claude-sonnet-4-6",
) -> VerificationResultMetadata:
    answering = _model_identity(answering_display)
    parsing = _model_identity(parsing_display)
    timestamp = "2026-04-16T12:00:00Z"
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=question_id,
        answering=answering,
        parsing=parsing,
        timestamp=timestamp,
        replicate=replicate,
    )
    return VerificationResultMetadata(
        question_id=question_id,
        template_id=template_id,
        failure=failure,
        question_text="What is 2+2?",
        answering=answering,
        parsing=parsing,
        execution_time=0.1,
        timestamp=timestamp,
        result_id=result_id,
        replicate=replicate,
        scenario_id=scenario_id,
        scenario_turn=scenario_turn,
        scenario_path=scenario_path,
        scenario_node=scenario_node,
    )


def make_template(
    *,
    raw_response: str = "4",
    trace_messages: list[dict[str, Any]] | None = None,
    verify_result: bool | None = True,
    field_results: dict[str, bool] | None = None,
) -> VerificationResultTemplate:
    return VerificationResultTemplate(
        raw_llm_response=raw_response,
        trace_messages=trace_messages or [],
        verify_result=verify_result,
        field_results=field_results,
        template_verification_performed=True,
    )


def make_pass(*, question_id: str = "q_pass", replicate: int | None = None) -> VerificationResult:
    return VerificationResult(
        metadata=make_metadata(question_id=question_id, replicate=replicate, failure=None),
        template=make_template(verify_result=True),
        rubric=VerificationResultRubric(),
    )


def make_failure(
    *,
    question_id: str = "q_fail",
    category: FailureCategory = FailureCategory.CONTENT,
    stage: str = "verify_template",
    reason: str = "mismatch",
    replicate: int | None = None,
    trace_messages: list[dict[str, Any]] | None = None,
) -> VerificationResult:
    # Failure.group is a computed field derived from category via
    # CATEGORY_TO_GROUP; never pass group= to the Failure constructor.
    return VerificationResult(
        metadata=make_metadata(
            question_id=question_id,
            replicate=replicate,
            failure=Failure(category=category, stage=stage, reason=reason),
        ),
        template=make_template(
            verify_result=False if category == FailureCategory.CONTENT else None,
            trace_messages=trace_messages or [],
        ),
        rubric=VerificationResultRubric(),
    )
