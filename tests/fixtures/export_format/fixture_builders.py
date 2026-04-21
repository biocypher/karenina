"""Deterministic builders for streaming JSON exporter tests.

These helpers are imported by the golden tests, the structural tests,
and the regen script. Fields that would otherwise be time- or
machine-dependent are pinned here so the resulting export is byte-
reproducible.
"""

from __future__ import annotations

from datetime import UTC, datetime

from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import Rubric
from karenina.schemas.entities.rubric import LLMRubricTrait
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    VerificationConfig,
    VerificationJob,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity

# Fixed constants so headers are deterministic across machines.
FIXED_EXPORT_TIMESTAMP = "2026-04-21 12:00:00 UTC"
FIXED_KARENINA_VERSION = "0.0.0-test"
FIXED_JOB_ID = "job-fixture-001"
FIXED_RUN_NAME = "fixture-run"
FIXED_START_TIME = 1_714_000_000.0
FIXED_END_TIME = 1_714_000_123.5
FIXED_RESULT_TIMESTAMP = datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC).isoformat()

# Non-ASCII payload (Latin-1 Supplement + CJK) to lock ensure_ascii=False behavior.
NON_ASCII_QUESTION_TEXT = "Qu'est-ce que la température idéale du café? 咖啡的理想温度是多少?"


def _make_model(name: str, interface: str = "langchain") -> ModelConfig:
    return ModelConfig(
        id=f"model-{name}",
        model_name=name,
        model_provider="openai",
        interface=interface,
        temperature=0.7,
    )


def build_empty_job() -> VerificationJob:
    """Minimal job with zero results, used by the empty golden fixture."""
    model = _make_model("gpt-4o")
    return VerificationJob(
        job_id=FIXED_JOB_ID,
        run_name=FIXED_RUN_NAME,
        status="completed",
        config=VerificationConfig(answering_models=[model], parsing_models=[model]),
        total_questions=0,
        successful_count=0,
        failed_count=0,
        start_time=FIXED_START_TIME,
        end_time=FIXED_END_TIME,
    )


def build_full_job() -> VerificationJob:
    """Job that will accompany the full-feature result set."""
    model = _make_model("gpt-4o")
    return VerificationJob(
        job_id=FIXED_JOB_ID,
        run_name=FIXED_RUN_NAME,
        status="completed",
        config=VerificationConfig(answering_models=[model], parsing_models=[model]),
        total_questions=2,
        successful_count=1,
        failed_count=1,
        start_time=FIXED_START_TIME,
        end_time=FIXED_END_TIME,
    )


def build_full_rubric() -> Rubric:
    """Rubric used only to populate shared_data.rubric_definition."""
    return Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="clarity",
                description="Response is clearly written and easy to read.",
                kind="boolean",
            ),
        ],
    )


def make_result(
    question_id: str,
    question_text: str,
    failure: Failure | None,
    replicate: int | None = None,
) -> VerificationResult:
    """Build a single deterministic VerificationResult for fixture assembly."""
    answering = ModelIdentity(interface="langchain", model_name="gpt-4o")
    parsing = ModelIdentity(interface="langchain", model_name="gpt-4o")
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=question_id,
        answering=answering,
        parsing=parsing,
        timestamp=FIXED_RESULT_TIMESTAMP,
        replicate=replicate,
    )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="no_template",
            failure=failure,
            question_text=question_text,
            answering=answering,
            parsing=parsing,
            replicate=replicate,
            execution_time=0.25,
            timestamp=FIXED_RESULT_TIMESTAMP,
            result_id=result_id,
        ),
        template=VerificationResultTemplate(
            raw_llm_response=f"raw response for {question_id}",
            template_verification_performed=True,
            verify_result=True,
        ),
    )


def build_full_results() -> VerificationResultSet:
    """Two-result set: one successful (with non-ASCII text), one failed."""
    success = make_result(
        question_id="q-unicode",
        question_text=NON_ASCII_QUESTION_TEXT,
        failure=None,
    )
    fail = make_result(
        question_id="q-timeout",
        question_text="Second question",
        failure=Failure(
            category=FailureCategory.TIMEOUT,
            stage="generate_answer",
            reason="timeout exceeded",
        ),
    )
    return VerificationResultSet(results=[success, fail])


def build_empty_results() -> VerificationResultSet:
    """Zero-result set for the empty golden fixture."""
    return VerificationResultSet(results=[])
