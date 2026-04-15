"""Unit tests for the central failure classifier."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.failure_classifier import classify_failure
from karenina.schemas.results.failure import FailureCategory, FailureGroup
from karenina.utils.errors import ErrorCategory
from tests.unit.benchmark.verification.stages.core._context_factory import make_context


@pytest.mark.unit
class TestClassifyFailure:
    def test_pass_when_no_error(self) -> None:
        ctx = make_context(verify_result=True, template_verification_performed=True)
        assert classify_failure(ctx) is None

    def test_autofail_recursion_wins_over_retry_exhaustion(self) -> None:
        ctx = make_context(
            failed_stage="RecursionLimitAutoFail",
            error_category=ErrorCategory.TIMEOUT,
            retry_counts={"timeout": {"used": 3, "budget": 3}},
        )
        f = classify_failure(ctx)
        assert f is not None
        assert f.category is FailureCategory.RECURSION_LIMIT
        assert f.group is FailureGroup.AUTOFAIL

    def test_content_fail_wins_over_retry_exhaustion(self) -> None:
        ctx = make_context(
            template_verification_performed=True,
            verify_result=False,
            error_category=ErrorCategory.TIMEOUT,
            retry_counts={"timeout": {"used": 3, "budget": 3}},
            response_timeout_partial=True,
        )
        f = classify_failure(ctx)
        assert f is not None
        assert f.category is FailureCategory.CONTENT

    def test_retry_exhausted_timeout(self) -> None:
        ctx = make_context(
            template_verification_performed=False,
            error_category=ErrorCategory.TIMEOUT,
            retry_counts={"timeout": {"used": 3, "budget": 3}},
            error_stage="generate_answer",
            error="request timed out after 120s",
        )
        f = classify_failure(ctx)
        assert f is not None
        assert f.category is FailureCategory.TIMEOUT
        assert f.group is FailureGroup.RETRY_EXHAUSTED
        assert f.stage == "generate_answer"
        assert f.details is not None
        assert f.details["error_message"] == "request timed out after 120s"

    def test_retry_consumed_but_not_exhausted_does_not_classify_retry(self) -> None:
        ctx = make_context(
            template_verification_performed=True,
            verify_result=True,
            retry_counts={"timeout": {"used": 1, "budget": 3}},
        )
        assert classify_failure(ctx) is None

    def test_abstention_beats_sufficiency(self) -> None:
        ctx = make_context(abstention_detected=True, sufficiency_detected=False)
        f = classify_failure(ctx)
        assert f is not None
        assert f.category is FailureCategory.ABSTENTION

    def test_sufficiency(self) -> None:
        ctx = make_context(abstention_detected=False, sufficiency_detected=True)
        f = classify_failure(ctx)
        assert f is not None
        assert f.category is FailureCategory.SUFFICIENCY

    def test_template_validation_error(self) -> None:
        ctx = make_context(template_validation_error="bad field", error_stage="validate_template")
        f = classify_failure(ctx)
        assert f is not None
        assert f.category is FailureCategory.TEMPLATE_VALIDATION
        assert f.stage == "validate_template"

    def test_unexpected_error_catchall(self) -> None:
        ctx = make_context(
            error_category=None,
            error="surprise",
            error_stage="generate_answer",
        )
        f = classify_failure(ctx)
        assert f is not None
        assert f.category is FailureCategory.UNEXPECTED_ERROR
        assert f.details is not None
        assert f.details["error_message"] == "surprise"

    def test_retry_counts_tolerates_none_values(self) -> None:
        ctx = make_context(
            error_category=ErrorCategory.TIMEOUT,
            retry_counts={"timeout": {"used": None, "budget": 3}},
        )
        # Must not raise. Either None-used => not exhausted => falls through to catchall (no error set => None).
        assert classify_failure(ctx) is None
