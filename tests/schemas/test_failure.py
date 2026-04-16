"""Tests for the unified Failure taxonomy schema."""

import pytest
from pydantic import ValidationError

from karenina.schemas.results.failure import (
    CATEGORY_TO_GROUP,
    Failure,
    FailureCategory,
    FailureGroup,
)


@pytest.mark.unit
class TestFailure:
    def test_group_derived_from_category(self):
        f = Failure(category=FailureCategory.TIMEOUT, stage="generate_answer", reason="budget exhausted")
        assert f.group is FailureGroup.RETRY_EXHAUSTED

    def test_all_categories_have_group(self):
        for cat in FailureCategory:
            assert cat in CATEGORY_TO_GROUP

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            Failure(category=FailureCategory.CONTENT, stage="verify_template", reason="nope", bogus=True)

    def test_reason_length_enforced(self):
        with pytest.raises(ValidationError):
            Failure(category=FailureCategory.CONTENT, stage="verify_template", reason="x" * 501)

    def test_details_optional_dict(self):
        f = Failure(
            category=FailureCategory.UNEXPECTED_ERROR,
            stage="parse_template",
            reason="boom",
            details={"error_message": "full stacktrace", "exc_class": "ValueError"},
        )
        assert f.details["exc_class"] == "ValueError"

    def test_group_ignores_init_input(self):
        f = Failure.model_validate(
            {
                "category": "content",
                "stage": "verify_template",
                "reason": "r",
                "group": "autofail",
            }
        )
        assert f.group is FailureGroup.CONTENT
