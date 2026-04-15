"""Tests for ``VerificationContext`` stage attribution fields.

Covers the ``error_stage`` / ``last_run_stage`` tracking introduced for the
failure-state harmonization work: ``mark_error`` records the originating
stage, and ``begin_stage`` tracks the most recently started stage so that
callers may omit the ``stage`` kwarg and still get a meaningful attribution.
"""

from __future__ import annotations

import pytest

from karenina.utils.errors import ErrorCategory
from tests.unit.benchmark.verification.stages.core._context_factory import make_context


@pytest.mark.unit
class TestStageContextErrorStage:
    def test_mark_error_records_stage(self) -> None:
        ctx = make_context()
        ctx.mark_error("boom", category=ErrorCategory.TIMEOUT, stage="generate_answer")
        assert ctx.error_stage == "generate_answer"
        assert ctx.error_category is ErrorCategory.TIMEOUT
        assert ctx.error == "boom"

    def test_mark_error_falls_back_to_last_run_stage(self) -> None:
        ctx = make_context()
        ctx.begin_stage("parse_template")
        ctx.mark_error("boom", category=ErrorCategory.TIMEOUT)
        assert ctx.error_stage == "parse_template"

    def test_last_run_stage_updates_as_stages_run(self) -> None:
        ctx = make_context()
        ctx.begin_stage("validate_template")
        assert ctx.last_run_stage == "validate_template"
        ctx.begin_stage("generate_answer")
        assert ctx.last_run_stage == "generate_answer"
