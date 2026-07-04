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

    def test_factory_routes_result_field_kwargs(self) -> None:
        """``verify_result`` routes through ``set_result_field``.

        Mirrors the production pipeline, where verify stages write to the
        result builder.
        """
        from karenina.benchmark.verification.stages.core.base import ArtifactKeys

        ctx = make_context(verify_result=True)
        assert ctx.get_result_field(ArtifactKeys.VERIFY_RESULT) is True

    def test_factory_routes_retry_counts_through_artifact(self) -> None:
        """``retry_counts`` routes through ``set_artifact``.

        The orchestrator writes ``RETRY_COUNTS`` via ``set_artifact`` inside
        its ``track_retries`` block; the classifier and caveat collector
        read the same dict via ``get_artifact``. The test factory must
        honour that boundary so unit tests exercise the correct path.
        """
        from karenina.benchmark.verification.stages.core.base import ArtifactKeys

        ctx = make_context(retry_counts={"timeout": {"used": 1, "budget": 3}})
        assert ctx.get_artifact(ArtifactKeys.RETRY_COUNTS) == {"timeout": {"used": 1, "budget": 3}}
        assert ctx.get_result_field(ArtifactKeys.RETRY_COUNTS) is None
