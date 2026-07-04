"""Tests for results_exporter JSON/CSV success field via Failure objects.

The CSV header ``success`` must be derived from ``metadata.failure is None``
without touching the legacy ``completed_without_errors`` attribute.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from io import StringIO
from pathlib import Path

import pytest

from karenina.benchmark.verification.stages.helpers.results_exporter import (
    export_verification_results_csv,
    export_verification_results_json_stream,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.results.verification_result_set import VerificationResultSet
from karenina.schemas.verification import (
    VerificationConfig,
    VerificationJob,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity


def _make_job() -> VerificationJob:
    """Construct a minimal VerificationJob suitable for exporter tests."""
    answering = ModelConfig(
        id="a1", model_provider="openai", model_name="gpt-4o", interface="langchain", temperature=0.1
    )
    parsing = ModelConfig(id="p1", model_provider="openai", model_name="gpt-4o", interface="langchain", temperature=0.1)
    return VerificationJob(
        job_id="job_1",
        run_name="run_1",
        status="completed",
        config=VerificationConfig(answering_models=[answering], parsing_models=[parsing]),
        total_questions=2,
        processed_count=2,
        successful_count=1,
        failed_count=1,
    )


def _make_result(question_id: str, failure: Failure | None) -> VerificationResult:
    """Construct a VerificationResult with only a Failure (no legacy fields)."""
    timestamp = datetime.now().isoformat()
    answering = ModelIdentity(interface="langchain", model_name="gpt-4o")
    parsing = ModelIdentity(interface="langchain", model_name="gpt-4o")
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=question_id,
        answering=answering,
        parsing=parsing,
        timestamp=timestamp,
    )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="no_template",
            failure=failure,
            question_text=f"Question {question_id}",
            answering=answering,
            parsing=parsing,
            execution_time=0.25,
            timestamp=timestamp,
            result_id=result_id,
        ),
        template=VerificationResultTemplate(raw_llm_response=""),
    )


@pytest.mark.unit
class TestResultsExporterSuccessField:
    """The ``success`` CSV column must reflect ``failure is None``."""

    def test_csv_marks_pass_when_failure_none(self) -> None:
        """A result with ``failure=None`` yields ``success=True`` in the CSV row."""
        job = _make_job()
        result_set = VerificationResultSet(results=[_make_result("q1", failure=None)])

        csv_text = export_verification_results_csv(job, result_set)
        reader = csv.DictReader(StringIO(csv_text))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["success"] == "True"
        assert rows[0]["error"] == ""

    def test_csv_marks_fail_when_failure_present(self) -> None:
        """A result carrying a Failure yields ``success=False`` and an error message."""
        job = _make_job()
        failure = Failure(
            category=FailureCategory.TIMEOUT,
            stage="generate_answer",
            reason="timeout exceeded",
        )
        result_set = VerificationResultSet(results=[_make_result("q1", failure=failure)])

        csv_text = export_verification_results_csv(job, result_set)
        reader = csv.DictReader(StringIO(csv_text))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["success"] == "False"
        assert "timeout exceeded" in rows[0]["error"]

    def test_json_includes_failure_field(self, tmp_path: Path) -> None:
        """JSON exports must expose the structured ``failure`` object and no legacy fields."""
        job = _make_job()
        failure = Failure(
            category=FailureCategory.TIMEOUT,
            stage="generate_answer",
            reason="timeout exceeded",
        )
        results = [_make_result("q1", failure=failure)]

        dst = tmp_path / "out.json"
        export_verification_results_json_stream(job, iter(results), out_path=dst)
        payload = json.loads(dst.read_text(encoding="utf-8"))

        assert len(payload["results"]) == 1
        metadata = payload["results"][0]["metadata"]
        assert "failure" in metadata
        assert metadata["failure"]["category"] == "timeout"
        # Legacy fields must be absent from the exported metadata
        assert "completed_without_errors" not in metadata
        assert "error" not in metadata
        assert "error_category" not in metadata
        assert "failed_stage" not in metadata
