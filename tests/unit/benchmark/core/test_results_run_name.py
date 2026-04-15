"""Tests for run_name preservation on bulk reimport."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from karenina.benchmark.core.results import ResultsManager
from karenina.schemas.verification import VerificationResult, VerificationResultMetadata
from karenina.schemas.verification.model_identity import ModelIdentity


def _make_minimal_result(question_id: str, run_name: str | None) -> dict:
    """Create a minimal serialized VerificationResult dict.

    Builds a valid nested structure matching the VerificationResult schema,
    including all required metadata fields.
    """
    answering = ModelIdentity(interface="langchain", model_name="gpt-4")
    parsing = ModelIdentity(interface="langchain", model_name="gpt-4")
    timestamp = datetime.now().isoformat()
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=question_id,
        answering=answering,
        parsing=parsing,
        timestamp=timestamp,
    )

    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            question_text="test",
            template_id="abc",
            run_name=run_name,
            answering=answering,
            parsing=parsing,
            execution_time=0.1,
            timestamp=timestamp,
            result_id=result_id,
            failure=None,
            caveats=[],
        ),
    )
    return result.model_dump(mode="json")


@pytest.mark.unit
class TestRunNamePreservation:
    """Bulk JSON reimport should preserve run_name grouping from metadata."""

    def test_multi_run_reimport_groups_by_metadata_run_name(self, tmp_path: Path) -> None:
        """Results from different runs should be stored under their original run_name."""
        data = [
            _make_minimal_result("q1", "run-alpha"),
            _make_minimal_result("q2", "run-alpha"),
            _make_minimal_result("q3", "run-beta"),
        ]
        json_file = tmp_path / "results.json"
        json_file.write_text(json.dumps(data))

        manager = ResultsManager.__new__(ResultsManager)
        manager._in_memory_results = {}

        manager.load_results_from_file(json_file, run_name="fallback")

        assert "run-alpha" in manager._in_memory_results
        assert "run-beta" in manager._in_memory_results
        assert len(manager._in_memory_results["run-alpha"]) == 2
        assert len(manager._in_memory_results["run-beta"]) == 1

    def test_fallback_run_name_when_metadata_is_none(self, tmp_path: Path) -> None:
        """Results without metadata.run_name should use the caller-provided fallback."""
        data = [_make_minimal_result("q1", None)]
        json_file = tmp_path / "results.json"
        json_file.write_text(json.dumps(data))

        manager = ResultsManager.__new__(ResultsManager)
        manager._in_memory_results = {}

        manager.load_results_from_file(json_file, run_name="my-fallback")

        assert "my-fallback" in manager._in_memory_results
