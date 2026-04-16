"""Tests for results I/O issue fixes.

Covers:
- Issue 021: CSV round-trip reconstructs nested VerificationResult from flat columns
"""

from pathlib import Path

import pytest

from karenina.benchmark.core.results_io import ResultsIOManager
from karenina.schemas.verification import VerificationResult, VerificationResultMetadata, VerificationResultTemplate
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import VerificationResultRubric


def _make_result(
    question_id: str = "q1",
    question_text: str = "What is 2+2?",
    answering_interface: str = "openai",
    answering_model_name: str = "gpt-4",
    parsing_interface: str = "openai",
    parsing_model_name: str = "gpt-4",
    tools: list[str] | None = None,
    raw_llm_response: str = "The answer is 4",
    verify_result: bool = True,
    execution_time: float = 1.5,
    timestamp: str = "2026-01-01T00:00:00",
    rubric_scores: dict[str, int | bool] | None = None,
    run_name: str | None = "test_run",
) -> VerificationResult:
    """Create a VerificationResult for testing CSV round-trip."""
    answering = ModelIdentity(
        interface=answering_interface,
        model_name=answering_model_name,
        tools=tools or [],
    )
    parsing = ModelIdentity(
        interface=parsing_interface,
        model_name=parsing_model_name,
    )
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=question_id,
        answering=answering,
        parsing=parsing,
        timestamp=timestamp,
    )
    metadata = VerificationResultMetadata(
        question_id=question_id,
        template_id="no_template",
        failure=None,
        caveats=[],
        question_text=question_text,
        answering=answering,
        parsing=parsing,
        execution_time=execution_time,
        timestamp=timestamp,
        result_id=result_id,
        run_name=run_name,
    )
    template = VerificationResultTemplate(
        raw_llm_response=raw_llm_response,
        verify_result=verify_result,
        parsed_llm_response={"value": "4"},
        parsed_gt_response={"value": "4"},
    )
    rubric = None
    if rubric_scores is not None:
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores=rubric_scores,
        )
    return VerificationResult(metadata=metadata, template=template, rubric=rubric)


@pytest.mark.unit
class TestCsvRoundTrip:
    """Issue 021: CSV round-trip must reconstruct nested VerificationResult."""

    def test_csv_round_trip_basic(self, tmp_path: Path) -> None:
        """Export a VerificationResult to CSV, reload it, and verify key fields match."""
        original = _make_result()
        results = {"q1_1": original}

        csv_content = ResultsIOManager.export_to_csv(results)
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        loaded = ResultsIOManager.load_from_csv(csv_file)

        assert len(loaded) == 1
        loaded_result = next(iter(loaded.values()))

        # Core metadata fields
        assert loaded_result.metadata.question_id == "q1"
        assert loaded_result.metadata.question_text == "What is 2+2?"
        assert loaded_result.metadata.failure is None
        assert loaded_result.metadata.execution_time == 1.5
        assert loaded_result.metadata.timestamp == "2026-01-01T00:00:00"
        assert loaded_result.metadata.run_name == "test_run"

        # Template fields
        assert loaded_result.template is not None
        assert loaded_result.template.raw_llm_response == "The answer is 4"
        assert loaded_result.template.verify_result is True

    def test_csv_round_trip_with_rubric_traits(self, tmp_path: Path) -> None:
        """Rubric trait scores survive the CSV round-trip."""
        original = _make_result(rubric_scores={"clarity": 4, "accuracy": 5})
        results = {"q1_1": original}

        csv_content = ResultsIOManager.export_to_csv(results)
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        loaded = ResultsIOManager.load_from_csv(csv_file)

        assert len(loaded) == 1
        loaded_result = next(iter(loaded.values()))
        assert loaded_result.rubric is not None
        all_scores = loaded_result.rubric.get_all_trait_scores()
        assert all_scores["clarity"] == 4
        assert all_scores["accuracy"] == 5

    def test_csv_round_trip_preserves_model_identity(self, tmp_path: Path) -> None:
        """Answering and parsing model identity reconstructed from display strings."""
        original = _make_result(
            answering_interface="anthropic",
            answering_model_name="claude-3-5-sonnet",
            parsing_interface="openai",
            parsing_model_name="gpt-4o",
            tools=["mcp__brave_search", "mcp__read_resource"],
        )
        results = {"q1_1": original}

        csv_content = ResultsIOManager.export_to_csv(results)
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        loaded = ResultsIOManager.load_from_csv(csv_file)

        loaded_result = next(iter(loaded.values()))
        assert loaded_result.metadata.answering.interface == "anthropic"
        assert loaded_result.metadata.answering.model_name == "claude-3-5-sonnet"
        assert set(loaded_result.metadata.answering.tools) == {"mcp__brave_search", "mcp__read_resource"}

        assert loaded_result.metadata.parsing.interface == "openai"
        assert loaded_result.metadata.parsing.model_name == "gpt-4o"
        assert loaded_result.metadata.parsing.tools == []

    def test_csv_load_does_not_silently_drop_rows(self, tmp_path: Path) -> None:
        """Export N results, load them back, and verify len(loaded) == N."""
        results = {}
        for i in range(5):
            r = _make_result(
                question_id=f"q{i}",
                question_text=f"Question {i}",
                timestamp=f"2026-01-01T00:00:0{i}",
            )
            results[f"q{i}_{i + 1}"] = r

        csv_content = ResultsIOManager.export_to_csv(results)
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        loaded = ResultsIOManager.load_from_csv(csv_file)
        assert len(loaded) == 5
