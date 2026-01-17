"""Unit tests for ResultsManager class.

Tests cover:
- Results storage and retrieval
- Run management
- Export functionality (JSON, CSV)
- Import functionality (JSON, CSV)
- Statistics and summary
- Results filtering and querying
"""

import json
from datetime import datetime

import pytest

from karenina import Benchmark
from karenina.benchmark.core.results import ResultsManager
from karenina.benchmark.core.results_io import ResultsIOHandler
from karenina.schemas.workflow import VerificationResult, VerificationResultMetadata, VerificationResultTemplate


# Helper function to create a sample VerificationResult
def create_sample_result(
    question_id: str = "q1",
    question_text: str = "What is 2+2?",
    success: bool = True,
    execution_time: float = 1.5,
) -> VerificationResult:
    """Create a sample VerificationResult for testing."""
    timestamp = datetime.now().isoformat()
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=question_id,
        answering_model="claude-3-5-sonnet",
        parsing_model="claude-3-5-sonnet",
        timestamp=timestamp,
    )

    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="template_hash",
            completed_without_errors=success,
            error=None if success else "Test error",
            question_text=question_text,
            answering_model="claude-3-5-sonnet",
            parsing_model="claude-3-5-sonnet",
            execution_time=execution_time,
            timestamp=timestamp,
            result_id=result_id,
            run_name="test_run",
        ),
        template=VerificationResultTemplate(
            raw_llm_response="The answer is 4.",
            parsed_llm_response={"value": "4"},
            parsed_gt_response={"value": "4"},
            verify_result=True,
            verify_granular_result={"value": True},
        ),
    )


@pytest.mark.unit
class TestResultsManagerInit:
    """Tests for ResultsManager initialization."""

    def test_init_with_benchmark_base(self) -> None:
        """Test ResultsManager initialization with BenchmarkBase."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        assert manager.base is benchmark
        assert manager._in_memory_results == {}


@pytest.mark.unit
class TestStoreVerificationResults:
    """Tests for store_verification_results method."""

    def test_store_results_with_run_name(self) -> None:
        """Test storing results with explicit run name."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {
            "q1": create_sample_result(question_id="q1"),
            "q2": create_sample_result(question_id="q2"),
        }
        manager.store_verification_results(results, run_name="my_run")

        assert "my_run" in manager._in_memory_results
        assert len(manager._in_memory_results["my_run"]) == 2

    def test_store_results_auto_generates_run_name(self) -> None:
        """Test storing results auto-generates run name if not provided."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {"q1": create_sample_result()}
        manager.store_verification_results(results)

        assert len(manager._in_memory_results) == 1
        run_name = list(manager._in_memory_results.keys())[0]
        assert run_name.startswith("verification_")

    def test_store_results_multiple_runs(self) -> None:
        """Test storing results from multiple runs."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results1 = {"q1": create_sample_result()}
        results2 = {"q2": create_sample_result(question_id="q2")}

        manager.store_verification_results(results1, run_name="run1")
        manager.store_verification_results(results2, run_name="run2")

        assert len(manager._in_memory_results) == 2
        assert "run1" in manager._in_memory_results
        assert "run2" in manager._in_memory_results


@pytest.mark.unit
class TestGetVerificationResults:
    """Tests for get_verification_results method."""

    def test_get_all_results(self) -> None:
        """Test getting all results from a run."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {
            "q1": create_sample_result(question_id="q1"),
            "q2": create_sample_result(question_id="q2"),
        }
        manager.store_verification_results(results, run_name="test_run")

        retrieved = manager.get_verification_results()
        assert len(retrieved) == 2

    def test_get_results_filtered_by_question_ids(self) -> None:
        """Test getting results filtered by question IDs."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {
            "q1": create_sample_result(question_id="q1"),
            "q2": create_sample_result(question_id="q2"),
            "q3": create_sample_result(question_id="q3"),
        }
        manager.store_verification_results(results, run_name="test_run")

        filtered = manager.get_verification_results(question_ids=["q1", "q3"])
        assert len(filtered) == 2
        assert "q1" in filtered
        assert "q3" in filtered
        assert "q2" not in filtered

    def test_get_results_filtered_by_run_name(self) -> None:
        """Test getting results filtered by run name."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results1 = {"q1": create_sample_result(question_id="q1")}
        results2 = {"q2": create_sample_result(question_id="q2")}

        manager.store_verification_results(results1, run_name="run1")
        manager.store_verification_results(results2, run_name="run2")

        run1_results = manager.get_verification_results(run_name="run1")
        assert len(run1_results) == 1
        assert "q1" in run1_results

    def test_get_results_from_nonexistent_run(self) -> None:
        """Test getting results from nonexistent run returns empty dict."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = manager.get_verification_results(run_name="nonexistent")
        assert results == {}


@pytest.mark.unit
class TestGetVerificationHistory:
    """Tests for get_verification_history method."""

    def test_get_history_all_runs(self) -> None:
        """Test getting verification history for all runs."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results1 = {"q1": create_sample_result(question_id="q1")}
        results2 = {"q2": create_sample_result(question_id="q2")}

        manager.store_verification_results(results1, run_name="run1")
        manager.store_verification_results(results2, run_name="run2")

        history = manager.get_verification_history()
        assert len(history) == 2
        assert "run1" in history
        assert "run2" in history

    def test_get_history_filtered_by_question_id(self) -> None:
        """Test getting history filtered by question ID."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results1 = {"q1": create_sample_result(question_id="q1")}
        results2 = {"q1": create_sample_result(question_id="q1"), "q2": create_sample_result(question_id="q2")}

        manager.store_verification_results(results1, run_name="run1")
        manager.store_verification_results(results2, run_name="run2")

        history = manager.get_verification_history(question_id="q1")
        assert len(history) == 2
        # Both runs should have q1 results
        assert "q1" in history["run1"]
        assert "q1" in history["run2"]


@pytest.mark.unit
class TestClearVerificationResults:
    """Tests for clear_verification_results method."""

    def test_clear_all_results_from_run(self) -> None:
        """Test clearing all results from a run."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {"q1": create_sample_result(), "q2": create_sample_result(question_id="q2")}
        manager.store_verification_results(results, run_name="test_run")

        count = manager.clear_verification_results(run_name="test_run")
        assert count == 2
        assert "test_run" not in manager._in_memory_results

    def test_clear_specific_questions_from_run(self) -> None:
        """Test clearing specific questions from a run."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {
            "q1": create_sample_result(question_id="q1"),
            "q2": create_sample_result(question_id="q2"),
            "q3": create_sample_result(question_id="q3"),
        }
        manager.store_verification_results(results, run_name="test_run")

        count = manager.clear_verification_results(question_ids=["q1", "q3"], run_name="test_run")
        assert count == 2

        remaining = manager.get_verification_results(run_name="test_run")
        assert len(remaining) == 1
        assert "q2" in remaining

    def test_clear_from_nonexistent_run(self) -> None:
        """Test clearing from nonexistent run returns 0."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        count = manager.clear_verification_results(run_name="nonexistent")
        assert count == 0


@pytest.mark.unit
class TestExportVerificationResults:
    """Tests for export_verification_results method."""

    def test_export_to_json(self) -> None:
        """Test exporting results to JSON format."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {"q1": create_sample_result()}
        manager.store_verification_results(results, run_name="test_run")

        json_output = manager.export_verification_results(format="json")
        data = json.loads(json_output)

        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["row_index"] == 1

    def test_export_to_csv(self) -> None:
        """Test exporting results to CSV format."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {"q1": create_sample_result()}
        manager.store_verification_results(results, run_name="test_run")

        csv_output = manager.export_verification_results(format="csv")

        assert "row_index,question_id" in csv_output
        assert "q1" in csv_output

    def test_export_unsupported_format_raises(self) -> None:
        """Test that unsupported format raises ValueError."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {"q1": create_sample_result()}
        manager.store_verification_results(results, run_name="test_run")

        with pytest.raises(ValueError, match="Unsupported export format"):
            manager.export_verification_results(format="xml")


@pytest.mark.unit
class TestGetVerificationSummary:
    """Tests for get_verification_summary method."""

    def test_summary_empty_results(self) -> None:
        """Test summary with no results."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        summary = manager.get_verification_summary()

        assert summary["total_results"] == 0
        assert summary["successful_count"] == 0
        assert summary["failed_count"] == 0
        assert summary["success_rate"] == 0.0

    def test_summary_with_results(self) -> None:
        """Test summary with actual results."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {
            "q1": create_sample_result(question_id="q1", success=True),
            "q2": create_sample_result(question_id="q2", success=False, execution_time=2.0),
            "q3": create_sample_result(question_id="q3", success=True, execution_time=1.0),
        }
        manager.store_verification_results(results, run_name="test_run")

        summary = manager.get_verification_summary(run_name="test_run")

        assert summary["total_results"] == 3
        assert summary["successful_count"] == 2
        assert summary["failed_count"] == 1
        assert summary["success_rate"] == pytest.approx(66.67, rel=0.1)
        assert summary["unique_questions"] == 3


@pytest.mark.unit
class TestGetResultsByQuestion:
    """Tests for get_results_by_question method."""

    def test_get_results_by_single_question(self) -> None:
        """Test getting results for a specific question."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {
            "q1": create_sample_result(question_id="q1"),
            "q2": create_sample_result(question_id="q2"),
        }
        manager.store_verification_results(results, run_name="test_run")

        q1_results = manager.get_results_by_question("q1")
        assert len(q1_results) == 1
        assert "q1" in q1_results

    def test_get_results_by_nonexistent_question(self) -> None:
        """Test getting results for nonexistent question returns empty dict."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = manager.get_results_by_question("nonexistent")
        assert results == {}


@pytest.mark.unit
class TestGetResultsByRun:
    """Tests for get_results_by_run method."""

    def test_get_results_by_run(self) -> None:
        """Test getting results for a specific run."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {"q1": create_sample_result()}
        manager.store_verification_results(results, run_name="my_run")

        run_results = manager.get_results_by_run("my_run")
        assert len(run_results) == 1

    def test_get_results_by_nonexistent_run(self) -> None:
        """Test getting results for nonexistent run returns empty dict."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = manager.get_results_by_run("nonexistent")
        assert results == {}


@pytest.mark.unit
class TestGetLatestResults:
    """Tests for get_latest_results method."""

    def test_get_latest_empty(self) -> None:
        """Test getting latest results when none exist."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = manager.get_latest_results()
        assert results == {}

    def test_get_latest_single_run(self) -> None:
        """Test getting latest results with single run."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {"q1": create_sample_result()}
        manager.store_verification_results(results, run_name="test_run")

        latest = manager.get_latest_results()
        assert len(latest) == 1


@pytest.mark.unit
class TestHasResults:
    """Tests for has_results method."""

    def test_has_results_true(self) -> None:
        """Test has_results returns True when results exist."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {"q1": create_sample_result()}
        manager.store_verification_results(results, run_name="test_run")

        assert manager.has_results() is True

    def test_has_results_false(self) -> None:
        """Test has_results returns False when no results exist."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        assert manager.has_results() is False

    def test_has_results_for_question(self) -> None:
        """Test has_results for specific question."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {
            "q1": create_sample_result(question_id="q1"),
            "q2": create_sample_result(question_id="q2"),
        }
        manager.store_verification_results(results, run_name="test_run")

        assert manager.has_results(question_id="q1") is True
        assert manager.has_results(question_id="q3") is False


@pytest.mark.unit
class TestGetAllRunNames:
    """Tests for get_all_run_names method."""

    def test_get_all_run_names_empty(self) -> None:
        """Test getting run names when none exist."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        assert manager.get_all_run_names() == []

    def test_get_all_run_names_sorted(self) -> None:
        """Test getting run names returns sorted list."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {"q1": create_sample_result()}
        manager.store_verification_results(results, run_name="zebra_run")
        manager.store_verification_results(results, run_name="alpha_run")
        manager.store_verification_results(results, run_name="beta_run")

        run_names = manager.get_all_run_names()
        assert run_names == ["alpha_run", "beta_run", "zebra_run"]


@pytest.mark.unit
class TestGetResultsStatisticsByRun:
    """Tests for get_results_statistics_by_run method."""

    def test_get_statistics_by_run(self) -> None:
        """Test getting statistics organized by run."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results1 = {"q1": create_sample_result(success=True)}
        results2 = {"q2": create_sample_result(question_id="q2", success=False)}

        manager.store_verification_results(results1, run_name="run1")
        manager.store_verification_results(results2, run_name="run2")

        stats = manager.get_results_statistics_by_run()

        assert len(stats) == 2
        assert "run1" in stats
        assert "run2" in stats
        assert stats["run1"]["successful_count"] == 1
        assert stats["run2"]["failed_count"] == 1


@pytest.mark.unit
class TestExportResultsToFile:
    """Tests for export_results_to_file method."""

    def test_export_to_json_file(self, tmp_path) -> None:
        """Test exporting results to JSON file."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {"q1": create_sample_result()}
        manager.store_verification_results(results, run_name="test_run")

        output_path = tmp_path / "results.json"
        manager.export_results_to_file(output_path, run_name="test_run")

        assert output_path.exists()

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_export_to_csv_file(self, tmp_path) -> None:
        """Test exporting results to CSV file."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {"q1": create_sample_result()}
        manager.store_verification_results(results, run_name="test_run")

        output_path = tmp_path / "results.csv"
        manager.export_results_to_file(output_path, run_name="test_run")

        assert output_path.exists()

        content = output_path.read_text()
        assert "row_index" in content

    def test_export_auto_detect_format(self, tmp_path) -> None:
        """Test auto format detection from file extension."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        results = {"q1": create_sample_result()}
        manager.store_verification_results(results, run_name="test_run")

        json_path = tmp_path / "data.json"
        csv_path = tmp_path / "data.csv"

        manager.export_results_to_file(json_path, run_name="test_run")
        manager.export_results_to_file(csv_path, run_name="test_run")

        assert json_path.exists()
        assert csv_path.exists()

    def test_export_unsupported_extension_raises(self, tmp_path) -> None:
        """Test that unsupported file extension raises ValueError."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        output_path = tmp_path / "results.xml"

        with pytest.raises(ValueError, match="Cannot determine format"):
            manager.export_results_to_file(output_path)


@pytest.mark.unit
class TestLoadResultsFromFile:
    """Tests for load_results_from_file method."""

    def test_load_nonexistent_file_raises(self, tmp_path) -> None:
        """Test that loading nonexistent file raises FileNotFoundError."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        with pytest.raises(FileNotFoundError):
            manager.load_results_from_file(tmp_path / "nonexistent.json")

    def test_load_unsupported_format_raises(self, tmp_path) -> None:
        """Test that loading unsupported format raises ValueError."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        xml_path = tmp_path / "results.xml"
        xml_path.write_text("<data></data>")

        with pytest.raises(ValueError, match="Unsupported file format"):
            manager.load_results_from_file(xml_path)

    def test_load_empty_json_returns_empty_dict(self, tmp_path) -> None:
        """Test loading empty JSON file returns empty dict."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        json_path = tmp_path / "results.json"
        json_path.write_text("[]")

        loaded = manager.load_results_from_file(json_path)
        assert len(loaded) == 0

    def test_load_empty_csv_returns_empty_dict(self, tmp_path) -> None:
        """Test loading CSV with only header returns empty dict."""
        benchmark = Benchmark.create(name="test")
        manager = ResultsManager(benchmark)

        csv_content = "row_index,question_id,question_text"
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(csv_content)

        loaded = manager.load_results_from_file(csv_path)
        assert len(loaded) == 0


@pytest.mark.unit
class TestEscapeCsvField:
    """Tests for ResultsIOHandler._escape_csv_field static method."""

    def test_escape_none(self) -> None:
        """Test escaping None returns empty string."""
        assert ResultsIOHandler._escape_csv_field(None) == ""

    def test_escape_simple_string(self) -> None:
        """Test escaping simple string returns as-is."""
        assert ResultsIOHandler._escape_csv_field("hello") == "hello"

    def test_escape_with_comma(self) -> None:
        """Test escaping string with comma wraps in quotes."""
        assert ResultsIOHandler._escape_csv_field("hello,world") == '"hello,world"'

    def test_escape_with_quote(self) -> None:
        """Test escaping string with quote doubles quotes."""
        assert ResultsIOHandler._escape_csv_field('say "hello"') == '"say ""hello"""'

    def test_escape_with_newline(self) -> None:
        """Test escaping string with newline wraps in quotes."""
        assert ResultsIOHandler._escape_csv_field("line1\nline2") == '"line1\nline2"'
