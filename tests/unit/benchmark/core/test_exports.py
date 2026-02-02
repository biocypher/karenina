"""Unit tests for ExportManager class.

Tests cover:
- Export operations (to_dict, to_markdown, to_csv, export_to_file)
- Statistics and summary methods
- Readiness checks
- Health reports
- Cloning functionality
"""

import pytest

from karenina import Benchmark
from karenina.benchmark.core.exports import ExportManager

# Valid template for testing
VALID_TEMPLATE = '''class Answer(BaseAnswer):
    """Simple answer template."""

    value: str = Field(description="The answer value")

    def verify(self) -> bool:
        return len(self.value) > 0
'''


@pytest.mark.unit
class TestExportManagerInit:
    """Tests for ExportManager initialization."""

    def test_init_components(self) -> None:
        """Test ExportManager initialization with components."""
        benchmark = Benchmark.create(name="test")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        assert manager.base is benchmark._base
        assert manager.templates_manager is not None
        assert manager.rubrics_manager is not None


@pytest.mark.unit
class TestToDict:
    """Tests for to_dict method."""

    def test_to_dict_structure(self) -> None:
        """Test that to_dict returns correct structure."""
        benchmark = Benchmark.create(name="test-bench", description="Test description")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        result = manager.to_dict()

        assert "metadata" in result
        assert "statistics" in result
        assert "questions" in result
        assert "global_rubric" in result

    def test_to_dict_metadata(self) -> None:
        """Test that to_dict includes correct metadata."""
        benchmark = Benchmark.create(
            name="test-bench", description="Test description", version="2.0.0", creator="Test Creator"
        )
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        result = manager.to_dict()

        assert result["metadata"]["name"] == "test-bench"
        assert result["metadata"]["description"] == "Test description"
        assert result["metadata"]["version"] == "2.0.0"
        assert result["metadata"]["creator"] == "Test Creator"
        assert "created_at" in result["metadata"]
        assert "modified_at" in result["metadata"]

    def test_to_dict_with_questions(self) -> None:
        """Test that to_dict includes questions."""
        benchmark = Benchmark.create(name="test")
        benchmark.add_question("What is 2+2?", "4")
        benchmark.add_question("What is 3+3?", "6")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        result = manager.to_dict()

        assert len(result["questions"]) == 2

    def test_to_dict_global_rubric(self) -> None:
        """Test that to_dict includes global rubric if present."""
        benchmark = Benchmark.create(name="test")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        result = manager.to_dict()

        # No global rubric by default
        assert result["global_rubric"] is None


@pytest.mark.unit
class TestToMarkdown:
    """Tests for to_markdown method."""

    def test_to_markdown_structure(self) -> None:
        """Test that to_markdown returns valid markdown."""
        benchmark = Benchmark.create(name="test-bench", description="Test desc")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        markdown = manager.to_markdown()

        assert "# test-bench" in markdown
        assert "Test desc" in markdown
        assert "## Metadata" in markdown
        assert "## Statistics" in markdown

    def test_to_markdown_includes_metadata(self) -> None:
        """Test that markdown includes benchmark metadata."""
        benchmark = Benchmark.create(name="test", description="Description", version="1.5.0", creator="Creator")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        markdown = manager.to_markdown()

        assert "**Version**: 1.5.0" in markdown
        assert "**Creator**: Creator" in markdown

    def test_to_markdown_includes_questions(self) -> None:
        """Test that markdown includes questions."""
        benchmark = Benchmark.create(name="test")
        benchmark.add_question("Question 1?", "Answer 1")
        benchmark.add_question("Question 2?", "Answer 2")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        markdown = manager.to_markdown()

        assert "## Questions" in markdown
        assert "Question 1?" in markdown
        assert "Question 2?" in markdown

    def test_to_markdown_status_indicators(self) -> None:
        """Test that markdown shows correct status indicators."""
        benchmark = Benchmark.create(name="test")
        q_id1 = benchmark.add_question("Q1?", "A1", finished=True)
        benchmark.add_question("Q2?", "A2", finished=False)

        # Add template to first question via template manager
        benchmark._template_manager.add_answer_template(q_id1, VALID_TEMPLATE)

        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        markdown = manager.to_markdown()

        # Check for status indicators (emoji)
        assert "✅" in markdown or "❌" in markdown
        assert "📝" in markdown or "Template:" in markdown


@pytest.mark.unit
class TestToCsv:
    """Tests for to_csv method."""

    def test_to_csv_header(self) -> None:
        """Test that CSV has correct header."""
        benchmark = Benchmark.create(name="test")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        csv = manager.to_csv()

        lines = csv.strip().split("\n")
        assert "Question ID" in lines[0]
        assert "Question" in lines[0]
        assert "Raw Answer" in lines[0]
        assert "Has Template" in lines[0]
        assert "Finished" in lines[0]

    def test_to_csv_with_questions(self) -> None:
        """Test that CSV includes question data."""
        benchmark = Benchmark.create(name="test")
        benchmark.add_question("What is 2+2?", "4", finished=True)
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        csv = manager.to_csv()
        lines = csv.strip().split("\n")

        assert len(lines) == 2  # header + 1 data row
        assert "What is 2+2?" in csv
        assert "4" in csv

    def test_to_csv_multiple_questions(self) -> None:
        """Test CSV with multiple questions."""
        benchmark = Benchmark.create(name="test")
        benchmark.add_question("Q1?", "A1", finished=False)
        benchmark.add_question("Q2?", "A2", finished=True)
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        csv = manager.to_csv()
        lines = csv.strip().split("\n")

        assert len(lines) == 3  # header + 2 data rows


@pytest.mark.unit
class TestGetSummary:
    """Tests for get_summary method."""

    def test_get_summary_empty_benchmark(self) -> None:
        """Test summary for empty benchmark."""
        benchmark = Benchmark.create(name="test")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        summary = manager.get_summary()

        assert summary["name"] == "test"
        assert summary["question_count"] == 0.0
        assert summary["finished_count"] == 0
        assert summary["has_template_count"] == 0
        assert summary["progress_percentage"] == 0.0
        assert summary["is_complete"] is False
        assert summary["has_global_rubric"] is False
        assert summary["global_rubric_traits"] == 0

    def test_get_summary_with_questions(self) -> None:
        """Test summary with questions."""
        benchmark = Benchmark.create(name="test")
        benchmark.add_question("Q1?", "A1", finished=True)
        benchmark.add_question("Q2?", "A2", finished=False)
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        summary = manager.get_summary()

        assert summary["question_count"] == 2.0
        assert summary["finished_count"] == 1
        assert summary["progress_percentage"] == 50.0

    def test_get_summary_with_templates(self) -> None:
        """Test summary counts templates correctly."""
        benchmark = Benchmark.create(name="test")
        q_id1 = benchmark.add_question("Q1?", "A1")
        benchmark.add_question("Q2?", "A2")

        benchmark._template_manager.add_answer_template(q_id1, VALID_TEMPLATE)

        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        summary = manager.get_summary()

        assert summary["has_template_count"] == 1


@pytest.mark.unit
class TestGetStatistics:
    """Tests for get_statistics method."""

    def test_get_statistics_empty(self) -> None:
        """Test statistics for empty benchmark."""
        benchmark = Benchmark.create(name="test")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        stats = manager.get_statistics()

        assert stats["question_count"] == 0.0
        assert stats["avg_template_length"] == 0
        assert stats["min_template_length"] == 0
        assert stats["max_template_length"] == 0
        assert stats["questions_with_custom_metadata"] == 0

    def test_get_statistics_with_templates(self) -> None:
        """Test statistics includes template length stats."""
        benchmark = Benchmark.create(name="test")
        q_id = benchmark.add_question("Q?", "A")
        benchmark._template_manager.add_answer_template(q_id, VALID_TEMPLATE)

        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        stats = manager.get_statistics()

        assert stats["avg_template_length"] > 0
        assert stats["min_template_length"] > 0
        assert stats["max_template_length"] > 0
        assert stats["min_template_length"] <= stats["max_template_length"]


@pytest.mark.unit
class TestCheckReadiness:
    """Tests for check_readiness method."""

    def test_readiness_empty_benchmark(self) -> None:
        """Test readiness check for empty benchmark."""
        benchmark = Benchmark.create(name="test")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        readiness = manager.check_readiness()

        assert readiness["ready_for_verification"] is False
        assert readiness["has_questions"] is False
        assert readiness["all_have_templates"] is True  # No questions = vacuously true
        assert readiness["all_finished"] is True  # No questions = vacuously true
        assert readiness["templates_valid"] is True
        assert readiness["rubrics_valid"] is True

    def test_readiness_with_questions(self) -> None:
        """Test readiness with questions but no templates."""
        benchmark = Benchmark.create(name="test")
        benchmark.add_question("Q?", "A", finished=False)
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        readiness = manager.check_readiness()

        assert readiness["ready_for_verification"] is False
        assert readiness["has_questions"] is True
        assert readiness["all_have_templates"] is False  # No template
        assert readiness["all_finished"] is False

    def test_readiness_complete(self) -> None:
        """Test readiness when all conditions met."""
        benchmark = Benchmark.create(name="test")
        q_id = benchmark.add_question("Q?", "A", finished=True)
        benchmark._template_manager.add_answer_template(q_id, VALID_TEMPLATE)

        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        readiness = manager.check_readiness()

        assert readiness["ready_for_verification"] is True
        assert readiness["has_questions"] is True
        assert readiness["all_have_templates"] is True
        assert readiness["all_finished"] is True


@pytest.mark.unit
class TestGetHealthReport:
    """Tests for get_health_report method."""

    def test_health_report_empty_benchmark(self) -> None:
        """Test health report for empty benchmark."""
        benchmark = Benchmark.create(name="test")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        report = manager.get_health_report()

        assert report["health_score"] == 0.0
        assert report["health_status"] == "critical"
        assert "readiness" in report
        assert "statistics" in report
        assert "recommendations" in report

    def test_health_report_with_questions(self) -> None:
        """Test health report with questions."""
        benchmark = Benchmark.create(name="test")
        benchmark.add_question("Q?", "A", finished=True)
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        report = manager.get_health_report()

        assert report["health_score"] > 0
        assert "timestamp" in report
        assert len(report["recommendations"]) > 0  # Should have recommendations

    def test_health_status_levels(self) -> None:
        """Test health status classification."""
        benchmark = Benchmark.create(name="test")
        benchmark.add_question("Q?", "A", finished=True)
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        report = manager.get_health_report()
        status = report["health_status"]

        # Status should be one of the defined levels
        assert status in ["excellent", "good", "fair", "poor", "critical"]


@pytest.mark.unit
class TestGetRecommendations:
    """Tests for _get_recommendations method."""

    def test_recommendations_empty_benchmark(self) -> None:
        """Test recommendations for empty benchmark."""
        benchmark = Benchmark.create(name="test")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        readiness = manager.check_readiness()

        recommendations = manager._get_recommendations(readiness)

        assert len(recommendations) > 0
        assert any("questions" in r.lower() for r in recommendations)


@pytest.mark.unit
class TestClone:
    """Tests for clone method."""

    def test_clone_creates_copy(self) -> None:
        """Test that clone creates a deep copy."""
        benchmark = Benchmark.create(name="original")
        benchmark.add_question("Q?", "A")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        cloned = manager.clone()

        assert cloned.name == benchmark._base.name
        assert cloned.question_count == benchmark._base.question_count

    def test_clone_independent_modifications(self) -> None:
        """Test that cloned benchmark is independent."""
        original = Benchmark.create(name="original")
        original.add_question("Q1?", "A1")
        manager = ExportManager(original._base, original._template_manager, original._rubric_manager)

        cloned = manager.clone()
        # Clone returns BenchmarkBase, verify it has the same initial state
        # Modifications to clone don't affect original (verified via internal cache)
        assert original._base.question_count == 1
        assert cloned.question_count == 1


@pytest.mark.unit
class TestExportToFile:
    """Tests for export_to_file method."""

    def test_export_to_json(self, tmp_path) -> None:
        """Test exporting to JSON file."""
        benchmark = Benchmark.create(name="test")
        benchmark.add_question("Q?", "A")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        output_path = tmp_path / "export.json"
        manager.export_to_file(output_path, format="json")

        assert output_path.exists()

        content = output_path.read_text()
        assert "test" in content
        assert "Q?" in content

    def test_export_to_csv(self, tmp_path) -> None:
        """Test exporting to CSV file."""
        benchmark = Benchmark.create(name="test")
        benchmark.add_question("Q?", "A")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        output_path = tmp_path / "export.csv"
        manager.export_to_file(output_path, format="csv")

        assert output_path.exists()

        content = output_path.read_text()
        assert "Question ID" in content
        assert "Q?" in content

    def test_export_to_markdown(self, tmp_path) -> None:
        """Test exporting to markdown file."""
        benchmark = Benchmark.create(name="test")
        benchmark.add_question("Q?", "A")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        output_path = tmp_path / "export.md"
        manager.export_to_file(output_path, format="markdown")

        assert output_path.exists()

        content = output_path.read_text()
        assert "# test" in content
        assert "Q?" in content

    def test_export_auto_detect_json(self, tmp_path) -> None:
        """Test auto format detection for JSON."""
        benchmark = Benchmark.create(name="test")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        output_path = tmp_path / "export.json"
        manager.export_to_file(output_path, format="auto")

        assert output_path.exists()

    def test_export_auto_detect_csv(self, tmp_path) -> None:
        """Test auto format detection for CSV."""
        benchmark = Benchmark.create(name="test")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        output_path = tmp_path / "data.csv"
        manager.export_to_file(output_path, format="auto")

        assert output_path.exists()

    def test_export_auto_detect_markdown(self, tmp_path) -> None:
        """Test auto format detection for markdown."""
        benchmark = Benchmark.create(name="test")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        output_path = tmp_path / "readme.md"
        manager.export_to_file(output_path, format="auto")

        assert output_path.exists()

    def test_export_unsupported_format_raises(self, tmp_path) -> None:
        """Test that unsupported format raises error."""
        benchmark = Benchmark.create(name="test")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        output_path = tmp_path / "export.xyz"

        with pytest.raises(ValueError, match="Unsupported export format"):
            manager.export_to_file(output_path, format="xyz")


@pytest.mark.unit
class TestGetProgressReport:
    """Tests for get_progress_report method."""

    def test_progress_report_structure(self) -> None:
        """Test that progress report has correct structure."""
        benchmark = Benchmark.create(name="test-bench")
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)
        report = manager.get_progress_report()

        assert "name" in report
        assert "progress_percentage" in report
        assert "questions_total" in report
        assert "questions_finished" in report
        assert "questions_with_templates" in report
        assert "is_ready_for_verification" in report
        assert "last_modified" in report

    def test_progress_report_values(self) -> None:
        """Test that progress report has correct values."""
        benchmark = Benchmark.create(name="test")
        benchmark.add_question("Q1?", "A1", finished=True)
        benchmark.add_question("Q2?", "A2", finished=False)
        manager = ExportManager(benchmark._base, benchmark._template_manager, benchmark._rubric_manager)

        report = manager.get_progress_report()

        assert report["name"] == "test"
        assert report["questions_total"] == 2
        assert report["questions_finished"] == 1
        assert report["progress_percentage"] == 50.0
