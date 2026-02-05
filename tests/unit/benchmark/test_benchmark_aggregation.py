"""Unit tests for Benchmark aggregation and export methods.

Tests cover:
- to_dict() export as dictionary
- to_csv() export as CSV format
- to_markdown() export as markdown
- get_summary() comprehensive statistics
- get_statistics() detailed statistics
- check_readiness() verification readiness
- get_health_report() comprehensive health status
"""

import csv
from io import StringIO

import pytest

from karenina import Benchmark


@pytest.mark.unit
def test_to_dict_empty_benchmark() -> None:
    """Test to_dict with empty benchmark."""
    benchmark = Benchmark.create(name="test")

    result = benchmark.to_dict()

    assert result["metadata"]["name"] == "test"
    assert result["metadata"]["version"] == "0.1.0"
    assert result["statistics"]["question_count"] == 0
    assert result["questions"] == []
    assert result["global_rubric"] is None


@pytest.mark.unit
def test_to_dict_with_questions() -> None:
    """Test to_dict includes questions."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False)

    result = benchmark.to_dict()

    assert len(result["questions"]) == 2
    assert result["statistics"]["question_count"] == 2


@pytest.mark.unit
def test_to_dict_includes_global_rubric() -> None:
    """Test to_dict includes global rubric."""
    from karenina.schemas.entities import LLMRubricTrait, Rubric

    benchmark = Benchmark.create(name="test")
    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)

    trait = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)
    rubric = Rubric(llm_traits=[trait])
    benchmark.set_global_rubric(rubric)

    result = benchmark.to_dict()

    assert result["global_rubric"] is not None
    assert result["global_rubric"]["llm_traits"][0]["name"] == "clarity"


@pytest.mark.unit
def test_to_csv_empty_benchmark() -> None:
    """Test to_csv with empty benchmark returns only header."""
    benchmark = Benchmark.create(name="test")

    csv_content = benchmark.to_csv()
    reader = csv.reader(StringIO(csv_content))

    rows = list(reader)
    assert len(rows) == 1  # Only header row
    assert "Question ID" in rows[0]


@pytest.mark.unit
def test_to_csv_with_questions() -> None:
    """Test to_csv exports question data."""
    benchmark = Benchmark.create(name="test")

    author = {"name": "Alice", "email": "alice@example.com"}
    benchmark.add_question("What is 2+2?", "4", question_id="q1", finished=True, author=author)
    benchmark.add_question("What is 3+3?", "6", question_id="q2", finished=False)

    csv_content = benchmark.to_csv()
    reader = csv.reader(StringIO(csv_content))

    rows = list(reader)
    assert len(rows) == 3  # Header + 2 data rows

    # Check header
    assert "Question ID" in rows[0]
    assert "Question" in rows[0]

    # Check data rows
    assert rows[1][0] == "q1"
    assert rows[1][2] == "4"  # Raw Answer
    # Note: Default template may be applied, so just check it's not empty
    assert rows[1][3] in ["Yes", "No"]  # Has Template
    assert rows[1][4] == "Yes"  # Finished
    assert rows[1][5] == "Alice"  # Author


@pytest.mark.unit
def test_to_csv_with_template() -> None:
    """Test to_csv shows template status."""
    benchmark = Benchmark.create(name="test")

    template = """
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: str = Field(description="The answer")

    def verify(self) -> bool:
        return self.value == "4"
"""

    benchmark.add_question("What is 2+2?", "4", question_id="q1", finished=True)
    benchmark.add_answer_template("q1", template)

    csv_content = benchmark.to_csv()
    reader = csv.reader(StringIO(csv_content))

    rows = list(reader)
    # Has Template column should be "Yes"
    assert rows[1][3] == "Yes"


@pytest.mark.unit
def test_get_summary_empty_benchmark() -> None:
    """Test get_summary with empty benchmark."""
    benchmark = Benchmark.create(name="empty-test")

    summary = benchmark.get_summary()

    assert summary["name"] == "empty-test"
    assert summary["question_count"] == 0
    assert summary["finished_count"] == 0
    assert summary["has_template_count"] == 0
    assert summary["has_rubric_count"] == 0
    assert summary["progress_percentage"] == 0.0
    assert summary["is_complete"] is False
    assert summary["has_global_rubric"] is False


@pytest.mark.unit
def test_get_summary_with_progress() -> None:
    """Test get_summary calculates progress correctly."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False)
    benchmark.add_question("Q3?", "A3", question_id="q3", finished=True)

    summary = benchmark.get_summary()

    assert summary["question_count"] == 3
    assert summary["finished_count"] == 2
    # Use approximate comparison for floating point
    assert 66.0 < summary["progress_percentage"] < 67.0


@pytest.mark.unit
def test_get_summary_with_global_rubric() -> None:
    """Test get_summary includes global rubric info."""
    from karenina.schemas.entities import LLMRubricTrait, Rubric

    benchmark = Benchmark.create(name="test")
    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)

    trait1 = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True, description="Response clarity")
    trait2 = LLMRubricTrait(name="safety", kind="boolean", higher_is_better=True, description="Safety check")
    rubric = Rubric(llm_traits=[trait1, trait2])
    benchmark.set_global_rubric(rubric)

    summary = benchmark.get_summary()

    assert summary["has_global_rubric"] is True
    # global_rubric_traits counts only LLM traits
    assert summary["global_rubric_traits"] == 2


@pytest.mark.unit
def test_get_statistics_empty_benchmark() -> None:
    """Test get_statistics with empty benchmark."""
    benchmark = Benchmark.create(name="test")

    stats = benchmark.get_statistics()

    assert stats["avg_template_length"] == 0
    assert stats["min_template_length"] == 0
    assert stats["max_template_length"] == 0
    assert stats["questions_with_custom_metadata"] == 0


@pytest.mark.unit
def test_get_statistics_with_templates() -> None:
    """Test get_statistics calculates template lengths."""
    benchmark = Benchmark.create(name="test")

    template1 = """
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: str = Field(description="The answer")

    def verify(self) -> bool:
        return True
"""

    template2 = """
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: str = Field(description="The answer")
    extra: str = Field(description="Extra field")

    def verify(self) -> bool:
        return True
"""

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=True)
    benchmark.add_answer_template("q1", template1)
    benchmark.add_answer_template("q2", template2)

    stats = benchmark.get_statistics()

    assert stats["avg_template_length"] > 0
    assert stats["min_template_length"] <= stats["max_template_length"]


@pytest.mark.unit
def test_get_statistics_custom_metadata() -> None:
    """Test get_statistics counts custom metadata."""
    benchmark = Benchmark.create(name="test")

    author = {"name": "Alice"}
    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True, author=author)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=True)
    benchmark.add_question("Q3?", "A3", question_id="q3", finished=True, custom_metadata={"difficulty": "easy"})

    stats = benchmark.get_statistics()

    # q1 has author, q3 has custom_metadata = 2 questions with custom metadata
    assert stats["questions_with_custom_metadata"] == 2


@pytest.mark.unit
def test_check_readiness_empty_benchmark() -> None:
    """Test check_readiness with empty benchmark."""
    benchmark = Benchmark.create(name="empty")

    readiness = benchmark.check_readiness()

    assert readiness["ready_for_verification"] is False
    assert readiness["has_questions"] is False


@pytest.mark.unit
def test_check_readiness_missing_templates() -> None:
    """Test check_readiness detects missing templates."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=True)

    readiness = benchmark.check_readiness()

    assert readiness["ready_for_verification"] is False
    assert readiness["missing_templates_count"] == 2
    assert readiness["all_have_templates"] is False


@pytest.mark.unit
def test_check_readiness_unfinished_questions() -> None:
    """Test check_readiness detects unfinished questions."""
    benchmark = Benchmark.create(name="test")

    template = """
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: str = Field(description="The answer")

    def verify(self) -> bool:
        return True
"""

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=False)
    benchmark.add_answer_template("q1", template)

    readiness = benchmark.check_readiness()

    assert readiness["ready_for_verification"] is False
    assert readiness["unfinished_count"] == 1
    assert readiness["all_finished"] is False


@pytest.mark.unit
def test_check_readiness_ready_benchmark() -> None:
    """Test check_readiness with fully ready benchmark."""
    benchmark = Benchmark.create(name="test")

    template = """
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: str = Field(description="The answer")

    def verify(self) -> bool:
        return True
"""

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_answer_template("q1", template)

    readiness = benchmark.check_readiness()

    assert readiness["ready_for_verification"] is True
    assert readiness["has_questions"] is True
    assert readiness["all_have_templates"] is True
    assert readiness["all_finished"] is True
    assert readiness["templates_valid"] is True


@pytest.mark.unit
def test_get_health_report_empty_benchmark() -> None:
    """Test get_health_report with empty benchmark."""
    benchmark = Benchmark.create(name="empty")

    report = benchmark.get_health_report()

    assert report["health_score"] == 0.0
    assert report["health_status"] == "critical"
    assert "Add questions to the benchmark" in report["recommendations"]


@pytest.mark.unit
def test_get_health_report_excellent() -> None:
    """Test get_health_report with excellent (ready) benchmark."""
    benchmark = Benchmark.create(name="test")

    template = """
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: str = Field(description="The answer")

    def verify(self) -> bool:
        return True
"""

    # Add multiple finished questions with templates
    for i in range(5):
        benchmark.add_question(f"Q{i}?", f"A{i}", question_id=f"q{i}", finished=True)
        benchmark.add_answer_template(f"q{i}", template)

    report = benchmark.get_health_report()

    assert report["health_score"] >= 90
    assert report["health_status"] == "excellent"
    assert report["readiness"]["ready_for_verification"] is True


@pytest.mark.unit
def test_get_health_report_levels() -> None:
    """Test health report status levels."""
    benchmark = Benchmark.create(name="test")

    template = """
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: str = Field(description="The answer")

    def verify(self) -> bool:
        return True
"""

    # Critical: no questions
    report = benchmark.get_health_report()
    assert report["health_status"] == "critical"

    # Fair: 1 question, not finished, with template (score = 20 + 0 + 25 + 15 + 0 = 60)
    benchmark.add_question("Q1?", "A1", question_id="q1", finished=False)
    benchmark.add_answer_template("q1", template)
    report = benchmark.get_health_report()
    assert report["health_status"] == "fair"  # 60 points = fair (50-74)

    # Good: 1 question, finished with template (score = 20 + 30 + 25 + 15 + 10 = 100)
    benchmark.mark_finished("q1")
    report = benchmark.get_health_report()
    assert report["health_status"] == "excellent"  # 100 points = excellent


@pytest.mark.unit
def test_get_health_report_includes_timestamp() -> None:
    """Test get_health_report includes timestamp."""
    benchmark = Benchmark.create(name="test")

    report = benchmark.get_health_report()

    assert "timestamp" in report
    assert isinstance(report["timestamp"], str)


@pytest.mark.unit
def test_to_markdown_empty_benchmark() -> None:
    """Test to_markdown with empty benchmark."""
    benchmark = Benchmark.create(name="test")

    markdown = benchmark.to_markdown()

    assert "# test" in markdown
    assert "## Metadata" in markdown
    assert "## Statistics" in markdown
    assert "0" in markdown  # Question count


@pytest.mark.unit
def test_to_markdown_with_questions() -> None:
    """Test to_markdown includes questions."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("What is 2+2?", "4", question_id="q1", finished=True)
    benchmark.add_question("What is 3+3?", "6", question_id="q2", finished=False)

    markdown = benchmark.to_markdown()

    assert "What is 2+2?" in markdown
    assert "What is 3+3?" in markdown
    assert "## Questions" in markdown


@pytest.mark.unit
def test_to_markdown_with_global_rubric() -> None:
    """Test to_markdown includes global rubric."""
    from karenina.schemas.entities import LLMRubricTrait, Rubric

    benchmark = Benchmark.create(name="test")
    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)

    trait = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True, description="Response clarity")
    rubric = Rubric(llm_traits=[trait])
    benchmark.set_global_rubric(rubric)

    markdown = benchmark.to_markdown()

    assert "## Global Rubric" in markdown
    assert "clarity" in markdown


@pytest.mark.unit
def test_get_progress() -> None:
    """Test get_progress method."""
    benchmark = Benchmark.create(name="test")

    assert benchmark.get_progress() == 0.0

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False)

    # 1 of 2 = 50%
    assert benchmark.get_progress() == 50.0


@pytest.mark.unit
def test_is_complete_property() -> None:
    """Test is_complete property."""
    benchmark = Benchmark.create(name="test")

    assert benchmark.is_complete is False

    template = """
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: str = Field(description="The answer")

    def verify(self) -> bool:
        return True
"""

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_answer_template("q1", template)

    assert benchmark.is_complete is True


@pytest.mark.unit
def test_question_count_property() -> None:
    """Test question_count property."""
    benchmark = Benchmark.create(name="test")

    assert benchmark.question_count == 0

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=True)

    assert benchmark.question_count == 2


@pytest.mark.unit
def test_finished_count_property() -> None:
    """Test finished_count property."""
    benchmark = Benchmark.create(name="test")

    assert benchmark.finished_count == 0

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False)
    benchmark.add_question("Q3?", "A3", question_id="q3", finished=True)

    assert benchmark.finished_count == 2


@pytest.mark.unit
def test_to_dict_roundtrip() -> None:
    """Test that to_dict output can be used to recreate benchmark structure."""
    benchmark = Benchmark.create(name="test", description="Test benchmark")

    benchmark.add_question("What is 2+2?", "4", question_id="q1", finished=True)

    result = benchmark.to_dict()

    # Verify all expected keys are present
    assert "metadata" in result
    assert "statistics" in result
    assert "questions" in result
    assert "global_rubric" in result

    # Verify metadata structure
    assert result["metadata"]["name"] == "test"
    assert result["metadata"]["description"] == "Test benchmark"
    assert result["metadata"]["version"] == "0.1.0"
