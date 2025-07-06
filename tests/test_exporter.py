"""Tests for the benchmark exporter module."""

import json

import pytest

from karenina.benchmark.exporter import export_verification_results_csv, export_verification_results_json
from karenina.benchmark.models import ModelConfiguration, VerificationConfig, VerificationJob, VerificationResult
from karenina.schemas.rubric_class import Rubric, RubricTrait


@pytest.fixture
def mock_global_rubric() -> Rubric:
    """Create a mock global rubric with common traits."""
    return Rubric(
        traits=[
            RubricTrait(
                name="Conciseness",
                description="Is the response concise?",
                kind="boolean",
                min_score=None,
                max_score=None,
            ),
            RubricTrait(
                name="Directness", description="Is the response direct?", kind="score", min_score=1, max_score=5
            ),
        ]
    )


@pytest.fixture
def mock_verification_job() -> VerificationJob:
    """Create a mock verification job."""
    answering_model = ModelConfiguration(
        id="answering-test",
        model_provider="openai",
        model_name="gpt-4",
        temperature=0.1,
        interface="langchain",
        system_prompt="You are a helpful assistant.",
    )

    parsing_model = ModelConfiguration(
        id="parsing-test",
        model_provider="openai",
        model_name="gpt-4",
        temperature=0.1,
        interface="langchain",
        system_prompt="Parse the response.",
    )

    config = VerificationConfig(
        answering_models=[answering_model],
        parsing_models=[parsing_model],
        replicate_count=1,
        rubric_enabled=True,
    )

    return VerificationJob(
        job_id="test-job-123",
        run_name="test-run",
        status="completed",
        config=config,
        total_questions=2,
        processed_count=2,
        successful_count=2,
        failed_count=0,
        percentage=100.0,
        start_time=1640995200.0,  # 2022-01-01 00:00:00 UTC
        end_time=1640995230.0,  # 2022-01-01 00:00:30 UTC
    )


@pytest.fixture
def mock_results_with_global_and_specific_rubrics() -> dict[str, VerificationResult]:
    """Create mock results with both global and question-specific rubrics."""
    return {
        "q1_answering-test_parsing-test": VerificationResult(
            question_id="q1",
            success=True,
            question_text="What is 2+2?",
            raw_llm_response="The answer is 4.",
            parsed_response={"answer": 4},
            verify_result=True,
            verify_granular_result={"correct": True},
            verify_rubric={
                "Conciseness": True,
                "Directness": 4,
                "specific_trait": 3,
            },
            answering_model="openai/gpt-4",
            parsing_model="openai/gpt-4",
            execution_time=1.5,
            timestamp="2022-01-01T00:00:00Z",
        ),
        "q2_answering-test_parsing-test": VerificationResult(
            question_id="q2",
            success=True,
            question_text="What is the capital of France?",
            raw_llm_response="The capital of France is Paris.",
            parsed_response={"answer": "Paris"},
            verify_result=True,
            verify_granular_result={"correct": True},
            verify_rubric={
                "Conciseness": False,
                "Directness": 2,
                "another_specific": True,
            },
            answering_model="openai/gpt-4",
            parsing_model="openai/gpt-4",
            execution_time=2.1,
            timestamp="2022-01-01T00:00:10Z",
        ),
    }


@pytest.fixture
def mock_results_global_only() -> dict[str, VerificationResult]:
    """Create mock results with only global rubrics."""
    return {
        "q1_answering-test_parsing-test": VerificationResult(
            question_id="q1",
            success=True,
            question_text="What is 2+2?",
            raw_llm_response="The answer is 4.",
            parsed_response={"answer": 4},
            verify_result=True,
            verify_granular_result={"correct": True},
            verify_rubric={
                "Conciseness": True,
                "Directness": 4,
            },
            answering_model="openai/gpt-4",
            parsing_model="openai/gpt-4",
            execution_time=1.5,
            timestamp="2022-01-01T00:00:00Z",
        ),
    }


@pytest.fixture
def mock_results_no_rubrics() -> dict[str, VerificationResult]:
    """Create mock results without any rubrics."""
    return {
        "q1_answering-test_parsing-test": VerificationResult(
            question_id="q1",
            success=True,
            question_text="What is 2+2?",
            raw_llm_response="The answer is 4.",
            parsed_response={"answer": 4},
            verify_result=True,
            verify_granular_result={"correct": True},
            verify_rubric=None,
            answering_model="openai/gpt-4",
            parsing_model="openai/gpt-4",
            execution_time=1.5,
            timestamp="2022-01-01T00:00:00Z",
        ),
    }


class TestExportVerificationResultsCSV:
    """Test cases for CSV export functionality."""

    def test_consolidate_question_specific_rubrics_with_global_rubric(
        self,
        mock_verification_job: VerificationJob,
        mock_results_with_global_and_specific_rubrics: dict[str, VerificationResult],
        mock_global_rubric: Rubric,
    ) -> None:
        """Test that question-specific rubrics are consolidated when global rubric is provided."""
        csv_content = export_verification_results_csv(
            mock_verification_job, mock_results_with_global_and_specific_rubrics, mock_global_rubric
        )

        lines = csv_content.strip().split("\n")
        headers = lines[0].split(",")

        # Check headers include global rubrics but not question-specific ones
        assert "rubric_Conciseness" in headers
        assert "rubric_Directness" in headers
        assert "question_specific_rubrics" in headers
        assert "rubric_specific_trait" not in headers
        assert "rubric_another_specific" not in headers

        # Check first row has global rubric values and consolidated question-specific
        first_row = lines[1]
        assert "True" in first_row  # Conciseness
        assert "4" in first_row  # Directness
        assert '""specific_trait"": 3' in first_row  # question-specific rubrics as JSON (CSV escaped)

        # Check second row
        second_row = lines[2]
        assert "False" in second_row  # Conciseness
        assert "2" in second_row  # Directness
        assert '""another_specific"": true' in second_row  # question-specific rubrics as JSON (CSV escaped)

    def test_all_global_rubrics_when_no_question_specific_exist(
        self, mock_verification_job, mock_results_global_only, mock_global_rubric
    ):
        """Test handling when only global rubrics exist."""
        csv_content = export_verification_results_csv(
            mock_verification_job, mock_results_global_only, mock_global_rubric
        )

        lines = csv_content.strip().split("\n")
        headers = lines[0].split(",")

        # Should not include question_specific_rubrics column when no question-specific rubrics exist
        assert "rubric_Conciseness" in headers
        assert "rubric_Directness" in headers
        assert "question_specific_rubrics" not in headers

    def test_no_global_rubric_all_rubrics_become_question_specific(
        self, mock_verification_job, mock_results_with_global_and_specific_rubrics
    ):
        """Test that all rubrics become question-specific when no global rubric is provided."""
        csv_content = export_verification_results_csv(
            mock_verification_job,
            mock_results_with_global_and_specific_rubrics,  # No global rubric
        )

        lines = csv_content.strip().split("\n")
        headers = lines[0].split(",")

        # All rubrics should be in question_specific_rubrics column
        assert "rubric_Conciseness" not in headers
        assert "rubric_Directness" not in headers
        assert "question_specific_rubrics" in headers

        # Check JSON consolidation (CSV escaped)
        first_row = lines[1]
        assert '""Conciseness"": true' in first_row
        assert '""Directness"": 4' in first_row
        assert '""specific_trait"": 3' in first_row

        second_row = lines[2]
        assert '""Conciseness"": false' in second_row
        assert '""Directness"": 2' in second_row
        assert '""another_specific"": true' in second_row

    def test_handle_empty_question_specific_rubrics(self, mock_verification_job, mock_results_global_only):
        """Test handling empty question-specific rubrics with proper JSON."""
        # Mock global rubric that includes some traits not in all results
        extended_global_rubric = Rubric(
            traits=[
                RubricTrait(
                    name="Conciseness",
                    description="Is the response concise?",
                    kind="boolean",
                    min_score=None,
                    max_score=None,
                ),
                RubricTrait(
                    name="Directness", description="Is the response direct?", kind="score", min_score=1, max_score=5
                ),
                RubricTrait(
                    name="extra_trait", description="Extra trait", kind="boolean", min_score=None, max_score=None
                ),
            ]
        )

        csv_content = export_verification_results_csv(
            mock_verification_job, mock_results_global_only, extended_global_rubric
        )

        lines = csv_content.strip().split("\n")

        # Should handle missing global traits gracefully
        first_row = lines[1]
        assert "True" in first_row  # Conciseness
        assert "4" in first_row  # Directness
        # extra_trait should be empty since it's not in the results

    def test_no_rubrics_at_all(self, mock_verification_job, mock_results_no_rubrics):
        """Test handling when no rubrics exist at all."""
        csv_content = export_verification_results_csv(mock_verification_job, mock_results_no_rubrics)

        lines = csv_content.strip().split("\n")
        headers = lines[0].split(",")

        # Should not include any rubric columns
        rubric_headers = [h for h in headers if h.startswith("rubric_") or h == "question_specific_rubrics"]
        assert len(rubric_headers) == 0

    def test_empty_question_specific_rubrics_json(
        self, mock_verification_job, mock_results_global_only, mock_global_rubric
    ):
        """Test that empty question-specific rubrics result in empty JSON object."""
        # Modify the results to have some results with empty question-specific rubrics
        results_with_mixed = mock_results_global_only.copy()
        results_with_mixed["q2_answering-test_parsing-test"] = VerificationResult(
            question_id="q2",
            success=True,
            question_text="Test question",
            raw_llm_response="Test response",
            parsed_response={"answer": "test"},
            verify_result=True,
            verify_granular_result={"correct": True},
            verify_rubric={
                "Conciseness": False,
                "Directness": 3,
                "specific_trait": 2,  # This is question-specific
            },
            answering_model="openai/gpt-4",
            parsing_model="openai/gpt-4",
            execution_time=1.2,
            timestamp="2022-01-01T00:00:05Z",
        )

        csv_content = export_verification_results_csv(mock_verification_job, results_with_mixed, mock_global_rubric)

        lines = csv_content.strip().split("\n")

        # First row should have empty question-specific rubrics
        first_row = lines[1]
        assert "{}" in first_row  # Empty JSON object for question-specific rubrics

        # Second row should have actual question-specific rubrics
        second_row = lines[2]
        assert '""specific_trait"": 2' in second_row


class TestExportVerificationResultsJSON:
    """Test cases for JSON export functionality."""

    def test_json_export_preserves_all_rubric_data(
        self, mock_verification_job, mock_results_with_global_and_specific_rubrics
    ):
        """Test that JSON export preserves all rubric data without consolidation."""
        json_content = export_verification_results_json(
            mock_verification_job, mock_results_with_global_and_specific_rubrics
        )

        data = json.loads(json_content)

        # Check that rubric data is preserved in full
        q1_result = data["results"]["q1_answering-test_parsing-test"]
        assert q1_result["verify_rubric"] == {
            "Conciseness": True,
            "Directness": 4,
            "specific_trait": 3,
        }

        q2_result = data["results"]["q2_answering-test_parsing-test"]
        assert q2_result["verify_rubric"] == {
            "Conciseness": False,
            "Directness": 2,
            "another_specific": True,
        }

    def test_json_export_includes_metadata(self, mock_verification_job, mock_results_with_global_and_specific_rubrics):
        """Test that JSON export includes comprehensive metadata."""
        json_content = export_verification_results_json(
            mock_verification_job, mock_results_with_global_and_specific_rubrics
        )

        data = json.loads(json_content)

        # Check metadata structure
        assert "metadata" in data
        assert "results" in data

        metadata = data["metadata"]
        assert "export_timestamp" in metadata
        assert "karenina_version" in metadata
        assert "job_id" in metadata
        assert metadata["job_id"] == "test-job-123"

        # Check verification config metadata
        assert "verification_config" in metadata
        assert "job_summary" in metadata

        job_summary = metadata["job_summary"]
        assert job_summary["total_questions"] == 2
        assert job_summary["successful_count"] == 2
        assert job_summary["failed_count"] == 0
