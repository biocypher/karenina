"""Integration test for CSV question extraction and template generation.

This test validates the complete workflow of:
1. Extracting questions from a CSV file
2. Generating answer templates using a custom OpenAI-compatible endpoint
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from karenina.domain.answers.generator import generate_answer_template
from karenina.domain.questions.extractor import extract_questions_from_file
from karenina.schemas.workflow import ModelConfig


@pytest.fixture
def csv_file_path(fixtures_dir: Path) -> Path:
    """Path to the test CSV fixture file."""
    return fixtures_dir / "data" / "sample_questions.csv"


@pytest.fixture
def openai_endpoint_config():
    """Configuration for the OpenAI-compatible endpoint."""
    return ModelConfig(
        id="glm-4.6-generator",
        model_name="glm-4.6",
        model_provider="openai",  # Not used for openai_endpoint interface
        interface="openai_endpoint",
        temperature=0.0,
        system_prompt="",  # Not used in template generation
        endpoint_base_url="http://codon-gpu-001.ebi.ac.uk:8000/v1",
        endpoint_api_key=SecretStr("mock"),
    )


@pytest.fixture
def mock_structured_output():
    """Mock structured output from _generate_structured_outputs."""
    return {
        "attributes": [
            {
                "name": "mentions_paris",
                "type": "bool",
                "ground_truth": True,
            }
        ],
        "field_descriptions": {
            "mentions_paris": "Answer with true if the response mentions Paris as the capital; otherwise answer false."
        },
    }


def test_csv_extraction(csv_file_path: Path):
    """Test that questions can be extracted from the CSV file."""
    # Extract questions from CSV
    questions_with_metadata = extract_questions_from_file(
        file_path=str(csv_file_path),
        question_column="Question",
        answer_column="Answer",
    )

    # Verify we got questions
    assert len(questions_with_metadata) > 0, "No questions extracted from CSV"

    # Check the structure of the first question
    first_question, _ = questions_with_metadata[0]

    assert first_question.question, "Question text is empty"
    assert first_question.raw_answer, "Answer text is empty"
    assert first_question.id, "Question ID not generated"


def test_csv_file_exists(csv_file_path: Path):
    """Verify the CSV file exists."""
    assert csv_file_path.exists(), f"CSV file not found at {csv_file_path}"
    assert csv_file_path.suffix == ".csv", f"File is not a CSV: {csv_file_path}"


def test_csv_has_required_columns(csv_file_path: Path):
    """Verify the CSV has the required Question and Answer columns."""
    import pandas as pd

    df = pd.read_csv(csv_file_path)

    assert "Question" in df.columns, "CSV missing 'Question' column"
    assert "Answer" in df.columns, "CSV missing 'Answer' column"

    # Check we have data
    assert len(df) > 0, "CSV file is empty"

    # Check for non-null values
    question_count = df["Question"].notna().sum()
    answer_count = df["Answer"].notna().sum()

    assert question_count > 0, "No questions with content"
    assert answer_count > 0, "No answers with content"


@pytest.mark.integration
def test_template_generation_with_openai_endpoint(
    csv_file_path: Path, openai_endpoint_config, mock_structured_output
):
    """Test template generation using mocked OpenAI-compatible endpoint."""
    # Extract questions from CSV
    questions_with_metadata = extract_questions_from_file(
        file_path=str(csv_file_path),
        question_column="Question",
        answer_column="Answer",
    )

    assert len(questions_with_metadata) > 0, "No questions extracted"

    # Take the first question for testing
    first_question, _ = questions_with_metadata[0]

    # Mock the structured output generation (bypasses LLM calls)
    with patch(
        "karenina.domain.answers.generator._generate_structured_outputs"
    ) as mock_gen:
        mock_gen.return_value = mock_structured_output

        # Generate template using the mocked endpoint
        template_code = generate_answer_template(
            question=first_question.question,
            raw_answer=first_question.raw_answer,
            config=openai_endpoint_config,
        )

        # Verify the template was generated
        assert template_code, "Template code is empty"
        assert "class Answer(BaseAnswer):" in template_code, "Template doesn't contain Answer class"
        assert "def verify(self)" in template_code, "Template doesn't contain verify method"


@pytest.mark.slow
@pytest.mark.integration
def test_batch_template_generation(
    csv_file_path: Path, openai_endpoint_config, mock_structured_output
):
    """Test template generation for multiple questions using mocked endpoint."""
    # Extract questions from CSV
    questions_with_metadata = extract_questions_from_file(
        file_path=str(csv_file_path),
        question_column="Question",
        answer_column="Answer",
    )

    # Test with first 3 questions
    num_questions = min(3, len(questions_with_metadata))

    templates = {}
    # Mock the structured output generation (bypasses LLM calls)
    with patch(
        "karenina.domain.answers.generator._generate_structured_outputs"
    ) as mock_gen:
        mock_gen.return_value = mock_structured_output

        for i in range(num_questions):
            question, _ = questions_with_metadata[i]

            template_code = generate_answer_template(
                question=question.question,
                raw_answer=question.raw_answer,
                config=openai_endpoint_config,
            )

            templates[question.id] = template_code

    # Verify all templates were generated
    assert len(templates) == num_questions, f"Expected {num_questions} templates, got {len(templates)}"


@pytest.mark.skip(reason="Debug test - only run manually when investigating model response issues")
def test_raw_model_response_debug(csv_file_path: Path, openai_endpoint_config):
    """Debug test to capture and display raw model responses for both phases."""
    # This test is skipped by default - only for manual debugging
    pass


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
