import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from karenina.questions.extractor import (
    extract_and_generate_questions,
    extract_questions_from_excel,
    generate_questions_file,
)


def test_question_id_hash_consistency() -> None:
    """Test that Question ID hash generation is consistent."""
    from karenina.schemas.question_class import Question

    q1 = Question(question="Test question", raw_answer="Answer", tags=[])
    q2 = Question(question="Test question", raw_answer="Answer", tags=[])

    assert q1.id == q2.id
    assert len(q1.id) == 32  # MD5 hash length


def test_extract_questions_from_excel() -> None:
    """Test extracting questions from an Excel file."""
    # Create a temporary Excel file with test data
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        df = pd.DataFrame({"Question": ["Test question 1", "Test question 2"], "Answer": ["Answer 1", "Answer 2"]})
        df.to_excel(tmp.name, sheet_name="Easy", index=False)
        tmp_path = tmp.name

    try:
        questions = extract_questions_from_excel(tmp_path)
        assert len(questions) == 2
        assert questions[0].question == "Test question 1"
        assert questions[0].raw_answer == "Answer 1"
        assert questions[1].question == "Test question 2"
        assert questions[1].raw_answer == "Answer 2"
        assert all(q.tags == [] for q in questions)
    finally:
        os.unlink(tmp_path)


def test_extract_questions_from_excel_missing_columns() -> None:
    """Test that missing required columns raise ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        df = pd.DataFrame({"WrongColumn": ["Test"]})
        df.to_excel(tmp.name, sheet_name="Easy", index=False)
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError):
            extract_questions_from_excel(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_generate_questions_file() -> None:
    """Test generating a Python file with questions."""
    from karenina.schemas.question_class import Question

    questions = [
        Question(
            question="Test question 1",
            raw_answer="Answer 1",
            tags=[],
        ),
        Question(
            question="Test question 2",
            raw_answer="Answer 2",
            tags=[],
        ),
    ]

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        generate_questions_file(questions, tmp.name)
        tmp_path = tmp.name

    try:
        # Read the generated file
        with open(tmp_path) as f:
            content = f.read()

        # Check that the file contains the expected content
        assert "from karenina.schemas.question_class import Question" in content
        assert "question_1" in content
        assert "question_2" in content
        assert "all_questions" in content
        assert "Test question 1" in content
        assert "Answer 1" in content
    finally:
        os.unlink(tmp_path)


def test_extract_and_generate_questions() -> None:
    """Test the complete question extraction and generation process."""
    # Create a temporary Excel file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as excel_tmp:
        df = pd.DataFrame({"Question": ["Test question 1", "Test question 2"], "Answer": ["Answer 1", "Answer 2"]})
        df.to_excel(excel_tmp.name, sheet_name="Easy", index=False)
        excel_path = excel_tmp.name

    # Create a temporary output file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as output_tmp:
        output_path = output_tmp.name

    try:
        # Run the extraction and generation
        extract_and_generate_questions(excel_path, output_path)

        # Verify the output file exists and has content
        assert Path(output_path).exists()
        with open(output_path) as f:
            content = f.read()
            assert "from karenina.schemas.question_class import Question" in content
            assert "Test question 1" in content
            assert "Answer 1" in content
            assert "all_questions" in content
    finally:
        os.unlink(excel_path)
        os.unlink(output_path)


def test_extract_and_generate_questions_file_not_found() -> None:
    """Test that FileNotFoundError is raised for non-existent Excel file."""
    with pytest.raises(FileNotFoundError) as exc_info:
        extract_and_generate_questions("/path/that/does/not/exist.xlsx", "output.py")

    assert "File not found" in str(exc_info.value)


def test_extract_and_generate_questions_no_valid_questions() -> None:
    """Test that ValueError is raised when no valid questions are found."""
    # Create an Excel file with empty/invalid data
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as excel_tmp:
        df = pd.DataFrame({"Question": ["", None], "Answer": ["", None]})
        df.to_excel(excel_tmp.name, sheet_name="Easy", index=False)
        excel_path = excel_tmp.name

    try:
        with pytest.raises(ValueError) as exc_info:
            extract_and_generate_questions(excel_path, "output.py")

        assert "No valid questions found" in str(exc_info.value)
    finally:
        os.unlink(excel_path)
