#!/usr/bin/env python3
"""
Question extractor for MCP benchmark files (Excel, CSV, TSV).

This module provides functionality to extract questions from various file formats and generate
Python files containing Question instances.

The main function extract_and_generate_questions() reads questions from files,
generates unique MD5 hash IDs, creates Question objects, and outputs a Python file
with all questions as individual variables and a list containing all questions.

Usage:
    from karenina.question_extractor import extract_and_generate_questions

    extract_and_generate_questions(
        file_path="data/questions.xlsx",
        output_path="karenina/questions.py",
        question_column="Question",
        answer_column="Answer"
    )
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..schemas.question_class import Question


def hash_question(question_text: str) -> str:
    """Generate a hash ID for a question."""
    return hashlib.md5(question_text.encode("utf-8")).hexdigest()


def read_file_to_dataframe(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Read a file (Excel, CSV, or TSV) into a pandas DataFrame.

    Args:
        file_path: Path to the file
        sheet_name: Sheet name for Excel files (optional)

    Returns:
        pandas DataFrame containing the file data

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = file_path_obj.suffix.lower()

    if file_extension in [".xlsx", ".xls"]:
        if sheet_name:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            return pd.read_excel(file_path)
    elif file_extension == ".csv":
        return pd.read_csv(file_path)
    elif file_extension in [".tsv", ".txt"]:
        return pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def get_file_preview(file_path: str, sheet_name: Optional[str] = None, max_rows: int = 100) -> Dict:
    """
    Get a preview of the file with column information and sample data.

    Args:
        file_path: Path to the file
        sheet_name: Sheet name for Excel files (optional)
        max_rows: Maximum number of rows to return in preview

    Returns:
        Dictionary containing file info, columns, and preview data
    """
    try:
        df = read_file_to_dataframe(file_path, sheet_name)

        # Get basic info
        total_rows = len(df)
        columns = df.columns.tolist()

        # Get preview data (first max_rows rows)
        preview_df = df.head(max_rows)

        # Convert to dict for JSON serialization
        preview_data = []
        for _, row in preview_df.iterrows():
            preview_data.append({col: str(val) if pd.notna(val) else "" for col, val in row.items()})

        return {
            "success": True,
            "total_rows": total_rows,
            "columns": columns,
            "preview_rows": len(preview_data),
            "data": preview_data,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def extract_questions_from_file(
    file_path: str, question_column: str, answer_column: str, sheet_name: Optional[str] = None
) -> List[Question]:
    """
    Extract questions from a file with flexible column selection.

    Args:
        file_path: Path to the file
        question_column: Name of the column containing questions
        answer_column: Name of the column containing answers
        sheet_name: Sheet name for Excel files (optional)

    Returns:
        List of Question instances

    Raises:
        ValueError: If required columns are missing
    """
    # Read the file
    df = read_file_to_dataframe(file_path, sheet_name)

    # Check if required columns exist
    required_columns = [question_column, answer_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in file: {missing_columns}")

    # Filter to only required columns and drop rows with missing data
    df_filtered = df[[question_column, answer_column]].dropna()

    # Convert to string and strip whitespace using vectorized operations
    df_filtered[question_column] = df_filtered[question_column].astype(str).str.strip()
    df_filtered[answer_column] = df_filtered[answer_column].astype(str).str.strip()

    # Filter out empty questions or answers
    df_filtered = df_filtered[(df_filtered[question_column] != "") & (df_filtered[answer_column] != "")]

    # Generate hashed IDs for all questions at once
    df_filtered["id"] = df_filtered[question_column].apply(hash_question)

    # Create Question instances using list comprehension
    questions = [
        Question(
            id=row["id"],
            question=row[question_column],
            raw_answer=row[answer_column],
            tags=[],  # No tags in the source data
        )
        for row in df_filtered.to_dict("records")
    ]

    return questions


def extract_questions_from_excel(excel_path: str) -> List[Question]:
    """
    Extract questions from the Easy sheet of the Excel file.

    This function is kept for backward compatibility.
    """
    return extract_questions_from_file(
        file_path=excel_path, question_column="Question", answer_column="Answer", sheet_name="Easy"
    )


def generate_questions_file(questions: List[Question], output_path: str):
    """Generate the questions.py file with all extracted questions."""

    # Create the file content
    content = """from karenina.schemas.question_class import Question

# Auto-generated questions from file

"""

    # Add each question as a variable
    for i, question in enumerate(questions):
        var_name = f"question_{i + 1}"
        content += f'''{var_name} = Question(
    id="{question.id}",
    question="""{question.question}""",
    raw_answer="""{question.raw_answer}""",
    tags={question.tags}
)

'''

    # Add a list containing all questions
    content += "# List of all questions\n"
    content += "all_questions = [\n"
    for i in range(len(questions)):
        content += f"    question_{i + 1},\n"
    content += "]\n"

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def questions_to_json(questions: List[Question]) -> Dict:
    """
    Convert questions to JSON format compatible with the webapp.

    Args:
        questions: List of Question instances

    Returns:
        Dictionary in the format expected by the webapp
    """
    result = {}
    for question in questions:
        result[question.id] = {
            "question": question.question,
            "raw_answer": question.raw_answer,
            # No answer_template - this should only be added after template generation
        }
    return result


def extract_and_generate_questions(
    file_path: str,
    output_path: str,
    question_column: str = "Question",
    answer_column: str = "Answer",
    sheet_name: Optional[str] = None,
    return_json: bool = False,
) -> Optional[Dict]:
    """
    Extract questions from file and generate a Python file with Question instances.

    Args:
        file_path: Path to the file containing questions (Excel, CSV, or TSV)
        output_path: Path where the generated Python file should be saved
        question_column: Name of the column containing questions
        answer_column: Name of the column containing answers
        sheet_name: Sheet name for Excel files (optional)
        return_json: If True, return JSON format instead of generating Python file

    Returns:
        If return_json is True, returns dictionary in webapp format

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing from the file
        Exception: For other errors during processing
    """

    # Validate input file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Extract questions
    questions = extract_questions_from_file(
        file_path=file_path, question_column=question_column, answer_column=answer_column, sheet_name=sheet_name
    )

    if not questions:
        raise ValueError("No valid questions found in the file")

    if return_json:
        return questions_to_json(questions)
    else:
        # Generate questions.py file
        generate_questions_file(questions, output_path)
        return None
