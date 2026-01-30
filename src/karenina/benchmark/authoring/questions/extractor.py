#!/usr/bin/env python3
"""
Question extractor for MCP benchmark files (Excel, CSV, TSV).

This module provides functionality to extract questions from various file formats and generate
Python files containing Question instances.

The main function extract_and_generate_questions() reads questions from files,
generates unique MD5 hash IDs, creates Question objects, and outputs a Python file
with all questions as individual variables and a list containing all questions.

Usage:
    from karenina.benchmark.authoring.questions import extract_and_generate_questions

    extract_and_generate_questions(
        file_path="data/questions.xlsx",
        output_path="karenina/questions.py",
        question_column="Question",
        answer_column="Answer"
    )
"""

from pathlib import Path
from typing import Any

import pandas as pd

from karenina.schemas.entities import Question


def read_file_to_dataframe(file_path: str, sheet_name: str | None = None) -> pd.DataFrame:
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


def get_file_preview(file_path: str, sheet_name: str | None = None, max_rows: int = 100) -> dict[str, Any]:
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
    file_path: str,
    question_column: str,
    answer_column: str,
    sheet_name: str | None = None,
    author_name_column: str | None = None,
    author_email_column: str | None = None,
    author_affiliation_column: str | None = None,
    url_column: str | None = None,
    keywords_columns: list[dict[str, str]] | None = None,
    # Deprecated: kept for backward compatibility
    keywords_column: str | None = None,
    keywords_separator: str = ",",
) -> list[tuple[Question, dict[str, Any]]]:
    """
    Extract questions from a file with flexible column selection and optional metadata.

    Args:
        file_path: Path to the file
        question_column: Name of the column containing questions
        answer_column: Name of the column containing answers
        sheet_name: Sheet name for Excel files (optional)
        author_name_column: Optional column name for author names
        author_email_column: Optional column name for author emails
        author_affiliation_column: Optional column name for author affiliations
        url_column: Optional column name for URLs
        keywords_columns: Optional list of keyword column configurations with individual separators
            e.g., [{"column": "keywords1", "separator": ","}, {"column": "keywords2", "separator": ";"}]
        keywords_column: (Deprecated) Optional single column name for keywords
        keywords_separator: (Deprecated) Separator for splitting keywords (default: ",")

    Returns:
        List of tuples containing (Question, metadata_dict)

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

    # Collect all columns we want to use (required + optional metadata)
    columns_to_use = [question_column, answer_column]
    metadata_columns: dict[str, Any] = {}

    # Add metadata columns if they exist in the file
    if author_name_column and author_name_column in df.columns:
        columns_to_use.append(author_name_column)
        metadata_columns["author_name"] = author_name_column

    if author_email_column and author_email_column in df.columns:
        columns_to_use.append(author_email_column)
        metadata_columns["author_email"] = author_email_column

    if author_affiliation_column and author_affiliation_column in df.columns:
        columns_to_use.append(author_affiliation_column)
        metadata_columns["author_affiliation"] = author_affiliation_column

    if url_column and url_column in df.columns:
        columns_to_use.append(url_column)
        metadata_columns["url"] = url_column

    # Handle backward compatibility for keywords
    effective_keywords_columns = keywords_columns
    if not effective_keywords_columns and keywords_column:
        # Old format: convert to new format
        effective_keywords_columns = [{"column": keywords_column, "separator": keywords_separator}]

    # Add all keyword columns
    if effective_keywords_columns:
        keyword_cols_info = []
        for kw_config in effective_keywords_columns:
            col_name = kw_config.get("column")
            separator = kw_config.get("separator", ",")
            if col_name and col_name in df.columns:
                columns_to_use.append(col_name)
                keyword_cols_info.append({"column": col_name, "separator": separator})
        if keyword_cols_info:
            metadata_columns["keywords_columns"] = keyword_cols_info

    # Filter to only the columns we need and drop rows with missing required data
    df_filtered = df[columns_to_use].dropna(subset=[question_column, answer_column])

    # Convert to string and strip whitespace for required columns
    df_filtered[question_column] = df_filtered[question_column].astype(str).str.strip()
    df_filtered[answer_column] = df_filtered[answer_column].astype(str).str.strip()

    # Filter out empty questions or answers
    df_filtered = df_filtered[(df_filtered[question_column] != "") & (df_filtered[answer_column] != "")]

    # Create Question instances with metadata
    results = []
    for _, row in df_filtered.iterrows():
        # Create the Question
        question = Question(
            question=row[question_column],
            raw_answer=row[answer_column],
            tags=[],  # No tags in the source data
        )

        # Extract metadata
        metadata: dict[str, Any] = {}

        # Author metadata
        author_data: dict[str, str] = {}
        if "author_name" in metadata_columns and pd.notna(row[metadata_columns["author_name"]]):
            author_data["name"] = str(row[metadata_columns["author_name"]]).strip()
        if "author_email" in metadata_columns and pd.notna(row[metadata_columns["author_email"]]):
            author_data["email"] = str(row[metadata_columns["author_email"]]).strip()
        if "author_affiliation" in metadata_columns and pd.notna(row[metadata_columns["author_affiliation"]]):
            author_data["affiliation"] = str(row[metadata_columns["author_affiliation"]]).strip()

        if author_data:
            metadata["author"] = {"@type": "Person", **author_data}

        # URL metadata
        if "url" in metadata_columns and pd.notna(row[metadata_columns["url"]]):
            url_value = str(row[metadata_columns["url"]]).strip()
            if url_value:
                metadata["url"] = url_value

        # Keywords metadata - handle multiple keyword columns
        if "keywords_columns" in metadata_columns:
            all_keywords = []
            for kw_col_info in metadata_columns["keywords_columns"]:
                col_name = kw_col_info["column"]
                separator = kw_col_info["separator"]
                if pd.notna(row[col_name]):
                    keywords_value = str(row[col_name]).strip()
                    if keywords_value:
                        keywords_list = [k.strip() for k in keywords_value.split(separator) if k.strip()]
                        all_keywords.extend(keywords_list)

            # Remove duplicates and sort for consistency
            if all_keywords:
                unique_keywords = sorted(set(all_keywords))
                metadata["keywords"] = unique_keywords

        results.append((question, metadata))

    return results


def extract_questions_from_excel(excel_path: str) -> list[Question]:
    """
    Extract questions from the Easy sheet of the Excel file.

    This function is kept for backward compatibility.
    """
    results = extract_questions_from_file(
        file_path=excel_path, question_column="Question", answer_column="Answer", sheet_name="Easy"
    )
    # Extract just the Question objects for backward compatibility
    return [question for question, _ in results]


def generate_questions_file(
    questions: list[Question] | list[tuple[Question, dict[str, Any]]], output_path: str
) -> None:
    """Generate the questions.py file with all extracted questions."""

    # Create the file content
    content = """from karenina.schemas.entities import Question

# Auto-generated questions from file

"""

    # Add each question as a variable
    question_objects = []
    for i, item in enumerate(questions):
        if isinstance(item, tuple):
            question, _ = item  # Ignore metadata for Python file generation
        else:
            question = item

        var_name = f"question_{i + 1}"
        question_objects.append(var_name)
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
    for var_name in question_objects:
        content += f"    {var_name},\n"
    content += "]\n"

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def questions_to_json(questions: list[Question] | list[tuple[Question, dict[str, Any]]]) -> dict[str, Any]:
    """
    Convert questions to JSON format compatible with the webapp.

    Args:
        questions: List of Question instances or tuples of (Question, metadata_dict)

    Returns:
        Dictionary in the format expected by the webapp
    """
    result = {}
    for item in questions:
        if isinstance(item, tuple):
            # New format: (Question, metadata)
            question, metadata = item
            question_data: dict[str, Any] = {
                "question": question.question,
                "raw_answer": question.raw_answer,
                # No answer_template - this should only be added after template generation
            }

            # Add metadata if present
            if metadata:
                question_data["metadata"] = metadata

            result[question.id] = question_data
        else:
            # Legacy format: just Question
            question = item
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
    sheet_name: str | None = None,
    return_json: bool = False,
    # Optional metadata columns
    author_name_column: str | None = None,
    author_email_column: str | None = None,
    author_affiliation_column: str | None = None,
    url_column: str | None = None,
    keywords_columns: list[dict[str, str]] | None = None,
    # Deprecated: kept for backward compatibility
    keywords_column: str | None = None,
    keywords_separator: str = ",",
) -> dict[str, Any] | None:
    """
    Extract questions from file and generate a Python file with Question instances.

    Args:
        file_path: Path to the file containing questions (Excel, CSV, or TSV)
        output_path: Path where the generated Python file should be saved
        question_column: Name of the column containing questions
        answer_column: Name of the column containing answers
        sheet_name: Sheet name for Excel files (optional)
        return_json: If True, return JSON format instead of generating Python file
        author_name_column: Optional column name for author names
        author_email_column: Optional column name for author emails
        author_affiliation_column: Optional column name for author affiliations
        url_column: Optional column name for URLs
        keywords_columns: Optional list of keyword column configurations with individual separators
            e.g., [{"column": "keywords1", "separator": ","}, {"column": "keywords2", "separator": ";"}]
        keywords_column: (Deprecated) Optional single column name for keywords
        keywords_separator: (Deprecated) Separator for splitting keywords (default: ",")

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

    # Extract questions with optional metadata
    questions = extract_questions_from_file(
        file_path=file_path,
        question_column=question_column,
        answer_column=answer_column,
        sheet_name=sheet_name,
        author_name_column=author_name_column,
        author_email_column=author_email_column,
        author_affiliation_column=author_affiliation_column,
        url_column=url_column,
        keywords_columns=keywords_columns,
        keywords_column=keywords_column,
        keywords_separator=keywords_separator,
    )

    if not questions:
        raise ValueError("No valid questions found in the file")

    if return_json:
        return questions_to_json(questions)
    else:
        # Generate questions.py file
        generate_questions_file(questions, output_path)
        return None
