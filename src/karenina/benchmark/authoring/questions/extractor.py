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
    answer_notes_column: str | None = None,
    custom_metadata_columns: list[str] | None = None,
) -> list[Question]:
    """Extract questions from a file with flexible column selection.

    Keywords, author info, and custom metadata columns are populated directly
    on each Question object.

    Args:
        file_path: Path to the file (CSV, TSV, or Excel)
        question_column: Name of the column containing questions
        answer_column: Name of the column containing answers
        sheet_name: Sheet name for Excel files (optional)
        author_name_column: Column for author names, populates Question.author
        author_email_column: Column for author emails, merged into Question.author
        author_affiliation_column: Column for author affiliations, merged into Question.author
        url_column: Column for URLs, stored in Question.custom_metadata["url"]
        keywords_columns: Keyword column configs with separators,
            e.g., [{"column": "Area", "separator": ","}]
        answer_notes_column: Column for answer interpretation notes
        custom_metadata_columns: Column names whose values are stored in
            Question.custom_metadata keyed by column name. Types are inferred
            by pandas (floats stay floats, strings stay strings).

    Returns:
        List of Question objects.

    Raises:
        ValueError: If required columns are missing
    """
    df = read_file_to_dataframe(file_path, sheet_name)

    # Validate required columns
    required_columns = [question_column, answer_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in file: {missing_columns}")

    # Collect all columns we need to read
    columns_to_use = list(required_columns)

    # Author columns
    author_col_map: dict[str, str] = {}  # schema key -> column name
    for schema_key, col_param in [
        ("name", author_name_column),
        ("email", author_email_column),
        ("affiliation", author_affiliation_column),
    ]:
        if col_param and col_param in df.columns:
            columns_to_use.append(col_param)
            author_col_map[schema_key] = col_param

    # URL column
    url_col: str | None = None
    if url_column and url_column in df.columns:
        columns_to_use.append(url_column)
        url_col = url_column

    # Answer notes column
    notes_col: str | None = None
    if answer_notes_column and answer_notes_column in df.columns:
        columns_to_use.append(answer_notes_column)
        notes_col = answer_notes_column

    # Keyword columns
    keyword_cols_info: list[dict[str, str]] = []
    if keywords_columns:
        for kw_config in keywords_columns:
            col_name = kw_config.get("column")
            separator = kw_config.get("separator", ",")
            if col_name and col_name in df.columns:
                columns_to_use.append(col_name)
                keyword_cols_info.append({"column": col_name, "separator": separator})

    # Custom metadata columns
    valid_custom_cols: list[str] = []
    if custom_metadata_columns:
        for col_name in custom_metadata_columns:
            if col_name in df.columns:
                columns_to_use.append(col_name)
                valid_custom_cols.append(col_name)

    # Deduplicate columns_to_use (a column could appear in multiple params)
    columns_to_use = list(dict.fromkeys(columns_to_use))

    # Filter and clean
    df_filtered = df[columns_to_use].dropna(subset=[question_column, answer_column])
    df_filtered[question_column] = df_filtered[question_column].astype(str).str.strip()
    df_filtered[answer_column] = df_filtered[answer_column].astype(str).str.strip()
    df_filtered = df_filtered[(df_filtered[question_column] != "") & (df_filtered[answer_column] != "")]

    # Build Question objects
    questions: list[Question] = []
    for _, row in df_filtered.iterrows():
        # Answer notes
        answer_notes_value = None
        if notes_col and pd.notna(row[notes_col]):
            answer_notes_value = str(row[notes_col]).strip() or None

        # Keywords
        question_keywords: list[str] = []
        for kw_col_info in keyword_cols_info:
            col_name = kw_col_info["column"]
            separator = kw_col_info["separator"]
            if pd.notna(row[col_name]):
                keywords_value = str(row[col_name]).strip()
                if keywords_value:
                    question_keywords.extend(k.strip() for k in keywords_value.split(separator) if k.strip())
        question_keywords = sorted(set(question_keywords))

        # Author
        author: dict[str, str] | None = None
        if author_col_map:
            author_data: dict[str, str] = {}
            for schema_key, col_name in author_col_map.items():
                if pd.notna(row[col_name]):
                    val = str(row[col_name]).strip()
                    if val:
                        author_data[schema_key] = val
            if author_data:
                author = {"@type": "Person", **author_data}

        # Custom metadata (url_column + custom_metadata_columns)
        custom_meta: dict[str, Any] = {}
        if url_col and pd.notna(row[url_col]):
            val = str(row[url_col]).strip()
            if val:
                custom_meta["url"] = val
        for col_name in valid_custom_cols:
            if pd.notna(row[col_name]):
                custom_meta[col_name] = row[col_name]

        questions.append(
            Question(
                question=row[question_column],
                raw_answer=row[answer_column],
                keywords=question_keywords,
                answer_notes=answer_notes_value,
                author=author,
                custom_metadata=custom_meta if custom_meta else None,
            )
        )

    return questions


def extract_questions_from_excel(excel_path: str) -> list[Question]:
    """Extract questions from the Easy sheet of the Excel file.

    This function is kept for backward compatibility.
    """
    return extract_questions_from_file(
        file_path=excel_path, question_column="Question", answer_column="Answer", sheet_name="Easy"
    )


def generate_questions_file(questions: list[Question], output_path: str) -> None:
    """Generate the questions.py file with all extracted questions."""

    # Create the file content
    content = """from karenina.schemas.entities import Question

# Auto-generated questions from file

"""

    # Add each question as a variable
    question_objects = []
    for i, question in enumerate(questions):
        var_name = f"question_{i + 1}"
        question_objects.append(var_name)
        answer_notes_line = ""
        if question.answer_notes:
            answer_notes_line = f",\n    answer_notes={repr(question.answer_notes)}"
        content += f"""{var_name} = Question(
    id={repr(question.id)},
    question={repr(question.question)},
    raw_answer={repr(question.raw_answer)},
    keywords={question.keywords!r}{answer_notes_line}
)

"""

    # Add a list containing all questions
    content += "# List of all questions\n"
    content += "all_questions = [\n"
    for var_name in question_objects:
        content += f"    {var_name},\n"
    content += "]\n"

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def questions_to_json(questions: list[Question]) -> dict[str, Any]:
    """Convert questions to JSON format compatible with the webapp.

    Args:
        questions: List of Question objects.

    Returns:
        Dictionary in the format expected by the webapp.
    """
    result = {}
    for question in questions:
        question_data: dict[str, Any] = {
            "question": question.question,
            "raw_answer": question.raw_answer,
        }

        if question.answer_notes:
            question_data["answer_notes"] = question.answer_notes

        if question.keywords:
            question_data["keywords"] = question.keywords

        result[question.id] = question_data
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
    answer_notes_column: str | None = None,
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
        answer_notes_column: Optional column name for answer interpretation notes

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
        answer_notes_column=answer_notes_column,
    )

    if not questions:
        raise ValueError("No valid questions found in the file")

    if return_json:
        return questions_to_json(questions)
    else:
        # Generate questions.py file
        generate_questions_file(questions, output_path)
        return None
