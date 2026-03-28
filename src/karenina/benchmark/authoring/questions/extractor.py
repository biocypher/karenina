#!/usr/bin/env python3
"""Question extractor for benchmark files (Excel, CSV, TSV).

Reads tabular data and returns Question objects with all metadata populated.

Usage:
    from karenina.benchmark.authoring.questions import extract_questions_from_file

    questions = extract_questions_from_file(
        file_path="data/questions.csv",
        question_column="Question",
        answer_column="Answer",
        keywords_columns=[{"column": "Area", "separator": ","}],
        custom_metadata_columns=["Complexity"],
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
