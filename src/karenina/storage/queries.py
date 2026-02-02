"""Query helpers for database views.

This module provides convenient functions for querying the database views
to get aggregated statistics and summaries.
"""

from typing import Any

from sqlalchemy import text

from .db_config import DBConfig
from .engine import get_session


def get_benchmark_summary(db_config: DBConfig, benchmark_name: str | None = None) -> list[dict[str, Any]]:
    """Get benchmark summary by querying tables directly.

    Args:
        db_config: Database configuration
        benchmark_name: Optional benchmark name to filter by

    Returns:
        List of benchmark summary dictionaries
    """
    with get_session(db_config) as session:
        query = """
            SELECT
                b.id as benchmark_id,
                b.name as benchmark_name,
                b.version,
                b.created_at,
                b.updated_at,
                COUNT(bq.id) as total_questions,
                SUM(CASE WHEN bq.finished THEN 1 ELSE 0 END) as finished_count,
                SUM(CASE WHEN NOT bq.finished THEN 1 ELSE 0 END) as unfinished_count
            FROM benchmarks b
            LEFT JOIN benchmark_questions bq ON b.id = bq.benchmark_id
        """
        if benchmark_name:
            query += f" WHERE b.name = '{benchmark_name}'"
        query += " GROUP BY b.id, b.name, b.version, b.created_at, b.updated_at"

        result = session.execute(text(query))
        return [dict(row._mapping) for row in result]


def get_verification_run_summary(db_config: DBConfig, run_name: str | None = None) -> list[dict[str, Any]]:
    """Get verification run summary from the verification_run_summary_view.

    Args:
        db_config: Database configuration
        run_name: Optional run name to filter by

    Returns:
        List of verification run summary dictionaries
    """
    with get_session(db_config) as session:
        query = "SELECT * FROM verification_run_summary_view"
        if run_name:
            query += f" WHERE run_name = '{run_name}'"
        query += " ORDER BY created_at DESC"

        result = session.execute(text(query))
        return [dict(row._mapping) for row in result]


def get_question_usage(db_config: DBConfig, min_benchmark_count: int = 0) -> list[dict[str, Any]]:
    """Get question usage statistics from the question_usage_view.

    Args:
        db_config: Database configuration
        min_benchmark_count: Minimum number of benchmarks using the question

    Returns:
        List of question usage dictionaries
    """
    with get_session(db_config) as session:
        query = "SELECT * FROM question_usage_view"
        if min_benchmark_count > 0:
            query += f" WHERE benchmark_count >= {min_benchmark_count}"
        query += " ORDER BY benchmark_count DESC"

        result = session.execute(text(query))
        return [dict(row._mapping) for row in result]


def get_latest_verification_results(
    db_config: DBConfig,
    benchmark_name: str | None = None,
    question_id: str | None = None,
) -> list[dict[str, Any]]:
    """Get latest verification results from the latest_verification_results_view.

    Args:
        db_config: Database configuration
        benchmark_name: Optional benchmark name to filter by
        question_id: Optional question ID to filter by

    Returns:
        List of latest verification result dictionaries
    """
    with get_session(db_config) as session:
        query = "SELECT * FROM latest_verification_results_view WHERE 1=1"
        if benchmark_name:
            query += f" AND benchmark_name = '{benchmark_name}'"
        if question_id:
            query += f" AND question_id = '{question_id}'"
        query += " ORDER BY created_at DESC"

        result = session.execute(text(query))
        return [dict(row._mapping) for row in result]


def get_model_performance(
    db_config: DBConfig,
    model_name: str | None = None,
    model_role: str | None = None,
) -> list[dict[str, Any]]:
    """Get model performance statistics from the model_performance_view.

    Args:
        db_config: Database configuration
        model_name: Optional model name to filter by
        model_role: Optional model role to filter by ('answering' or 'parsing')

    Returns:
        List of model performance dictionaries
    """
    with get_session(db_config) as session:
        query = "SELECT * FROM model_performance_view WHERE 1=1"
        if model_name:
            query += f" AND model_name = '{model_name}'"
        if model_role:
            query += f" AND model_role = '{model_role}'"
        query += " ORDER BY total_runs DESC"

        result = session.execute(text(query))
        return [dict(row._mapping) for row in result]


def get_rubric_scores_aggregate(
    db_config: DBConfig,
    question_id: str | None = None,
    min_evaluations: int = 0,
) -> list[dict[str, Any]]:
    """Get aggregated rubric scores from the rubric_scores_aggregate_view.

    Args:
        db_config: Database configuration
        question_id: Optional question ID to filter by
        min_evaluations: Minimum number of evaluations required

    Returns:
        List of rubric score aggregate dictionaries
    """
    with get_session(db_config) as session:
        query = "SELECT * FROM rubric_scores_aggregate_view WHERE 1=1"
        if question_id:
            query += f" AND question_id = '{question_id}'"
        if min_evaluations > 0:
            query += f" AND rubric_evaluations_count >= {min_evaluations}"
        query += " ORDER BY total_evaluations DESC"

        result = session.execute(text(query))
        return [dict(row._mapping) for row in result]


def get_verification_history_timeline(
    db_config: DBConfig,
    benchmark_name: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get verification history timeline from the verification_history_timeline_view.

    Args:
        db_config: Database configuration
        benchmark_name: Optional benchmark name to filter by
        limit: Maximum number of records to return

    Returns:
        List of verification history dictionaries
    """
    with get_session(db_config) as session:
        query = "SELECT * FROM verification_history_timeline_view WHERE 1=1"
        if benchmark_name:
            query += f" AND benchmark_name = '{benchmark_name}'"
        query += f" ORDER BY created_at DESC LIMIT {limit}"

        result = session.execute(text(query))
        return [dict(row._mapping) for row in result]


def get_rubric_traits_by_type(
    db_config: DBConfig,
    benchmark_name: str | None = None,
    question_id: str | None = None,
) -> list[dict[str, Any]]:
    """Get rubric traits categorized by type from the rubric_traits_by_type_view.

    Args:
        db_config: Database configuration
        benchmark_name: Optional benchmark name to filter by
        question_id: Optional question ID to filter by

    Returns:
        List of rubric trait dictionaries
    """
    with get_session(db_config) as session:
        query = "SELECT * FROM rubric_traits_by_type_view WHERE 1=1"
        if benchmark_name:
            query += f" AND benchmark_name = '{benchmark_name}'"
        if question_id:
            query += f" AND question_id = '{question_id}'"
        query += " ORDER BY benchmark_name, question_id"

        result = session.execute(text(query))
        return [dict(row._mapping) for row in result]


def get_database_statistics(db_config: DBConfig) -> dict[str, Any]:
    """Get overall database statistics.

    Args:
        db_config: Database configuration

    Returns:
        Dictionary with database statistics
    """
    with get_session(db_config) as session:
        stats = {}

        # Count benchmarks
        result = session.execute(text("SELECT COUNT(*) as count FROM benchmarks"))
        stats["total_benchmarks"] = result.scalar()

        # Count questions
        result = session.execute(text("SELECT COUNT(*) as count FROM questions"))
        stats["total_questions"] = result.scalar()

        # Count verification runs
        result = session.execute(text("SELECT COUNT(*) as count FROM verification_runs"))
        stats["total_verification_runs"] = result.scalar()

        # Count verification results
        result = session.execute(text("SELECT COUNT(*) as count FROM verification_results"))
        stats["total_verification_results"] = result.scalar()

        # Get latest verification run
        result = session.execute(
            text("SELECT run_name, created_at FROM verification_runs ORDER BY created_at DESC LIMIT 1")
        )
        latest_run = result.first()
        if latest_run:
            stats["latest_verification_run"] = latest_run._mapping["run_name"]
            stats["latest_verification_date"] = str(latest_run._mapping["created_at"])

        return stats
