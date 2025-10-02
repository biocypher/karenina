"""Database views for Karenina storage.

This module defines SQL views that provide convenient aggregated queries
for common use cases like benchmark summaries, verification statistics,
and model performance analysis.
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine


def create_benchmark_summary_view(engine: Engine) -> None:
    """Create or replace the benchmark_summary_view.

    Provides a quick overview of all benchmarks with question counts and status.
    """
    view_sql = text(
        """
        CREATE OR REPLACE VIEW benchmark_summary_view AS
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
        GROUP BY b.id, b.name, b.version, b.created_at, b.updated_at
        """
    )
    with engine.begin() as conn:
        # For SQLite, use DROP VIEW IF EXISTS instead of CREATE OR REPLACE
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS benchmark_summary_view"))
            view_sql = text(
                """
                CREATE VIEW benchmark_summary_view AS
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
                GROUP BY b.id, b.name, b.version, b.created_at, b.updated_at
                """
            )
        conn.execute(view_sql)


def create_verification_run_summary_view(engine: Engine) -> None:
    """Create or replace the verification_run_summary_view.

    Aggregates verification run statistics including success rates and timing.
    """
    view_sql_sqlite = """
        CREATE VIEW verification_run_summary_view AS
        SELECT
            vr.id as run_id,
            vr.run_name,
            vr.benchmark_id,
            b.name as benchmark_name,
            vr.status,
            vr.total_questions,
            vr.processed_count,
            vr.successful_count,
            vr.failed_count,
            CASE
                WHEN vr.processed_count > 0
                THEN CAST(vr.successful_count AS REAL) / vr.processed_count * 100
                ELSE 0
            END as success_rate,
            CASE
                WHEN vr.start_time IS NOT NULL AND vr.end_time IS NOT NULL
                THEN (julianday(vr.end_time) - julianday(vr.start_time)) * 86400
                ELSE NULL
            END as total_duration_seconds,
            vr.start_time,
            vr.end_time,
            vr.created_at
        FROM verification_runs vr
        JOIN benchmarks b ON vr.benchmark_id = b.id
    """

    view_sql_postgres = """
        CREATE OR REPLACE VIEW verification_run_summary_view AS
        SELECT
            vr.id as run_id,
            vr.run_name,
            vr.benchmark_id,
            b.name as benchmark_name,
            vr.status,
            vr.total_questions,
            vr.processed_count,
            vr.successful_count,
            vr.failed_count,
            CASE
                WHEN vr.processed_count > 0
                THEN (vr.successful_count::FLOAT / vr.processed_count) * 100
                ELSE 0
            END as success_rate,
            EXTRACT(EPOCH FROM (vr.end_time - vr.start_time)) as total_duration_seconds,
            vr.start_time,
            vr.end_time,
            vr.created_at
        FROM verification_runs vr
        JOIN benchmarks b ON vr.benchmark_id = b.id
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS verification_run_summary_view"))
            conn.execute(text(view_sql_sqlite))
        else:
            conn.execute(text(view_sql_postgres))


def create_question_usage_view(engine: Engine) -> None:
    """Create or replace the question_usage_view.

    Shows which benchmarks use each question.
    """
    view_sql_base = """
        SELECT
            q.id as question_id,
            q.question_text,
            COUNT(DISTINCT bq.benchmark_id) as benchmark_count,
            GROUP_CONCAT(DISTINCT b.name) as benchmark_names
        FROM questions q
        LEFT JOIN benchmark_questions bq ON q.id = bq.question_id
        LEFT JOIN benchmarks b ON bq.benchmark_id = b.id
        GROUP BY q.id, q.question_text
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS question_usage_view"))
            conn.execute(
                text(
                    f"""
                CREATE VIEW question_usage_view AS
                {view_sql_base}
            """
                )
            )
        else:
            # PostgreSQL uses STRING_AGG instead of GROUP_CONCAT
            view_sql = """
                CREATE OR REPLACE VIEW question_usage_view AS
                SELECT
                    q.id as question_id,
                    q.question_text,
                    COUNT(DISTINCT bq.benchmark_id) as benchmark_count,
                    STRING_AGG(DISTINCT b.name, ', ') as benchmark_names
                FROM questions q
                LEFT JOIN benchmark_questions bq ON q.id = bq.question_id
                LEFT JOIN benchmarks b ON bq.benchmark_id = b.id
                GROUP BY q.id, q.question_text
            """
            conn.execute(text(view_sql))


def create_latest_verification_results_view(engine: Engine) -> None:
    """Create or replace the latest_verification_results_view.

    Shows the most recent verification result for each question-model combination.
    """
    view_sql = """
        SELECT
            vres.*,
            vr.run_name,
            vr.benchmark_id,
            b.name as benchmark_name
        FROM verification_results vres
        JOIN verification_runs vr ON vres.run_id = vr.id
        JOIN benchmarks b ON vr.benchmark_id = b.id
        WHERE vres.id IN (
            SELECT vr2.id
            FROM verification_results vr2
            JOIN verification_runs run2 ON vr2.run_id = run2.id
            WHERE vr2.question_id = vres.question_id
              AND vr2.answering_model = vres.answering_model
              AND vr2.parsing_model = vres.parsing_model
              AND run2.benchmark_id = vr.benchmark_id
            ORDER BY vr2.created_at DESC
            LIMIT 1
        )
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS latest_verification_results_view"))
            conn.execute(text(f"CREATE VIEW latest_verification_results_view AS {view_sql}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW latest_verification_results_view AS {view_sql}"))


def create_model_performance_view(engine: Engine) -> None:
    """Create or replace the model_performance_view.

    Aggregates performance statistics per model.
    """
    view_sql = """
        SELECT
            model_name,
            model_role,
            total_runs,
            successful_runs,
            failed_runs,
            CASE
                WHEN total_runs > 0
                THEN success_rate
                ELSE 0
            END as success_rate_pct,
            avg_execution_time
        FROM (
            SELECT
                answering_model as model_name,
                'answering' as model_role,
                COUNT(*) as total_runs,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_runs,
                SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed_runs,
                AVG(execution_time) as avg_execution_time
            FROM verification_results
            GROUP BY answering_model

            UNION ALL

            SELECT
                parsing_model as model_name,
                'parsing' as model_role,
                COUNT(*) as total_runs,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_runs,
                SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed_runs,
                AVG(execution_time) as avg_execution_time
            FROM verification_results
            GROUP BY parsing_model
        ) model_stats
    """

    view_sql_with_rate_sqlite = f"""
        SELECT
            model_name,
            model_role,
            total_runs,
            successful_runs,
            failed_runs,
            CASE
                WHEN total_runs > 0
                THEN CAST(successful_runs AS REAL) / total_runs * 100
                ELSE 0
            END as success_rate_pct,
            avg_execution_time
        FROM (
            {view_sql.split("FROM (")[1].split(") model_stats")[0]}
        ) model_stats
    """

    view_sql_with_rate_postgres = f"""
        SELECT
            model_name,
            model_role,
            total_runs,
            successful_runs,
            failed_runs,
            CASE
                WHEN total_runs > 0
                THEN (successful_runs::FLOAT / total_runs) * 100
                ELSE 0
            END as success_rate_pct,
            avg_execution_time
        FROM (
            {view_sql.split("FROM (")[1].split(") model_stats")[0]}
        ) model_stats
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS model_performance_view"))
            conn.execute(text(f"CREATE VIEW model_performance_view AS {view_sql_with_rate_sqlite}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW model_performance_view AS {view_sql_with_rate_postgres}"))


def create_rubric_scores_aggregate_view(engine: Engine) -> None:
    """Create or replace the rubric_scores_aggregate_view.

    Aggregates average rubric scores per question across all runs.
    """
    # This view is complex because rubric scores are stored as JSON
    # We'll create a simplified version that just shows stats
    view_sql = """
        SELECT
            vr.question_id,
            q.question_text,
            COUNT(*) as total_evaluations,
            SUM(CASE WHEN vr.verify_rubric IS NOT NULL THEN 1 ELSE 0 END) as rubric_evaluations_count,
            AVG(vr.execution_time) as avg_execution_time
        FROM verification_results vr
        JOIN questions q ON vr.question_id = q.id
        GROUP BY vr.question_id, q.question_text
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS rubric_scores_aggregate_view"))
            conn.execute(text(f"CREATE VIEW rubric_scores_aggregate_view AS {view_sql}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW rubric_scores_aggregate_view AS {view_sql}"))


def create_verification_history_timeline_view(engine: Engine) -> None:
    """Create or replace the verification_history_timeline_view.

    Chronological timeline of all verification runs with key metrics.
    """
    view_sql = """
        SELECT
            vr.id as run_id,
            vr.run_name,
            vr.benchmark_id,
            b.name as benchmark_name,
            vr.status,
            vr.total_questions,
            vr.successful_count,
            vr.failed_count,
            vr.start_time,
            vr.end_time,
            vr.created_at,
            COUNT(DISTINCT res.answering_model) as unique_answering_models,
            COUNT(DISTINCT res.parsing_model) as unique_parsing_models
        FROM verification_runs vr
        JOIN benchmarks b ON vr.benchmark_id = b.id
        LEFT JOIN verification_results res ON vr.id = res.run_id
        GROUP BY vr.id, vr.run_name, vr.benchmark_id, b.name, vr.status,
                 vr.total_questions, vr.successful_count, vr.failed_count,
                 vr.start_time, vr.end_time, vr.created_at
        ORDER BY vr.created_at DESC
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS verification_history_timeline_view"))
            conn.execute(text(f"CREATE VIEW verification_history_timeline_view AS {view_sql}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW verification_history_timeline_view AS {view_sql}"))


def create_failed_verifications_view(engine: Engine) -> None:
    """Create or replace the failed_verifications_view.

    All failed verifications with error messages for debugging.
    """
    view_sql = """
        SELECT
            vr.id as result_id,
            vr.run_id,
            vrun.run_name,
            vrun.benchmark_id,
            b.name as benchmark_name,
            vr.question_id,
            vr.question_text,
            vr.answering_model,
            vr.parsing_model,
            vr.error,
            vr.raw_llm_response,
            vr.execution_time,
            vr.timestamp,
            vr.created_at
        FROM verification_results vr
        JOIN verification_runs vrun ON vr.run_id = vrun.id
        JOIN benchmarks b ON vrun.benchmark_id = b.id
        WHERE vr.success = 0
        ORDER BY vr.created_at DESC
    """

    # SQLite uses 0/1 for boolean, PostgreSQL uses FALSE
    view_sql_postgres = view_sql.replace("vr.success = 0", "vr.success = FALSE")

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS failed_verifications_view"))
            conn.execute(text(f"CREATE VIEW failed_verifications_view AS {view_sql}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW failed_verifications_view AS {view_sql_postgres}"))


def create_rubric_traits_by_type_view(engine: Engine) -> None:
    """Create or replace the rubric_traits_by_type_view.

    Categorizes all rubric traits by type (Global/QuestionSpecific, LLM/Manual).
    This view extracts rubric traits from the benchmark_questions table where
    they are stored as JSON.
    """
    # Note: This is a simplified view. Full JSON extraction would require
    # database-specific JSON functions. This provides the structure.
    view_sql = """
        SELECT
            bq.benchmark_id,
            b.name as benchmark_name,
            bq.question_id,
            q.question_text,
            bq.question_rubric,
            bq.created_at
        FROM benchmark_questions bq
        JOIN benchmarks b ON bq.benchmark_id = b.id
        JOIN questions q ON bq.question_id = q.id
        WHERE bq.question_rubric IS NOT NULL
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS rubric_traits_by_type_view"))
            conn.execute(text(f"CREATE VIEW rubric_traits_by_type_view AS {view_sql}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW rubric_traits_by_type_view AS {view_sql}"))


def create_all_views(engine: Engine) -> None:
    """Create all database views.

    Args:
        engine: SQLAlchemy engine instance
    """
    create_benchmark_summary_view(engine)
    create_verification_run_summary_view(engine)
    create_question_usage_view(engine)
    create_latest_verification_results_view(engine)
    create_model_performance_view(engine)
    create_rubric_scores_aggregate_view(engine)
    create_verification_history_timeline_view(engine)
    create_failed_verifications_view(engine)
    create_rubric_traits_by_type_view(engine)


def drop_all_views(engine: Engine) -> None:
    """Drop all database views.

    Args:
        engine: SQLAlchemy engine instance
    """
    views = [
        "benchmark_summary_view",
        "verification_run_summary_view",
        "question_usage_view",
        "latest_verification_results_view",
        "model_performance_view",
        "rubric_scores_aggregate_view",
        "verification_history_timeline_view",
        "failed_verifications_view",
        "rubric_traits_by_type_view",
    ]

    with engine.begin() as conn:
        for view in views:
            conn.execute(text(f"DROP VIEW IF EXISTS {view}"))
