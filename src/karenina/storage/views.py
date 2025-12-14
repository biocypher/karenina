"""Database views for Karenina storage.

This module defines SQL views that provide convenient aggregated queries
for common use cases like benchmark summaries, verification statistics,
and model performance analysis.

All views use the flattened column naming scheme from the auto-generated
VerificationResultModel (e.g., metadata_answering_model, template_verify_result).
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
    Uses flattened column names (metadata_*).
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
              AND vr2.metadata_answering_model = vres.metadata_answering_model
              AND vr2.metadata_parsing_model = vres.metadata_parsing_model
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

    Aggregates performance statistics per model using flattened column names.
    """
    inner_sql = """
        SELECT
            metadata_answering_model as model_name,
            'answering' as model_role,
            COUNT(*) as total_runs,
            SUM(CASE WHEN metadata_completed_without_errors THEN 1 ELSE 0 END) as successful_runs,
            SUM(CASE WHEN NOT metadata_completed_without_errors THEN 1 ELSE 0 END) as failed_runs,
            AVG(metadata_execution_time) as avg_execution_time
        FROM verification_results
        GROUP BY metadata_answering_model

        UNION ALL

        SELECT
            metadata_parsing_model as model_name,
            'parsing' as model_role,
            COUNT(*) as total_runs,
            SUM(CASE WHEN metadata_completed_without_errors THEN 1 ELSE 0 END) as successful_runs,
            SUM(CASE WHEN NOT metadata_completed_without_errors THEN 1 ELSE 0 END) as failed_runs,
            AVG(metadata_execution_time) as avg_execution_time
        FROM verification_results
        GROUP BY metadata_parsing_model
    """

    view_sql_sqlite = f"""
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
            {inner_sql}
        ) model_stats
    """

    view_sql_postgres = f"""
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
            {inner_sql}
        ) model_stats
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS model_performance_view"))
            conn.execute(text(f"CREATE VIEW model_performance_view AS {view_sql_sqlite}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW model_performance_view AS {view_sql_postgres}"))


def create_rubric_scores_aggregate_view(engine: Engine) -> None:
    """Create or replace the rubric_scores_aggregate_view.

    Aggregates rubric evaluation counts per question using flattened column names.
    """
    view_sql = """
        SELECT
            vr.question_id,
            q.question_text,
            COUNT(*) as total_evaluations,
            SUM(CASE WHEN vr.rubric_rubric_evaluation_performed THEN 1 ELSE 0 END) as rubric_evaluations_count,
            SUM(CASE WHEN vr.djr_deep_judgment_rubric_performed THEN 1 ELSE 0 END) as deep_judgment_rubric_count,
            AVG(vr.metadata_execution_time) as avg_execution_time
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
    Uses flattened column names.
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
            COUNT(DISTINCT res.metadata_answering_model) as unique_answering_models,
            COUNT(DISTINCT res.metadata_parsing_model) as unique_parsing_models
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
    Uses flattened column names.
    """
    view_sql = """
        SELECT
            vr.id as result_id,
            vr.run_id,
            vrun.run_name,
            vrun.benchmark_id,
            b.name as benchmark_name,
            vr.question_id,
            vr.metadata_question_text as question_text,
            vr.metadata_answering_model as answering_model,
            vr.metadata_parsing_model as parsing_model,
            vr.metadata_error as error,
            vr.template_raw_llm_response as raw_llm_response,
            vr.metadata_execution_time as execution_time,
            vr.metadata_timestamp as timestamp,
            vr.created_at
        FROM verification_results vr
        JOIN verification_runs vrun ON vr.run_id = vrun.id
        JOIN benchmarks b ON vrun.benchmark_id = b.id
        WHERE vr.metadata_completed_without_errors = 0
        ORDER BY vr.created_at DESC
    """

    # SQLite uses 0/1 for boolean, PostgreSQL uses FALSE
    view_sql_postgres = view_sql.replace(
        "vr.metadata_completed_without_errors = 0", "vr.metadata_completed_without_errors = FALSE"
    )

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


# ============================================================================
# MCP Query Views - Optimized for quick inspection via MCP tools
# ============================================================================


def create_model_comparison_view(engine: Engine) -> None:
    """Create or replace the model_comparison_view.

    Compares pass rates across different model combinations.
    Ideal for MCP queries: "Which model performs best on this benchmark?"
    """
    view_sql_sqlite = """
        SELECT
            vr.metadata_question_id as question_id,
            vr.metadata_answering_model as answering_model,
            vr.metadata_parsing_model as parsing_model,
            run.benchmark_id,
            b.name as benchmark_name,
            COUNT(*) as result_count,
            AVG(CASE WHEN vr.template_verify_result = 'true' OR vr.template_verify_result = 1 THEN 1.0 ELSE 0.0 END) as template_pass_rate,
            AVG(CASE WHEN vr.metadata_completed_without_errors THEN 1.0 ELSE 0.0 END) as completion_rate,
            AVG(vr.metadata_execution_time) as avg_execution_time,
            SUM(CASE WHEN vr.template_abstention_detected THEN 1 ELSE 0 END) as abstention_count,
            SUM(CASE WHEN vr.dj_deep_judgment_performed THEN 1 ELSE 0 END) as deep_judgment_count
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        JOIN benchmarks b ON run.benchmark_id = b.id
        GROUP BY vr.metadata_question_id, vr.metadata_answering_model, vr.metadata_parsing_model,
                 run.benchmark_id, b.name
    """

    view_sql_postgres = """
        SELECT
            vr.metadata_question_id as question_id,
            vr.metadata_answering_model as answering_model,
            vr.metadata_parsing_model as parsing_model,
            run.benchmark_id,
            b.name as benchmark_name,
            COUNT(*) as result_count,
            AVG(CASE WHEN vr.template_verify_result::text = 'true' THEN 1.0 ELSE 0.0 END) as template_pass_rate,
            AVG(CASE WHEN vr.metadata_completed_without_errors THEN 1.0 ELSE 0.0 END) as completion_rate,
            AVG(vr.metadata_execution_time) as avg_execution_time,
            SUM(CASE WHEN vr.template_abstention_detected THEN 1 ELSE 0 END) as abstention_count,
            SUM(CASE WHEN vr.dj_deep_judgment_performed THEN 1 ELSE 0 END) as deep_judgment_count
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        JOIN benchmarks b ON run.benchmark_id = b.id
        GROUP BY vr.metadata_question_id, vr.metadata_answering_model, vr.metadata_parsing_model,
                 run.benchmark_id, b.name
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS model_comparison_view"))
            conn.execute(text(f"CREATE VIEW model_comparison_view AS {view_sql_sqlite}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW model_comparison_view AS {view_sql_postgres}"))


def create_question_analysis_view(engine: Engine) -> None:
    """Create or replace the question_analysis_view.

    Analyzes question difficulty and reliability across all evaluations.
    Ideal for MCP queries: "Which questions are hardest?" "Where do models struggle?"
    """
    view_sql_sqlite = """
        SELECT
            vr.metadata_question_id as question_id,
            vr.metadata_template_id as template_id,
            vr.metadata_question_text as question_text,
            COUNT(*) as total_attempts,
            COUNT(DISTINCT vr.metadata_answering_model) as unique_answering_models,
            COUNT(DISTINCT vr.metadata_parsing_model) as unique_parsing_models,
            AVG(CASE WHEN vr.template_verify_result = 'true' OR vr.template_verify_result = 1 THEN 1.0 ELSE 0.0 END) as overall_pass_rate,
            AVG(CASE WHEN vr.metadata_completed_without_errors THEN 1.0 ELSE 0.0 END) as completion_rate,
            SUM(CASE WHEN vr.template_abstention_detected THEN 1 ELSE 0 END) as total_abstentions,
            SUM(CASE WHEN vr.template_embedding_override_applied THEN 1 ELSE 0 END) as embedding_overrides,
            AVG(vr.template_embedding_similarity_score) as avg_embedding_similarity,
            AVG(vr.metadata_execution_time) as avg_execution_time,
            MIN(vr.metadata_timestamp) as first_evaluated,
            MAX(vr.metadata_timestamp) as last_evaluated
        FROM verification_results vr
        GROUP BY vr.metadata_question_id, vr.metadata_template_id, vr.metadata_question_text
    """

    view_sql_postgres = """
        SELECT
            vr.metadata_question_id as question_id,
            vr.metadata_template_id as template_id,
            vr.metadata_question_text as question_text,
            COUNT(*) as total_attempts,
            COUNT(DISTINCT vr.metadata_answering_model) as unique_answering_models,
            COUNT(DISTINCT vr.metadata_parsing_model) as unique_parsing_models,
            AVG(CASE WHEN vr.template_verify_result::text = 'true' THEN 1.0 ELSE 0.0 END) as overall_pass_rate,
            AVG(CASE WHEN vr.metadata_completed_without_errors THEN 1.0 ELSE 0.0 END) as completion_rate,
            SUM(CASE WHEN vr.template_abstention_detected THEN 1 ELSE 0 END) as total_abstentions,
            SUM(CASE WHEN vr.template_embedding_override_applied THEN 1 ELSE 0 END) as embedding_overrides,
            AVG(vr.template_embedding_similarity_score) as avg_embedding_similarity,
            AVG(vr.metadata_execution_time) as avg_execution_time,
            MIN(vr.metadata_timestamp) as first_evaluated,
            MAX(vr.metadata_timestamp) as last_evaluated
        FROM verification_results vr
        GROUP BY vr.metadata_question_id, vr.metadata_template_id, vr.metadata_question_text
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS question_analysis_view"))
            conn.execute(text(f"CREATE VIEW question_analysis_view AS {view_sql_sqlite}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW question_analysis_view AS {view_sql_postgres}"))


def create_trend_analysis_view(engine: Engine) -> None:
    """Create or replace the trend_analysis_view.

    Tracks performance trends over time per model.
    Ideal for MCP queries: "How has model X improved over time?"
    """
    view_sql_sqlite = """
        SELECT
            vr.metadata_answering_model as answering_model,
            vr.metadata_parsing_model as parsing_model,
            DATE(vr.created_at) as evaluation_date,
            run.benchmark_id,
            b.name as benchmark_name,
            COUNT(*) as evaluations,
            AVG(CASE WHEN vr.template_verify_result = 'true' OR vr.template_verify_result = 1 THEN 1.0 ELSE 0.0 END) as pass_rate,
            AVG(CASE WHEN vr.metadata_completed_without_errors THEN 1.0 ELSE 0.0 END) as completion_rate,
            AVG(vr.metadata_execution_time) as avg_execution_time,
            COUNT(DISTINCT vr.metadata_question_id) as unique_questions
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        JOIN benchmarks b ON run.benchmark_id = b.id
        GROUP BY vr.metadata_answering_model, vr.metadata_parsing_model, DATE(vr.created_at),
                 run.benchmark_id, b.name
        ORDER BY evaluation_date DESC
    """

    view_sql_postgres = """
        SELECT
            vr.metadata_answering_model as answering_model,
            vr.metadata_parsing_model as parsing_model,
            DATE(vr.created_at) as evaluation_date,
            run.benchmark_id,
            b.name as benchmark_name,
            COUNT(*) as evaluations,
            AVG(CASE WHEN vr.template_verify_result::text = 'true' THEN 1.0 ELSE 0.0 END) as pass_rate,
            AVG(CASE WHEN vr.metadata_completed_without_errors THEN 1.0 ELSE 0.0 END) as completion_rate,
            AVG(vr.metadata_execution_time) as avg_execution_time,
            COUNT(DISTINCT vr.metadata_question_id) as unique_questions
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        JOIN benchmarks b ON run.benchmark_id = b.id
        GROUP BY vr.metadata_answering_model, vr.metadata_parsing_model, DATE(vr.created_at),
                 run.benchmark_id, b.name
        ORDER BY evaluation_date DESC
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS trend_analysis_view"))
            conn.execute(text(f"CREATE VIEW trend_analysis_view AS {view_sql_sqlite}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW trend_analysis_view AS {view_sql_postgres}"))


def create_import_history_view(engine: Engine) -> None:
    """Create or replace the import_history_view.

    Tracks all imported verification results with source metadata.
    Ideal for MCP queries: "Where did these results come from?"
    """
    view_sql = """
        SELECT
            im.id as import_id,
            im.run_id,
            vr.run_name,
            im.import_source,
            im.source_format_version,
            im.source_file_name,
            im.source_file_hash,
            im.original_run_name,
            im.original_timestamp,
            im.import_timestamp,
            im.imported_result_count,
            vr.benchmark_id,
            b.name as benchmark_name,
            vr.status as run_status
        FROM import_metadata im
        JOIN verification_runs vr ON im.run_id = vr.id
        JOIN benchmarks b ON vr.benchmark_id = b.id
        ORDER BY im.import_timestamp DESC
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS import_history_view"))
            conn.execute(text(f"CREATE VIEW import_history_view AS {view_sql}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW import_history_view AS {view_sql}"))


def create_deep_judgment_analysis_view(engine: Engine) -> None:
    """Create or replace the deep_judgment_analysis_view.

    Analyzes deep judgment usage and effectiveness.
    Ideal for MCP queries: "How effective is deep judgment?"
    """
    view_sql = """
        SELECT
            vr.metadata_answering_model as answering_model,
            vr.metadata_parsing_model as parsing_model,
            run.benchmark_id,
            b.name as benchmark_name,
            COUNT(*) as total_evaluations,
            SUM(CASE WHEN vr.dj_deep_judgment_enabled THEN 1 ELSE 0 END) as dj_enabled_count,
            SUM(CASE WHEN vr.dj_deep_judgment_performed THEN 1 ELSE 0 END) as dj_performed_count,
            AVG(vr.dj_deep_judgment_model_calls) as avg_dj_model_calls,
            AVG(vr.dj_deep_judgment_excerpt_retry_count) as avg_excerpt_retries,
            SUM(CASE WHEN vr.djr_deep_judgment_rubric_performed THEN 1 ELSE 0 END) as djr_performed_count,
            AVG(vr.djr_total_deep_judgment_model_calls) as avg_djr_model_calls,
            AVG(vr.djr_total_traits_evaluated) as avg_traits_evaluated
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        JOIN benchmarks b ON run.benchmark_id = b.id
        GROUP BY vr.metadata_answering_model, vr.metadata_parsing_model, run.benchmark_id, b.name
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS deep_judgment_analysis_view"))
            conn.execute(text(f"CREATE VIEW deep_judgment_analysis_view AS {view_sql}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW deep_judgment_analysis_view AS {view_sql}"))


def create_all_views(engine: Engine) -> None:
    """Create all database views.

    Args:
        engine: SQLAlchemy engine instance
    """
    # Core views
    create_benchmark_summary_view(engine)
    create_verification_run_summary_view(engine)
    create_question_usage_view(engine)
    create_latest_verification_results_view(engine)
    create_model_performance_view(engine)
    create_rubric_scores_aggregate_view(engine)
    create_verification_history_timeline_view(engine)
    create_failed_verifications_view(engine)
    create_rubric_traits_by_type_view(engine)

    # MCP query views
    create_model_comparison_view(engine)
    create_question_analysis_view(engine)
    create_trend_analysis_view(engine)
    create_import_history_view(engine)
    create_deep_judgment_analysis_view(engine)


def drop_all_views(engine: Engine) -> None:
    """Drop all database views.

    Args:
        engine: SQLAlchemy engine instance
    """
    views = [
        # Core views
        "benchmark_summary_view",
        "verification_run_summary_view",
        "question_usage_view",
        "latest_verification_results_view",
        "model_performance_view",
        "rubric_scores_aggregate_view",
        "verification_history_timeline_view",
        "failed_verifications_view",
        "rubric_traits_by_type_view",
        # MCP query views
        "model_comparison_view",
        "question_analysis_view",
        "trend_analysis_view",
        "import_history_view",
        "deep_judgment_analysis_view",
    ]

    with engine.begin() as conn:
        for view in views:
            conn.execute(text(f"DROP VIEW IF EXISTS {view}"))
