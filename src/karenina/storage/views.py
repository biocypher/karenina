"""Database views for Karenina storage.

This module defines SQL views that provide convenient aggregated queries
for common use cases like benchmark summaries, verification statistics,
and model performance analysis.

All views use the flattened column naming scheme from the auto-generated
VerificationResultModel (e.g., metadata_answering_model, template_verify_result).
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

# ============================================================================
# Template Analysis Views
# ============================================================================


def create_template_results_view(engine: Engine) -> None:
    """Create or replace the template_results_view.

    Shows verification results with benchmark context and question details.
    One row per verification result (overall pass/fail).
    """
    view_sql = """
        SELECT
            DATE(vr.created_at) as verification_date,
            run.run_name,
            b.name as benchmark_name,
            vr.question_id,
            vr.metadata_question_text as question_text,
            vr.template_verify_result as verify_result
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        JOIN benchmarks b ON run.benchmark_id = b.id
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS template_results_view"))
            conn.execute(text(f"CREATE VIEW template_results_view AS {view_sql}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW template_results_view AS {view_sql}"))


def create_template_attributes_view(engine: Engine) -> None:
    """Create or replace the template_attributes_view.

    Shows verification results disaggregated by template attribute.
    One row per attribute in the parsed template response.
    Uses JSON functions to explode parsed_gt_response and parsed_llm_response.
    """
    # SQLite version using json_each
    view_sql_sqlite = """
        SELECT
            DATE(vr.created_at) as verification_date,
            run.run_name,
            b.name as benchmark_name,
            vr.question_id,
            vr.metadata_question_text as question_text,
            gt.key as attribute_name,
            gt.value as gt_value,
            json_extract(vr.template_parsed_llm_response, '$.' || gt.key) as llm_value,
            CASE
                WHEN gt.value = json_extract(vr.template_parsed_llm_response, '$.' || gt.key)
                THEN 1
                ELSE 0
            END as attribute_match
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        JOIN benchmarks b ON run.benchmark_id = b.id
        CROSS JOIN json_each(vr.template_parsed_gt_response) as gt
        WHERE vr.template_parsed_gt_response IS NOT NULL
    """

    # PostgreSQL version using jsonb_each_text
    view_sql_postgres = """
        SELECT
            DATE(vr.created_at) as verification_date,
            run.run_name,
            b.name as benchmark_name,
            vr.question_id,
            vr.metadata_question_text as question_text,
            gt.key as attribute_name,
            gt.value as gt_value,
            vr.template_parsed_llm_response ->> gt.key as llm_value,
            CASE
                WHEN gt.value = (vr.template_parsed_llm_response ->> gt.key)
                THEN 1
                ELSE 0
            END as attribute_match
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        JOIN benchmarks b ON run.benchmark_id = b.id
        CROSS JOIN LATERAL jsonb_each_text(vr.template_parsed_gt_response::jsonb) as gt
        WHERE vr.template_parsed_gt_response IS NOT NULL
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("DROP VIEW IF EXISTS template_attributes_view"))
            conn.execute(text(f"CREATE VIEW template_attributes_view AS {view_sql_sqlite}"))
        else:
            conn.execute(text("DROP VIEW IF EXISTS template_attributes_view"))
            conn.execute(text(f"CREATE VIEW template_attributes_view AS {view_sql_postgres}"))


def create_all_views(engine: Engine) -> None:
    """Create all database views.

    Args:
        engine: SQLAlchemy engine instance
    """
    create_template_results_view(engine)
    create_template_attributes_view(engine)


def drop_all_views(engine: Engine) -> None:
    """Drop all database views.

    Args:
        engine: SQLAlchemy engine instance
    """
    views = [
        "template_results_view",
        "template_attributes_view",
    ]

    with engine.begin() as conn:
        for view in views:
            conn.execute(text(f"DROP VIEW IF EXISTS {view}"))
