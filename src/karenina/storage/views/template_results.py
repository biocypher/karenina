"""Template Results View.

View Name: template_results_view

Description:
    Shows verification results with benchmark context and question details.
    One row per verification result (overall pass/fail).

Columns:
    - result_id: Unique identifier for the verification result
    - verification_date: Date the verification was performed
    - run_name: Name of the verification run
    - benchmark_name: Name of the benchmark
    - question_id: Unique identifier for the question
    - question_text: The question text
    - verify_result: Boolean indicating if the answer was correct

Source Tables:
    - verification_results (vr)
    - verification_runs (run)
    - benchmarks (b)

Example Query:
    SELECT * FROM template_results_view
    WHERE benchmark_name = 'my_benchmark'
    AND verify_result = 1;
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

VIEW_NAME = "template_results_view"


def create_template_results_view(engine: Engine) -> None:
    """Create or replace the template_results_view."""
    view_sql = """
        SELECT
            vr.metadata_result_id as result_id,
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
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW {VIEW_NAME} AS {view_sql}"))


def drop_template_results_view(engine: Engine) -> None:
    """Drop the template_results_view if it exists."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
