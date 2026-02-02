"""template_results_view

Primary view for verification outcomes. One row per result with pass/fail status,
result_id, run_id, run_name, question_id, and benchmark_name. Use for pass rate
analysis, filtering failed questions, and joining to other result-level views.

Columns:
    result_id (TEXT): Unique identifier for the verification result
    run_id (TEXT): Unique identifier for the verification run (UUID)
    verification_date (TIMESTAMP): Timestamp when the verification was performed
    run_name (TEXT): Name of the verification run
    benchmark_name (TEXT): Name of the benchmark
    question_id (TEXT): Unique identifier for the question (MD5 hash)
    question_text (TEXT): The question content
    verify_result (BOOLEAN): Verification outcome (true=pass, false=fail, NULL=not evaluated)
    replicate (INTEGER): Replicate number (NULL for single runs, 1/2/3/... for replicated)

Keys:
    Primary: result_id
    Joins: result_id → results_metadata_view, raw_llm_answers_view, rubric_traits_view, result_mcp_servers_view
           run_id → combination_info_view.run_id
           run_name → combination_info_view.run_name
           question_id → question_attributes_view.question_id

Example:
    SELECT run_name, COUNT(*) as total, SUM(verify_result) as passed,
           ROUND(100.0 * SUM(verify_result) / COUNT(*), 2) as pass_rate
    FROM template_results_view GROUP BY run_name;
"""

from sqlalchemy.engine import Engine

from .utils import create_view_safe, drop_view_safe

VIEW_NAME = "template_results_view"

_VIEW_SQL = """
    SELECT
        vr.metadata_result_id as result_id,
        run.id as run_id,
        vr.created_at as verification_date,
        run.run_name,
        b.name as benchmark_name,
        vr.question_id,
        vr.metadata_question_text as question_text,
        vr.template_verify_result as verify_result,
        vr.metadata_replicate as replicate
    FROM verification_results vr
    JOIN verification_runs run ON vr.run_id = run.id
    JOIN benchmarks b ON run.benchmark_id = b.id
"""


def create_template_results_view(engine: Engine) -> None:
    """Create or replace the template_results_view."""
    create_view_safe(engine, VIEW_NAME, _VIEW_SQL)


def drop_template_results_view(engine: Engine) -> None:
    """Drop the template_results_view if it exists."""
    drop_view_safe(engine, VIEW_NAME)
