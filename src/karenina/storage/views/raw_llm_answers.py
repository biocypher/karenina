"""raw_llm_answers_view

Raw LLM response text per result. One row per result with non-NULL responses.
Use for full response analysis, text inspection, and debugging model outputs.

Columns:
    result_id (TEXT): Unique identifier for the verification result
    question_id (TEXT): Unique identifier for the question (MD5 hash)
    raw_llm_response (TEXT): The complete raw text response from the answering LLM

Keys:
    Primary: result_id
    Joins: result_id → template_results_view.result_id
           question_id → template_results_view.question_id

Example:
    SELECT r.result_id, r.raw_llm_response FROM raw_llm_answers_view r
    JOIN template_results_view t ON r.result_id = t.result_id
    WHERE t.verify_result = 0;
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

VIEW_NAME = "raw_llm_answers_view"


def create_raw_llm_answers_view(engine: Engine) -> None:
    """Create or replace the raw_llm_answers_view."""
    view_sql = """
        SELECT
            vr.metadata_result_id as result_id,
            vr.question_id,
            vr.template_raw_llm_response as raw_llm_response
        FROM verification_results vr
        WHERE vr.template_raw_llm_response IS NOT NULL
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql}"))
        else:
            conn.execute(text(f"CREATE OR REPLACE VIEW {VIEW_NAME} AS {view_sql}"))


def drop_raw_llm_answers_view(engine: Engine) -> None:
    """Drop the raw_llm_answers_view if it exists."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
