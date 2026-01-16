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

from sqlalchemy.engine import Engine

from .utils import create_view_safe, drop_view_safe

VIEW_NAME = "raw_llm_answers_view"

_VIEW_SQL = """
    SELECT
        vr.metadata_result_id as result_id,
        vr.question_id,
        vr.template_raw_llm_response as raw_llm_response
    FROM verification_results vr
    WHERE vr.template_raw_llm_response IS NOT NULL
"""


def create_raw_llm_answers_view(engine: Engine) -> None:
    """Create or replace the raw_llm_answers_view."""
    create_view_safe(engine, VIEW_NAME, _VIEW_SQL)


def drop_raw_llm_answers_view(engine: Engine) -> None:
    """Drop the raw_llm_answers_view if it exists."""
    drop_view_safe(engine, VIEW_NAME)
