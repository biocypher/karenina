"""deep_judgment_rubric_traits_view

Deep judgment rubric scores per result. One row per trait per result, distinguishing
DJ scores from standard scores in same run via is_deep_judgment flag. Use for
comparing deep judgment vs standard rubric evaluations.

Columns:
    result_id (TEXT): Unique identifier for the verification result
    trait_name (TEXT): Name of the rubric trait
    trait_type (TEXT): Type of trait (always "llm" for deep judgment)
    score (TEXT): The score value (bool/int as text)
    is_deep_judgment (INTEGER): 1 if from deep judgment, 0 if standard evaluation

Keys:
    Primary: result_id + trait_name + is_deep_judgment
    Joins: result_id → template_results_view.result_id

Example:
    SELECT result_id, trait_name, score, is_deep_judgment
    FROM deep_judgment_rubric_traits_view
    ORDER BY result_id, trait_name, is_deep_judgment;
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

VIEW_NAME = "deep_judgment_rubric_traits_view"


def create_deep_judgment_rubric_traits_view(engine: Engine) -> None:
    """Create or replace the deep_judgment_rubric_traits_view."""
    # SQLite version using json_each
    view_sql_sqlite = """
        SELECT
            vr.metadata_result_id as result_id,
            trait.key as trait_name,
            'llm' as trait_type,
            trait.value as score,
            1 as is_deep_judgment
        FROM verification_results vr
        CROSS JOIN json_each(vr.djr_deep_judgment_rubric_scores) as trait
        WHERE vr.djr_deep_judgment_rubric_scores IS NOT NULL
          AND vr.djr_deep_judgment_rubric_performed = 1

        UNION ALL

        SELECT
            vr.metadata_result_id as result_id,
            trait.key as trait_name,
            'llm' as trait_type,
            trait.value as score,
            0 as is_deep_judgment
        FROM verification_results vr
        CROSS JOIN json_each(vr.djr_standard_rubric_scores) as trait
        WHERE vr.djr_standard_rubric_scores IS NOT NULL
          AND vr.djr_deep_judgment_rubric_performed = 1
    """

    # PostgreSQL version using jsonb_each
    view_sql_postgres = """
        SELECT
            vr.metadata_result_id as result_id,
            trait.key as trait_name,
            'llm' as trait_type,
            trait.value::text as score,
            1 as is_deep_judgment
        FROM verification_results vr
        CROSS JOIN LATERAL jsonb_each(vr.djr_deep_judgment_rubric_scores::jsonb) as trait
        WHERE vr.djr_deep_judgment_rubric_scores IS NOT NULL
          AND vr.djr_deep_judgment_rubric_performed = TRUE

        UNION ALL

        SELECT
            vr.metadata_result_id as result_id,
            trait.key as trait_name,
            'llm' as trait_type,
            trait.value::text as score,
            0 as is_deep_judgment
        FROM verification_results vr
        CROSS JOIN LATERAL jsonb_each(vr.djr_standard_rubric_scores::jsonb) as trait
        WHERE vr.djr_standard_rubric_scores IS NOT NULL
          AND vr.djr_deep_judgment_rubric_performed = TRUE
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_sqlite}"))
        else:
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_postgres}"))


def drop_deep_judgment_rubric_traits_view(engine: Engine) -> None:
    """Drop the deep_judgment_rubric_traits_view if it exists."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
