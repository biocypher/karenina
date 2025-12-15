"""Deep Judgment Rubric Traits View.

View Name: deep_judgment_rubric_traits_view

Description:
    Shows deep judgment rubric trait scores with one row per trait per result.
    Separates deep judgment scores from standard scores evaluated in the same run.
    Only includes results where deep judgment rubric evaluation was performed.

Columns:
    - result_id: Unique identifier for the verification result (for joining)
    - trait_name: Name of the rubric trait
    - trait_type: Type of trait (always "llm" for deep judgment)
    - score: The score value (bool or int)
    - is_deep_judgment: 1 if from deep judgment evaluation, 0 if standard evaluation

Source Tables:
    - verification_results (vr)

Source Columns:
    - djr_deep_judgment_rubric_scores: Deep judgment evaluated trait scores (JSON dict)
    - djr_standard_rubric_scores: Standard (non-DJ) trait scores in DJ runs (JSON dict)

Note:
    Only includes results where djr_deep_judgment_rubric_performed = 1.
    Deep judgment is always LLM-based, hence trait_type is always "llm".
    The is_deep_judgment flag distinguishes between DJ and standard evaluations.

Example Query:
    -- Compare DJ vs standard scores for same traits
    SELECT result_id, trait_name, score, is_deep_judgment
    FROM deep_judgment_rubric_traits_view
    ORDER BY result_id, trait_name, is_deep_judgment;

    -- Count traits evaluated with deep judgment
    SELECT COUNT(DISTINCT trait_name) as dj_traits
    FROM deep_judgment_rubric_traits_view
    WHERE is_deep_judgment = 1;
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
