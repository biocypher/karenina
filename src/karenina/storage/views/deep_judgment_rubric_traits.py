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

from sqlalchemy.engine import Engine

from .utils import create_view_safe, drop_view_safe

VIEW_NAME = "deep_judgment_rubric_traits_view"

# SQLite version using json_each
_SQLITE_SQL = """
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
_POSTGRES_SQL = """
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


def create_deep_judgment_rubric_traits_view(engine: Engine) -> None:
    """Create or replace the deep_judgment_rubric_traits_view."""
    create_view_safe(engine, VIEW_NAME, _SQLITE_SQL, _POSTGRES_SQL)


def drop_deep_judgment_rubric_traits_view(engine: Engine) -> None:
    """Drop the deep_judgment_rubric_traits_view if it exists."""
    drop_view_safe(engine, VIEW_NAME)
