"""Rubric Traits View.

View Name: rubric_traits_view

Description:
    Shows rubric trait scores with one row per trait per result.
    Explodes JSON trait score dictionaries into individual rows.
    Combines all four trait types (LLM, regex, callable, metric) via UNION ALL.

Columns:
    - result_id: Unique identifier for the verification result (for joining)
    - trait_name: Name of the rubric trait
    - trait_type: Type of trait ("llm", "regex", "callable", "metric")
    - score: The score value (bool, int, or JSON for metric traits)

Source Tables:
    - verification_results (vr)

Source Columns:
    - rubric_llm_trait_scores: LLM-evaluated trait scores (JSON dict)
    - rubric_regex_trait_scores: Regex pattern match results (JSON dict)
    - rubric_callable_trait_scores: Custom Python function scores (JSON dict)
    - rubric_metric_trait_scores: Confusion matrix metrics (JSON dict of dicts)

Note:
    Only includes results where rubric_rubric_evaluation_performed = 1.
    Empty JSON dicts produce no rows.
    For metric traits, score is a JSON object (e.g., {"precision": 0.8, "recall": 0.9}).

Example Query:
    -- Count trait pass rates by trait name
    SELECT trait_name, trait_type,
           SUM(CASE WHEN score = 1 OR score = 'true' THEN 1 ELSE 0 END) as passes,
           COUNT(*) as total
    FROM rubric_traits_view
    WHERE trait_type IN ('llm', 'regex', 'callable')
    GROUP BY trait_name, trait_type;

    -- Get all metric trait results
    SELECT result_id, trait_name, score
    FROM rubric_traits_view
    WHERE trait_type = 'metric';
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

VIEW_NAME = "rubric_traits_view"


def create_rubric_traits_view(engine: Engine) -> None:
    """Create or replace the rubric_traits_view."""
    # SQLite version using json_each
    view_sql_sqlite = """
        SELECT
            vr.metadata_result_id as result_id,
            trait.key as trait_name,
            'llm' as trait_type,
            trait.value as score
        FROM verification_results vr
        CROSS JOIN json_each(vr.rubric_llm_trait_scores) as trait
        WHERE vr.rubric_llm_trait_scores IS NOT NULL
          AND vr.rubric_rubric_evaluation_performed = 1

        UNION ALL

        SELECT
            vr.metadata_result_id as result_id,
            trait.key as trait_name,
            'regex' as trait_type,
            trait.value as score
        FROM verification_results vr
        CROSS JOIN json_each(vr.rubric_regex_trait_scores) as trait
        WHERE vr.rubric_regex_trait_scores IS NOT NULL
          AND vr.rubric_rubric_evaluation_performed = 1

        UNION ALL

        SELECT
            vr.metadata_result_id as result_id,
            trait.key as trait_name,
            'callable' as trait_type,
            trait.value as score
        FROM verification_results vr
        CROSS JOIN json_each(vr.rubric_callable_trait_scores) as trait
        WHERE vr.rubric_callable_trait_scores IS NOT NULL
          AND vr.rubric_rubric_evaluation_performed = 1

        UNION ALL

        SELECT
            vr.metadata_result_id as result_id,
            trait.key as trait_name,
            'metric' as trait_type,
            trait.value as score
        FROM verification_results vr
        CROSS JOIN json_each(vr.rubric_metric_trait_scores) as trait
        WHERE vr.rubric_metric_trait_scores IS NOT NULL
          AND vr.rubric_rubric_evaluation_performed = 1
    """

    # PostgreSQL version using jsonb_each
    view_sql_postgres = """
        SELECT
            vr.metadata_result_id as result_id,
            trait.key as trait_name,
            'llm' as trait_type,
            trait.value::text as score
        FROM verification_results vr
        CROSS JOIN LATERAL jsonb_each(vr.rubric_llm_trait_scores::jsonb) as trait
        WHERE vr.rubric_llm_trait_scores IS NOT NULL
          AND vr.rubric_rubric_evaluation_performed = TRUE

        UNION ALL

        SELECT
            vr.metadata_result_id as result_id,
            trait.key as trait_name,
            'regex' as trait_type,
            trait.value::text as score
        FROM verification_results vr
        CROSS JOIN LATERAL jsonb_each(vr.rubric_regex_trait_scores::jsonb) as trait
        WHERE vr.rubric_regex_trait_scores IS NOT NULL
          AND vr.rubric_rubric_evaluation_performed = TRUE

        UNION ALL

        SELECT
            vr.metadata_result_id as result_id,
            trait.key as trait_name,
            'callable' as trait_type,
            trait.value::text as score
        FROM verification_results vr
        CROSS JOIN LATERAL jsonb_each(vr.rubric_callable_trait_scores::jsonb) as trait
        WHERE vr.rubric_callable_trait_scores IS NOT NULL
          AND vr.rubric_rubric_evaluation_performed = TRUE

        UNION ALL

        SELECT
            vr.metadata_result_id as result_id,
            trait.key as trait_name,
            'metric' as trait_type,
            trait.value::text as score
        FROM verification_results vr
        CROSS JOIN LATERAL jsonb_each(vr.rubric_metric_trait_scores::jsonb) as trait
        WHERE vr.rubric_metric_trait_scores IS NOT NULL
          AND vr.rubric_rubric_evaluation_performed = TRUE
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_sqlite}"))
        else:
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_postgres}"))


def drop_rubric_traits_view(engine: Engine) -> None:
    """Drop the rubric_traits_view if it exists."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
