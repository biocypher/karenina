"""Models Used View.

View Name: models_used_view

Description:
    Shows all distinct model names used throughout verification runs,
    with flags indicating whether each model was used for answering
    (generation) and/or parsing.

Columns:
    - model_name: Name of the model
    - used_for_answering: 1 if the model was used to generate answers, 0 otherwise
    - used_for_parsing: 1 if the model was used to parse responses, 0 otherwise

Source Tables:
    - verification_results (vr)

Source Columns:
    - metadata_answering_model: The model used for generating answers
    - metadata_parsing_model: The model used for parsing responses

Example Query:
    -- List all models and their roles
    SELECT * FROM models_used_view
    ORDER BY model_name;

    -- Find models used for both answering and parsing
    SELECT model_name FROM models_used_view
    WHERE used_for_answering = 1 AND used_for_parsing = 1;

    -- Find models used only for answering
    SELECT model_name FROM models_used_view
    WHERE used_for_answering = 1 AND used_for_parsing = 0;
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

VIEW_NAME = "models_used_view"


def create_models_used_view(engine: Engine) -> None:
    """Create or replace the models_used_view."""
    # This query unions answering and parsing models, then aggregates
    # to show which roles each model was used for
    view_sql = """
        SELECT
            model_name,
            MAX(is_answering) as used_for_answering,
            MAX(is_parsing) as used_for_parsing
        FROM (
            SELECT
                metadata_answering_model as model_name,
                1 as is_answering,
                0 as is_parsing
            FROM verification_results
            WHERE metadata_answering_model IS NOT NULL
            UNION ALL
            SELECT
                metadata_parsing_model as model_name,
                0 as is_answering,
                1 as is_parsing
            FROM verification_results
            WHERE metadata_parsing_model IS NOT NULL
        )
        GROUP BY model_name
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql}"))
        else:
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql}"))


def drop_models_used_view(engine: Engine) -> None:
    """Drop the models_used_view if it exists."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
