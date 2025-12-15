"""Combination Info View.

View Name: combination_info_view

Description:
    Shows distinct combinations of run, answering model, and parsing model
    with an indicator of whether MCP tooling was attached and the number of replicates.
    Useful for understanding which model configurations were used in each run.

Columns:
    - run_id: Unique identifier for the verification run
    - run_name: Name of the verification run
    - answering_model: Name of the model that generated answers
    - parsing_model: Name of the model that parsed responses
    - has_mcp: Boolean indicating if MCP servers were configured (1=yes, 0=no)
    - replicate_count: Number of replicates used for this combination

Source Tables:
    - verification_results (vr)
    - verification_runs (run)

Source Columns:
    - metadata_answering_model: The answering model name
    - metadata_parsing_model: The parsing model name
    - metadata_answering_replicate: Replicate number (used to count replicates)
    - template_answering_mcp_servers: JSON array of MCP server names (used to determine has_mcp)

Example Query:
    -- List all combinations for a run
    SELECT * FROM combination_info_view
    WHERE run_name = 'my_run';

    -- Find all runs that used MCP
    SELECT DISTINCT run_name FROM combination_info_view
    WHERE has_mcp = 1;

    -- Find combinations with multiple replicates
    SELECT * FROM combination_info_view
    WHERE replicate_count > 1;
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

VIEW_NAME = "combination_info_view"


def create_combination_info_view(engine: Engine) -> None:
    """Create or replace the combination_info_view."""
    # SQLite version
    view_sql_sqlite = """
        SELECT
            run.id as run_id,
            run.run_name,
            vr.metadata_answering_model as answering_model,
            vr.metadata_parsing_model as parsing_model,
            MAX(CASE
                WHEN vr.template_answering_mcp_servers IS NOT NULL
                     AND json_array_length(vr.template_answering_mcp_servers) > 0
                THEN 1
                ELSE 0
            END) as has_mcp,
            COUNT(DISTINCT vr.metadata_answering_replicate) as replicate_count
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        GROUP BY run.id, run.run_name, vr.metadata_answering_model, vr.metadata_parsing_model
    """

    # PostgreSQL version
    view_sql_postgres = """
        SELECT
            run.id as run_id,
            run.run_name,
            vr.metadata_answering_model as answering_model,
            vr.metadata_parsing_model as parsing_model,
            MAX(CASE
                WHEN vr.template_answering_mcp_servers IS NOT NULL
                     AND jsonb_array_length(vr.template_answering_mcp_servers::jsonb) > 0
                THEN 1
                ELSE 0
            END) as has_mcp,
            COUNT(DISTINCT vr.metadata_answering_replicate) as replicate_count
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        GROUP BY run.id, run.run_name, vr.metadata_answering_model, vr.metadata_parsing_model
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_sqlite}"))
        else:
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_postgres}"))


def drop_combination_info_view(engine: Engine) -> None:
    """Drop the combination_info_view if it exists."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
