"""run_mcp_servers_view

MCP servers configured per run. One row per MCP server per run. Shows servers
that were *configured* (not necessarily used). Use for understanding MCP tooling
setup across runs. Join via run_id to template_results_view.

Columns:
    run_id (TEXT): Unique identifier for the verification run (UUID)
    run_name (TEXT): Name of the verification run
    mcp_server (TEXT): Name of an MCP server configured (e.g., 'brave_search', 'filesystem')

Keys:
    Primary: run_id + mcp_server
    Joins: run_id → template_results_view.run_id, combination_info_view.run_id
           run_name → template_results_view.run_name

Example:
    SELECT run_name, mcp_server FROM run_mcp_servers_view
    ORDER BY run_name, mcp_server;
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

VIEW_NAME = "run_mcp_servers_view"


def create_run_mcp_servers_view(engine: Engine) -> None:
    """Create or replace the run_mcp_servers_view."""
    # SQLite version using json_each
    view_sql_sqlite = """
        SELECT DISTINCT
            run.id as run_id,
            run.run_name,
            mcp.value as mcp_server
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        CROSS JOIN json_each(vr.template_answering_mcp_servers) as mcp
        WHERE vr.template_answering_mcp_servers IS NOT NULL
          AND json_array_length(vr.template_answering_mcp_servers) > 0
    """

    # PostgreSQL version using jsonb_array_elements_text
    view_sql_postgres = """
        SELECT DISTINCT
            run.id as run_id,
            run.run_name,
            mcp.value as mcp_server
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        CROSS JOIN LATERAL jsonb_array_elements_text(vr.template_answering_mcp_servers::jsonb) as mcp(value)
        WHERE vr.template_answering_mcp_servers IS NOT NULL
          AND jsonb_array_length(vr.template_answering_mcp_servers::jsonb) > 0
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_sqlite}"))
        else:
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_postgres}"))


def drop_run_mcp_servers_view(engine: Engine) -> None:
    """Drop the run_mcp_servers_view if it exists."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))


# Backward compatibility aliases
create_result_mcp_servers_view = create_run_mcp_servers_view
drop_result_mcp_servers_view = drop_run_mcp_servers_view
