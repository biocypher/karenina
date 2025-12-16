"""result_mcp_servers_view

MCP servers configured per result. One row per MCP server per result. Shows servers
that were configured for each verification result. Use for understanding MCP tooling
setup at the individual result level. Join via result_id to template_results_view.

Columns:
    result_id (TEXT): Unique identifier for the verification result
    mcp_server (TEXT): Name of an MCP server configured (e.g., 'brave_search', 'filesystem')

Keys:
    Primary: result_id + mcp_server
    Joins: result_id → template_results_view.result_id, results_metadata_view.result_id

Example:
    SELECT result_id, mcp_server FROM result_mcp_servers_view
    ORDER BY result_id, mcp_server;
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

VIEW_NAME = "result_mcp_servers_view"


def create_result_mcp_servers_view(engine: Engine) -> None:
    """Create or replace the result_mcp_servers_view."""
    # SQLite version using json_each
    view_sql_sqlite = """
        SELECT DISTINCT
            vr.metadata_result_id as result_id,
            mcp.value as mcp_server
        FROM verification_results vr
        CROSS JOIN json_each(vr.template_answering_mcp_servers) as mcp
        WHERE vr.template_answering_mcp_servers IS NOT NULL
          AND json_array_length(vr.template_answering_mcp_servers) > 0
    """

    # PostgreSQL version using jsonb_array_elements_text
    view_sql_postgres = """
        SELECT DISTINCT
            vr.metadata_result_id as result_id,
            mcp.value as mcp_server
        FROM verification_results vr
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


def drop_result_mcp_servers_view(engine: Engine) -> None:
    """Drop the result_mcp_servers_view if it exists."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
