"""Run MCP Servers View.

View Name: run_mcp_servers_view

Description:
    Shows one row per MCP server configured for each verification run.
    Provides a compact representation of which MCP servers were attached
    to each run, avoiding duplication across individual results.

    Note: This shows servers that were *configured*, not necessarily *used*.
    For actual tool usage, see result_tools_used_view.

Columns:
    - run_id: Unique identifier for the verification run
    - run_name: Name of the verification run
    - mcp_server: Name of an MCP server configured for this run

Source Tables:
    - verification_results (vr)
    - verification_runs (run)

Source Column:
    - template_answering_mcp_servers: JSON array of MCP server names
      Example: ["brave_search", "filesystem", "postgres"]

JSON Functions Used:
    - SQLite: json_each(), json_array_length()
    - PostgreSQL: jsonb_array_elements_text(), jsonb_array_length()

Example Query:
    -- List all MCP servers for each run
    SELECT run_name, mcp_server
    FROM run_mcp_servers_view
    ORDER BY run_name, mcp_server;

    -- Count runs by MCP server
    SELECT mcp_server, COUNT(DISTINCT run_id) as run_count
    FROM run_mcp_servers_view
    GROUP BY mcp_server
    ORDER BY run_count DESC;

    -- Find runs that used a specific MCP server
    SELECT run_id, run_name FROM run_mcp_servers_view
    WHERE mcp_server = 'brave_search';
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
