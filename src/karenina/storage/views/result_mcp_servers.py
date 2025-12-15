"""Result MCP Servers View.

View Name: result_mcp_servers_view

Description:
    Shows one row per MCP server that was configured for each verification result.
    Unnests the template_answering_mcp_servers JSON array to create a normalized
    view of which MCP servers were attached to each result.

    Note: This shows servers that were *configured*, not necessarily *used*.
    For actual tool usage, see result_tools_used_view.

Columns:
    - result_id: Unique identifier for the verification result
    - mcp_server: Name of an MCP server configured for this result

Source Tables:
    - verification_results (vr)

Source Column:
    - template_answering_mcp_servers: JSON array of MCP server names
      Example: ["brave_search", "filesystem", "postgres"]

JSON Functions Used:
    - SQLite: json_each(), json_array_length()
    - PostgreSQL: jsonb_array_elements_text(), jsonb_array_length()

Example Query:
    -- Count results by MCP server
    SELECT mcp_server, COUNT(DISTINCT result_id) as result_count
    FROM result_mcp_servers_view
    GROUP BY mcp_server
    ORDER BY result_count DESC;

    -- Find results that used a specific MCP server
    SELECT result_id FROM result_mcp_servers_view
    WHERE mcp_server = 'brave_search';
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

VIEW_NAME = "result_mcp_servers_view"


def create_result_mcp_servers_view(engine: Engine) -> None:
    """Create or replace the result_mcp_servers_view."""
    # SQLite version using json_each
    view_sql_sqlite = """
        SELECT
            vr.metadata_result_id as result_id,
            mcp.value as mcp_server
        FROM verification_results vr
        CROSS JOIN json_each(vr.template_answering_mcp_servers) as mcp
        WHERE vr.template_answering_mcp_servers IS NOT NULL
          AND json_array_length(vr.template_answering_mcp_servers) > 0
    """

    # PostgreSQL version using jsonb_array_elements_text
    view_sql_postgres = """
        SELECT
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
