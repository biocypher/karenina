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

from sqlalchemy.engine import Engine

from .utils import create_view_safe, drop_view_safe

VIEW_NAME = "result_mcp_servers_view"

# SQLite version using json_each
_SQLITE_SQL = """
    SELECT DISTINCT
        vr.metadata_result_id as result_id,
        mcp.value as mcp_server
    FROM verification_results vr
    CROSS JOIN json_each(vr.template_answering_mcp_servers) as mcp
    WHERE vr.template_answering_mcp_servers IS NOT NULL
      AND json_array_length(vr.template_answering_mcp_servers) > 0
"""

# PostgreSQL version using jsonb_array_elements_text
_POSTGRES_SQL = """
    SELECT DISTINCT
        vr.metadata_result_id as result_id,
        mcp.value as mcp_server
    FROM verification_results vr
    CROSS JOIN LATERAL jsonb_array_elements_text(vr.template_answering_mcp_servers::jsonb) as mcp(value)
    WHERE vr.template_answering_mcp_servers IS NOT NULL
      AND jsonb_array_length(vr.template_answering_mcp_servers::jsonb) > 0
"""


def create_result_mcp_servers_view(engine: Engine) -> None:
    """Create or replace the result_mcp_servers_view."""
    create_view_safe(engine, VIEW_NAME, _SQLITE_SQL, _POSTGRES_SQL)


def drop_result_mcp_servers_view(engine: Engine) -> None:
    """Drop the result_mcp_servers_view if it exists."""
    drop_view_safe(engine, VIEW_NAME)
