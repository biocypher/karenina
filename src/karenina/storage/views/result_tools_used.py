"""Result Tools Used View.

View Name: result_tools_used_view

Description:
    Shows one row per tool that was actually invoked during verification.
    Extracts the tools_used array from template_agent_metrics JSON object.

    Note: This shows tools that were *actually called*, not just configured.
    For configured MCP servers, see result_mcp_servers_view.

Columns:
    - result_id: Unique identifier for the verification result
    - tool_name: Name of a tool that was invoked (e.g., 'mcp__brave_search')

Source Tables:
    - verification_results (vr)

Source Column:
    - template_agent_metrics: JSON object containing agent execution metrics
      Example: {
          "iterations": 3,
          "tool_calls": 5,
          "tools_used": ["mcp__brave_search", "mcp__read_resource"],
          "suspect_failed_tool_calls": 0,
          "suspect_failed_tools": []
      }

JSON Functions Used:
    - SQLite: json_each(), json_extract()
    - PostgreSQL: jsonb_array_elements_text(), -> operator

Example Query:
    -- Count how often each tool was used across results
    SELECT tool_name, COUNT(DISTINCT result_id) as result_count
    FROM result_tools_used_view
    GROUP BY tool_name
    ORDER BY result_count DESC;

    -- Find results that used a specific tool
    SELECT result_id FROM result_tools_used_view
    WHERE tool_name = 'mcp__brave_search';

    -- Join with result_mcp_servers_view to compare configured vs used
    SELECT
        s.result_id,
        s.mcp_server as configured,
        t.tool_name as used
    FROM result_mcp_servers_view s
    LEFT JOIN result_tools_used_view t
        ON s.result_id = t.result_id
        AND t.tool_name LIKE '%' || s.mcp_server || '%';
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

VIEW_NAME = "result_tools_used_view"


def create_result_tools_used_view(engine: Engine) -> None:
    """Create or replace the result_tools_used_view."""
    # SQLite version using json_each on nested array
    view_sql_sqlite = """
        SELECT
            vr.metadata_result_id as result_id,
            tool.value as tool_name
        FROM verification_results vr
        CROSS JOIN json_each(json_extract(vr.template_agent_metrics, '$.tools_used')) as tool
        WHERE vr.template_agent_metrics IS NOT NULL
          AND json_extract(vr.template_agent_metrics, '$.tools_used') IS NOT NULL
    """

    # PostgreSQL version using jsonb operators
    view_sql_postgres = """
        SELECT
            vr.metadata_result_id as result_id,
            tool.value as tool_name
        FROM verification_results vr
        CROSS JOIN LATERAL jsonb_array_elements_text(
            (vr.template_agent_metrics::jsonb) -> 'tools_used'
        ) as tool(value)
        WHERE vr.template_agent_metrics IS NOT NULL
          AND (vr.template_agent_metrics::jsonb) -> 'tools_used' IS NOT NULL
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_sqlite}"))
        else:
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_postgres}"))


def drop_result_tools_used_view(engine: Engine) -> None:
    """Drop the result_tools_used_view if it exists."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
