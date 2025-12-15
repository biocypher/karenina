"""result_tools_used_view

Tools actually invoked during verification. One row per tool per result. Shows
tools that were *actually called* during agent execution. Use for analyzing
tool usage patterns and comparing to configured MCP servers.

Columns:
    result_id (TEXT): Unique identifier for the verification result
    tool_name (TEXT): Name of the tool invoked (e.g., 'mcp__brave_search')

Keys:
    Primary: result_id + tool_name
    Joins: result_id → template_results_view.result_id

Example:
    SELECT tool_name, COUNT(DISTINCT result_id) as result_count
    FROM result_tools_used_view GROUP BY tool_name ORDER BY result_count DESC;
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
