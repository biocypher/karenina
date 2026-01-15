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

from sqlalchemy.engine import Engine

from .utils import create_view_safe, drop_view_safe

VIEW_NAME = "result_tools_used_view"

# SQLite version using json_each on nested array
_SQLITE_SQL = """
    SELECT
        vr.metadata_result_id as result_id,
        tool.value as tool_name
    FROM verification_results vr
    CROSS JOIN json_each(json_extract(vr.template_agent_metrics, '$.tools_used')) as tool
    WHERE vr.template_agent_metrics IS NOT NULL
      AND json_extract(vr.template_agent_metrics, '$.tools_used') IS NOT NULL
"""

# PostgreSQL version using jsonb operators
_POSTGRES_SQL = """
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


def create_result_tools_used_view(engine: Engine) -> None:
    """Create or replace the result_tools_used_view."""
    create_view_safe(engine, VIEW_NAME, _SQLITE_SQL, _POSTGRES_SQL)


def drop_result_tools_used_view(engine: Engine) -> None:
    """Drop the result_tools_used_view if it exists."""
    drop_view_safe(engine, VIEW_NAME)
