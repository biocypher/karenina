"""Results Metadata View.

View Name: results_metadata_view

Description:
    Comprehensive execution metadata including token counts, timing, agent metrics,
    and status flags. One row per verification result.

Columns:
    - result_id: Unique identifier for the verification result
    - has_template_results: 1 if template verification was performed, 0 otherwise
    - has_rubric_results: 1 if rubric evaluation was performed, 0 otherwise
    - execution_time: Total execution time in seconds
    - timestamp: ISO timestamp of verification
    - input_tokens: Total input tokens across all stages (may be NULL)
    - output_tokens: Total output tokens across all stages (may be NULL)
    - total_tokens: Total tokens across all stages (may be NULL)
    - agent_iterations: Number of agent think-act cycles (NULL if no agent)
    - agent_tool_calls: Total tool invocations (NULL if no agent)
    - has_mcp: 1 if MCP servers were configured, 0 otherwise
    - used_full_trace: 1 if full trace was used, 0 if only final AI message
    - has_trace_extraction_error: 1 if trace extraction failed, 0 otherwise
    - abstention_detected: Whether model abstained (bool or NULL)
    - abstention_reasoning: Explanation text if abstention detected (NULL otherwise)
    - embedding_check_performed: 1 if embedding check ran, 0 otherwise
    - recursion_limit_reached: 1 if agent hit recursion limit, 0 otherwise
    - completed_without_errors: Overall success flag

Source Tables:
    - verification_results (vr)

JSON Fields Used:
    - template_usage_metadata: Token counts structure
    - template_agent_metrics: Agent execution metrics

Example Query:
    -- Find results with high token usage
    SELECT result_id, total_tokens, execution_time
    FROM results_metadata_view
    WHERE total_tokens > 10000
    ORDER BY total_tokens DESC;

    -- Find results where abstention was detected
    SELECT result_id, abstention_reasoning
    FROM results_metadata_view
    WHERE abstention_detected = 1;
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

VIEW_NAME = "results_metadata_view"


def create_results_metadata_view(engine: Engine) -> None:
    """Create or replace the results_metadata_view."""
    # SQLite version
    view_sql_sqlite = """
        SELECT
            vr.metadata_result_id as result_id,
            COALESCE(vr.template_template_verification_performed, 0) as has_template_results,
            COALESCE(vr.rubric_rubric_evaluation_performed, 0) as has_rubric_results,
            vr.metadata_execution_time as execution_time,
            vr.metadata_timestamp as timestamp,
            CAST(json_extract(vr.template_usage_metadata, '$.total.input_tokens') AS INTEGER) as input_tokens,
            CAST(json_extract(vr.template_usage_metadata, '$.total.output_tokens') AS INTEGER) as output_tokens,
            CAST(json_extract(vr.template_usage_metadata, '$.total.total_tokens') AS INTEGER) as total_tokens,
            CAST(json_extract(vr.template_agent_metrics, '$.iterations') AS INTEGER) as agent_iterations,
            CAST(json_extract(vr.template_agent_metrics, '$.tool_calls') AS INTEGER) as agent_tool_calls,
            CASE
                WHEN vr.template_answering_mcp_servers IS NOT NULL
                     AND json_array_length(vr.template_answering_mcp_servers) > 0
                THEN 1
                ELSE 0
            END as has_mcp,
            COALESCE(vr.used_full_trace, 1) as used_full_trace,
            CASE WHEN vr.trace_extraction_error IS NOT NULL THEN 1 ELSE 0 END as has_trace_extraction_error,
            vr.template_abstention_detected as abstention_detected,
            vr.template_abstention_reasoning as abstention_reasoning,
            COALESCE(vr.template_embedding_check_performed, 0) as embedding_check_performed,
            COALESCE(vr.template_recursion_limit_reached, 0) as recursion_limit_reached,
            vr.metadata_completed_without_errors as completed_without_errors
        FROM verification_results vr
    """

    # PostgreSQL version
    view_sql_postgres = """
        SELECT
            vr.metadata_result_id as result_id,
            COALESCE(vr.template_template_verification_performed::INTEGER, 0) as has_template_results,
            COALESCE(vr.rubric_rubric_evaluation_performed::INTEGER, 0) as has_rubric_results,
            vr.metadata_execution_time as execution_time,
            vr.metadata_timestamp as timestamp,
            ((vr.template_usage_metadata::jsonb) -> 'total' ->> 'input_tokens')::INTEGER as input_tokens,
            ((vr.template_usage_metadata::jsonb) -> 'total' ->> 'output_tokens')::INTEGER as output_tokens,
            ((vr.template_usage_metadata::jsonb) -> 'total' ->> 'total_tokens')::INTEGER as total_tokens,
            ((vr.template_agent_metrics::jsonb) ->> 'iterations')::INTEGER as agent_iterations,
            ((vr.template_agent_metrics::jsonb) ->> 'tool_calls')::INTEGER as agent_tool_calls,
            CASE
                WHEN vr.template_answering_mcp_servers IS NOT NULL
                     AND jsonb_array_length(vr.template_answering_mcp_servers::jsonb) > 0
                THEN 1
                ELSE 0
            END as has_mcp,
            COALESCE(vr.used_full_trace::INTEGER, 1) as used_full_trace,
            CASE WHEN vr.trace_extraction_error IS NOT NULL THEN 1 ELSE 0 END as has_trace_extraction_error,
            vr.template_abstention_detected as abstention_detected,
            vr.template_abstention_reasoning as abstention_reasoning,
            COALESCE(vr.template_embedding_check_performed::INTEGER, 0) as embedding_check_performed,
            COALESCE(vr.template_recursion_limit_reached::INTEGER, 0) as recursion_limit_reached,
            vr.metadata_completed_without_errors as completed_without_errors
        FROM verification_results vr
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_sqlite}"))
        else:
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_postgres}"))


def drop_results_metadata_view(engine: Engine) -> None:
    """Drop the results_metadata_view if it exists."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
