"""combination_info_view

Distinct run/model/replicate combinations. Shows which model configurations were
used in each run, with replicate counts and MCP flags. Use for understanding
experiment design and filtering by model configuration.

Columns:
    run_id (TEXT): Unique identifier for the verification run (UUID)
    run_name (TEXT): Name of the verification run
    answering_model (TEXT): Name of the model that generated answers
    parsing_model (TEXT): Name of the model that parsed responses
    has_mcp (INTEGER): 1 if MCP servers were configured, 0 otherwise
    replicate_count (INTEGER): Number of replicates used for this combination

Keys:
    Primary: run_id + answering_model + parsing_model
    Joins: run_id → template_results_view.run_id
           run_name → template_results_view.run_name

Example:
    SELECT * FROM combination_info_view WHERE has_mcp = 1;
    SELECT * FROM combination_info_view WHERE replicate_count > 1;
"""

from sqlalchemy.engine import Engine

from .utils import create_view_safe, drop_view_safe

VIEW_NAME = "combination_info_view"

# SQLite version
_SQLITE_SQL = """
    SELECT
        run.id as run_id,
        run.run_name,
        vr.metadata_answering_model_name as answering_model,
        vr.metadata_parsing_model_name as parsing_model,
        MAX(CASE
            WHEN vr.template_answering_mcp_servers IS NOT NULL
                 AND json_array_length(vr.template_answering_mcp_servers) > 0
            THEN 1
            ELSE 0
        END) as has_mcp,
        COUNT(DISTINCT vr.metadata_replicate) as replicate_count
    FROM verification_results vr
    JOIN verification_runs run ON vr.run_id = run.id
    GROUP BY run.id, run.run_name, vr.metadata_answering_model_name, vr.metadata_parsing_model_name
"""

# PostgreSQL version
_POSTGRES_SQL = """
    SELECT
        run.id as run_id,
        run.run_name,
        vr.metadata_answering_model_name as answering_model,
        vr.metadata_parsing_model_name as parsing_model,
        MAX(CASE
            WHEN vr.template_answering_mcp_servers IS NOT NULL
                 AND jsonb_array_length(vr.template_answering_mcp_servers::jsonb) > 0
            THEN 1
            ELSE 0
        END) as has_mcp,
        COUNT(DISTINCT vr.metadata_replicate) as replicate_count
    FROM verification_results vr
    JOIN verification_runs run ON vr.run_id = run.id
    GROUP BY run.id, run.run_name, vr.metadata_answering_model_name, vr.metadata_parsing_model_name
"""


def create_combination_info_view(engine: Engine) -> None:
    """Create or replace the combination_info_view."""
    create_view_safe(engine, VIEW_NAME, _SQLITE_SQL, _POSTGRES_SQL)


def drop_combination_info_view(engine: Engine) -> None:
    """Drop the combination_info_view if it exists."""
    drop_view_safe(engine, VIEW_NAME)
