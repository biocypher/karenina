"""question_attributes_view

Question metadata and Pydantic-typed template attributes. One row per attribute
per question. Use for understanding answer template structure and expected
attribute types (bool, str, int, float, list, dict).

Columns:
    benchmark_name (TEXT): Name of the benchmark
    question_id (TEXT): Unique identifier for the question (MD5 hash)
    question_text (TEXT): The question content
    attribute_name (TEXT): Name of the template attribute (e.g., 'gene_name', 'tissue')
    attribute_type (TEXT): Pydantic type (bool, str, int, float, list, dict, None)

Keys:
    Primary: question_id + attribute_name
    Joins: question_id → template_results_view.question_id, template_attributes_view.question_id

Example:
    SELECT attribute_name, attribute_type FROM question_attributes_view
    WHERE question_id = 'abc123';
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

VIEW_NAME = "question_attributes_view"


def create_question_attributes_view(engine: Engine) -> None:
    """Create or replace the question_attributes_view."""
    # SQLite version using json_each
    # gt.type returns: null, true, false, integer, real, text, array, object
    # We map to Pydantic types: None, bool, int, float, str, list, dict
    view_sql_sqlite = """
        SELECT DISTINCT
            b.name as benchmark_name,
            vr.question_id,
            vr.metadata_question_text as question_text,
            gt.key as attribute_name,
            CASE gt.type
                WHEN 'true' THEN 'bool'
                WHEN 'false' THEN 'bool'
                WHEN 'text' THEN 'str'
                WHEN 'integer' THEN 'int'
                WHEN 'real' THEN 'float'
                WHEN 'array' THEN 'list'
                WHEN 'object' THEN 'dict'
                WHEN 'null' THEN 'None'
                ELSE gt.type
            END as attribute_type
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        JOIN benchmarks b ON run.benchmark_id = b.id
        CROSS JOIN json_each(vr.template_parsed_gt_response) as gt
        WHERE vr.template_parsed_gt_response IS NOT NULL
    """

    # PostgreSQL version using jsonb_each
    # jsonb_typeof() returns: null, boolean, number, string, array, object
    view_sql_postgres = """
        SELECT DISTINCT
            b.name as benchmark_name,
            vr.question_id,
            vr.metadata_question_text as question_text,
            gt.key as attribute_name,
            CASE jsonb_typeof(gt.value)
                WHEN 'boolean' THEN 'bool'
                WHEN 'string' THEN 'str'
                WHEN 'number' THEN 'float'
                WHEN 'array' THEN 'list'
                WHEN 'object' THEN 'dict'
                WHEN 'null' THEN 'None'
                ELSE jsonb_typeof(gt.value)
            END as attribute_type
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        JOIN benchmarks b ON run.benchmark_id = b.id
        CROSS JOIN LATERAL jsonb_each(vr.template_parsed_gt_response::jsonb) as gt
        WHERE vr.template_parsed_gt_response IS NOT NULL
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_sqlite}"))
        else:
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_postgres}"))


def drop_question_attributes_view(engine: Engine) -> None:
    """Drop the question_attributes_view if it exists."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
