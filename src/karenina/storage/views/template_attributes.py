"""Template Attributes View.

View Name: template_attributes_view

Description:
    Shows verification results disaggregated by template attribute.
    One row per attribute in the parsed template response.
    Useful for analyzing which specific attributes the model got right or wrong.

Columns:
    - result_id: Unique identifier for the verification result
    - verification_date: Date the verification was performed
    - run_name: Name of the verification run
    - benchmark_name: Name of the benchmark
    - question_id: Unique identifier for the question
    - question_text: The question text
    - attribute_name: Name of the template attribute (e.g., 'gene_name', 'tissue')
    - gt_value: Ground truth value for this attribute
    - llm_value: Value extracted from the LLM response
    - attribute_match: 1 if gt_value == llm_value, 0 otherwise

Source Tables:
    - verification_results (vr)
    - verification_runs (run)
    - benchmarks (b)

JSON Functions Used:
    - SQLite: json_each(), json_extract()
    - PostgreSQL: jsonb_each_text(), ->> operator

Example Query:
    SELECT attribute_name,
           SUM(attribute_match) as correct,
           COUNT(*) as total
    FROM template_attributes_view
    GROUP BY attribute_name;
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

VIEW_NAME = "template_attributes_view"


def create_template_attributes_view(engine: Engine) -> None:
    """Create or replace the template_attributes_view."""
    # SQLite version using json_each
    view_sql_sqlite = """
        SELECT
            vr.metadata_result_id as result_id,
            DATE(vr.created_at) as verification_date,
            run.run_name,
            b.name as benchmark_name,
            vr.question_id,
            vr.metadata_question_text as question_text,
            gt.key as attribute_name,
            gt.value as gt_value,
            json_extract(vr.template_parsed_llm_response, '$.' || gt.key) as llm_value,
            CASE
                WHEN gt.value = json_extract(vr.template_parsed_llm_response, '$.' || gt.key)
                THEN 1
                ELSE 0
            END as attribute_match
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        JOIN benchmarks b ON run.benchmark_id = b.id
        CROSS JOIN json_each(vr.template_parsed_gt_response) as gt
        WHERE vr.template_parsed_gt_response IS NOT NULL
    """

    # PostgreSQL version using jsonb_each_text
    view_sql_postgres = """
        SELECT
            vr.metadata_result_id as result_id,
            DATE(vr.created_at) as verification_date,
            run.run_name,
            b.name as benchmark_name,
            vr.question_id,
            vr.metadata_question_text as question_text,
            gt.key as attribute_name,
            gt.value as gt_value,
            vr.template_parsed_llm_response ->> gt.key as llm_value,
            CASE
                WHEN gt.value = (vr.template_parsed_llm_response ->> gt.key)
                THEN 1
                ELSE 0
            END as attribute_match
        FROM verification_results vr
        JOIN verification_runs run ON vr.run_id = run.id
        JOIN benchmarks b ON run.benchmark_id = b.id
        CROSS JOIN LATERAL jsonb_each_text(vr.template_parsed_gt_response::jsonb) as gt
        WHERE vr.template_parsed_gt_response IS NOT NULL
    """

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_sqlite}"))
        else:
            conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
            conn.execute(text(f"CREATE VIEW {VIEW_NAME} AS {view_sql_postgres}"))


def drop_template_attributes_view(engine: Engine) -> None:
    """Drop the template_attributes_view if it exists."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
