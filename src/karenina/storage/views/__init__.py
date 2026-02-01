"""Database views for Karenina storage.

This package provides SQL views that offer convenient aggregated queries
for common use cases like benchmark summaries, verification statistics,
and model performance analysis.

Available Views:
    - template_results_view: One row per verification result (pass/fail)
    - template_attributes_view: One row per template attribute (for detailed analysis)
    - result_mcp_servers_view: One row per MCP server configured for a result
    - result_tools_used_view: One row per tool actually invoked during verification
    - combination_info_view: Distinct run/model combinations with MCP flag
    - models_used_view: All models used, with flags for answering/parsing roles
    - question_attributes_view: Attributes per question with Pydantic types
    - raw_llm_answers_view: Raw LLM response text per result
    - results_metadata_view: Execution metadata (tokens, timing, status flags)
    - rubric_traits_view: One row per rubric trait per result
    - deep_judgment_rubric_traits_view: Deep judgment rubric traits per result

All views use the flattened column naming scheme from the auto-generated
VerificationResultModel (e.g., metadata_answering_interface, metadata_answering_model_name, template_verify_result).

Usage:
    from karenina.storage.views import create_all_views, drop_all_views

    # Create all views
    create_all_views(engine)

    # Or create individual views
    from karenina.storage.views import create_template_results_view
    create_template_results_view(engine)
"""

from sqlalchemy.engine import Engine

# Import view modules for direct access (used by karenina-mcp)
from . import (
    combination_info,
    deep_judgment_rubric_traits,
    models_used,
    question_attributes,
    raw_llm_answers,
    result_mcp_servers,
    result_tools_used,
    results_metadata,
    rubric_traits,
    template_attributes,
    template_results,
)
from .combination_info import (
    VIEW_NAME as COMBINATION_INFO_VIEW,
)
from .combination_info import (
    create_combination_info_view,
    drop_combination_info_view,
)
from .deep_judgment_rubric_traits import (
    VIEW_NAME as DEEP_JUDGMENT_RUBRIC_TRAITS_VIEW,
)
from .deep_judgment_rubric_traits import (
    create_deep_judgment_rubric_traits_view,
    drop_deep_judgment_rubric_traits_view,
)
from .models_used import (
    VIEW_NAME as MODELS_USED_VIEW,
)
from .models_used import (
    create_models_used_view,
    drop_models_used_view,
)
from .question_attributes import (
    VIEW_NAME as QUESTION_ATTRIBUTES_VIEW,
)
from .question_attributes import (
    create_question_attributes_view,
    drop_question_attributes_view,
)
from .raw_llm_answers import (
    VIEW_NAME as RAW_LLM_ANSWERS_VIEW,
)
from .raw_llm_answers import (
    create_raw_llm_answers_view,
    drop_raw_llm_answers_view,
)
from .result_mcp_servers import (
    VIEW_NAME as RESULT_MCP_SERVERS_VIEW,
)
from .result_mcp_servers import (
    create_result_mcp_servers_view,
    drop_result_mcp_servers_view,
)
from .result_tools_used import (
    VIEW_NAME as RESULT_TOOLS_USED_VIEW,
)
from .result_tools_used import (
    create_result_tools_used_view,
    drop_result_tools_used_view,
)
from .results_metadata import (
    VIEW_NAME as RESULTS_METADATA_VIEW,
)
from .results_metadata import (
    create_results_metadata_view,
    drop_results_metadata_view,
)
from .rubric_traits import (
    VIEW_NAME as RUBRIC_TRAITS_VIEW,
)
from .rubric_traits import (
    create_rubric_traits_view,
    drop_rubric_traits_view,
)
from .template_attributes import (
    VIEW_NAME as TEMPLATE_ATTRIBUTES_VIEW,
)
from .template_attributes import (
    create_template_attributes_view,
    drop_template_attributes_view,
)
from .template_results import (
    VIEW_NAME as TEMPLATE_RESULTS_VIEW,
)
from .template_results import (
    create_template_results_view,
    drop_template_results_view,
)

# All view names for reference
ALL_VIEW_NAMES = [
    TEMPLATE_RESULTS_VIEW,
    TEMPLATE_ATTRIBUTES_VIEW,
    RESULT_MCP_SERVERS_VIEW,
    RESULT_TOOLS_USED_VIEW,
    COMBINATION_INFO_VIEW,
    MODELS_USED_VIEW,
    QUESTION_ATTRIBUTES_VIEW,
    RAW_LLM_ANSWERS_VIEW,
    RESULTS_METADATA_VIEW,
    RUBRIC_TRAITS_VIEW,
    DEEP_JUDGMENT_RUBRIC_TRAITS_VIEW,
]

__all__ = [
    # View modules (for karenina-mcp docstring access)
    "combination_info",
    "deep_judgment_rubric_traits",
    "models_used",
    "question_attributes",
    "raw_llm_answers",
    "result_mcp_servers",
    "result_tools_used",
    "results_metadata",
    "rubric_traits",
    "template_attributes",
    "template_results",
    # View names
    "ALL_VIEW_NAMES",
    "TEMPLATE_RESULTS_VIEW",
    "TEMPLATE_ATTRIBUTES_VIEW",
    "RESULT_MCP_SERVERS_VIEW",
    "RESULT_TOOLS_USED_VIEW",
    "COMBINATION_INFO_VIEW",
    "MODELS_USED_VIEW",
    "QUESTION_ATTRIBUTES_VIEW",
    "RAW_LLM_ANSWERS_VIEW",
    "RESULTS_METADATA_VIEW",
    "RUBRIC_TRAITS_VIEW",
    "DEEP_JUDGMENT_RUBRIC_TRAITS_VIEW",
    # Aggregate functions
    "create_all_views",
    "drop_all_views",
    # Individual create functions
    "create_template_results_view",
    "create_template_attributes_view",
    "create_result_mcp_servers_view",
    "create_result_tools_used_view",
    "create_combination_info_view",
    "create_models_used_view",
    "create_question_attributes_view",
    "create_raw_llm_answers_view",
    "create_results_metadata_view",
    "create_rubric_traits_view",
    "create_deep_judgment_rubric_traits_view",
    # Individual drop functions
    "drop_template_results_view",
    "drop_template_attributes_view",
    "drop_result_mcp_servers_view",
    "drop_result_tools_used_view",
    "drop_combination_info_view",
    "drop_models_used_view",
    "drop_question_attributes_view",
    "drop_raw_llm_answers_view",
    "drop_results_metadata_view",
    "drop_rubric_traits_view",
    "drop_deep_judgment_rubric_traits_view",
]


def create_all_views(engine: Engine) -> None:
    """Create all database views.

    Args:
        engine: SQLAlchemy engine instance
    """
    create_template_results_view(engine)
    create_template_attributes_view(engine)
    create_result_mcp_servers_view(engine)
    create_result_tools_used_view(engine)
    create_combination_info_view(engine)
    create_models_used_view(engine)
    create_question_attributes_view(engine)
    create_raw_llm_answers_view(engine)
    create_results_metadata_view(engine)
    create_rubric_traits_view(engine)
    create_deep_judgment_rubric_traits_view(engine)


def drop_all_views(engine: Engine) -> None:
    """Drop all database views.

    Args:
        engine: SQLAlchemy engine instance
    """
    drop_template_results_view(engine)
    drop_template_attributes_view(engine)
    drop_result_mcp_servers_view(engine)
    drop_result_tools_used_view(engine)
    drop_combination_info_view(engine)
    drop_models_used_view(engine)
    drop_question_attributes_view(engine)
    drop_raw_llm_answers_view(engine)
    drop_results_metadata_view(engine)
    drop_rubric_traits_view(engine)
    drop_deep_judgment_rubric_traits_view(engine)
