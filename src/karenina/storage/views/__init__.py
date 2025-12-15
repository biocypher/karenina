"""Database views for Karenina storage.

This package provides SQL views that offer convenient aggregated queries
for common use cases like benchmark summaries, verification statistics,
and model performance analysis.

Available Views:
    - template_results_view: One row per verification result (pass/fail)
    - template_attributes_view: One row per template attribute (for detailed analysis)
    - result_mcp_servers_view: One row per MCP server configured for a result
    - result_tools_used_view: One row per tool actually invoked during verification

All views use the flattened column naming scheme from the auto-generated
VerificationResultModel (e.g., metadata_answering_model, template_verify_result).

Usage:
    from karenina.storage.views import create_all_views, drop_all_views

    # Create all views
    create_all_views(engine)

    # Or create individual views
    from karenina.storage.views import create_template_results_view
    create_template_results_view(engine)
"""

from sqlalchemy.engine import Engine

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
]

__all__ = [
    # View names
    "ALL_VIEW_NAMES",
    "TEMPLATE_RESULTS_VIEW",
    "TEMPLATE_ATTRIBUTES_VIEW",
    "RESULT_MCP_SERVERS_VIEW",
    "RESULT_TOOLS_USED_VIEW",
    # Aggregate functions
    "create_all_views",
    "drop_all_views",
    # Individual create functions
    "create_template_results_view",
    "create_template_attributes_view",
    "create_result_mcp_servers_view",
    "create_result_tools_used_view",
    # Individual drop functions
    "drop_template_results_view",
    "drop_template_attributes_view",
    "drop_result_mcp_servers_view",
    "drop_result_tools_used_view",
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


def drop_all_views(engine: Engine) -> None:
    """Drop all database views.

    Args:
        engine: SQLAlchemy engine instance
    """
    drop_template_results_view(engine)
    drop_template_attributes_view(engine)
    drop_result_mcp_servers_view(engine)
    drop_result_tools_used_view(engine)
