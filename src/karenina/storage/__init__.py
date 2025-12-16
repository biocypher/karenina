"""Database storage module for Karenina.

This module provides SQLAlchemy-based persistence for benchmarks and verification results.

The VerificationResultModel is auto-generated from Pydantic schemas to ensure
the database schema stays in sync with the domain models automatically.
"""

from .base import Base
from .db_config import DBConfig
from .engine import (
    close_engine,
    create_session_factory,
    drop_database,
    get_engine,
    get_session,
    init_database,
    reset_database,
)
from .generated_models import FLATTEN_CONFIG, VerificationResultModel
from .models import (
    BenchmarkModel,
    BenchmarkQuestionModel,
    ImportMetadataModel,
    QuestionModel,
    VerificationRunModel,
)
from .operations import (
    import_verification_results,
    load_benchmark,
    load_verification_results,
    save_benchmark,
    save_verification_results,
)
from .queries import (
    get_benchmark_summary,
    get_database_statistics,
    get_failed_verifications,
    get_latest_verification_results,
    get_model_performance,
    get_question_usage,
    get_rubric_scores_aggregate,
    get_rubric_traits_by_type,
    get_verification_history_timeline,
    get_verification_run_summary,
)

__all__ = [
    # Configuration
    "DBConfig",
    "FLATTEN_CONFIG",
    # Engine and session management
    "get_engine",
    "get_session",
    "create_session_factory",
    "init_database",
    "drop_database",
    "reset_database",
    "close_engine",
    # Models
    "Base",
    "BenchmarkModel",
    "QuestionModel",
    "BenchmarkQuestionModel",
    "VerificationRunModel",
    "VerificationResultModel",
    "ImportMetadataModel",
    # Operations
    "save_benchmark",
    "load_benchmark",
    "save_verification_results",
    "load_verification_results",
    "import_verification_results",
    # Query helpers
    "get_benchmark_summary",
    "get_verification_run_summary",
    "get_question_usage",
    "get_latest_verification_results",
    "get_model_performance",
    "get_rubric_scores_aggregate",
    "get_verification_history_timeline",
    "get_failed_verifications",
    "get_rubric_traits_by_type",
    "get_database_statistics",
]
