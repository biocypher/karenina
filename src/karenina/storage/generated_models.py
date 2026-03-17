"""Auto-generated SQLAlchemy models from Pydantic schemas.

This module uses the PydanticSQLAlchemyMapper to automatically generate
the VerificationResultModel from the VerificationResult Pydantic class.
This eliminates manual column definition and ensures the schema stays
in sync with the domain model.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from ..schemas.verification import VerificationResult
from .auto_mapper import PydanticSQLAlchemyMapper
from .base import Base

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


# Configuration for how to flatten each component of VerificationResult
FLATTEN_CONFIG: dict[str, dict[str, Any]] = {
    "metadata": {"prefix": "metadata_", "optional": False},
    "template": {"prefix": "template_", "optional": True},
    "rubric": {"prefix": "rubric_", "optional": True},
    "deep_judgment": {"prefix": "dj_", "optional": True},
    "deep_judgment_rubric": {"prefix": "djr_", "optional": True},
    # Root-level fields don't need prefixes
    "evaluation_input": {"prefix": "", "optional": True},
    "used_full_trace": {"prefix": "", "optional": False},
    "trace_extraction_error": {"prefix": "", "optional": True},
}


def _generate_verification_result_model() -> type:
    """Generate the VerificationResultModel class dynamically.

    This function creates a SQLAlchemy ORM model that mirrors the
    VerificationResult Pydantic model, with all nested fields flattened
    into prefixed columns for high queryability.

    Returns:
        Generated SQLAlchemy model class
    """
    mapper = PydanticSQLAlchemyMapper()

    # Generate column specifications from Pydantic model
    columns = mapper.generate_columns_for_result(VerificationResult, FLATTEN_CONFIG)

    # Define additional columns not from Pydantic (ORM-specific)
    # Note: question_id is duplicated (also in metadata_question_id) for the foreign key relationship
    extra_columns: dict[str, Column[Any]] = {
        "id": Column(Integer, primary_key=True, autoincrement=True),
        "run_id": Column(String(36), ForeignKey("verification_runs.id", ondelete="CASCADE"), nullable=False),
        "question_id": Column(String(32), ForeignKey("questions.id", ondelete="CASCADE"), nullable=False),
        "created_at": Column(DateTime, default=lambda: datetime.now(UTC), nullable=False),
    }

    # Define composite indexes for common query patterns
    composite_indexes = [
        ("run_id", "metadata_question_id"),  # Results for a run
        ("metadata_answering_interface", "metadata_answering_model_name"),  # Answering model lookup
        ("metadata_parsing_interface", "metadata_parsing_model_name"),  # Parsing model lookup
        ("metadata_question_id", "metadata_answering_model_name", "metadata_timestamp"),  # Question history
    ]

    # Build table args with indexes
    table_args = tuple(Index(f"idx_vr_{'_'.join(cols)}", *cols) for cols in composite_indexes) + (
        {"extend_existing": True},
    )

    # Create the model class
    model = mapper.create_model_class(
        base=Base,
        name="VerificationResultModel",
        tablename="verification_results",
        columns=columns,
        extra_columns=extra_columns,
        relationships={
            "run": relationship("VerificationRunModel", back_populates="results"),
            "question": relationship("QuestionModel", back_populates="verification_results"),
        },
        table_args=table_args,
    )

    return model


# Generate the model at module load time
# This creates the class once and caches it
VerificationResultModel = _generate_verification_result_model()


def get_column_names() -> list[str]:
    """Get all column names from the generated model.

    Useful for debugging and introspection.
    """
    if hasattr(VerificationResultModel, "__table__"):
        return [c.name for c in VerificationResultModel.__table__.columns]
    return []


def get_indexed_columns() -> list[str]:
    """Get column names that have individual indexes.

    Useful for debugging and introspection.
    """
    if hasattr(VerificationResultModel, "__table__"):
        indexed = []
        for c in VerificationResultModel.__table__.columns:
            if c.index:
                indexed.append(c.name)
        return indexed
    return []


def print_schema_info() -> None:
    """Print schema information for debugging.

    Shows all columns, their types, and indexes.
    """
    if not hasattr(VerificationResultModel, "__table__"):
        logger.info("Model has no __table__ attribute")
        return

    table = VerificationResultModel.__table__
    logger.info("Table: %s", table.name)
    logger.info("Columns (%d):", len(table.columns))

    for col in table.columns:
        index_marker = " [INDEXED]" if col.index else ""
        nullable = "NULL" if col.nullable else "NOT NULL"
        logger.info("  %s: %s %s%s", col.name, col.type, nullable, index_marker)

    logger.info("Indexes (%d):", len(table.indexes))
    for idx in table.indexes:
        cols = ", ".join(c.name for c in idx.columns)
        logger.info("  %s: (%s)", idx.name, cols)
