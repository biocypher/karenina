"""Tests for the results_metadata SQLAlchemy view column contract.

Guards the public column surface of ``results_metadata_view`` against
accidental regressions. The legacy ``completed_without_errors`` /
``error`` / ``error_category`` / ``failed_stage`` fields must no longer
appear; the unified failure taxonomy columns plus ``caveats`` must.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect

from karenina.storage.base import Base
from karenina.storage.generated_models import VerificationResultModel  # noqa: F401  (registers table on Base)
from karenina.storage.models import (  # noqa: F401  (registers tables on Base)
    BenchmarkModel,
    BenchmarkQuestionModel,
    ImportMetadataModel,
    QuestionModel,
    VerificationRunModel,
)
from karenina.storage.views.results_metadata import create_results_metadata_view


def _reflect_view_columns() -> set[str]:
    """Create tables + the view in an in-memory SQLite and reflect column names.

    Returns:
        Set of column names present on ``results_metadata_view``.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    create_results_metadata_view(engine)
    return {col["name"] for col in inspect(engine).get_columns("results_metadata_view")}


@pytest.mark.unit
class TestResultsMetadataView:
    """Contract tests for the ``results_metadata_view`` column surface."""

    def test_legacy_columns_removed(self) -> None:
        """Legacy error-state columns must no longer appear on the view."""
        cols = _reflect_view_columns()
        for legacy in (
            "metadata_completed_without_errors",
            "metadata_error",
            "metadata_error_category",
            "metadata_failed_stage",
            "completed_without_errors",
            "error",
            "error_category",
            "failed_stage",
        ):
            assert legacy not in cols, f"Legacy column still present: {legacy}"

    def test_failure_columns_present(self) -> None:
        """New failure-taxonomy columns and caveats must be exposed."""
        cols = _reflect_view_columns()
        for added in (
            "metadata_failure_category",
            "metadata_failure_group",
            "metadata_failure_stage",
            "metadata_failure_reason",
            "metadata_caveats",
        ):
            assert added in cols, f"New column missing: {added}"
