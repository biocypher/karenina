"""Shared helpers and base test classes for DataFrame integration tests.

This module consolidates common code used across DataFrame integration tests:
- test_dataframe_integration.py
- test_dataframe_integration_deep_judgment.py
- test_dataframe_integration_rubrics.py

It provides:
- _create_metadata(): Helper to create VerificationResultMetadata objects
- CommonColumnTestMixin: Mixin class with shared DataFrame consistency tests
- PandasOperationsTestMixin: Mixin class with shared pandas operations tests

Usage:
    from tests.integration.dataframe_helpers import (
        create_metadata,
        CommonColumnTestMixin,
        PandasOperationsTestMixin,
    )
"""

from datetime import UTC, datetime

import pandas as pd

from karenina.schemas.results import JudgmentResults, RubricResults, TemplateResults
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultMetadata,
)
from karenina.schemas.verification.model_identity import ModelIdentity


def create_metadata(
    question_id: str,
    answering_model: str = "claude-haiku-4-5",
    completed: bool = True,
    error: str | None = None,
) -> VerificationResultMetadata:
    """Helper to create metadata with computed result_id.

    Args:
        question_id: Unique identifier for the question
        answering_model: Model used for answering (default: claude-haiku-4-5)
        completed: Whether the verification completed without errors
        error: Optional error message if verification failed

    Returns:
        VerificationResultMetadata instance with computed result_id
    """
    timestamp = datetime.now(UTC).isoformat()
    _answering = ModelIdentity(interface="langchain", model_name=answering_model)
    _parsing = ModelIdentity(interface="langchain", model_name="claude-haiku-4-5")
    return VerificationResultMetadata(
        question_id=question_id,
        template_id="test-template-id",
        completed_without_errors=completed,
        error=error,
        question_text=f"Question text for {question_id}",
        raw_answer="Expected answer",
        answering=_answering,
        parsing=_parsing,
        execution_time=1.5,
        timestamp=timestamp,
        result_id=VerificationResultMetadata.compute_result_id(
            question_id=question_id,
            answering=_answering,
            parsing=_parsing,
            timestamp=timestamp,
        ),
    )


# =============================================================================
# Base Test Mixins
# =============================================================================


class CommonColumnTestMixin:
    """Mixin class providing common column consistency tests.

    Subclasses should provide:
    - verification_results_list fixture (list[VerificationResult])
    - Define which result types to test via class attributes:
        - test_template: bool = True
        - test_rubric: bool = True
        - test_judgment: bool = True
    """

    test_template: bool = True
    test_rubric: bool = True
    test_judgment: bool = True

    def test_common_columns_consistency(self, verification_results_list: list[VerificationResult]):
        """Test that common columns are consistent across all DataFrame types."""
        # Common columns that should exist in all DataFrames
        common_columns = [
            "completed_without_errors",
            "question_id",
            "answering_model",
        ]

        if self.test_template:
            template_results = TemplateResults(results=verification_results_list)
            template_df = template_results.to_dataframe()
            for col in common_columns:
                assert col in template_df.columns, f"TemplateResults missing: {col}"

        if self.test_rubric:
            # Only test if results have rubric data
            has_rubric = any(r.rubric is not None for r in verification_results_list)
            if has_rubric:
                rubric_results = RubricResults(results=verification_results_list)
                rubric_df = rubric_results.to_dataframe(trait_type="all")
                for col in common_columns:
                    assert col in rubric_df.columns, f"RubricResults missing: {col}"

        if self.test_judgment:
            # Only test if results have deep judgment data
            has_judgment = any(r.deep_judgment is not None for r in verification_results_list)
            if has_judgment:
                judgment_results = JudgmentResults(results=verification_results_list)
                judgment_df = judgment_results.to_dataframe()
                for col in common_columns:
                    assert col in judgment_df.columns, f"JudgmentResults missing: {col}"

    def test_status_column_first(self, verification_results_list: list[VerificationResult]):
        """Test that status column appears first in all DataFrames."""
        if self.test_template:
            template_results = TemplateResults(results=verification_results_list)
            template_df = template_results.to_dataframe()
            assert template_df.columns[0] == "completed_without_errors"

        if self.test_rubric:
            has_rubric = any(r.rubric is not None for r in verification_results_list)
            if has_rubric:
                rubric_results = RubricResults(results=verification_results_list)
                rubric_df = rubric_results.to_dataframe(trait_type="all")
                assert rubric_df.columns[0] == "completed_without_errors"

        if self.test_judgment:
            has_judgment = any(r.deep_judgment is not None for r in verification_results_list)
            if has_judgment:
                judgment_results = JudgmentResults(results=verification_results_list)
                judgment_df = judgment_results.to_dataframe()
                assert judgment_df.columns[0] == "completed_without_errors"


class PandasOperationsTestMixin:
    """Mixin class providing common pandas operations tests.

    Subclasses should provide:
    - A fixture that returns a DataFrame for testing (via _get_test_dataframe method)

    Override _get_test_dataframe() to return the appropriate DataFrame.
    """

    def _get_test_dataframe(self, verification_results_list: list[VerificationResult]) -> pd.DataFrame:
        """Override this method to return the DataFrame for testing.

        Default implementation returns TemplateResults DataFrame.
        """
        template_results = TemplateResults(results=verification_results_list)
        return template_results.to_dataframe()

    def test_groupby_operations(self, verification_results_list: list[VerificationResult]):
        """Test pandas groupby operations on DataFrame."""
        df = self._get_test_dataframe(verification_results_list)

        # Test groupby question_id
        grouped = df.groupby("question_id")
        assert len(grouped) > 0

    def test_filtering_operations(self, verification_results_list: list[VerificationResult]):
        """Test pandas filtering operations on DataFrame."""
        df = self._get_test_dataframe(verification_results_list)

        # Filter to successful results only
        successful = df[df["completed_without_errors"]]
        assert len(successful) >= 0  # May be 0 if all failed

        # Filter to specific question
        if len(df) > 0:
            first_question = df["question_id"].iloc[0]
            question_df = df[df["question_id"] == first_question]
            assert len(question_df) > 0
            assert (question_df["question_id"] == first_question).all()
