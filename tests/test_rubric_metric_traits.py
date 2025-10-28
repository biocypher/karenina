"""Tests for MetricRubricTrait functionality.

This module tests:
- Schema validation (valid/invalid metric trait configurations)
- Rubric integration (get_metric_trait_names, trait inclusion)
- Precision-only metric computation (TP/TN instructions)
"""

import pytest
from pydantic import ValidationError

from karenina.schemas import MetricRubricTrait, Rubric, RubricTrait


class TestMetricRubricTraitValidation:
    """Test schema validation for MetricRubricTrait."""

    def test_valid_metric_trait_with_both_instructions(self):
        """Test valid metric trait with TP and TN instructions."""
        trait = MetricRubricTrait(
            name="extraction_precision",
            description="Evaluate precision of model extractions",
            metrics=["precision"],
            tp_instructions=["correctly identifies key information", "mentions mechanism"],
            tn_instructions=["includes off-topic content", "mentions side effects"],
            repeated_extraction=True,
        )

        assert trait.name == "extraction_precision"
        assert trait.metrics == ["precision"]
        assert len(trait.tp_instructions) == 2
        assert len(trait.tn_instructions) == 2
        assert trait.repeated_extraction is True

    def test_valid_metric_trait_default_repeated_extraction(self):
        """Test that repeated_extraction defaults to True."""
        trait = MetricRubricTrait(
            name="test_trait",
            metrics=["precision"],
            tp_instructions=["correct"],
            tn_instructions=["incorrect"],
        )

        assert trait.repeated_extraction is True

    def test_invalid_metric_missing_tp_instructions(self):
        """Test that TP instructions are required."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_trait",
                metrics=["precision"],
                tp_instructions=[],  # Empty TP instructions
                tn_instructions=["incorrect match"],
            )

        error_msg = str(exc_info.value).lower()
        assert "tp" in error_msg or "correct" in error_msg

    def test_invalid_metric_missing_tn_instructions(self):
        """Test that TN instructions are required."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_trait",
                metrics=["precision"],
                tp_instructions=["correct match"],
                tn_instructions=[],  # Empty TN instructions
            )

        error_msg = str(exc_info.value).lower()
        assert "tn" in error_msg or "incorrect" in error_msg

    def test_invalid_empty_instructions(self):
        """Test that at least one instruction is required in each bucket."""
        with pytest.raises(ValidationError):
            MetricRubricTrait(
                name="bad_trait",
                metrics=["precision"],
                tp_instructions=[],
                tn_instructions=[],
            )

    def test_invalid_metric_name(self):
        """Test that only 'precision' is a valid metric."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_metric",
                metrics=["recall"],  # Recall is no longer supported
                tp_instructions=["correct"],
                tn_instructions=["incorrect"],
            )

        error_msg = str(exc_info.value).lower()
        assert "metric" in error_msg or "precision" in error_msg

    def test_invalid_unsupported_metrics(self):
        """Test that unsupported metrics (recall, f1, etc.) are rejected."""
        unsupported = ["recall", "f1", "accuracy", "specificity"]
        for metric in unsupported:
            with pytest.raises(ValidationError):
                MetricRubricTrait(
                    name="test",
                    metrics=[metric],
                    tp_instructions=["correct"],
                    tn_instructions=["incorrect"],
                )

    def test_empty_metrics_list(self):
        """Test that at least one metric must be specified."""
        with pytest.raises(ValidationError):
            MetricRubricTrait(
                name="no_metrics",
                metrics=[],  # Empty metrics list
                tp_instructions=["correct"],
                tn_instructions=["incorrect"],
            )


class TestRubricIntegration:
    """Test MetricRubricTrait integration with Rubric."""

    def test_rubric_with_metric_traits(self):
        """Test creating a rubric with metric traits."""
        trait1 = MetricRubricTrait(
            name="precision_trait",
            metrics=["precision"],
            tp_instructions=["correct extraction"],
            tn_instructions=["incorrect extraction"],
        )

        rubric = Rubric(
            traits=[],
            manual_traits=[],
            metric_traits=[trait1],
        )

        assert len(rubric.metric_traits) == 1
        assert rubric.metric_traits[0].name == "precision_trait"

    def test_get_metric_trait_names(self):
        """Test get_metric_trait_names() method."""
        trait1 = MetricRubricTrait(
            name="trait_a",
            metrics=["precision"],
            tp_instructions=["correct"],
            tn_instructions=["incorrect"],
        )
        trait2 = MetricRubricTrait(
            name="trait_b",
            metrics=["precision"],
            tp_instructions=["correct"],
            tn_instructions=["incorrect"],
        )

        rubric = Rubric(metric_traits=[trait1, trait2])
        names = rubric.get_metric_trait_names()

        assert names == ["trait_a", "trait_b"]

    def test_get_trait_names_includes_metric_traits(self):
        """Test that get_trait_names() includes metric traits."""
        llm_trait = RubricTrait(name="llm_trait", kind="boolean")
        metric_trait = MetricRubricTrait(
            name="metric_trait",
            metrics=["precision"],
            tp_instructions=["correct"],
            tn_instructions=["incorrect"],
        )

        rubric = Rubric(traits=[llm_trait], metric_traits=[metric_trait])
        all_names = rubric.get_trait_names()

        assert "llm_trait" in all_names
        assert "metric_trait" in all_names
        assert len(all_names) == 2
