"""Tests for MetricRubricTrait functionality.

This module tests:
- Schema validation (valid/invalid metric trait configurations)
- Rubric integration (get_metric_trait_names, trait inclusion)
- VerificationResult serialization with metric data
"""

import pytest
from pydantic import ValidationError

from karenina.schemas import MetricRubricTrait, Rubric, RubricTrait


class TestMetricRubricTraitValidation:
    """Test schema validation for MetricRubricTrait."""

    def test_valid_metric_trait_all_buckets(self):
        """Test valid metric trait with all instruction buckets."""
        trait = MetricRubricTrait(
            name="diagnosis_accuracy",
            description="Evaluate diagnosis accuracy using confusion matrix",
            metrics=["precision", "recall", "accuracy", "f1"],
            tp_instructions=["correctly identifies condition"],
            tn_instructions=["correctly rules out condition"],
            fp_instructions=["incorrectly identifies condition"],
            fn_instructions=["misses actual condition"],
            repeated_extraction=True,
        )

        assert trait.name == "diagnosis_accuracy"
        assert len(trait.metrics) == 4
        assert len(trait.tp_instructions) == 1
        assert trait.repeated_extraction is True

    def test_valid_metric_trait_partial_buckets(self):
        """Test valid metric trait with only precision metrics (TP + FP)."""
        trait = MetricRubricTrait(
            name="precision_only",
            description="Only evaluate precision",
            metrics=["precision"],
            tp_instructions=["correct match"],
            fp_instructions=["incorrect match"],
        )

        assert "precision" in trait.metrics
        assert len(trait.tp_instructions) == 1
        assert len(trait.fp_instructions) == 1

    def test_invalid_metric_precision_missing_buckets(self):
        """Test that precision requires TP + FP instructions."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_precision",
                metrics=["precision"],
                tp_instructions=["correct match"],
                # Missing fp_instructions
            )

        assert "precision" in str(exc_info.value).lower()

    def test_invalid_metric_recall_missing_buckets(self):
        """Test that recall requires TP + FN instructions."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_recall",
                metrics=["recall"],
                tp_instructions=["correct match"],
                # Missing fn_instructions
            )

        assert "recall" in str(exc_info.value).lower()

    def test_invalid_metric_specificity_missing_buckets(self):
        """Test that specificity requires TN + FP instructions."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_specificity",
                metrics=["specificity"],
                tn_instructions=["correct non-match"],
                # Missing fp_instructions
            )

        assert "specificity" in str(exc_info.value).lower()

    def test_invalid_metric_accuracy_missing_buckets(self):
        """Test that accuracy requires all four buckets."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_accuracy",
                metrics=["accuracy"],
                tp_instructions=["correct match"],
                fp_instructions=["incorrect match"],
                # Missing tn_instructions and fn_instructions
            )

        assert "accuracy" in str(exc_info.value).lower()

    def test_invalid_metric_f1_missing_buckets(self):
        """Test that F1 requires TP + FP + FN instructions."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_f1",
                metrics=["f1"],
                tp_instructions=["correct match"],
                fp_instructions=["incorrect match"],
                # Missing fn_instructions
            )

        assert "f1" in str(exc_info.value).lower()

    def test_invalid_empty_instructions(self):
        """Test that at least one instruction bucket must be non-empty."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="empty_trait",
                metrics=["precision"],
            )

        assert "at least one" in str(exc_info.value).lower()

    def test_invalid_metric_name(self):
        """Test that invalid metric names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_metric",
                metrics=["invalid_metric"],
                tp_instructions=["correct match"],
                fp_instructions=["incorrect match"],
            )

        assert "invalid" in str(exc_info.value).lower() or "valid" in str(exc_info.value).lower()

    def test_repeated_extraction_default(self):
        """Test that repeated_extraction defaults to True."""
        trait = MetricRubricTrait(
            name="default_dedup",
            metrics=["precision"],
            tp_instructions=["correct"],
            fp_instructions=["incorrect"],
        )

        assert trait.repeated_extraction is True


class TestRubricWithMetricTraits:
    """Test Rubric class integration with metric traits."""

    def test_rubric_with_metric_traits(self):
        """Test creating rubric with metric traits."""
        trait = MetricRubricTrait(
            name="metric1",
            metrics=["precision"],
            tp_instructions=["correct"],
            fp_instructions=["incorrect"],
        )

        rubric = Rubric(traits=[], metric_traits=[trait])

        assert len(rubric.metric_traits) == 1
        assert rubric.metric_traits[0].name == "metric1"

    def test_get_metric_trait_names(self):
        """Test get_metric_trait_names() method."""
        trait1 = MetricRubricTrait(
            name="metric1",
            metrics=["precision"],
            tp_instructions=["correct"],
            fp_instructions=["incorrect"],
        )
        trait2 = MetricRubricTrait(
            name="metric2",
            metrics=["recall"],
            tp_instructions=["correct"],
            fn_instructions=["missed"],
        )

        rubric = Rubric(traits=[], metric_traits=[trait1, trait2])
        names = rubric.get_metric_trait_names()

        assert names == ["metric1", "metric2"]

    def test_get_trait_names_includes_metric_traits(self):
        """Test that get_trait_names() includes metric traits."""

        llm_trait = RubricTrait(name="llm1", kind="score", min_score=0, max_score=10)
        metric_trait = MetricRubricTrait(
            name="metric1",
            metrics=["precision"],
            tp_instructions=["correct"],
            fp_instructions=["incorrect"],
        )

        rubric = Rubric(traits=[llm_trait], metric_traits=[metric_trait])
        names = rubric.get_trait_names()

        assert "llm1" in names
        assert "metric1" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
