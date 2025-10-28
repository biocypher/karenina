"""Tests for MetricRubricTrait functionality.

This module tests:
- Schema validation for both evaluation modes (tp_only and full_matrix)
- Rubric integration (get_metric_trait_names, trait inclusion)
- Metric computability validation for each mode
"""

import pytest
from pydantic import ValidationError

from karenina.schemas import MetricRubricTrait, Rubric, RubricTrait


class TestMetricRubricTraitTPOnlyMode:
    """Test schema validation for MetricRubricTrait in tp_only mode."""

    def test_valid_tp_only_mode_with_precision(self):
        """Test valid tp_only mode trait with precision metric."""
        trait = MetricRubricTrait(
            name="extraction_precision",
            description="Evaluate precision of model extractions",
            evaluation_mode="tp_only",
            metrics=["precision"],
            tp_instructions=["correctly identifies key information", "mentions mechanism"],
            repeated_extraction=True,
        )

        assert trait.name == "extraction_precision"
        assert trait.evaluation_mode == "tp_only"
        assert trait.metrics == ["precision"]
        assert len(trait.tp_instructions) == 2
        assert len(trait.tn_instructions) == 0  # TN not required in tp_only
        assert trait.repeated_extraction is True

    def test_valid_tp_only_mode_with_recall_and_f1(self):
        """Test valid tp_only mode trait with recall and f1 metrics."""
        trait = MetricRubricTrait(
            name="extraction_recall_f1",
            description="Evaluate recall and f1 of model extractions",
            evaluation_mode="tp_only",
            metrics=["recall", "f1"],
            tp_instructions=["mentions key term A", "mentions key term B"],
        )

        assert trait.evaluation_mode == "tp_only"
        assert "recall" in trait.metrics
        assert "f1" in trait.metrics
        assert len(trait.tp_instructions) == 2

    def test_valid_tp_only_mode_all_valid_metrics(self):
        """Test tp_only mode with all valid metrics (precision, recall, f1)."""
        trait = MetricRubricTrait(
            name="all_metrics",
            evaluation_mode="tp_only",
            metrics=["precision", "recall", "f1"],
            tp_instructions=["term A", "term B"],
        )

        assert len(trait.metrics) == 3
        assert set(trait.metrics) == {"precision", "recall", "f1"}

    def test_invalid_tp_only_missing_tp_instructions(self):
        """Test that TP instructions are required in tp_only mode."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_trait",
                evaluation_mode="tp_only",
                metrics=["precision"],
                tp_instructions=[],  # Empty TP instructions
            )

        error_msg = str(exc_info.value).lower()
        assert "tp" in error_msg or "must be provided" in error_msg

    def test_invalid_tp_only_specificity_not_computable(self):
        """Test that specificity cannot be computed in tp_only mode (requires TN)."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_trait",
                evaluation_mode="tp_only",
                metrics=["specificity"],  # Requires TN count
                tp_instructions=["term A"],
            )

        error_msg = str(exc_info.value).lower()
        assert "specificity" in error_msg or "not available" in error_msg

    def test_invalid_tp_only_accuracy_not_computable(self):
        """Test that accuracy cannot be computed in tp_only mode (requires TN)."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_trait",
                evaluation_mode="tp_only",
                metrics=["accuracy"],  # Requires TN count
                tp_instructions=["term A"],
            )

        error_msg = str(exc_info.value).lower()
        assert "accuracy" in error_msg or "not available" in error_msg

    def test_tp_only_default_mode(self):
        """Test that tp_only is the default evaluation mode."""
        trait = MetricRubricTrait(
            name="default_mode",
            metrics=["precision"],
            tp_instructions=["term A"],
        )

        assert trait.evaluation_mode == "tp_only"


class TestMetricRubricTraitFullMatrixMode:
    """Test schema validation for MetricRubricTrait in full_matrix mode."""

    def test_valid_full_matrix_with_all_metrics(self):
        """Test valid full_matrix mode trait with all five metrics."""
        trait = MetricRubricTrait(
            name="full_evaluation",
            description="Complete confusion matrix evaluation",
            evaluation_mode="full_matrix",
            metrics=["precision", "recall", "specificity", "accuracy", "f1"],
            tp_instructions=["correct term A", "correct term B"],
            tn_instructions=["incorrect term X", "incorrect term Y"],
            repeated_extraction=True,
        )

        assert trait.name == "full_evaluation"
        assert trait.evaluation_mode == "full_matrix"
        assert len(trait.metrics) == 5
        assert len(trait.tp_instructions) == 2
        assert len(trait.tn_instructions) == 2
        assert trait.repeated_extraction is True

    def test_valid_full_matrix_with_subset_of_metrics(self):
        """Test full_matrix mode with a subset of valid metrics."""
        trait = MetricRubricTrait(
            name="partial_metrics",
            evaluation_mode="full_matrix",
            metrics=["precision", "specificity"],
            tp_instructions=["correct term"],
            tn_instructions=["incorrect term"],
        )

        assert trait.evaluation_mode == "full_matrix"
        assert set(trait.metrics) == {"precision", "specificity"}

    def test_invalid_full_matrix_missing_tp_instructions(self):
        """Test that TP instructions are required in full_matrix mode."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_trait",
                evaluation_mode="full_matrix",
                metrics=["precision"],
                tp_instructions=[],  # Empty TP instructions
                tn_instructions=["incorrect term"],
            )

        error_msg = str(exc_info.value).lower()
        assert "tp" in error_msg or "must be provided" in error_msg

    def test_invalid_full_matrix_missing_tn_instructions(self):
        """Test that TN instructions are required in full_matrix mode."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_trait",
                evaluation_mode="full_matrix",
                metrics=["precision"],
                tp_instructions=["correct term"],
                tn_instructions=[],  # Empty TN instructions
            )

        error_msg = str(exc_info.value).lower()
        assert "tn" in error_msg or "must be provided" in error_msg

    def test_full_matrix_allows_specificity(self):
        """Test that specificity is computable in full_matrix mode."""
        trait = MetricRubricTrait(
            name="with_specificity",
            evaluation_mode="full_matrix",
            metrics=["specificity"],
            tp_instructions=["correct"],
            tn_instructions=["incorrect"],
        )

        assert "specificity" in trait.metrics

    def test_full_matrix_allows_accuracy(self):
        """Test that accuracy is computable in full_matrix mode."""
        trait = MetricRubricTrait(
            name="with_accuracy",
            evaluation_mode="full_matrix",
            metrics=["accuracy"],
            tp_instructions=["correct"],
            tn_instructions=["incorrect"],
        )

        assert "accuracy" in trait.metrics


class TestMetricRubricTraitGeneralValidation:
    """Test general validation rules for MetricRubricTrait."""

    def test_invalid_evaluation_mode(self):
        """Test that invalid evaluation modes are rejected."""
        with pytest.raises(ValidationError):
            MetricRubricTrait(
                name="bad_mode",
                evaluation_mode="invalid_mode",  # Not 'tp_only' or 'full_matrix'
                metrics=["precision"],
                tp_instructions=["correct"],
            )

    def test_invalid_metric_name(self):
        """Test that invalid metric names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MetricRubricTrait(
                name="bad_metric",
                evaluation_mode="tp_only",
                metrics=["fake_metric"],  # Not a valid metric
                tp_instructions=["correct"],
            )

        error_msg = str(exc_info.value).lower()
        assert "metric" in error_msg

    def test_empty_metrics_list(self):
        """Test that at least one metric must be specified."""
        with pytest.raises(ValidationError):
            MetricRubricTrait(
                name="no_metrics",
                evaluation_mode="tp_only",
                metrics=[],  # Empty metrics list
                tp_instructions=["correct"],
            )

    def test_valid_repeated_extraction_default(self):
        """Test that repeated_extraction defaults to True."""
        trait = MetricRubricTrait(
            name="default_repeated",
            evaluation_mode="tp_only",
            metrics=["precision"],
            tp_instructions=["term"],
        )

        assert trait.repeated_extraction is True

    def test_valid_repeated_extraction_false(self):
        """Test that repeated_extraction can be set to False."""
        trait = MetricRubricTrait(
            name="no_repeated",
            evaluation_mode="tp_only",
            metrics=["precision"],
            tp_instructions=["term"],
            repeated_extraction=False,
        )

        assert trait.repeated_extraction is False


class TestRubricIntegration:
    """Test MetricRubricTrait integration with Rubric."""

    def test_rubric_with_metric_traits_tp_only(self):
        """Test creating a rubric with tp_only metric traits."""
        trait1 = MetricRubricTrait(
            name="precision_trait",
            evaluation_mode="tp_only",
            metrics=["precision"],
            tp_instructions=["correct extraction"],
        )

        rubric = Rubric(
            traits=[],
            manual_traits=[],
            metric_traits=[trait1],
        )

        assert len(rubric.metric_traits) == 1
        assert rubric.metric_traits[0].name == "precision_trait"
        assert rubric.metric_traits[0].evaluation_mode == "tp_only"

    def test_rubric_with_metric_traits_full_matrix(self):
        """Test creating a rubric with full_matrix metric traits."""
        trait1 = MetricRubricTrait(
            name="full_metric_trait",
            evaluation_mode="full_matrix",
            metrics=["precision", "recall", "specificity"],
            tp_instructions=["correct"],
            tn_instructions=["incorrect"],
        )

        rubric = Rubric(
            traits=[],
            manual_traits=[],
            metric_traits=[trait1],
        )

        assert len(rubric.metric_traits) == 1
        assert rubric.metric_traits[0].evaluation_mode == "full_matrix"

    def test_rubric_with_mixed_mode_metric_traits(self):
        """Test rubric with both tp_only and full_matrix traits."""
        trait1 = MetricRubricTrait(
            name="tp_only_trait",
            evaluation_mode="tp_only",
            metrics=["precision"],
            tp_instructions=["correct"],
        )
        trait2 = MetricRubricTrait(
            name="full_matrix_trait",
            evaluation_mode="full_matrix",
            metrics=["accuracy"],
            tp_instructions=["correct"],
            tn_instructions=["incorrect"],
        )

        rubric = Rubric(metric_traits=[trait1, trait2])

        assert len(rubric.metric_traits) == 2
        assert rubric.metric_traits[0].evaluation_mode == "tp_only"
        assert rubric.metric_traits[1].evaluation_mode == "full_matrix"

    def test_get_metric_trait_names(self):
        """Test get_metric_trait_names() method."""
        trait1 = MetricRubricTrait(
            name="trait_a",
            evaluation_mode="tp_only",
            metrics=["precision"],
            tp_instructions=["correct"],
        )
        trait2 = MetricRubricTrait(
            name="trait_b",
            evaluation_mode="full_matrix",
            metrics=["accuracy"],
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
            evaluation_mode="tp_only",
            metrics=["precision"],
            tp_instructions=["correct"],
        )

        rubric = Rubric(traits=[llm_trait], metric_traits=[metric_trait])
        all_names = rubric.get_trait_names()

        assert "llm_trait" in all_names
        assert "metric_trait" in all_names
        assert len(all_names) == 2

    def test_rubric_validation_unique_trait_names_across_types(self):
        """Test that trait names must be unique across LLM, manual, and metric traits."""
        # This test documents expected behavior - the validation happens at the API level
        llm_trait = RubricTrait(name="same_name", kind="boolean")
        metric_trait = MetricRubricTrait(
            name="same_name",  # Duplicate name
            evaluation_mode="tp_only",
            metrics=["precision"],
            tp_instructions=["correct"],
        )

        # The Rubric model itself doesn't enforce uniqueness, but the API does
        rubric = Rubric(traits=[llm_trait], metric_traits=[metric_trait])

        # Both traits are present - uniqueness is enforced at API level
        assert len(rubric.get_trait_names()) == 2  # Counts both, even with duplicate name
