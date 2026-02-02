"""Integration tests for MetricRubricTrait evaluation.

These tests verify that MetricRubricTrait correctly configures confusion matrix
analysis and computes precision/recall/F1 metrics. The tests focus on:

1. Trait configuration and validation
2. Metric computation formulas (mathematically verified)
3. Integration with Rubric structures
4. Edge cases (divide by zero, empty lists)

Note: MetricRubricTrait uses LLM for the actual classification step (categorizing
instructions into tp/fn/fp/tn). These tests verify the metric calculation logic
and trait configuration, not the LLM classification itself.

Metric formulas tested:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
- Specificity = TN / (TN + FP)
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
"""

import pytest

from karenina.benchmark.verification.evaluators import MetricTraitEvaluator
from karenina.schemas.domain import MetricRubricTrait, Rubric

# =============================================================================
# MetricRubricTrait Configuration Tests
# =============================================================================


@pytest.mark.integration
class TestMetricRubricTraitConfiguration:
    """Test MetricRubricTrait configuration with trace-relevant scenarios."""

    def test_tp_only_mode_for_citation_extraction(self):
        """Verify tp_only mode works for citation extraction scenario."""
        # Simulate: checking if response mentions expected references
        trait = MetricRubricTrait(
            name="reference_coverage",
            evaluation_mode="tp_only",
            metrics=["precision", "recall", "f1"],
            tp_instructions=[
                "Mentions Tsujimoto et al., Science, 1985",
                "Mentions Hockenbery et al., Nature, 1990",
                "Mentions Adams & Cory, Science, 1998",
            ],
            description="Check if response covers key references",
        )

        assert trait.evaluation_mode == "tp_only"
        assert len(trait.tp_instructions) == 3
        assert trait.get_required_buckets() == {"tp", "fn", "fp"}

    def test_full_matrix_mode_for_content_analysis(self):
        """Verify full_matrix mode works for content analysis scenario."""
        # Simulate: checking for correct inclusions and exclusions
        trait = MetricRubricTrait(
            name="content_accuracy",
            evaluation_mode="full_matrix",
            metrics=["precision", "recall", "specificity", "accuracy", "f1"],
            tp_instructions=[
                "Mentions BCL2 gene",
                "Discusses apoptosis regulation",
                "References cancer research",
            ],
            tn_instructions=[
                "Claims BCL2 is pro-apoptotic",  # Should NOT be present (it's anti-apoptotic)
                "States BCL2 is on chromosome 1",  # Should NOT be present (it's on 18)
            ],
            description="Check content accuracy",
        )

        assert trait.evaluation_mode == "full_matrix"
        assert len(trait.tp_instructions) == 3
        assert len(trait.tn_instructions) == 2
        assert trait.get_required_buckets() == {"tp", "fn", "tn", "fp"}

    def test_single_metric_configuration(self):
        """Verify trait works with single metric."""
        trait = MetricRubricTrait(
            name="recall_only",
            evaluation_mode="tp_only",
            metrics=["recall"],
            tp_instructions=["test instruction"],
        )

        assert trait.metrics == ["recall"]

    def test_repeated_extraction_default_true(self):
        """Verify repeated_extraction defaults to True."""
        trait = MetricRubricTrait(
            name="test",
            evaluation_mode="tp_only",
            metrics=["precision"],
            tp_instructions=["test"],
        )

        assert trait.repeated_extraction is True

    def test_repeated_extraction_can_be_disabled(self):
        """Verify repeated_extraction can be set to False."""
        trait = MetricRubricTrait(
            name="test",
            evaluation_mode="tp_only",
            metrics=["precision"],
            tp_instructions=["test"],
            repeated_extraction=False,
        )

        assert trait.repeated_extraction is False


# =============================================================================
# Metric Computation Tests (Mathematically Verified)
# =============================================================================


@pytest.mark.integration
class TestMetricComputation:
    """Test metric computation formulas.

    These tests verify the mathematical correctness of metric calculations
    by using MetricTraitEvaluator's internal _compute_metrics method.

    Note: _compute_metrics was moved from RubricEvaluator to MetricTraitEvaluator
    during the flaw-001 refactoring to extract metric trait evaluation.
    """

    @pytest.fixture
    def metric_evaluator(self):
        """Get metric trait evaluator for metric computation.

        Note: _compute_metrics was moved from RubricEvaluator to MetricTraitEvaluator
        during the flaw-001 refactoring.
        """
        from karenina.schemas.workflow import ModelConfig

        model_config = ModelConfig(
            id="test-metric",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            temperature=0.0,
            interface="langchain",
        )
        return MetricTraitEvaluator(llm=None, model_config=model_config)

    def test_perfect_extraction_all_metrics_one(self, metric_evaluator):
        """Verify perfect extraction: P=1, R=1, F1=1."""
        # All TP instructions found, no FN or FP
        tp = ["item1", "item2", "item3"]
        fn = []
        fp = []
        tn = []

        metrics = metric_evaluator._compute_metrics(tp, tn, fp, fn, ["precision", "recall", "f1"])

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_partial_extraction_recall_less_than_one(self, metric_evaluator):
        """Verify partial extraction: recall < 1 when some items missed."""
        # 2 out of 4 TP instructions found
        tp = ["item1", "item2"]
        fn = ["item3", "item4"]  # Missed items
        fp = []
        tn = []

        metrics = metric_evaluator._compute_metrics(tp, tn, fp, fn, ["precision", "recall", "f1"])

        # Precision = 2 / (2 + 0) = 1.0
        assert metrics["precision"] == 1.0
        # Recall = 2 / (2 + 2) = 0.5
        assert metrics["recall"] == 0.5
        # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 2/3
        assert abs(metrics["f1"] - (2 / 3)) < 0.0001

    def test_false_positives_precision_less_than_one(self, metric_evaluator):
        """Verify false positives: precision < 1 when extra items included."""
        # 2 TP but also 2 FP
        tp = ["correct1", "correct2"]
        fn = []
        fp = ["wrong1", "wrong2"]
        tn = []

        metrics = metric_evaluator._compute_metrics(tp, tn, fp, fn, ["precision", "recall", "f1"])

        # Precision = 2 / (2 + 2) = 0.5
        assert metrics["precision"] == 0.5
        # Recall = 2 / (2 + 0) = 1.0
        assert metrics["recall"] == 1.0
        # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 2/3
        assert abs(metrics["f1"] - (2 / 3)) < 0.0001

    def test_no_matches_handles_divide_by_zero(self, metric_evaluator):
        """Verify no matches case: returns 0 instead of divide by zero error."""
        # No TP found
        tp = []
        fn = ["missed1", "missed2"]
        fp = ["wrong1"]
        tn = []

        metrics = metric_evaluator._compute_metrics(tp, tn, fp, fn, ["precision", "recall", "f1"])

        # Precision = 0 / (0 + 1) = 0.0
        assert metrics["precision"] == 0.0
        # Recall = 0 / (0 + 2) = 0.0
        assert metrics["recall"] == 0.0
        # F1 = 0 (since precision + recall = 0)
        assert metrics["f1"] == 0.0

    def test_all_empty_lists_handles_gracefully(self, metric_evaluator):
        """Verify all empty lists returns zeros."""
        tp = []
        fn = []
        fp = []
        tn = []

        metrics = metric_evaluator._compute_metrics(tp, tn, fp, fn, ["precision", "recall", "f1"])

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_specificity_calculation(self, metric_evaluator):
        """Verify specificity formula: TN / (TN + FP)."""
        tp = ["found"]
        fn = []
        fp = ["wrong1"]
        tn = ["correctly_absent1", "correctly_absent2", "correctly_absent3"]

        metrics = metric_evaluator._compute_metrics(tp, tn, fp, fn, ["specificity"])

        # Specificity = 3 / (3 + 1) = 0.75
        assert metrics["specificity"] == 0.75

    def test_accuracy_calculation(self, metric_evaluator):
        """Verify accuracy formula: (TP + TN) / (TP + TN + FP + FN)."""
        tp = ["correct1", "correct2"]
        tn = ["absent1", "absent2", "absent3"]
        fp = ["wrong1"]
        fn = ["missed1", "missed2"]

        metrics = metric_evaluator._compute_metrics(tp, tn, fp, fn, ["accuracy"])

        # Accuracy = (2 + 3) / (2 + 3 + 1 + 2) = 5/8 = 0.625
        assert metrics["accuracy"] == 0.625

    def test_all_full_matrix_metrics(self, metric_evaluator):
        """Verify all metrics in full matrix mode."""
        tp = ["tp1", "tp2"]  # 2
        tn = ["tn1", "tn2", "tn3"]  # 3
        fp = ["fp1"]  # 1
        fn = ["fn1"]  # 1

        metrics = metric_evaluator._compute_metrics(
            tp, tn, fp, fn, ["precision", "recall", "specificity", "accuracy", "f1"]
        )

        # Precision = 2 / (2 + 1) = 2/3
        assert abs(metrics["precision"] - (2 / 3)) < 0.0001
        # Recall = 2 / (2 + 1) = 2/3
        assert abs(metrics["recall"] - (2 / 3)) < 0.0001
        # Specificity = 3 / (3 + 1) = 0.75
        assert metrics["specificity"] == 0.75
        # Accuracy = (2 + 3) / (2 + 3 + 1 + 1) = 5/7
        assert abs(metrics["accuracy"] - (5 / 7)) < 0.0001
        # F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3
        assert abs(metrics["f1"] - (2 / 3)) < 0.0001


# =============================================================================
# Confusion Matrix Interpretation Tests
# =============================================================================


@pytest.mark.integration
class TestConfusionMatrixInterpretation:
    """Test understanding of confusion matrix buckets in context."""

    def test_tp_only_mode_buckets(self):
        """Verify tp_only mode bucket interpretation."""
        trait = MetricRubricTrait(
            name="entity_check",
            evaluation_mode="tp_only",
            metrics=["precision", "recall"],
            tp_instructions=[
                "Gene BCL2",
                "Chromosome 18",
                "Apoptosis inhibitor",
            ],
        )

        # In tp_only mode:
        # - TP: Instructions that ARE found in the answer
        # - FN: Instructions that are NOT found (missed)
        # - FP: Extra content in answer not matching any instruction
        # - TN: Cannot be computed (no explicit negative set)
        buckets = trait.get_required_buckets()
        assert buckets == {"tp", "fn", "fp"}
        assert "tn" not in buckets

    def test_full_matrix_mode_buckets(self):
        """Verify full_matrix mode bucket interpretation."""
        trait = MetricRubricTrait(
            name="content_check",
            evaluation_mode="full_matrix",
            metrics=["accuracy"],
            tp_instructions=["BCL2 inhibits apoptosis"],  # Should be present
            tn_instructions=["BCL2 promotes apoptosis"],  # Should NOT be present
        )

        # In full_matrix mode:
        # - TP: TP instructions found in answer
        # - FN: TP instructions missing from answer
        # - TN: TN instructions correctly absent
        # - FP: TN instructions incorrectly present
        buckets = trait.get_required_buckets()
        assert buckets == {"tp", "fn", "tn", "fp"}


# =============================================================================
# Integration with Rubric Tests
# =============================================================================


@pytest.mark.integration
class TestMetricRubricTraitWithRubric:
    """Test MetricRubricTrait integration with Rubric structures."""

    def test_rubric_with_single_metric_trait(self):
        """Verify Rubric can contain single metric trait."""
        trait = MetricRubricTrait(
            name="entity_extraction",
            evaluation_mode="tp_only",
            metrics=["precision", "recall", "f1"],
            tp_instructions=["mitochondria", "apoptosis"],
        )

        rubric = Rubric(metric_traits=[trait])

        assert len(rubric.metric_traits) == 1
        assert rubric.get_metric_trait_names() == ["entity_extraction"]

    def test_rubric_with_multiple_metric_traits(self):
        """Verify Rubric can contain multiple metric traits."""
        trait1 = MetricRubricTrait(
            name="gene_extraction",
            evaluation_mode="tp_only",
            metrics=["recall"],
            tp_instructions=["BCL2", "TP53", "BRCA1"],
        )
        trait2 = MetricRubricTrait(
            name="pathway_extraction",
            evaluation_mode="tp_only",
            metrics=["precision"],
            tp_instructions=["apoptosis", "cell cycle"],
        )

        rubric = Rubric(metric_traits=[trait1, trait2])

        assert len(rubric.metric_traits) == 2
        assert set(rubric.get_metric_trait_names()) == {"gene_extraction", "pathway_extraction"}

    def test_rubric_with_mixed_trait_types(self):
        """Verify Rubric can contain metric traits alongside other trait types."""
        from karenina.schemas.domain import RegexTrait

        metric_trait = MetricRubricTrait(
            name="entity_check",
            evaluation_mode="tp_only",
            metrics=["f1"],
            tp_instructions=["BCL2"],
        )
        regex_trait = RegexTrait(
            name="has_citations",
            pattern=r"\[\d+\]",
            description="Has numeric citations",
        )

        rubric = Rubric(
            metric_traits=[metric_trait],
            regex_traits=[regex_trait],
        )

        assert len(rubric.metric_traits) == 1
        assert len(rubric.regex_traits) == 1
        assert set(rubric.get_trait_names()) == {"entity_check", "has_citations"}

    def test_metric_trait_serialization_roundtrip(self):
        """Verify MetricRubricTrait survives serialization."""
        trait = MetricRubricTrait(
            name="test_trait",
            evaluation_mode="tp_only",
            metrics=["precision", "recall", "f1"],
            tp_instructions=["instruction1", "instruction2"],
            description="Test description",
        )

        rubric = Rubric(metric_traits=[trait])

        # Serialize and deserialize
        data = rubric.model_dump()
        restored = Rubric(**data)

        assert len(restored.metric_traits) == 1
        restored_trait = restored.metric_traits[0]
        assert restored_trait.name == "test_trait"
        assert restored_trait.evaluation_mode == "tp_only"
        assert restored_trait.metrics == ["precision", "recall", "f1"]
        assert restored_trait.tp_instructions == ["instruction1", "instruction2"]


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================


@pytest.mark.integration
class TestMetricRubricTraitEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_instruction(self):
        """Verify trait works with single instruction."""
        trait = MetricRubricTrait(
            name="single",
            evaluation_mode="tp_only",
            metrics=["recall"],
            tp_instructions=["only instruction"],
        )

        assert len(trait.tp_instructions) == 1

    def test_many_instructions(self):
        """Verify trait works with many instructions."""
        instructions = [f"instruction_{i}" for i in range(50)]
        trait = MetricRubricTrait(
            name="many",
            evaluation_mode="tp_only",
            metrics=["precision"],
            tp_instructions=instructions,
        )

        assert len(trait.tp_instructions) == 50

    def test_instructions_with_special_characters(self):
        """Verify trait handles special characters in instructions."""
        trait = MetricRubricTrait(
            name="special",
            evaluation_mode="tp_only",
            metrics=["recall"],
            tp_instructions=[
                "Gene: BCL2 (B-cell lymphoma 2)",
                "Chromosome 18q21.33",
                "Reference: [1]",
                "Value > 0.5 and < 1.0",
            ],
        )

        assert len(trait.tp_instructions) == 4

    def test_instructions_with_unicode(self):
        """Verify trait handles unicode in instructions."""
        trait = MetricRubricTrait(
            name="unicode",
            evaluation_mode="tp_only",
            metrics=["precision"],
            tp_instructions=[
                "α-synuclein protein",
                "β-amyloid plaques",
                "Δ mutation",
            ],
        )

        assert len(trait.tp_instructions) == 3
        assert "α-synuclein protein" in trait.tp_instructions

    def test_long_instruction_text(self):
        """Verify trait handles long instruction text."""
        long_instruction = "This is a very long instruction " * 20
        trait = MetricRubricTrait(
            name="long",
            evaluation_mode="tp_only",
            metrics=["recall"],
            tp_instructions=[long_instruction],
        )

        assert len(trait.tp_instructions[0]) > 500


# =============================================================================
# Metric Formula Verification Tests
# =============================================================================


@pytest.mark.integration
class TestMetricFormulas:
    """Test that metric formulas match standard definitions."""

    @pytest.fixture
    def metric_evaluator(self):
        """Get metric trait evaluator for metric computation."""
        from karenina.schemas.workflow import ModelConfig

        model_config = ModelConfig(
            id="test-metric",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            temperature=0.0,
            interface="langchain",
        )
        return MetricTraitEvaluator(llm=None, model_config=model_config)

    def test_precision_matches_sklearn_definition(self, metric_evaluator):
        """Verify precision matches sklearn: TP / (TP + FP)."""
        # sklearn definition: precision = tp / (tp + fp)
        tp = ["a", "b", "c"]  # 3
        fp = ["x", "y"]  # 2
        fn = ["z"]  # 1
        tn = []

        metrics = metric_evaluator._compute_metrics(tp, tn, fp, fn, ["precision"])

        expected = 3 / (3 + 2)  # 0.6
        assert metrics["precision"] == expected

    def test_recall_matches_sklearn_definition(self, metric_evaluator):
        """Verify recall matches sklearn: TP / (TP + FN)."""
        # sklearn definition: recall = tp / (tp + fn)
        tp = ["a", "b"]  # 2
        fn = ["c", "d", "e"]  # 3
        fp = ["x"]  # 1
        tn = []

        metrics = metric_evaluator._compute_metrics(tp, tn, fp, fn, ["recall"])

        expected = 2 / (2 + 3)  # 0.4
        assert metrics["recall"] == expected

    def test_f1_matches_sklearn_definition(self, metric_evaluator):
        """Verify F1 matches sklearn: 2 * (precision * recall) / (precision + recall)."""
        tp = ["a", "b", "c"]  # 3
        fp = ["x"]  # 1
        fn = ["y"]  # 1
        tn = []

        metrics = metric_evaluator._compute_metrics(tp, tn, fp, fn, ["precision", "recall", "f1"])

        precision = 3 / (3 + 1)  # 0.75
        recall = 3 / (3 + 1)  # 0.75
        expected_f1 = 2 * (precision * recall) / (precision + recall)  # 0.75

        assert metrics["precision"] == precision
        assert metrics["recall"] == recall
        assert metrics["f1"] == expected_f1

    def test_f1_equals_precision_equals_recall_when_fp_equals_fn(self, metric_evaluator):
        """Verify F1 = precision = recall when FP = FN."""
        tp = ["a", "b"]  # 2
        fp = ["x"]  # 1
        fn = ["y"]  # 1 (same as FP)
        tn = []

        metrics = metric_evaluator._compute_metrics(tp, tn, fp, fn, ["precision", "recall", "f1"])

        # When FP = FN, precision = recall = F1
        assert metrics["precision"] == metrics["recall"]
        assert metrics["f1"] == metrics["precision"]
