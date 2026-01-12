"""Tests for multi-objective scoring in GEPA integration.

Run with: uv run pytest tests/integrations/gepa/test_multi_objective.py -v
"""

from unittest.mock import MagicMock

import pytest

# =============================================================================
# Config Class Tests
# =============================================================================


class TestTraitSelectionMode:
    """Tests for TraitSelectionMode enum."""

    def test_enum_values(self):
        """Verify enum values exist."""
        from karenina.integrations.gepa import TraitSelectionMode

        assert TraitSelectionMode.ALL == "all"
        assert TraitSelectionMode.NONE == "none"
        assert TraitSelectionMode.CUSTOM == "custom"


class TestMetricObjectiveConfig:
    """Tests for MetricObjectiveConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from karenina.integrations.gepa import MetricObjectiveConfig

        config = MetricObjectiveConfig()
        assert config.include_precision is False
        assert config.include_recall is False
        assert config.include_f1 is True

    def test_get_enabled_metrics_default(self):
        """Test get_enabled_metrics with defaults."""
        from karenina.integrations.gepa import MetricObjectiveConfig

        config = MetricObjectiveConfig()
        assert config.get_enabled_metrics() == ["f1"]

    def test_get_enabled_metrics_all(self):
        """Test get_enabled_metrics with all enabled."""
        from karenina.integrations.gepa import MetricObjectiveConfig

        config = MetricObjectiveConfig(
            include_precision=True,
            include_recall=True,
            include_f1=True,
        )
        assert config.get_enabled_metrics() == ["precision", "recall", "f1"]

    def test_get_enabled_metrics_none(self):
        """Test get_enabled_metrics with none enabled."""
        from karenina.integrations.gepa import MetricObjectiveConfig

        config = MetricObjectiveConfig(
            include_precision=False,
            include_recall=False,
            include_f1=False,
        )
        assert config.get_enabled_metrics() == []


class TestObjectiveConfig:
    """Tests for ObjectiveConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode

        config = ObjectiveConfig()
        assert config.include_template is True
        assert config.trait_mode == TraitSelectionMode.ALL
        assert config.selected_traits is None

    def test_custom_mode_requires_traits(self):
        """Test that CUSTOM mode requires selected_traits."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode

        with pytest.raises(ValueError, match="selected_traits must be provided"):
            ObjectiveConfig(
                trait_mode=TraitSelectionMode.CUSTOM,
                selected_traits=None,
            )

    def test_custom_mode_with_traits(self):
        """Test that CUSTOM mode works with selected_traits."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode

        config = ObjectiveConfig(
            trait_mode=TraitSelectionMode.CUSTOM,
            selected_traits=["clarity", "safety"],
        )
        assert config.trait_mode == TraitSelectionMode.CUSTOM
        assert config.selected_traits == ["clarity", "safety"]

    def test_should_include_trait_all_mode(self):
        """Test should_include_trait with ALL mode."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode

        config = ObjectiveConfig(trait_mode=TraitSelectionMode.ALL)
        assert config.should_include_trait("clarity") is True
        assert config.should_include_trait("any_trait") is True

    def test_should_include_trait_none_mode(self):
        """Test should_include_trait with NONE mode."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode

        config = ObjectiveConfig(trait_mode=TraitSelectionMode.NONE)
        assert config.should_include_trait("clarity") is False
        assert config.should_include_trait("any_trait") is False

    def test_should_include_trait_custom_mode(self):
        """Test should_include_trait with CUSTOM mode."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode

        config = ObjectiveConfig(
            trait_mode=TraitSelectionMode.CUSTOM,
            selected_traits=["clarity", "safety"],
        )
        assert config.should_include_trait("clarity") is True
        assert config.should_include_trait("safety") is True
        assert config.should_include_trait("other") is False


# =============================================================================
# Scoring Function Tests
# =============================================================================


class TestComputeObjectiveScores:
    """Tests for compute_objective_scores function."""

    def _create_mock_result(
        self,
        template_pass: bool | None = None,
        rubric_scores: dict | None = None,
    ) -> MagicMock:
        """Helper to create mock VerificationResult."""
        mock_result = MagicMock()

        # Setup template
        if template_pass is not None:
            mock_result.template = MagicMock()
            mock_result.template.verify_result = template_pass
        else:
            mock_result.template = None

        # Setup rubric
        if rubric_scores is not None:
            mock_result.rubric = MagicMock()
            mock_result.rubric.rubric_evaluation_performed = True
            mock_result.rubric.get_all_trait_scores.return_value = rubric_scores
        else:
            mock_result.rubric = None

        return mock_result

    def test_template_only(self):
        """Test scoring with template only."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode, compute_objective_scores

        config = ObjectiveConfig(
            include_template=True,
            trait_mode=TraitSelectionMode.NONE,
        )
        mock_result = self._create_mock_result(template_pass=True)

        scores = compute_objective_scores(mock_result, "claude-haiku", config)

        assert scores == {"claude-haiku:template": 1.0}

    def test_template_fail(self):
        """Test scoring with template fail."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode, compute_objective_scores

        config = ObjectiveConfig(
            include_template=True,
            trait_mode=TraitSelectionMode.NONE,
        )
        mock_result = self._create_mock_result(template_pass=False)

        scores = compute_objective_scores(mock_result, "claude-haiku", config)

        assert scores == {"claude-haiku:template": 0.0}

    def test_boolean_rubric_trait(self):
        """Test scoring with boolean rubric trait."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode, compute_objective_scores

        config = ObjectiveConfig(
            include_template=False,
            trait_mode=TraitSelectionMode.ALL,
        )
        mock_result = self._create_mock_result(
            rubric_scores={"mentions_safety": True, "has_citations": False}
        )

        scores = compute_objective_scores(mock_result, "claude-haiku", config)

        assert scores == {
            "claude-haiku:mentions_safety": 1.0,
            "claude-haiku:has_citations": 0.0,
        }

    def test_integer_rubric_trait(self):
        """Test scoring with integer rubric trait (1-5 scale)."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode, compute_objective_scores

        config = ObjectiveConfig(
            include_template=False,
            trait_mode=TraitSelectionMode.ALL,
        )
        mock_result = self._create_mock_result(rubric_scores={"clarity": 4})

        scores = compute_objective_scores(mock_result, "claude-haiku", config)

        assert scores == {"claude-haiku:clarity": 0.8}  # 4/5 = 0.8

    def test_metric_trait_f1_only(self):
        """Test scoring with metric trait using f1 only."""
        from karenina.integrations.gepa import (
            MetricObjectiveConfig,
            ObjectiveConfig,
            TraitSelectionMode,
            compute_objective_scores,
        )

        config = ObjectiveConfig(
            include_template=False,
            trait_mode=TraitSelectionMode.ALL,
            metric_config=MetricObjectiveConfig(include_f1=True),
        )
        mock_result = self._create_mock_result(
            rubric_scores={"entity_extraction": {"precision": 0.9, "recall": 0.85, "f1": 0.87}}
        )

        scores = compute_objective_scores(mock_result, "claude-haiku", config)

        assert scores == {"claude-haiku:entity_extraction_f1": 0.87}

    def test_metric_trait_all_metrics(self):
        """Test scoring with metric trait using all metrics."""
        from karenina.integrations.gepa import (
            MetricObjectiveConfig,
            ObjectiveConfig,
            TraitSelectionMode,
            compute_objective_scores,
        )

        config = ObjectiveConfig(
            include_template=False,
            trait_mode=TraitSelectionMode.ALL,
            metric_config=MetricObjectiveConfig(
                include_precision=True,
                include_recall=True,
                include_f1=True,
            ),
        )
        mock_result = self._create_mock_result(
            rubric_scores={"entity_extraction": {"precision": 0.9, "recall": 0.85, "f1": 0.87}}
        )

        scores = compute_objective_scores(mock_result, "claude-haiku", config)

        assert scores == {
            "claude-haiku:entity_extraction_precision": 0.9,
            "claude-haiku:entity_extraction_recall": 0.85,
            "claude-haiku:entity_extraction_f1": 0.87,
        }

    def test_custom_trait_selection(self):
        """Test scoring with custom trait selection."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode, compute_objective_scores

        config = ObjectiveConfig(
            include_template=False,
            trait_mode=TraitSelectionMode.CUSTOM,
            selected_traits=["clarity"],
        )
        mock_result = self._create_mock_result(
            rubric_scores={"clarity": 4, "safety": 5, "completeness": 3}
        )

        scores = compute_objective_scores(mock_result, "claude-haiku", config)

        # Only clarity should be included
        assert scores == {"claude-haiku:clarity": 0.8}

    def test_compound_keys_multiple_models(self):
        """Test that compound keys work for differentiating models."""
        from karenina.integrations.gepa import ObjectiveConfig, compute_objective_scores

        config = ObjectiveConfig(include_template=True)
        mock_result = self._create_mock_result(template_pass=True)

        scores_haiku = compute_objective_scores(mock_result, "claude-haiku", config)
        scores_sonnet = compute_objective_scores(mock_result, "claude-sonnet", config)

        assert "claude-haiku:template" in scores_haiku
        assert "claude-sonnet:template" in scores_sonnet
        assert scores_haiku != scores_sonnet  # Different keys

    def test_full_integration(self):
        """Test with template and multiple rubric trait types."""
        from karenina.integrations.gepa import (
            MetricObjectiveConfig,
            ObjectiveConfig,
            TraitSelectionMode,
            compute_objective_scores,
        )

        config = ObjectiveConfig(
            include_template=True,
            trait_mode=TraitSelectionMode.ALL,
            metric_config=MetricObjectiveConfig(include_f1=True),
        )
        mock_result = self._create_mock_result(
            template_pass=True,
            rubric_scores={
                "clarity": 4,  # int -> 0.8
                "mentions_safety": True,  # bool -> 1.0
                "entity_extraction": {"precision": 0.9, "recall": 0.85, "f1": 0.87},  # metric -> 0.87
            },
        )

        scores = compute_objective_scores(mock_result, "claude-haiku", config)

        assert scores == {
            "claude-haiku:template": 1.0,
            "claude-haiku:clarity": 0.8,
            "claude-haiku:mentions_safety": 1.0,
            "claude-haiku:entity_extraction_f1": 0.87,
        }


# =============================================================================
# Utility Function Tests
# =============================================================================


def test_compute_improvement_positive():
    """Test improvement calculation with positive improvement."""
    from karenina.integrations.gepa import compute_improvement

    improvement = compute_improvement(baseline_score=0.5, optimized_score=0.6)
    expected = (0.6 - 0.5) / 0.5  # 0.2 = 20%
    assert abs(improvement - expected) < 1e-6


def test_compute_improvement_zero_baseline():
    """Test improvement calculation with zero baseline."""
    from karenina.integrations.gepa import compute_improvement

    improvement = compute_improvement(baseline_score=0.0, optimized_score=0.5)
    assert improvement == 0.5


def test_extract_failed_fields():
    """Test extraction of failed template fields."""
    from karenina.integrations.gepa import extract_failed_fields

    mock_result = MagicMock()
    mock_result.template = MagicMock()
    mock_result.template.field_results = {
        "field_a": True,
        "field_b": False,
        "field_c": {"passed": True},
        "field_d": {"passed": False},
    }

    failed = extract_failed_fields(mock_result)
    assert set(failed) == {"field_b", "field_d"}


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in multi-objective scoring."""

    def test_single_model_no_rubrics(self):
        """Test single model with template only (no rubrics)."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode, compute_objective_scores

        config = ObjectiveConfig(
            include_template=True,
            trait_mode=TraitSelectionMode.NONE,
        )

        # Mock result with no rubric
        mock_result = MagicMock()
        mock_result.template = MagicMock()
        mock_result.template.verify_result = True
        mock_result.rubric = None

        scores = compute_objective_scores(mock_result, "claude-haiku", config)

        # Should have exactly one objective
        assert scores == {"claude-haiku:template": 1.0}
        assert len(scores) == 1

    def test_single_model_no_rubrics_template_fail(self):
        """Test single model with failing template."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode, compute_objective_scores

        config = ObjectiveConfig(
            include_template=True,
            trait_mode=TraitSelectionMode.NONE,
        )

        mock_result = MagicMock()
        mock_result.template = MagicMock()
        mock_result.template.verify_result = False
        mock_result.rubric = None

        scores = compute_objective_scores(mock_result, "claude-haiku", config)

        assert scores == {"claude-haiku:template": 0.0}

    def test_rubric_present_but_not_evaluated(self):
        """Test when rubric exists but evaluation was not performed."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode, compute_objective_scores

        config = ObjectiveConfig(
            include_template=True,
            trait_mode=TraitSelectionMode.ALL,
        )

        mock_result = MagicMock()
        mock_result.template = MagicMock()
        mock_result.template.verify_result = True
        mock_result.rubric = MagicMock()
        mock_result.rubric.rubric_evaluation_performed = False  # Not evaluated

        scores = compute_objective_scores(mock_result, "claude-haiku", config)

        # Should only have template, no rubric traits
        assert scores == {"claude-haiku:template": 1.0}

    def test_empty_rubric_scores(self):
        """Test when rubric was evaluated but returned no traits."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode, compute_objective_scores

        config = ObjectiveConfig(
            include_template=True,
            trait_mode=TraitSelectionMode.ALL,
        )

        mock_result = MagicMock()
        mock_result.template = MagicMock()
        mock_result.template.verify_result = True
        mock_result.rubric = MagicMock()
        mock_result.rubric.rubric_evaluation_performed = True
        mock_result.rubric.get_all_trait_scores.return_value = {}  # Empty

        scores = compute_objective_scores(mock_result, "claude-haiku", config)

        # Should only have template
        assert scores == {"claude-haiku:template": 1.0}

    def test_no_template_no_rubrics_raises_error(self):
        """Test misconfiguration: no template and no rubrics raises ValueError."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode

        with pytest.raises(ValueError, match="must include at least one objective"):
            ObjectiveConfig(
                include_template=False,
                trait_mode=TraitSelectionMode.NONE,
            )

    def test_consistent_keys_across_multiple_calls(self):
        """Verify same config produces same keys for different results."""
        from karenina.integrations.gepa import ObjectiveConfig, TraitSelectionMode, compute_objective_scores

        config = ObjectiveConfig(
            include_template=True,
            trait_mode=TraitSelectionMode.ALL,
        )

        def make_result(template_pass: bool, traits: dict):
            mock = MagicMock()
            mock.template = MagicMock()
            mock.template.verify_result = template_pass
            mock.rubric = MagicMock()
            mock.rubric.rubric_evaluation_performed = True
            mock.rubric.get_all_trait_scores.return_value = traits
            return mock

        result1 = make_result(True, {"clarity": 4, "safety": True})
        result2 = make_result(False, {"clarity": 3, "safety": False})

        scores1 = compute_objective_scores(result1, "claude-haiku", config)
        scores2 = compute_objective_scores(result2, "claude-haiku", config)

        # Same keys, different values
        assert set(scores1.keys()) == set(scores2.keys())
        assert scores1["claude-haiku:template"] == 1.0
        assert scores2["claude-haiku:template"] == 0.0
