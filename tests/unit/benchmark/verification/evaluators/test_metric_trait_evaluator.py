"""Unit tests for MetricTraitEvaluator surviving methods.

Tests cover _deduplicate_excerpts, _deduplicate_confusion_lists,
and _compute_metrics, which remain after removing the dead
_parse_metric_trait_response method.
"""

import pytest

from karenina.schemas.config.models import ModelConfig


def _make_model_config() -> ModelConfig:
    """Create a test ModelConfig."""
    return ModelConfig(
        id="test-model",
        model_name="test-model",
        model_provider="anthropic",
        interface="claude_agent_sdk",
    )


def _make_evaluator():
    """Create a MetricTraitEvaluator with a mock LLM."""
    from unittest.mock import MagicMock

    from karenina.benchmark.verification.evaluators.rubric.metric_trait import MetricTraitEvaluator

    mock_llm = MagicMock()
    return MetricTraitEvaluator(mock_llm, model_config=_make_model_config())


@pytest.mark.unit
class TestDeduplicateExcerpts:
    """Tests for MetricTraitEvaluator._deduplicate_excerpts."""

    def test_no_duplicates(self):
        evaluator = _make_evaluator()
        result = evaluator._deduplicate_excerpts(["asthma", "bronchitis"])
        assert result == ["asthma", "bronchitis"]

    def test_case_insensitive_dedup(self):
        evaluator = _make_evaluator()
        result = evaluator._deduplicate_excerpts(["Asthma", "asthma", "ASTHMA"])
        assert result == ["Asthma"]

    def test_preserves_first_occurrence(self):
        evaluator = _make_evaluator()
        result = evaluator._deduplicate_excerpts(["COPD", "asthma", "copd"])
        assert result == ["COPD", "asthma"]

    def test_empty_strings_removed(self):
        evaluator = _make_evaluator()
        result = evaluator._deduplicate_excerpts(["asthma", "", "  ", "bronchitis"])
        assert result == ["asthma", "bronchitis"]

    def test_empty_list(self):
        evaluator = _make_evaluator()
        result = evaluator._deduplicate_excerpts([])
        assert result == []


@pytest.mark.unit
class TestDeduplicateConfusionLists:
    """Tests for MetricTraitEvaluator._deduplicate_confusion_lists."""

    def test_deduplicates_all_buckets(self):
        evaluator = _make_evaluator()
        confusion = {
            "tp": ["asthma", "Asthma"],
            "fn": ["copd", "COPD", "pneumonia"],
            "fp": ["cancer"],
            "tn": [],
        }
        result = evaluator._deduplicate_confusion_lists(confusion)
        assert result["tp"] == ["asthma"]
        assert result["fn"] == ["copd", "pneumonia"]
        assert result["fp"] == ["cancer"]
        assert result["tn"] == []


@pytest.mark.unit
class TestComputeMetrics:
    """Tests for MetricTraitEvaluator._compute_metrics."""

    def test_precision(self):
        evaluator = _make_evaluator()
        result = evaluator._compute_metrics(tp=["a", "b"], tn=[], fp=["c"], fn=[], requested_metrics=["precision"])
        assert result["precision"] == pytest.approx(2 / 3)

    def test_recall(self):
        evaluator = _make_evaluator()
        result = evaluator._compute_metrics(tp=["a", "b"], tn=[], fp=[], fn=["c"], requested_metrics=["recall"])
        assert result["recall"] == pytest.approx(2 / 3)

    def test_f1(self):
        evaluator = _make_evaluator()
        result = evaluator._compute_metrics(tp=["a", "b"], tn=[], fp=["c"], fn=["d"], requested_metrics=["f1"])
        # precision = 2/3, recall = 2/3, f1 = 2*(2/3*2/3)/(2/3+2/3) = 2/3
        assert result["f1"] == pytest.approx(2 / 3)

    def test_specificity(self):
        evaluator = _make_evaluator()
        result = evaluator._compute_metrics(
            tp=[], tn=["a", "b", "c"], fp=["d"], fn=[], requested_metrics=["specificity"]
        )
        assert result["specificity"] == pytest.approx(3 / 4)

    def test_accuracy(self):
        evaluator = _make_evaluator()
        result = evaluator._compute_metrics(tp=["a"], tn=["b"], fp=["c"], fn=["d"], requested_metrics=["accuracy"])
        assert result["accuracy"] == pytest.approx(2 / 4)

    def test_zero_denominator_returns_zero(self):
        evaluator = _make_evaluator()
        result = evaluator._compute_metrics(tp=[], tn=[], fp=[], fn=[], requested_metrics=["precision", "recall", "f1"])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_multiple_metrics(self):
        evaluator = _make_evaluator()
        result = evaluator._compute_metrics(
            tp=["a", "b"],
            tn=["c"],
            fp=["d"],
            fn=["e"],
            requested_metrics=["precision", "recall", "f1", "accuracy", "specificity"],
        )
        assert len(result) == 5
        assert all(isinstance(v, float) for v in result.values())
