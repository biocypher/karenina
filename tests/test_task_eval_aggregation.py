"""Tests for StepEval rubric results aggregation across replicates."""

from karenina.benchmark.task_eval.models import StepEval
from karenina.schemas.workflow.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)


def test_aggregate_rubric_results_all_trait_types() -> None:
    """Test aggregation with LLM, manual, and metric traits."""
    step_eval = StepEval()

    # Create 3 replicates with all trait types
    step_eval.verification_results["trace_1"] = [
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=True,
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response 1",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores={"clarity": 4, "analysis_quality": 3},
                manual_trait_scores={"has_citation": True},
                metric_trait_scores={"entity_extraction": {"precision": 0.8, "recall": 0.9, "f1": 0.85}},
                metric_trait_confusion_lists={"entity_extraction": {"tp": ["Alice"], "tn": [], "fp": [], "fn": []}},
                evaluation_rubric={
                    "traits": [
                        {"name": "clarity", "description": "Clarity", "kind": "score", "min_score": 1, "max_score": 5},
                        {
                            "name": "analysis_quality",
                            "description": "Quality",
                            "kind": "score",
                            "min_score": 1,
                            "max_score": 5,
                        },
                    ],
                    "manual_traits": [
                        {
                            "name": "has_citation",
                            "description": "Has citation",
                            "pattern": r"\[\d+\]",
                            "callable_name": None,
                            "case_sensitive": True,
                            "invert_result": False,
                        }
                    ],
                    "metric_traits": [],
                },
            ),
        ),
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=True,
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response 2",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores={"clarity": 5, "analysis_quality": 4},
                manual_trait_scores={"has_citation": True},
                metric_trait_scores={"entity_extraction": {"precision": 0.9, "recall": 0.95, "f1": 0.925}},
                metric_trait_confusion_lists={"entity_extraction": {"tp": ["Bob"], "tn": [], "fp": [], "fn": []}},
                evaluation_rubric={
                    "traits": [
                        {"name": "clarity", "description": "Clarity", "kind": "score", "min_score": 1, "max_score": 5},
                        {
                            "name": "analysis_quality",
                            "description": "Quality",
                            "kind": "score",
                            "min_score": 1,
                            "max_score": 5,
                        },
                    ],
                    "manual_traits": [
                        {
                            "name": "has_citation",
                            "description": "Has citation",
                            "pattern": r"\[\d+\]",
                            "callable_name": None,
                            "case_sensitive": True,
                            "invert_result": False,
                        }
                    ],
                    "metric_traits": [],
                },
            ),
        ),
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=True,
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response 3",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores={"clarity": 4, "analysis_quality": 2},
                manual_trait_scores={"has_citation": False},
                metric_trait_scores={"entity_extraction": {"precision": 0.85, "recall": 0.92, "f1": 0.88}},
                metric_trait_confusion_lists={"entity_extraction": {"tp": ["Charlie"], "tn": [], "fp": [], "fn": []}},
                evaluation_rubric={
                    "traits": [
                        {"name": "clarity", "description": "Clarity", "kind": "score", "min_score": 1, "max_score": 5},
                        {
                            "name": "analysis_quality",
                            "description": "Quality",
                            "kind": "score",
                            "min_score": 1,
                            "max_score": 5,
                        },
                    ],
                    "manual_traits": [
                        {
                            "name": "has_citation",
                            "description": "Has citation",
                            "pattern": r"\[\d+\]",
                            "callable_name": None,
                            "case_sensitive": True,
                            "invert_result": False,
                        }
                    ],
                    "metric_traits": [],
                },
            ),
        ),
    ]

    # Aggregate
    aggregated = step_eval.aggregate_rubric_results()

    # Verify structure
    assert "trace_1" in aggregated
    assert "llm" in aggregated["trace_1"]
    assert "manual" in aggregated["trace_1"]
    assert "metric" in aggregated["trace_1"]

    # Verify LLM scores (averaged)
    assert aggregated["trace_1"]["llm"]["clarity"] == 4.333333333333333  # (4+5+4)/3
    assert aggregated["trace_1"]["llm"]["analysis_quality"] == 3.0  # (3+4+2)/3

    # Verify manual traits (pass rate)
    assert aggregated["trace_1"]["manual"]["has_citation"] == 0.6666666666666666  # 2/3

    # Verify metric traits (averaged, no confusion)
    assert "entity_extraction" in aggregated["trace_1"]["metric"]
    assert "metrics" in aggregated["trace_1"]["metric"]["entity_extraction"]
    assert "confusion" not in aggregated["trace_1"]["metric"]["entity_extraction"]  # Omitted by design

    entity_metrics = aggregated["trace_1"]["metric"]["entity_extraction"]["metrics"]
    assert abs(entity_metrics["precision"] - 0.85) < 0.001  # (0.8+0.9+0.85)/3
    assert abs(entity_metrics["recall"] - 0.9233333333333333) < 0.001  # (0.9+0.95+0.92)/3
    assert abs(entity_metrics["f1"] - 0.885) < 0.001  # (0.85+0.925+0.88)/3


def test_aggregate_rubric_results_single_replicate() -> None:
    """Test that single replicate is returned as-is without modification."""
    step_eval = StepEval()

    step_eval.verification_results["trace_1"] = [
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=True,
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores={"clarity": 4},
                manual_trait_scores={"has_citation": True},
                evaluation_rubric={
                    "traits": [
                        {"name": "clarity", "description": "Clarity", "kind": "score", "min_score": 1, "max_score": 5}
                    ],
                    "manual_traits": [
                        {
                            "name": "has_citation",
                            "description": "Has citation",
                            "pattern": r"\[\d+\]",
                            "callable_name": None,
                            "case_sensitive": True,
                            "invert_result": False,
                        }
                    ],
                    "metric_traits": [],
                },
            ),
        )
    ]

    aggregated = step_eval.aggregate_rubric_results()

    # Single replicate should be returned as-is
    assert "trace_1" in aggregated
    assert aggregated["trace_1"]["llm"]["clarity"] == 4  # Not converted to float
    assert aggregated["trace_1"]["manual"]["has_citation"] is True  # Still boolean, not pass rate


def test_aggregate_rubric_results_with_failures() -> None:
    """Test that failed replicates are excluded and count is tracked."""
    step_eval = StepEval()

    step_eval.verification_results["trace_1"] = [
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=True,
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response 1",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores={"clarity": 4},
                evaluation_rubric={
                    "traits": [
                        {"name": "clarity", "description": "Clarity", "kind": "score", "min_score": 1, "max_score": 5}
                    ],
                    "manual_traits": [],
                    "metric_traits": [],
                },
            ),
        ),
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=False,  # FAILED
                error="Test error",
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=False,
            ),
        ),
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=True,
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response 3",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores={"clarity": 5},
                evaluation_rubric={
                    "traits": [
                        {"name": "clarity", "description": "Clarity", "kind": "score", "min_score": 1, "max_score": 5}
                    ],
                    "manual_traits": [],
                    "metric_traits": [],
                },
            ),
        ),
    ]

    aggregated = step_eval.aggregate_rubric_results()

    # Verify failed replicate was excluded
    assert aggregated["trace_1"]["llm"]["clarity"] == 4.5  # (4+5)/2, not (4+5)/3

    # Verify failure count is tracked
    assert "failed_replicate_count" in aggregated["trace_1"]
    assert aggregated["trace_1"]["failed_replicate_count"] == 1


def test_aggregate_rubric_results_all_failures() -> None:
    """Test that all-failed replicates return only failure count."""
    step_eval = StepEval()

    step_eval.verification_results["trace_1"] = [
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=False,
                error="Error 1",
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=False,
            ),
        ),
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=False,
                error="Error 2",
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=False,
            ),
        ),
    ]

    aggregated = step_eval.aggregate_rubric_results()

    # All failed - should only have failure count
    assert "trace_1" in aggregated
    assert aggregated["trace_1"] == {"failed_replicate_count": 2}


def test_aggregate_rubric_results_empty_stepeval() -> None:
    """Test empty StepEval returns empty dict."""
    step_eval = StepEval()

    aggregated = step_eval.aggregate_rubric_results()

    assert aggregated == {}


def test_aggregate_rubric_results_missing_traits() -> None:
    """Test that missing traits in some replicates are handled gracefully."""
    step_eval = StepEval()

    step_eval.verification_results["trace_1"] = [
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=True,
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response 1",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores={"clarity": 4},
                evaluation_rubric={
                    "traits": [
                        {"name": "clarity", "description": "Clarity", "kind": "score", "min_score": 1, "max_score": 5}
                    ],
                    "manual_traits": [],
                    "metric_traits": [],
                },
            ),
        ),
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=True,
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response 2",
            ),
            # No rubric data at all
        ),
    ]

    aggregated = step_eval.aggregate_rubric_results()

    # Should still work with partial data
    assert "trace_1" in aggregated
    assert "llm" in aggregated["trace_1"]
    assert aggregated["trace_1"]["llm"]["clarity"] == 4.0  # Only one replicate had this trait


def test_aggregate_metric_confusion_omitted() -> None:
    """Verify that confusion matrices are omitted from aggregated results."""
    step_eval = StepEval()

    step_eval.verification_results["trace_1"] = [
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=True,
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response 1",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                metric_trait_scores={"entity_extraction": {"precision": 0.8, "recall": 0.9}},
                metric_trait_confusion_lists={
                    "entity_extraction": {"tp": ["Alice", "Bob"], "tn": [], "fp": ["Charlie"], "fn": ["Dave"]}
                },
            ),
        ),
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=True,
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response 2",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                metric_trait_scores={"entity_extraction": {"precision": 0.9, "recall": 0.95}},
                metric_trait_confusion_lists={
                    "entity_extraction": {"tp": ["Eve", "Frank"], "tn": [], "fp": [], "fn": ["Grace"]}
                },
            ),
        ),
    ]

    aggregated = step_eval.aggregate_rubric_results()

    # Verify confusion matrices are not included
    assert "metric" in aggregated["trace_1"]
    assert "entity_extraction" in aggregated["trace_1"]["metric"]
    assert "metrics" in aggregated["trace_1"]["metric"]["entity_extraction"]
    assert "confusion" not in aggregated["trace_1"]["metric"]["entity_extraction"]

    # Verify metrics are averaged
    assert abs(aggregated["trace_1"]["metric"]["entity_extraction"]["metrics"]["precision"] - 0.85) < 0.001
    assert abs(aggregated["trace_1"]["metric"]["entity_extraction"]["metrics"]["recall"] - 0.925) < 0.001


def test_aggregate_multiple_metric_names() -> None:
    """Test averaging when metrics have different subsets of metric names."""
    step_eval = StepEval()

    step_eval.verification_results["trace_1"] = [
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=True,
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response 1",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                metric_trait_scores={"entity_extraction": {"precision": 0.8, "recall": 0.9, "f1": 0.85}},
            ),
        ),
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="trace_1",
                template_id="test_template",
                completed_without_errors=True,
                question_text="Test question",
                answering_model="gpt-4.1-mini",
                parsing_model="gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2025-11-11T00:00:00",
            ),
            template=VerificationResultTemplate(
                raw_llm_response="Test response 2",
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                metric_trait_scores={
                    "entity_extraction": {"precision": 0.9, "recall": 0.95}  # Missing f1
                },
            ),
        ),
    ]

    aggregated = step_eval.aggregate_rubric_results()

    # Should average metrics that are present
    entity_metrics = aggregated["trace_1"]["metric"]["entity_extraction"]["metrics"]
    assert abs(entity_metrics["precision"] - 0.85) < 0.001  # (0.8+0.9)/2
    assert abs(entity_metrics["recall"] - 0.925) < 0.001  # (0.9+0.95)/2
    assert abs(entity_metrics["f1"] - 0.85) < 0.001  # Only one replicate had this, so just that value
