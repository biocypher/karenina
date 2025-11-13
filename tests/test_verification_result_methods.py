"""Tests for VerificationResult convenience methods (rubric_results, get_trait_by_name, get_all_scores)."""

from karenina.schemas.workflow.verification import VerificationResult


def test_rubric_results_property_all_trait_types() -> None:
    """Test rubric_results property with all trait types."""
    # Create a VerificationResult with all trait types
    result = VerificationResult(
        question_id="test_q1",
        template_id="test_t1",
        completed_without_errors=True,
        question_text="Test question",
        raw_llm_response="Test response",
        answering_model="gpt-4.1-mini",
        parsing_model="gpt-4.1-mini",
        execution_time=1.0,
        timestamp="2025-11-11T00:00:00",
        # Rubric evaluation results
        verify_rubric={"clarity": 4, "analysis_quality": 2, "mentions_regulatory_elements": True},
        evaluation_rubric={
            "traits": [
                {
                    "name": "clarity",
                    "description": "Clarity of response",
                    "kind": "score",
                    "min_score": 1,
                    "max_score": 5,
                },
                {
                    "name": "analysis_quality",
                    "description": "Quality of analysis",
                    "kind": "score",
                    "min_score": 1,
                    "max_score": 5,
                },
            ],
            "manual_traits": [
                {
                    "name": "mentions_regulatory_elements",
                    "description": "Mentions regulatory elements",
                    "pattern": r"(?i)\b(TATA|binding\s+site|motif)\b",
                    "callable_name": None,
                    "case_sensitive": True,
                    "invert_result": False,
                }
            ],
            "metric_traits": [
                {
                    "name": "feature_identification",
                    "description": "Feature identification accuracy",
                    "evaluation_mode": "tp_only",
                    "metrics": ["precision", "recall", "f1"],
                    "tp_instructions": ["TATA box identified", "Sp1 sites identified"],
                    "tn_instructions": [],
                    "repeated_extraction": True,
                }
            ],
        },
        metric_trait_confusion_lists={
            "feature_identification": {
                "tp": ["TATA box identified", "Sp1 sites identified"],
                "tn": [],
                "fp": [],
                "fn": [],
            }
        },
        metric_trait_metrics={"feature_identification": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
    )

    # Get rubric results
    rubric_data = result.rubric_results

    # Verify structure
    assert "llm" in rubric_data
    assert "manual" in rubric_data
    assert "metric" in rubric_data

    # Verify LLM traits
    assert rubric_data["llm"] == {"clarity": 4, "analysis_quality": 2}

    # Verify manual traits
    assert rubric_data["manual"] == {"mentions_regulatory_elements": True}

    # Verify metric traits
    assert "feature_identification" in rubric_data["metric"]
    metric_data = rubric_data["metric"]["feature_identification"]
    assert "metrics" in metric_data
    assert "confusion" in metric_data
    assert metric_data["metrics"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert metric_data["confusion"] == {
        "tp": ["TATA box identified", "Sp1 sites identified"],
        "tn": [],
        "fp": [],
        "fn": [],
    }


def test_rubric_results_property_empty() -> None:
    """Test rubric_results property with no rubric data."""
    result = VerificationResult(
        question_id="test_q1",
        template_id="test_t1",
        completed_without_errors=True,
        question_text="Test question",
        raw_llm_response="Test response",
        answering_model="gpt-4.1-mini",
        parsing_model="gpt-4.1-mini",
        execution_time=1.0,
        timestamp="2025-11-11T00:00:00",
    )

    # Should return empty dict
    rubric_data = result.rubric_results
    assert rubric_data == {}


def test_rubric_results_property_partial_data() -> None:
    """Test rubric_results property with only some trait types."""
    # Test with only LLM traits
    result = VerificationResult(
        question_id="test_q1",
        template_id="test_t1",
        completed_without_errors=True,
        question_text="Test question",
        raw_llm_response="Test response",
        answering_model="gpt-4.1-mini",
        parsing_model="gpt-4.1-mini",
        execution_time=1.0,
        timestamp="2025-11-11T00:00:00",
        verify_rubric={"clarity": 4},
        evaluation_rubric={
            "traits": [
                {
                    "name": "clarity",
                    "description": "Clarity of response",
                    "kind": "score",
                    "min_score": 1,
                    "max_score": 5,
                }
            ],
            "manual_traits": [],
            "metric_traits": [],
        },
    )

    rubric_data = result.rubric_results
    assert "llm" in rubric_data
    assert "manual" not in rubric_data
    assert "metric" not in rubric_data
    assert rubric_data["llm"] == {"clarity": 4}

    # Test with only metric traits
    result = VerificationResult(
        question_id="test_q1",
        template_id="test_t1",
        completed_without_errors=True,
        question_text="Test question",
        raw_llm_response="Test response",
        answering_model="gpt-4.1-mini",
        parsing_model="gpt-4.1-mini",
        execution_time=1.0,
        timestamp="2025-11-11T00:00:00",
        metric_trait_metrics={"entity_extraction": {"precision": 0.85, "recall": 0.92}},
        metric_trait_confusion_lists={
            "entity_extraction": {"tp": ["Alice", "Bob"], "tn": [], "fp": ["Charlie"], "fn": []}
        },
    )

    rubric_data = result.rubric_results
    assert "llm" not in rubric_data
    assert "manual" not in rubric_data
    assert "metric" in rubric_data
    assert rubric_data["metric"]["entity_extraction"]["metrics"] == {"precision": 0.85, "recall": 0.92}


def test_get_trait_by_name_llm() -> None:
    """Test get_trait_by_name for LLM traits."""
    result = VerificationResult(
        question_id="test_q1",
        template_id="test_t1",
        completed_without_errors=True,
        question_text="Test question",
        raw_llm_response="Test response",
        answering_model="gpt-4.1-mini",
        parsing_model="gpt-4.1-mini",
        execution_time=1.0,
        timestamp="2025-11-11T00:00:00",
        verify_rubric={"clarity": 4},
        evaluation_rubric={
            "traits": [
                {
                    "name": "clarity",
                    "description": "Clarity of response",
                    "kind": "score",
                    "min_score": 1,
                    "max_score": 5,
                }
            ],
            "manual_traits": [],
            "metric_traits": [],
        },
    )

    value, trait_type = result.get_trait_by_name("clarity")
    assert value == 4
    assert trait_type == "llm"


def test_get_trait_by_name_manual() -> None:
    """Test get_trait_by_name for manual traits."""
    result = VerificationResult(
        question_id="test_q1",
        template_id="test_t1",
        completed_without_errors=True,
        question_text="Test question",
        raw_llm_response="Test response",
        answering_model="gpt-4.1-mini",
        parsing_model="gpt-4.1-mini",
        execution_time=1.0,
        timestamp="2025-11-11T00:00:00",
        verify_rubric={"has_citation": True},
        evaluation_rubric={
            "traits": [],
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
    )

    value, trait_type = result.get_trait_by_name("has_citation")
    assert value is True
    assert trait_type == "manual"


def test_get_trait_by_name_metric() -> None:
    """Test get_trait_by_name for metric traits."""
    result = VerificationResult(
        question_id="test_q1",
        template_id="test_t1",
        completed_without_errors=True,
        question_text="Test question",
        raw_llm_response="Test response",
        answering_model="gpt-4.1-mini",
        parsing_model="gpt-4.1-mini",
        execution_time=1.0,
        timestamp="2025-11-11T00:00:00",
        metric_trait_metrics={"entity_extraction": {"precision": 0.85, "recall": 0.92}},
        metric_trait_confusion_lists={"entity_extraction": {"tp": ["Alice"], "tn": [], "fp": [], "fn": []}},
    )

    value, trait_type = result.get_trait_by_name("entity_extraction")
    assert trait_type == "metric"
    assert "metrics" in value
    assert "confusion" in value
    assert value["metrics"] == {"precision": 0.85, "recall": 0.92}


def test_get_trait_by_name_not_found() -> None:
    """Test get_trait_by_name returns None for non-existent trait."""
    result = VerificationResult(
        question_id="test_q1",
        template_id="test_t1",
        completed_without_errors=True,
        question_text="Test question",
        raw_llm_response="Test response",
        answering_model="gpt-4.1-mini",
        parsing_model="gpt-4.1-mini",
        execution_time=1.0,
        timestamp="2025-11-11T00:00:00",
    )

    result_tuple = result.get_trait_by_name("nonexistent_trait")
    assert result_tuple is None


def test_get_all_scores_all_types() -> None:
    """Test get_all_scores with all trait types."""
    result = VerificationResult(
        question_id="test_q1",
        template_id="test_t1",
        completed_without_errors=True,
        question_text="Test question",
        raw_llm_response="Test response",
        answering_model="gpt-4.1-mini",
        parsing_model="gpt-4.1-mini",
        execution_time=1.0,
        timestamp="2025-11-11T00:00:00",
        verify_rubric={"clarity": 4, "analysis_quality": 2, "mentions_regulatory_elements": True},
        evaluation_rubric={
            "traits": [
                {"name": "clarity", "description": "Clarity", "kind": "score", "min_score": 1, "max_score": 5},
                {
                    "name": "analysis_quality",
                    "description": "Analysis",
                    "kind": "score",
                    "min_score": 1,
                    "max_score": 5,
                },
            ],
            "manual_traits": [
                {
                    "name": "mentions_regulatory_elements",
                    "description": "Mentions regulatory elements",
                    "pattern": r"(?i)\b(TATA)\b",
                    "callable_name": None,
                    "case_sensitive": True,
                    "invert_result": False,
                }
            ],
            "metric_traits": [],
        },
        metric_trait_metrics={"feature_identification": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
        metric_trait_confusion_lists={"feature_identification": {"tp": ["TATA box"], "tn": [], "fp": [], "fn": []}},
    )

    scores = result.get_all_scores()

    # Check all scores are present
    assert scores["clarity"] == 4
    assert scores["analysis_quality"] == 2
    assert scores["mentions_regulatory_elements"] is True
    assert scores["feature_identification.precision"] == 1.0
    assert scores["feature_identification.recall"] == 1.0
    assert scores["feature_identification.f1"] == 1.0

    # Verify confusion data is NOT included
    assert "feature_identification.tp" not in scores
    assert "feature_identification.confusion" not in scores


def test_get_all_scores_empty() -> None:
    """Test get_all_scores with no rubric data."""
    result = VerificationResult(
        question_id="test_q1",
        template_id="test_t1",
        completed_without_errors=True,
        question_text="Test question",
        raw_llm_response="Test response",
        answering_model="gpt-4.1-mini",
        parsing_model="gpt-4.1-mini",
        execution_time=1.0,
        timestamp="2025-11-11T00:00:00",
    )

    scores = result.get_all_scores()
    assert scores == {}


def test_get_all_scores_metric_only() -> None:
    """Test get_all_scores with only metric traits."""
    result = VerificationResult(
        question_id="test_q1",
        template_id="test_t1",
        completed_without_errors=True,
        question_text="Test question",
        raw_llm_response="Test response",
        answering_model="gpt-4.1-mini",
        parsing_model="gpt-4.1-mini",
        execution_time=1.0,
        timestamp="2025-11-11T00:00:00",
        metric_trait_metrics={
            "entity_extraction": {"precision": 0.85, "recall": 0.92, "f1": 0.88},
            "feature_identification": {"precision": 1.0, "recall": 0.95},
        },
        metric_trait_confusion_lists={
            "entity_extraction": {"tp": ["Alice"], "tn": [], "fp": [], "fn": []},
            "feature_identification": {"tp": ["TATA"], "tn": [], "fp": [], "fn": ["Sp1"]},
        },
    )

    scores = result.get_all_scores()

    # Check all metric scores are flattened with dot notation
    assert scores["entity_extraction.precision"] == 0.85
    assert scores["entity_extraction.recall"] == 0.92
    assert scores["entity_extraction.f1"] == 0.88
    assert scores["feature_identification.precision"] == 1.0
    assert scores["feature_identification.recall"] == 0.95

    # Verify no confusion data
    assert "entity_extraction.tp" not in scores
    assert "feature_identification.fn" not in scores


def test_rubric_results_missing_confusion_lists() -> None:
    """Test rubric_results when metric_trait_confusion_lists is None."""
    result = VerificationResult(
        question_id="test_q1",
        template_id="test_t1",
        completed_without_errors=True,
        question_text="Test question",
        raw_llm_response="Test response",
        answering_model="gpt-4.1-mini",
        parsing_model="gpt-4.1-mini",
        execution_time=1.0,
        timestamp="2025-11-11T00:00:00",
        metric_trait_metrics={"feature_identification": {"precision": 1.0, "recall": 1.0}},
        metric_trait_confusion_lists=None,  # Explicitly None
    )

    rubric_data = result.rubric_results

    # Should still have metric data with empty confusion
    assert "metric" in rubric_data
    assert "feature_identification" in rubric_data["metric"]
    assert rubric_data["metric"]["feature_identification"]["confusion"] == {}
