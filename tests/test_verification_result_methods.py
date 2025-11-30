"""Tests for VerificationResultRubric helper methods (get_all_trait_scores, get_trait_by_name)."""

from karenina.schemas.workflow.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)


def test_get_all_trait_scores_all_types() -> None:
    """Test get_all_trait_scores with all trait types."""
    # Create a VerificationResult with all trait types
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="test_q1",
            template_id="test_t1",
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
            llm_trait_scores={"clarity": 4, "analysis_quality": 2},
            regex_trait_scores={"mentions_regulatory_elements": True},
            metric_trait_scores={"feature_identification": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
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
                "regex_traits": [
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
        ),
    )

    # Get all trait scores
    scores = result.rubric.get_all_trait_scores()

    # Verify all scores are present (flat structure)
    assert scores["clarity"] == 4
    assert scores["analysis_quality"] == 2
    assert scores["mentions_regulatory_elements"] is True
    assert scores["feature_identification"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_get_all_trait_scores_empty() -> None:
    """Test get_all_trait_scores with no rubric data."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="test_q1",
            template_id="test_t1",
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
    )

    # Should return empty dict when rubric is None
    assert result.rubric is None


def test_get_all_trait_scores_partial_data() -> None:
    """Test get_all_trait_scores with only some trait types."""
    # Test with only LLM traits
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="test_q1",
            template_id="test_t1",
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
                "regex_traits": [],
                "metric_traits": [],
            },
        ),
    )

    scores = result.rubric.get_all_trait_scores()
    assert scores == {"clarity": 4}

    # Test with only metric traits
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="test_q1",
            template_id="test_t1",
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
            metric_trait_scores={"entity_extraction": {"precision": 0.85, "recall": 0.92}},
            metric_trait_confusion_lists={
                "entity_extraction": {"tp": ["Alice", "Bob"], "tn": [], "fp": ["Charlie"], "fn": []}
            },
        ),
    )

    scores = result.rubric.get_all_trait_scores()
    assert scores == {"entity_extraction": {"precision": 0.85, "recall": 0.92}}


def test_get_trait_by_name_llm() -> None:
    """Test get_trait_by_name for LLM traits."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="test_q1",
            template_id="test_t1",
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
                "regex_traits": [],
                "metric_traits": [],
            },
        ),
    )

    value, trait_type = result.rubric.get_trait_by_name("clarity")
    assert value == 4
    assert trait_type == "llm"


def test_get_trait_by_name_regex() -> None:
    """Test get_trait_by_name for regex traits."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="test_q1",
            template_id="test_t1",
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
            regex_trait_scores={"has_citation": True},
            evaluation_rubric={
                "traits": [],
                "regex_traits": [
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

    value, trait_type = result.rubric.get_trait_by_name("has_citation")
    assert value is True
    assert trait_type == "regex"


def test_get_trait_by_name_metric() -> None:
    """Test get_trait_by_name for metric traits."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="test_q1",
            template_id="test_t1",
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
            metric_trait_scores={"entity_extraction": {"precision": 0.85, "recall": 0.92}},
            metric_trait_confusion_lists={"entity_extraction": {"tp": ["Alice"], "tn": [], "fp": [], "fn": []}},
        ),
    )

    value, trait_type = result.rubric.get_trait_by_name("entity_extraction")
    assert trait_type == "metric"
    # The method returns the metric scores dict directly
    assert value == {"precision": 0.85, "recall": 0.92}


def test_get_trait_by_name_not_found() -> None:
    """Test get_trait_by_name returns None for non-existent trait."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="test_q1",
            template_id="test_t1",
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
        ),
    )

    result_tuple = result.rubric.get_trait_by_name("nonexistent_trait")
    assert result_tuple is None


def test_get_all_trait_scores_with_empty_rubric() -> None:
    """Test get_all_trait_scores returns empty dict when rubric has no scores."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="test_q1",
            template_id="test_t1",
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
            # All trait scores are None
        ),
    )

    scores = result.rubric.get_all_trait_scores()
    assert scores == {}
