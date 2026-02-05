"""Unit tests for Rubric schema classes.

Tests cover:
- MetricRubricTrait validation and configuration
- Rubric class trait management
- RubricEvaluation
- merge_rubrics function
"""

import pytest
from pydantic import ValidationError

from karenina.schemas.entities import (
    CallableTrait,
    LLMRubricTrait,
    MetricRubricTrait,
    RegexTrait,
    Rubric,
    RubricEvaluation,
    merge_rubrics,
)

# =============================================================================
# MetricRubricTrait Tests
# =============================================================================


@pytest.mark.unit
def test_metric_rubric_trait_tp_only_mode() -> None:
    """Test MetricRubricTrait with tp_only mode."""
    trait = MetricRubricTrait(
        name="entity_extraction",
        evaluation_mode="tp_only",
        metrics=["precision", "recall", "f1"],
        tp_instructions=["mitochondria", "apoptosis"],
    )

    assert trait.name == "entity_extraction"
    assert trait.evaluation_mode == "tp_only"
    assert trait.metrics == ["precision", "recall", "f1"]
    assert trait.tp_instructions == ["mitochondria", "apoptosis"]


@pytest.mark.unit
def test_metric_rubric_trait_full_matrix_mode() -> None:
    """Test MetricRubricTrait with full_matrix mode."""
    trait = MetricRubricTrait(
        name="entity_extraction",
        evaluation_mode="full_matrix",
        metrics=["precision", "recall", "specificity", "accuracy", "f1"],
        tp_instructions=["mitochondria", "apoptosis"],
        tn_instructions=["nucleus", "ribosome"],
    )

    assert trait.evaluation_mode == "full_matrix"
    assert trait.tn_instructions == ["nucleus", "ribosome"]


@pytest.mark.unit
def test_metric_rubric_trait_empty_tp_instructions_raises_error() -> None:
    """Test that empty tp_instructions raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        MetricRubricTrait(
            name="test",
            evaluation_mode="tp_only",
            metrics=["precision"],
        )

    assert "TP instructions must be provided" in str(exc_info.value)


@pytest.mark.unit
def test_metric_rubric_trait_invalid_metric_name() -> None:
    """Test that invalid metric names raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        MetricRubricTrait(
            name="test",
            evaluation_mode="tp_only",
            metrics=["invalid_metric"],
            tp_instructions=["test"],
        )

    assert "Invalid metric names" in str(exc_info.value)


@pytest.mark.unit
def test_metric_rubric_trait_specificity_requires_full_matrix() -> None:
    """Test that specificity metric requires full_matrix mode."""
    with pytest.raises(ValidationError) as exc_info:
        MetricRubricTrait(
            name="test",
            evaluation_mode="tp_only",
            metrics=["specificity"],  # Not available in tp_only mode
            tp_instructions=["test"],
        )

    assert "not available in tp_only mode" in str(exc_info.value)


@pytest.mark.unit
def test_metric_rubric_trait_full_matrix_requires_tn_instructions() -> None:
    """Test that full_matrix mode requires tn_instructions."""
    with pytest.raises(ValidationError) as exc_info:
        MetricRubricTrait(
            name="test",
            evaluation_mode="full_matrix",
            metrics=["precision"],
            tp_instructions=["test"],
            # tn_instructions missing
        )

    assert "TN instructions must be provided in full_matrix mode" in str(exc_info.value)


@pytest.mark.unit
def test_metric_rubric_trait_repeated_extraction_default() -> None:
    """Test that repeated_extraction defaults to True."""
    trait = MetricRubricTrait(
        name="test",
        evaluation_mode="tp_only",
        metrics=["precision"],
        tp_instructions=["test"],
    )

    assert trait.repeated_extraction is True


@pytest.mark.unit
def test_metric_rubric_trait_repeated_extraction_disabled() -> None:
    """Test setting repeated_extraction to False."""
    trait = MetricRubricTrait(
        name="test",
        evaluation_mode="tp_only",
        metrics=["precision"],
        tp_instructions=["test"],
        repeated_extraction=False,
    )

    assert trait.repeated_extraction is False


@pytest.mark.unit
def test_metric_rubric_trait_get_required_buckets_tp_only() -> None:
    """Test get_required_buckets for tp_only mode."""
    trait = MetricRubricTrait(
        name="test",
        evaluation_mode="tp_only",
        metrics=["precision", "recall"],
        tp_instructions=["test"],
    )

    buckets = trait.get_required_buckets()
    assert buckets == {"tp", "fn", "fp"}


@pytest.mark.unit
def test_metric_rubric_trait_get_required_buckets_full_matrix() -> None:
    """Test get_required_buckets for full_matrix mode."""
    trait = MetricRubricTrait(
        name="test",
        evaluation_mode="full_matrix",
        metrics=["accuracy"],
        tp_instructions=["test1"],
        tn_instructions=["test2"],
    )

    buckets = trait.get_required_buckets()
    assert buckets == {"tp", "fn", "tn", "fp"}


# =============================================================================
# Rubric Class Tests
# =============================================================================


@pytest.mark.unit
def test_rubric_empty() -> None:
    """Test creating empty rubric."""
    rubric = Rubric()

    assert rubric.llm_traits == []
    assert rubric.regex_traits == []
    assert rubric.callable_traits == []
    assert rubric.metric_traits == []


@pytest.mark.unit
def test_rubric_with_llm_traits() -> None:
    """Test Rubric with LLM traits."""
    trait1 = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True, description="Response clarity")
    trait2 = LLMRubricTrait(name="quality", kind="score", higher_is_better=True, min_score=1, max_score=5)

    rubric = Rubric(llm_traits=[trait1, trait2])

    assert len(rubric.llm_traits) == 2
    assert rubric.get_llm_trait_names() == ["clarity", "quality"]


@pytest.mark.unit
def test_rubric_with_regex_traits() -> None:
    """Test Rubric with regex traits."""
    trait1 = RegexTrait(
        name="has_email",
        pattern=r"\S+@\S+",
        higher_is_better=True,
    )
    trait2 = RegexTrait(
        name="has_citation",
        pattern=r"\[\d+\]",
        higher_is_better=True,
    )

    rubric = Rubric(regex_traits=[trait1, trait2])

    assert len(rubric.regex_traits) == 2
    assert rubric.get_regex_trait_names() == ["has_email", "has_citation"]


@pytest.mark.unit
def test_rubric_with_callable_traits() -> None:
    """Test Rubric with callable traits."""
    import cloudpickle

    trait1 = CallableTrait(
        name="min_length",
        kind="boolean",
        callable_code=cloudpickle.dumps(lambda x: len(x) >= 10),
        higher_is_better=True,
    )

    rubric = Rubric(callable_traits=[trait1])

    assert len(rubric.callable_traits) == 1
    assert rubric.get_callable_trait_names() == ["min_length"]


@pytest.mark.unit
def test_rubric_with_metric_traits() -> None:
    """Test Rubric with metric traits."""
    trait1 = MetricRubricTrait(
        name="entity_extraction",
        evaluation_mode="tp_only",
        metrics=["precision", "recall"],
        tp_instructions=["mitochondria"],
    )

    rubric = Rubric(metric_traits=[trait1])

    assert len(rubric.metric_traits) == 1
    assert rubric.get_metric_trait_names() == ["entity_extraction"]


@pytest.mark.unit
def test_rubric_get_trait_names() -> None:
    """Test get_trait_names returns all trait names."""
    llm_trait = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)
    regex_trait = RegexTrait(
        name="has_email",
        pattern=r"\S+@\S+",
        higher_is_better=True,
    )
    import cloudpickle

    callable_trait = CallableTrait(
        name="min_length",
        kind="boolean",
        callable_code=cloudpickle.dumps(lambda x: len(x) >= 10),
        higher_is_better=True,
    )
    metric_trait = MetricRubricTrait(
        name="entity_check",
        evaluation_mode="tp_only",
        metrics=["precision"],
        tp_instructions=["test"],
    )

    rubric = Rubric(
        llm_traits=[llm_trait],
        regex_traits=[regex_trait],
        callable_traits=[callable_trait],
        metric_traits=[metric_trait],
    )

    names = rubric.get_trait_names()
    assert set(names) == {"clarity", "has_email", "min_length", "entity_check"}


@pytest.mark.unit
def test_rubric_get_trait_max_scores() -> None:
    """Test get_trait_max_scores returns max scores for score-based traits."""
    trait1 = LLMRubricTrait(
        name="clarity",
        kind="boolean",
        higher_is_better=True,
    )
    trait2 = LLMRubricTrait(
        name="quality",
        kind="score",
        min_score=1,
        max_score=10,
        higher_is_better=True,
    )
    import cloudpickle

    trait3 = CallableTrait(
        name="readability",
        kind="score",
        callable_code=cloudpickle.dumps(lambda _: 5),
        min_score=1,
        max_score=5,
        higher_is_better=True,
    )

    rubric = Rubric(llm_traits=[trait1, trait2], callable_traits=[trait3])

    max_scores = rubric.get_trait_max_scores()

    # Boolean traits not included
    assert "clarity" not in max_scores
    assert max_scores["quality"] == 10
    assert max_scores["readability"] == 5


@pytest.mark.unit
def test_rubric_get_trait_directionalities() -> None:
    """Test get_trait_directionalities returns higher_is_better values."""
    llm_trait = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)
    regex_trait = RegexTrait(
        name="no_profanity",
        pattern=r"\bbadword\b",
        higher_is_better=True,
    )
    import cloudpickle

    callable_trait = CallableTrait(
        name="shortness",
        kind="boolean",
        callable_code=cloudpickle.dumps(lambda x: len(x) < 100),
        higher_is_better=False,
    )

    rubric = Rubric(
        llm_traits=[llm_trait],
        regex_traits=[regex_trait],
        callable_traits=[callable_trait],
    )

    directionalities = rubric.get_trait_directionalities()

    assert directionalities["clarity"] is True
    assert directionalities["no_profanity"] is True
    assert directionalities["shortness"] is False


@pytest.mark.unit
def test_rubric_validate_evaluation_success() -> None:
    """Test validate_evaluation with valid evaluation."""
    llm_trait = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)
    score_trait = LLMRubricTrait(
        name="quality",
        kind="score",
        min_score=1,
        max_score=5,
        higher_is_better=True,
    )
    regex_trait = RegexTrait(
        name="has_email",
        pattern=r"\S+@\S+",
        higher_is_better=True,
    )
    import cloudpickle

    callable_trait = CallableTrait(
        name="min_length",
        kind="boolean",
        callable_code=cloudpickle.dumps(lambda x: len(x) >= 10),
        higher_is_better=True,
    )

    rubric = Rubric(
        llm_traits=[llm_trait, score_trait],
        regex_traits=[regex_trait],
        callable_traits=[callable_trait],
    )

    evaluation = {
        "clarity": True,
        "quality": 4,
        "has_email": True,
        "min_length": False,
    }

    assert rubric.validate_evaluation(evaluation) is True


@pytest.mark.unit
def test_rubric_validate_evaluation_missing_trait() -> None:
    """Test validate_evaluation fails when trait is missing."""
    trait = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)

    rubric = Rubric(llm_traits=[trait])

    # Missing "clarity" key
    evaluation = {}

    assert rubric.validate_evaluation(evaluation) is False


@pytest.mark.unit
def test_rubric_validate_evaluation_extra_trait() -> None:
    """Test validate_evaluation fails when extra trait present."""
    trait = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)

    rubric = Rubric(llm_traits=[trait])

    # Extra "unknown" key
    evaluation = {"clarity": True, "unknown": True}

    assert rubric.validate_evaluation(evaluation) is False


@pytest.mark.unit
def test_rubric_validate_evaluation_invalid_score_type() -> None:
    """Test validate_evaluation fails with wrong type."""
    trait = LLMRubricTrait(
        name="quality",
        kind="score",
        min_score=1,
        max_score=5,
        higher_is_better=True,
    )

    rubric = Rubric(llm_traits=[trait])

    # Should be int, not bool
    evaluation = {"quality": True}

    assert rubric.validate_evaluation(evaluation) is False


@pytest.mark.unit
def test_rubric_validate_evaluation_score_out_of_range() -> None:
    """Test validate_evaluation fails with score out of range."""
    trait = LLMRubricTrait(
        name="quality",
        kind="score",
        min_score=1,
        max_score=5,
        higher_is_better=True,
    )

    rubric = Rubric(llm_traits=[trait])

    # Score 6 is outside 1-5 range
    evaluation = {"quality": 6}

    assert rubric.validate_evaluation(evaluation) is False


@pytest.mark.unit
def test_rubric_extra_fields_forbidden() -> None:
    """Test that extra fields are rejected."""
    with pytest.raises(ValidationError):
        Rubric(
            llm_traits=[],
            extra_field="not_allowed",
        )


# =============================================================================
# RubricEvaluation Tests
# =============================================================================


@pytest.mark.unit
def test_rubric_evaluation_creation() -> None:
    """Test RubricEvaluation creation."""
    evaluation = RubricEvaluation(trait_scores={"clarity": True, "quality": 4})

    assert evaluation.trait_scores == {"clarity": True, "quality": 4}


@pytest.mark.unit
def test_rubric_evaluation_extra_fields_forbidden() -> None:
    """Test that extra fields are rejected."""
    with pytest.raises(ValidationError):
        RubricEvaluation(
            trait_scores={"clarity": True},
            extra_field="not_allowed",
        )


# =============================================================================
# merge_rubrics Function Tests
# =============================================================================


@pytest.mark.unit
def test_merge_rubrics_both_none() -> None:
    """Test merge_rubrics with both rubrics as None."""
    result = merge_rubrics(None, None)

    assert result is None


@pytest.mark.unit
def test_merge_rubrics_global_only() -> None:
    """Test merge_rubrics with only global rubric."""
    global_trait = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)
    global_rubric = Rubric(llm_traits=[global_trait])

    result = merge_rubrics(global_rubric, None)

    assert result is global_rubric


@pytest.mark.unit
def test_merge_rubrics_question_only() -> None:
    """Test merge_rubrics with only question rubric."""
    question_trait = LLMRubricTrait(name="specificity", kind="boolean", higher_is_better=True)
    question_rubric = Rubric(llm_traits=[question_trait])

    result = merge_rubrics(None, question_rubric)

    assert result is question_rubric


@pytest.mark.unit
def test_merge_rubrics_no_conflicts() -> None:
    """Test merge_rubrics with no trait name conflicts."""
    global_trait = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)
    question_trait = LLMRubricTrait(name="specificity", kind="boolean", higher_is_better=True)

    global_rubric = Rubric(llm_traits=[global_trait])
    question_rubric = Rubric(llm_traits=[question_trait])

    result = merge_rubrics(global_rubric, question_rubric)

    assert len(result.llm_traits) == 2
    assert result.get_llm_trait_names() == ["clarity", "specificity"]


@pytest.mark.unit
def test_merge_rubrics_with_conflict_raises_error() -> None:
    """Test merge_rubrics raises ValueError on trait name conflict."""
    global_trait = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)
    question_trait = LLMRubricTrait(name="clarity", kind="score", min_score=1, max_score=5, higher_is_better=True)

    global_rubric = Rubric(llm_traits=[global_trait])
    question_rubric = Rubric(llm_traits=[question_trait])

    with pytest.raises(ValueError) as exc_info:
        merge_rubrics(global_rubric, question_rubric)

    assert "Trait name conflicts" in str(exc_info.value)
    assert "clarity" in str(exc_info.value)


@pytest.mark.unit
def test_merge_rubrics_all_trait_types() -> None:
    """Test merge_rubrics merges all trait types."""
    global_llm = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)
    global_regex = RegexTrait(
        name="has_email",
        pattern=r"\S+@\S+",
        higher_is_better=True,
    )

    question_llm = LLMRubricTrait(name="specificity", kind="boolean", higher_is_better=True)
    question_regex = RegexTrait(
        name="has_citation",
        pattern=r"\[\d+\]",
        higher_is_better=True,
    )
    question_metric = MetricRubricTrait(
        name="entity_check",
        evaluation_mode="tp_only",
        metrics=["precision"],
        tp_instructions=["test"],
    )

    global_rubric = Rubric(llm_traits=[global_llm], regex_traits=[global_regex])
    question_rubric = Rubric(
        llm_traits=[question_llm],
        regex_traits=[question_regex],
        metric_traits=[question_metric],
    )

    result = merge_rubrics(global_rubric, question_rubric)

    assert len(result.llm_traits) == 2
    assert len(result.regex_traits) == 2
    assert len(result.metric_traits) == 1
    assert set(result.get_trait_names()) == {
        "clarity",
        "has_email",
        "specificity",
        "has_citation",
        "entity_check",
    }


@pytest.mark.unit
def test_valid_metrics_constants() -> None:
    """Test that VALID_METRICS constants are correctly defined."""
    from karenina.schemas.entities.rubric import (
        VALID_METRICS,
        VALID_METRICS_FULL_MATRIX,
        VALID_METRICS_TP_ONLY,
    )

    assert {"precision", "recall", "f1"} == VALID_METRICS_TP_ONLY
    assert {"precision", "recall", "specificity", "accuracy", "f1"} == VALID_METRICS_FULL_MATRIX
    assert {"precision", "recall", "specificity", "accuracy", "f1"} == VALID_METRICS


@pytest.mark.unit
def test_metric_requirements_constant() -> None:
    """Test that METRIC_REQUIREMENTS constant is correctly defined."""
    from karenina.schemas.entities.rubric import METRIC_REQUIREMENTS

    assert METRIC_REQUIREMENTS["precision"] == {"tp", "fp"}
    assert METRIC_REQUIREMENTS["recall"] == {"tp", "fn"}
    assert METRIC_REQUIREMENTS["specificity"] == {"tn", "fp"}
    assert METRIC_REQUIREMENTS["accuracy"] == {"tp", "tn", "fp", "fn"}
    assert METRIC_REQUIREMENTS["f1"] == {"tp", "fp", "fn"}
