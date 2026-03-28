"""Tests for trait provenance tracking in merge_rubrics().

Verifies that merge_rubrics() returns a tuple of (Rubric | None, dict[str, str] | None)
where the dict maps trait names to their source ("global" or "question_specific").
"""

import pytest

from karenina.schemas.entities.rubric import (
    LLMRubricTrait,
    MetricRubricTrait,
    RegexRubricTrait,
    Rubric,
    merge_rubrics,
)


@pytest.mark.unit
def test_merge_rubrics_returns_tuple() -> None:
    """merge_rubrics returns a two-element tuple."""
    global_trait = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)
    global_rubric = Rubric(llm_traits=[global_trait])

    result = merge_rubrics(global_rubric, None)

    assert isinstance(result, tuple)
    assert len(result) == 2


@pytest.mark.unit
def test_merge_rubrics_none_none_returns_none_provenance() -> None:
    """merge_rubrics(None, None) returns (None, None)."""
    rubric, provenance = merge_rubrics(None, None)

    assert rubric is None
    assert provenance is None


@pytest.mark.unit
def test_merge_rubrics_global_only_provenance() -> None:
    """All traits from a global-only rubric get 'global' provenance."""
    g1 = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)
    g2 = RegexRubricTrait(name="has_email", pattern=r"\S+@\S+", higher_is_better=True)
    global_rubric = Rubric(llm_traits=[g1], regex_traits=[g2])

    rubric, provenance = merge_rubrics(global_rubric, None)

    assert rubric is global_rubric
    assert provenance == {"clarity": "global", "has_email": "global"}


@pytest.mark.unit
def test_merge_rubrics_question_only_provenance() -> None:
    """All traits from a question-only rubric get 'question_specific' provenance."""
    q1 = LLMRubricTrait(name="specificity", kind="boolean", higher_is_better=True)
    q2 = RegexRubricTrait(name="has_citation", pattern=r"\[\d+\]", higher_is_better=True)
    question_rubric = Rubric(llm_traits=[q1], regex_traits=[q2])

    rubric, provenance = merge_rubrics(None, question_rubric)

    assert rubric is question_rubric
    assert provenance == {"specificity": "question_specific", "has_citation": "question_specific"}


@pytest.mark.unit
def test_merge_rubrics_both_provenance() -> None:
    """Global traits get 'global', question traits get 'question_specific'."""
    g_llm = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)
    g_regex = RegexRubricTrait(name="has_email", pattern=r"\S+@\S+", higher_is_better=True)
    q_llm = LLMRubricTrait(name="specificity", kind="boolean", higher_is_better=True)
    q_metric = MetricRubricTrait(
        name="entity_check",
        evaluation_mode="tp_only",
        metrics=["precision"],
        tp_instructions=["test"],
    )

    global_rubric = Rubric(llm_traits=[g_llm], regex_traits=[g_regex])
    question_rubric = Rubric(llm_traits=[q_llm], metric_traits=[q_metric])

    rubric, provenance = merge_rubrics(global_rubric, question_rubric)

    assert provenance == {
        "clarity": "global",
        "has_email": "global",
        "specificity": "question_specific",
        "entity_check": "question_specific",
    }


@pytest.mark.unit
def test_merge_rubrics_merged_object_is_correct() -> None:
    """The returned Rubric object contains all traits from both inputs."""
    g_llm = LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)
    q_llm = LLMRubricTrait(name="specificity", kind="boolean", higher_is_better=True)

    global_rubric = Rubric(llm_traits=[g_llm])
    question_rubric = Rubric(llm_traits=[q_llm])

    rubric, provenance = merge_rubrics(global_rubric, question_rubric)

    assert rubric is not None
    assert len(rubric.llm_traits) == 2
    assert rubric.get_llm_trait_names() == ["clarity", "specificity"]


@pytest.mark.unit
def test_merge_rubrics_provenance_covers_all_trait_types() -> None:
    """Provenance dict includes entries from multiple trait types."""
    from karenina.schemas.entities.rubric import AgenticRubricTrait

    g_llm = LLMRubricTrait(name="g_llm", kind="boolean", higher_is_better=True)
    g_regex = RegexRubricTrait(name="g_regex", pattern=r"\d+", higher_is_better=True)
    q_metric = MetricRubricTrait(
        name="q_metric",
        evaluation_mode="tp_only",
        metrics=["precision"],
        tp_instructions=["test"],
    )
    q_agentic = AgenticRubricTrait(name="q_agentic", kind="boolean", description="Check something.")

    global_rubric = Rubric(llm_traits=[g_llm], regex_traits=[g_regex])
    question_rubric = Rubric(metric_traits=[q_metric], agentic_traits=[q_agentic])

    rubric, provenance = merge_rubrics(global_rubric, question_rubric)

    assert provenance["g_llm"] == "global"
    assert provenance["g_regex"] == "global"
    assert provenance["q_metric"] == "question_specific"
    assert provenance["q_agentic"] == "question_specific"


@pytest.mark.unit
def test_merge_rubrics_conflict_still_raises() -> None:
    """Conflicts still raise ValueError even with the new return type."""
    global_rubric = Rubric(llm_traits=[LLMRubricTrait(name="clarity", kind="boolean", higher_is_better=True)])
    question_rubric = Rubric(
        llm_traits=[LLMRubricTrait(name="clarity", kind="score", min_score=1, max_score=5, higher_is_better=True)]
    )

    with pytest.raises(ValueError, match="Same-type trait name conflicts"):
        merge_rubrics(global_rubric, question_rubric)
