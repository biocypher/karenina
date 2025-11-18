"""Tests for question-specific rubric functionality."""

import pytest

from karenina.schemas.domain import LLMRubricTrait, Rubric, merge_rubrics


class TestQuestionRubrics:
    """Test question-specific rubric merging functionality."""

    def test_merge_rubrics_both_none(self) -> None:
        """Test merging when both rubrics are None."""
        result = merge_rubrics(None, None)
        assert result is None

    def test_merge_rubrics_global_only(self) -> None:
        """Test merging when only global rubric exists."""
        global_rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="clarity", description="Is the answer clear?", kind="boolean"),
                LLMRubricTrait(
                    name="accuracy", description="Is the answer accurate?", kind="score", min_score=1, max_score=5
                ),
            ]
        )

        result = merge_rubrics(global_rubric, None)
        assert result == global_rubric
        assert len(result.llm_traits) == 2

    def test_merge_rubrics_question_only(self) -> None:
        """Test merging when only question rubric exists."""
        question_rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="specificity", description="Is the answer specific to this question?", kind="boolean"
                )
            ]
        )

        result = merge_rubrics(None, question_rubric)
        assert result == question_rubric
        assert len(result.llm_traits) == 1

    def test_merge_rubrics_both_present(self) -> None:
        """Test merging when both rubrics exist."""
        global_rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="clarity", description="Is the answer clear?", kind="boolean"),
                LLMRubricTrait(
                    name="accuracy", description="Is the answer accurate?", kind="score", min_score=1, max_score=5
                ),
            ]
        )

        question_rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="specificity", description="Is the answer specific to this question?", kind="boolean"
                ),
                LLMRubricTrait(
                    name="depth", description="How deep is the analysis?", kind="score", min_score=1, max_score=3
                ),
            ]
        )

        result = merge_rubrics(global_rubric, question_rubric)
        assert result is not None
        assert len(result.llm_traits) == 4

        # Check that all traits are present
        trait_names = {trait.name for trait in result.llm_traits}
        assert trait_names == {"clarity", "accuracy", "specificity", "depth"}

        # Verify trait order (global first, then question)
        assert result.llm_traits[0].name == "clarity"
        assert result.llm_traits[1].name == "accuracy"
        assert result.llm_traits[2].name == "specificity"
        assert result.llm_traits[3].name == "depth"

    def test_merge_rubrics_name_conflicts(self) -> None:
        """Test that merging fails when trait names conflict."""
        global_rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(name="clarity", description="Is the answer clear?", kind="boolean"),
                LLMRubricTrait(
                    name="accuracy", description="Is the answer accurate?", kind="score", min_score=1, max_score=5
                ),
            ]
        )

        question_rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="clarity",
                    description="Different definition of clarity",
                    kind="score",
                    min_score=1,
                    max_score=3,
                ),
                LLMRubricTrait(name="specificity", description="Is the answer specific?", kind="boolean"),
            ]
        )

        with pytest.raises(ValueError, match="Trait name conflicts between global and question rubrics: {'clarity'}"):
            merge_rubrics(global_rubric, question_rubric)

    def test_merge_rubrics_preserves_trait_properties(self) -> None:
        """Test that trait properties are preserved during merging."""
        global_rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="clarity", description="Is the answer clear?", kind="boolean")]
        )

        question_rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="depth", description="How deep is the analysis?", kind="score", min_score=2, max_score=10
                )
            ]
        )

        result = merge_rubrics(global_rubric, question_rubric)
        assert result is not None

        # Find the traits in the result
        clarity_trait = next(t for t in result.llm_traits if t.name == "clarity")
        depth_trait = next(t for t in result.llm_traits if t.name == "depth")

        # Verify properties are preserved
        assert clarity_trait.kind == "boolean"
        assert clarity_trait.description == "Is the answer clear?"
        # Boolean traits get default scores of 1-5 even if not specified
        assert clarity_trait.min_score == 1
        assert clarity_trait.max_score == 5

        assert depth_trait.kind == "score"
        assert depth_trait.description == "How deep is the analysis?"
        assert depth_trait.min_score == 2
        assert depth_trait.max_score == 10

    def test_merge_rubrics_empty_rubrics(self) -> None:
        """Test merging rubrics with no traits."""
        empty_global = Rubric(llm_traits=[])
        empty_question = Rubric(llm_traits=[])

        result = merge_rubrics(empty_global, empty_question)
        assert result is not None
        assert len(result.llm_traits) == 0

    def test_merge_rubrics_mixed_empty(self) -> None:
        """Test merging when one rubric is empty."""
        global_rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="clarity", description="Is the answer clear?", kind="boolean")]
        )
        empty_question = Rubric(llm_traits=[])

        result = merge_rubrics(global_rubric, empty_question)
        assert result is not None
        assert len(result.llm_traits) == 1
        assert result.llm_traits[0].name == "clarity"

        # Test reverse
        empty_global = Rubric(llm_traits=[])
        question_rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="specificity", description="Is the answer specific?", kind="boolean")]
        )

        result = merge_rubrics(empty_global, question_rubric)
        assert result is not None
        assert len(result.llm_traits) == 1
        assert result.llm_traits[0].name == "specificity"
