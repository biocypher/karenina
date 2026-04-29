"""Structural tests for the redesigned DeepJudgmentPromptBuilder.

These tests pin the new prompt layout (Role / Principles / Anti-patterns /
Output handoff) and the integer-anchor scoring helper. They are independent of
the existing bias and task_eval_mode tests.
"""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.prompts.deep_judgment.rubric.deep_judgment import (
    build_integer_score_labels,
)
from karenina.schemas.entities import LLMRubricTrait

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def boolean_trait() -> LLMRubricTrait:
    return LLMRubricTrait(
        name="GroundedAnswer",
        description="Answer is grounded in the retrieved evidence.",
        kind="boolean",
        higher_is_better=True,
    )


@pytest.fixture
def score_trait() -> LLMRubricTrait:
    return LLMRubricTrait(
        name="Clarity",
        description="Answer is clear and well-structured.",
        kind="score",
        min_score=1,
        max_score=5,
        higher_is_better=True,
    )


# ----------------------------------------------------------------------
# build_integer_score_labels
# ----------------------------------------------------------------------


class TestBuildIntegerScoreLabels:
    def test_one_to_three_uses_lookup(self) -> None:
        result = build_integer_score_labels(1, 3)
        assert result == [
            (1, "Does not meet"),
            (2, "Partially meets"),
            (3, "Fully meets"),
        ]

    def test_one_to_five_uses_lookup(self) -> None:
        result = build_integer_score_labels(1, 5)
        assert result == [
            (1, "Poor"),
            (2, "Below average"),
            (3, "Adequate"),
            (4, "Strong"),
            (5, "Excellent"),
        ]

    def test_one_to_seven_uses_lookup(self) -> None:
        result = build_integer_score_labels(1, 7)
        assert [pair[0] for pair in result] == [1, 2, 3, 4, 5, 6, 7]
        assert result[0] == (1, "Very poor")
        assert result[-1] == (7, "Excellent")
        assert len(result) == 7

    def test_one_to_ten_uses_lookup(self) -> None:
        result = build_integer_score_labels(1, 10)
        assert [pair[0] for pair in result] == list(range(1, 11))
        assert result[0][1].lower().startswith("poor") or result[0][1].lower().startswith("very")
        assert result[-1] == (10, "Excellent")

    def test_zero_to_two_falls_back_to_ladder(self) -> None:
        # Atypical range: triggers the fallback ladder.
        result = build_integer_score_labels(0, 2)
        assert [pair[0] for pair in result] == [0, 1, 2]
        # First label is the lowest rung; last is the highest.
        assert "lowest" in result[0][1].lower() or result[0][1].lower() == "lowest"
        assert "highest" in result[-1][1].lower() or result[-1][1].lower() == "highest"

    def test_two_to_six_falls_back_to_ladder(self) -> None:
        # Range length 5 but not starting at 1 -> fallback.
        result = build_integer_score_labels(2, 6)
        assert [pair[0] for pair in result] == [2, 3, 4, 5, 6]
        # Five labels, all distinct.
        labels = [label for _, label in result]
        assert len(set(labels)) == 5

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError):
            build_integer_score_labels(5, 1)
