"""Tests for dynamic rubric fields on VerificationResultRubric."""

import pytest

from karenina.schemas.verification.result_components import VerificationResultRubric


@pytest.mark.unit
class TestDynamicRubricResultFields:
    """Verify the dynamic_rubric_skipped_traits and dynamic_rubric_promoted_traits fields."""

    def test_fields_default_to_none(self) -> None:
        """Both dynamic rubric fields should default to None when not provided."""
        result = VerificationResultRubric()

        assert result.dynamic_rubric_skipped_traits is None
        assert result.dynamic_rubric_promoted_traits is None

    def test_fields_can_be_set_with_values(self) -> None:
        """Both dynamic rubric fields should accept their expected types."""
        skipped = {"safety_tone": "concept not present in response"}
        promoted = ["clarity", "citation_format"]

        result = VerificationResultRubric(
            dynamic_rubric_skipped_traits=skipped,
            dynamic_rubric_promoted_traits=promoted,
        )

        assert result.dynamic_rubric_skipped_traits == skipped
        assert result.dynamic_rubric_promoted_traits == promoted

    def test_model_dump_includes_fields(self) -> None:
        """model_dump output should contain the dynamic rubric fields."""
        skipped = {"hedging": "no hedging language detected"}
        promoted = ["specificity"]

        result = VerificationResultRubric(
            rubric_evaluation_performed=True,
            dynamic_rubric_skipped_traits=skipped,
            dynamic_rubric_promoted_traits=promoted,
        )
        dumped = result.model_dump()

        assert "dynamic_rubric_skipped_traits" in dumped
        assert "dynamic_rubric_promoted_traits" in dumped
        assert dumped["dynamic_rubric_skipped_traits"] == skipped
        assert dumped["dynamic_rubric_promoted_traits"] == promoted

    def test_model_dump_includes_fields_when_none(self) -> None:
        """model_dump should include the fields even when they are None."""
        result = VerificationResultRubric()
        dumped = result.model_dump()

        assert "dynamic_rubric_skipped_traits" in dumped
        assert "dynamic_rubric_promoted_traits" in dumped
        assert dumped["dynamic_rubric_skipped_traits"] is None
        assert dumped["dynamic_rubric_promoted_traits"] is None

    def test_empty_dict_and_list_accepted(self) -> None:
        """Empty dict and empty list should be accepted without error."""
        result = VerificationResultRubric(
            dynamic_rubric_skipped_traits={},
            dynamic_rubric_promoted_traits=[],
        )

        assert result.dynamic_rubric_skipped_traits == {}
        assert result.dynamic_rubric_promoted_traits == []
