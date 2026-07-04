"""Tests for Benchmark facade DynamicRubric persistence through checkpoint.

Verifies that set_global_dynamic_rubric() on the Benchmark facade persists
the DynamicRubric into the checkpoint so it survives save/load cycles,
and that clearing (passing None) removes it from the checkpoint.
"""

import pytest

from karenina import Benchmark
from karenina.schemas.entities import LLMRubricTrait
from karenina.schemas.entities.rubric import DynamicRubric


@pytest.mark.unit
class TestDynamicRubricFacadePersistence:
    """Tests for Benchmark.set_global_dynamic_rubric() checkpoint persistence."""

    def _make_dynamic_rubric(self) -> DynamicRubric:
        """Create a simple DynamicRubric with one LLM trait for testing."""
        return DynamicRubric(
            llm_traits=[
                LLMRubricTrait(
                    name="safety",
                    description="Checks response safety.",
                    summary="Response safety assessment",
                    kind="boolean",
                ),
            ],
        )

    def test_set_persists_to_checkpoint(self) -> None:
        """Setting a DynamicRubric persists traits into the checkpoint rating array."""
        benchmark = Benchmark.create(name="persist-test")
        rubric = self._make_dynamic_rubric()

        benchmark.set_global_dynamic_rubric(rubric)

        # Verify checkpoint contains the serialized dynamic rubric traits
        ratings = benchmark._base._checkpoint.rating or []
        dynamic_ratings = [r for r in ratings if r.additionalType == "karenina:GlobalDynamicRubricTrait"]
        assert len(dynamic_ratings) > 0, "Expected dynamic rubric traits in checkpoint ratings after set"

    def test_clear_removes_from_checkpoint(self) -> None:
        """Clearing the DynamicRubric (passing None) removes traits from checkpoint."""
        benchmark = Benchmark.create(name="clear-test")
        rubric = self._make_dynamic_rubric()

        # Set first, so there is something in the checkpoint
        benchmark.set_global_dynamic_rubric(rubric)
        # Confirm it was persisted
        ratings = benchmark._base._checkpoint.rating or []
        assert any(r.additionalType == "karenina:GlobalDynamicRubricTrait" for r in ratings), (
            "Precondition: rubric should be in checkpoint before clearing"
        )

        # Now clear
        benchmark.set_global_dynamic_rubric(None)

        # Verify checkpoint no longer contains dynamic rubric traits
        ratings_after = benchmark._base._checkpoint.rating or []
        dynamic_after = [r for r in ratings_after if r.additionalType == "karenina:GlobalDynamicRubricTrait"]
        assert len(dynamic_after) == 0, "Expected no dynamic rubric traits in checkpoint after clearing"

    def test_retrieve_after_set_returns_correct_rubric(self) -> None:
        """get_global_dynamic_rubric() returns the rubric set via the facade."""
        benchmark = Benchmark.create(name="retrieve-test")
        rubric = self._make_dynamic_rubric()

        benchmark.set_global_dynamic_rubric(rubric)

        retrieved = benchmark.get_global_dynamic_rubric()
        assert retrieved is not None, "Expected non-None rubric after set"
        assert len(retrieved.llm_traits) == 1
        assert retrieved.llm_traits[0].name == "safety"
