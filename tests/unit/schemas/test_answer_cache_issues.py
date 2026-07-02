"""Tests for BaseAnswer caching bug fixes.

Covers:
- Issue 132: _compute_field_results() mutable cache never invalidated
"""

import pytest

from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.verified_field import VerifiedField
from karenina.schemas.primitives.comparisons import ExactMatch


@pytest.mark.unit
class TestFieldResultsCacheInvalidation:
    """Issue 132: _compute_field_results cache should be clearable."""

    def _make_answer_class(self) -> type[BaseAnswer]:
        """Create a simple VerifiedField-based answer class for testing."""

        class TestAnswer(BaseAnswer):
            name: str = VerifiedField(
                description="The name",
                ground_truth="alice",
                verify_with=ExactMatch(),
            )

        return TestAnswer

    def test_cache_populated_on_first_call(self) -> None:
        """First call to _compute_field_results should populate cache."""
        cls = self._make_answer_class()
        answer = cls(name="alice")
        result = answer._compute_field_results()
        assert result == {"name": True}
        # Cache should now be stored
        assert "_field_results" in answer.__dict__

    def test_cache_returns_same_on_second_call(self) -> None:
        """Second call returns cached result without recomputation."""
        cls = self._make_answer_class()
        answer = cls(name="alice")
        result1 = answer._compute_field_results()
        result2 = answer._compute_field_results()
        assert result1 is result2  # Same object (cached)

    def test_clear_verification_cache_exists(self) -> None:
        """_clear_verification_cache method should exist on BaseAnswer."""
        cls = self._make_answer_class()
        answer = cls(name="alice")
        assert hasattr(answer, "_clear_verification_cache")

    def test_clear_cache_allows_recomputation(self) -> None:
        """After clearing cache, _compute_field_results should recompute."""
        cls = self._make_answer_class()
        answer = cls(name="alice")

        # First call: cache populated, name matches
        result1 = answer._compute_field_results()
        assert result1 == {"name": True}

        # Mutate the field value
        answer.name = "bob"

        # Without clearing: stale cache returns old result
        stale = answer._compute_field_results()
        assert stale == {"name": True}  # Still cached

        # Clear cache
        answer._clear_verification_cache()

        # After clearing: recomputes with new value
        result2 = answer._compute_field_results()
        assert result2 == {"name": False}  # bob != alice

    def test_clear_cache_is_idempotent(self) -> None:
        """Clearing cache when no cache exists should not raise."""
        cls = self._make_answer_class()
        answer = cls(name="alice")
        # No cache yet; clearing should not raise
        answer._clear_verification_cache()
        # And computing should still work
        result = answer._compute_field_results()
        assert result == {"name": True}

    def test_clear_cache_invalidates_scores_and_ground_truths_too(self) -> None:
        """Clearing the results cache must also drop the companion caches.

        ``_compute_field_results`` populates three caches at once
        (``_field_results``, ``_field_scores``, ``_resolved_ground_truths``).
        If ``_clear_verification_cache`` only cleared one of them, a stale
        graded score or stale ground-truth dict could survive a field mutation
        and diverge from the recomputed binary results.
        """
        cls = self._make_answer_class()
        answer = cls(name="alice")

        # Populate all three caches.
        answer._compute_field_results()
        assert "_field_results" in answer.__dict__
        assert "_field_scores" in answer.__dict__
        assert "_resolved_ground_truths" in answer.__dict__

        answer._clear_verification_cache()
        assert "_field_results" not in answer.__dict__
        assert "_field_scores" not in answer.__dict__
        assert "_resolved_ground_truths" not in answer.__dict__

    def test_results_and_scores_stay_consistent_after_recompute(self) -> None:
        """After mutation + clear, binary results and graded scores agree.

        ``verify_granular`` reads ``_field_scores`` and ``verify`` reads
        ``_field_results``; the two maps are computed together so they must
        never disagree after a cache clear + recompute.
        """
        cls = self._make_answer_class()
        answer = cls(name="alice")
        assert answer.verify() is True
        assert answer.verify_granular() == 1.0

        answer.name = "bob"
        answer._clear_verification_cache()

        assert answer.verify() is False
        assert answer.verify_granular() == 0.0
        # The two caches were repopulated together by the same call.
        assert set(answer._compute_field_results()) == set(answer._compute_field_scores())
