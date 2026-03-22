"""Tests for issue 113: add_question() with explicit duplicate ID corrupts state.

When a user explicitly passes a question_id that already exists, the checkpoint
gets a duplicate entry while the cache silently overwrites. This causes counts
to diverge between checkpoint and cache. The fix is to raise ValueError before
appending when a duplicate explicit ID is detected.
"""

import pytest

from karenina import Benchmark


@pytest.mark.unit
class TestAddQuestionDuplicateId:
    """Tests for issue 113: duplicate explicit question ID detection."""

    def test_duplicate_explicit_id_raises_value_error(self) -> None:
        """add_question() should raise ValueError when an explicit duplicate ID is given."""
        benchmark = Benchmark.create(name="test-dup-id")
        benchmark.add_question("What is 2+2?", "4", question_id="q1")

        with pytest.raises(ValueError, match="already exists"):
            benchmark.add_question("What is 3+3?", "6", question_id="q1")

    def test_duplicate_explicit_id_does_not_corrupt_checkpoint(self) -> None:
        """After a rejected duplicate, checkpoint and cache should remain consistent."""
        benchmark = Benchmark.create(name="test-dup-state")
        benchmark.add_question("What is 2+2?", "4", question_id="q1")

        with pytest.raises(ValueError):
            benchmark.add_question("What is 3+3?", "6", question_id="q1")

        # Checkpoint should have exactly one entry
        assert len(benchmark._checkpoint.dataFeedElement) == 1
        # Cache should also have exactly one entry
        assert len(benchmark._questions_cache) == 1

    def test_auto_generated_ids_do_not_collide(self) -> None:
        """Auto-generated IDs should still auto-increment to avoid collisions."""
        benchmark = Benchmark.create(name="test-auto-id")
        id1 = benchmark.add_question("What is 2+2?", "4")
        id2 = benchmark.add_question("What is 2+2?", "4")

        # Same question text should get different IDs via auto-incrementing
        assert id1 != id2
        assert len(benchmark._checkpoint.dataFeedElement) == 2
        assert len(benchmark._questions_cache) == 2
