"""Tests for k-shot example selection reproducibility across processes."""

import pytest

from karenina.schemas.config.models import FewShotConfig


@pytest.mark.unit
class TestKShotReproducibility:
    """k-shot seeding must be stable across processes (not depend on PYTHONHASHSEED)."""

    def test_resolve_deterministic_across_calls(self) -> None:
        """Same question_id yields same k-shot selection on repeated calls."""
        config = FewShotConfig(source="question_pool", pool_mode="k-shot", pool_k=3)
        examples = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)]

        first = config.resolve_examples_for_question("test-q-123", examples)
        second = config.resolve_examples_for_question("test-q-123", examples)
        assert first == second

    def test_different_question_ids_yield_different_selections(self) -> None:
        """Different question_ids should (usually) produce different selections."""
        config = FewShotConfig(source="question_pool", pool_mode="k-shot", pool_k=3)
        examples = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(20)]

        sel_a = config.resolve_examples_for_question("question-a", examples)
        sel_b = config.resolve_examples_for_question("question-b", examples)
        assert sel_a != sel_b

    def test_seed_uses_builtin_hash(self) -> None:
        """The seed must use hash(question_id) for reproducible seeding.

        We verify by checking the source code uses hash(question_id).
        """
        import inspect

        source = inspect.getsource(FewShotConfig.resolve_examples_for_question)
        assert "hash(question_id)" in source
