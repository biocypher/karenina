"""Tests for task_helpers utility functions."""

from dataclasses import dataclass

import pytest

from karenina.benchmark.verification.utils.task_helpers import (
    model_sort_key,
    replicate_range,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import FinishedTemplate, VerificationConfig


@dataclass
class _StubModel:
    """Lightweight stand-in for ModelConfig in sort key tests.

    Avoids ModelConfig validators that require manual_traces or non-None id
    for certain interface values.
    """

    id: str | None = None
    model_name: str | None = None


def _make_model(model_id: str, model_name: str | None = None) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        model_name=model_name or model_id,
        model_provider="anthropic",
        interface="langchain",
        system_prompt="test",
        temperature=0.1,
    )


def _make_template(question_id: str = "q1") -> FinishedTemplate:
    return FinishedTemplate(
        question_id=question_id,
        question_text="What is 2+2?",
        question_preview="What is 2+2?",
        template_code="class Answer(BaseAnswer): pass",
        last_modified="2026-01-01T00:00:00",
    )


# =============================================================================
# model_sort_key
# =============================================================================


@pytest.mark.unit
class TestModelSortKey:
    """Tests for the model_sort_key helper."""

    def test_prefers_id(self) -> None:
        """When both id and model_name are set, id wins."""
        model = _make_model("alpha", model_name="beta")
        assert model_sort_key(model) == "alpha"

    def test_falls_back_to_model_name(self) -> None:
        """When id is None, falls back to model_name."""
        model = _StubModel(id=None, model_name="fallback-name")
        assert model_sort_key(model) == "fallback-name"

    def test_empty_string_when_both_none(self) -> None:
        """When both id and model_name are None, returns empty string."""
        model = _StubModel(id=None, model_name=None)
        assert model_sort_key(model) == ""


# =============================================================================
# replicate_range
# =============================================================================


@pytest.mark.unit
class TestReplicateRange:
    """Tests for the replicate_range helper."""

    def test_single_returns_none_list(self) -> None:
        """count=1 returns [None] (no replicate numbering)."""
        assert replicate_range(1) == [None]

    def test_zero_returns_none_list(self) -> None:
        """count=0 returns [None] (degenerate case, same as single)."""
        assert replicate_range(0) == [None]

    def test_multiple_returns_one_indexed_range(self) -> None:
        """count=3 returns [1, 2, 3]."""
        assert replicate_range(3) == [1, 2, 3]


# =============================================================================
# Task ordering (integration test for batch_runner sort logic)
# =============================================================================


@pytest.mark.unit
class TestTaskOrdering:
    """Tests for task queue ordering in batch_runner."""

    def test_prefix_cache_groups_by_answering_model(self) -> None:
        """prefix_cache mode sorts by (answering_model, question, parsing_model)."""
        from karenina.benchmark.verification.batch_runner import generate_task_queue
        from karenina.benchmark.verification.utils.task_helpers import model_sort_key

        model_a = _make_model("model-a")
        model_b = _make_model("model-b")

        config = VerificationConfig(
            answering_models=[model_b, model_a],
            parsing_models=[model_a],
            task_ordering="prefix_cache",
        )

        templates = [_make_template("q2"), _make_template("q0"), _make_template("q1")]

        task_queue = generate_task_queue(templates, config)

        # Apply the same sort that batch_runner applies after generation
        task_queue.sort(
            key=lambda t: (
                model_sort_key(t["answering_model"]),
                t["question_id"],
                model_sort_key(t["parsing_model"]),
                t.get("replicate") or 0,
            )
        )

        answering_ids = [model_sort_key(t["answering_model"]) for t in task_queue]
        question_ids = [t["question_id"] for t in task_queue]

        # All model-a tasks come before all model-b tasks
        assert answering_ids == ["model-a", "model-a", "model-a", "model-b", "model-b", "model-b"]
        # Within each model group, questions are sorted
        assert question_ids == ["q0", "q1", "q2", "q0", "q1", "q2"]

    def test_generation_order_preserves_loop_order(self) -> None:
        """generation_order keeps the original expansion order from generate_task_queue."""
        from karenina.benchmark.verification.batch_runner import generate_task_queue

        model_a = _make_model("model-a")
        model_b = _make_model("model-b")

        config = VerificationConfig(
            answering_models=[model_b, model_a],
            parsing_models=[model_a],
            task_ordering="generation_order",
        )

        templates = [_make_template("q2"), _make_template("q0")]

        task_queue = generate_task_queue(templates, config)
        # generation_order is a no-op, so the loop order is: template x answering x parsing
        # q2/model-b, q2/model-a, q0/model-b, q0/model-a
        question_ids = [t["question_id"] for t in task_queue]
        answering_ids = [model_sort_key(t["answering_model"]) for t in task_queue]

        assert question_ids == ["q2", "q2", "q0", "q0"]
        assert answering_ids == ["model-b", "model-a", "model-b", "model-a"]
