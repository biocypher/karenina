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


# =============================================================================
# stamp_agentic_trait_overrides
# =============================================================================


def _make_deep_agent_model(
    model_id: str,
    request_timeout: float | None = None,
    retry_policy=None,
) -> ModelConfig:
    """Build a ModelConfig with claude_agent_sdk interface (agent_tier=deep_agent)."""
    return ModelConfig(
        id=model_id,
        model_name=model_id,
        model_provider="anthropic",
        interface="claude_agent_sdk",
        request_timeout=request_timeout,
        retry_policy=retry_policy,
    )


def _make_pipeline_config(
    request_timeout: float = 600.0,
    timeout_max_attempts: int = 5,
) -> VerificationConfig:
    """Build a VerificationConfig with non-default timeout and retry policy."""
    from karenina.utils.retry_policy import (
        CategoryRetryConfig,
        RetryPolicy,
        TimeoutEscalationConfig,
    )

    retry_policy = RetryPolicy(
        timeout=CategoryRetryConfig(
            max_attempts=timeout_max_attempts,
            backoff_min=2.0,
            backoff_max=20.0,
        ),
        timeout_escalation=TimeoutEscalationConfig(
            strategy="additive",
            increment=120.0,
            max_timeout=1200.0,
        ),
    )
    return VerificationConfig(
        answering_models=[_make_model("base-ans")],
        parsing_models=[_make_model("base-parse")],
        request_timeout=request_timeout,
        retry_policy=retry_policy,
        evaluation_mode="template_and_rubric",
    )


@pytest.mark.unit
class TestStampAgenticTraitOverrides:
    """Tests for stamp_agentic_trait_overrides helper.

    Verifies the helper propagates pipeline-level retry_policy and
    request_timeout onto AgenticRubricTrait.model_override the same way
    the scenario per-node ModelOverride stamping does, while preserving
    explicit fields and avoiding rebuilding rubrics in the common case.
    """

    def test_returns_none_for_none_input(self) -> None:
        """None rubric short-circuits to None."""
        from karenina.benchmark.verification.utils.task_helpers import (
            stamp_agentic_trait_overrides,
        )

        config = _make_pipeline_config()
        assert stamp_agentic_trait_overrides(None, config) is None

    def test_returns_same_rubric_when_no_agentic_traits(self) -> None:
        """Rubrics without agentic traits are returned unchanged (identity)."""
        from karenina.benchmark.verification.utils.task_helpers import (
            stamp_agentic_trait_overrides,
        )
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        config = _make_pipeline_config()
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="quality",
                    description="A quality trait",
                    kind="boolean",
                    higher_is_better=True,
                )
            ],
        )
        result = stamp_agentic_trait_overrides(rubric, config)
        assert result is rubric

    def test_returns_same_rubric_when_no_overrides(self) -> None:
        """Agentic traits without model_override leave the rubric unchanged."""
        from karenina.benchmark.verification.utils.task_helpers import (
            stamp_agentic_trait_overrides,
        )
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric

        config = _make_pipeline_config()
        trait = AgenticRubricTrait(
            name="investigate",
            description="Run an investigation",
            kind="boolean",
            higher_is_better=True,
        )
        rubric = Rubric(agentic_traits=[trait])
        result = stamp_agentic_trait_overrides(rubric, config)
        assert result is rubric

    def test_stamps_unset_override_fields(self) -> None:
        """Both request_timeout and retry_policy are populated when None on the override."""
        from karenina.benchmark.verification.utils.task_helpers import (
            stamp_agentic_trait_overrides,
        )
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric

        config = _make_pipeline_config(request_timeout=600.0, timeout_max_attempts=5)
        override = _make_deep_agent_model("agent-1")
        trait = AgenticRubricTrait(
            name="investigate",
            description="Run an investigation",
            kind="boolean",
            higher_is_better=True,
            model_override=override,
        )
        rubric = Rubric(agentic_traits=[trait])

        result = stamp_agentic_trait_overrides(rubric, config)

        assert result is not rubric
        stamped_trait = result.agentic_traits[0]
        assert stamped_trait.model_override is not None
        assert stamped_trait.model_override.request_timeout == 600.0
        assert stamped_trait.model_override.retry_policy is not None
        assert stamped_trait.model_override.retry_policy.timeout.max_attempts == 5
        assert stamped_trait.model_override.retry_policy.timeout_escalation is not None
        assert stamped_trait.model_override.retry_policy.timeout_escalation.strategy == "additive"

        # Original rubric / trait / override are not mutated
        assert rubric.agentic_traits[0].model_override is override
        assert override.request_timeout is None
        assert override.retry_policy is None

    def test_preserves_explicit_override_fields(self) -> None:
        """Existing request_timeout and retry_policy on the override are preserved."""
        from karenina.benchmark.verification.utils.task_helpers import (
            stamp_agentic_trait_overrides,
        )
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric
        from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy

        config = _make_pipeline_config(request_timeout=600.0, timeout_max_attempts=5)
        explicit_policy = RetryPolicy(
            timeout=CategoryRetryConfig(max_attempts=9, backoff_min=1.0, backoff_max=2.0),
        )
        override = _make_deep_agent_model(
            "agent-1",
            request_timeout=42.0,
            retry_policy=explicit_policy,
        )
        trait = AgenticRubricTrait(
            name="investigate",
            description="Run an investigation",
            kind="boolean",
            higher_is_better=True,
            model_override=override,
        )
        rubric = Rubric(agentic_traits=[trait])

        result = stamp_agentic_trait_overrides(rubric, config)
        # No stamping needed; helper returns the original instance
        assert result is rubric
        stamped_trait = result.agentic_traits[0]
        assert stamped_trait.model_override is override
        assert stamped_trait.model_override.request_timeout == 42.0
        assert stamped_trait.model_override.retry_policy is explicit_policy
        assert stamped_trait.model_override.retry_policy.timeout.max_attempts == 9

    def test_partial_stamp_only_unset_fields(self) -> None:
        """Mixed override (one field set, one unset) only fills the unset field."""
        from karenina.benchmark.verification.utils.task_helpers import (
            stamp_agentic_trait_overrides,
        )
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric

        config = _make_pipeline_config(request_timeout=600.0, timeout_max_attempts=5)
        override = _make_deep_agent_model("agent-1", request_timeout=42.0)
        trait = AgenticRubricTrait(
            name="investigate",
            description="Run an investigation",
            kind="boolean",
            higher_is_better=True,
            model_override=override,
        )
        rubric = Rubric(agentic_traits=[trait])

        result = stamp_agentic_trait_overrides(rubric, config)
        assert result is not rubric
        stamped_override = result.agentic_traits[0].model_override
        assert stamped_override is not None
        # Explicit timeout preserved
        assert stamped_override.request_timeout == 42.0
        # Retry policy stamped from config
        assert stamped_override.retry_policy is not None
        assert stamped_override.retry_policy.timeout.max_attempts == 5

    def test_only_traits_with_overrides_are_rebuilt(self) -> None:
        """Traits without overrides are not re-instantiated even when stamping occurs."""
        from karenina.benchmark.verification.utils.task_helpers import (
            stamp_agentic_trait_overrides,
        )
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric

        config = _make_pipeline_config()
        plain_trait = AgenticRubricTrait(
            name="plain",
            description="No override",
            kind="boolean",
            higher_is_better=True,
        )
        override = _make_deep_agent_model("agent-1")
        override_trait = AgenticRubricTrait(
            name="overridden",
            description="With override",
            kind="boolean",
            higher_is_better=True,
            model_override=override,
        )
        rubric = Rubric(agentic_traits=[plain_trait, override_trait])

        result = stamp_agentic_trait_overrides(rubric, config)
        assert result is not rubric
        # plain_trait passed through by identity (slot 0)
        assert result.agentic_traits[0] is plain_trait
        # overridden trait was rebuilt (slot 1)
        assert result.agentic_traits[1] is not override_trait
        new_override = result.agentic_traits[1].model_override
        assert new_override is not None
        assert new_override.request_timeout == 600.0
        assert new_override.retry_policy is not None

    def test_merge_rubrics_for_task_applies_stamping(self) -> None:
        """Integration: merge_rubrics_for_task pipes the merged rubric through stamping."""
        from karenina.benchmark.verification.utils.task_helpers import (
            merge_rubrics_for_task,
        )
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric

        config = _make_pipeline_config(request_timeout=600.0, timeout_max_attempts=5)
        override = _make_deep_agent_model("agent-1")
        trait = AgenticRubricTrait(
            name="investigate",
            description="Run an investigation",
            kind="boolean",
            higher_is_better=True,
            model_override=override,
        )
        global_rubric = Rubric(agentic_traits=[trait])
        template = _make_template()

        merged, _ = merge_rubrics_for_task(global_rubric, template, config)
        assert merged is not None
        merged_override = merged.agentic_traits[0].model_override
        assert merged_override is not None
        assert merged_override.request_timeout == 600.0
        assert merged_override.retry_policy is not None
        assert merged_override.retry_policy.timeout.max_attempts == 5

        # Original global rubric / trait / override are not mutated
        assert global_rubric.agentic_traits[0].model_override is override
        assert override.request_timeout is None
        assert override.retry_policy is None
