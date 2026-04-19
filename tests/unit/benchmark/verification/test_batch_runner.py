"""Tests for batch_runner task queue generation and parsing_only guard."""

import pytest

from karenina.benchmark.verification.batch_runner import (
    _apply_retry_config,
    generate_task_queue,
    run_verification_batch,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import FinishedTemplate, VerificationConfig


def _make_model(model_id: str) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        model_name=model_id,
        model_provider="anthropic",
        interface="langchain",
        system_prompt="test",
        temperature=0.1,
    )


def _make_template(question_id: str = "q1", raw_answer: str | None = None) -> FinishedTemplate:
    return FinishedTemplate(
        question_id=question_id,
        question_text="What is 2+2?",
        question_preview="What is 2+2?",
        template_code="class Answer(BaseAnswer): pass",
        raw_answer=raw_answer,
        last_modified="2026-01-01T00:00:00",
    )


HAIKU = _make_model("haiku")
SONNET = _make_model("sonnet")


# =============================================================================
# generate_task_queue: normal mode
# =============================================================================


@pytest.mark.unit
class TestTaskQueueGeneration:
    """Verify normal task queue expansion."""

    def test_cross_product_of_models(self) -> None:
        """Normal mode produces answering x parsing cross product."""
        config = VerificationConfig(
            parsing_models=[HAIKU, SONNET],
            answering_models=[HAIKU, SONNET],
        )
        templates = [_make_template()]

        tasks = generate_task_queue(templates, config)

        assert len(tasks) == 4  # 2 answering x 2 parsing

    def test_replicates_expand(self) -> None:
        """Replicates multiply the task count."""
        config = VerificationConfig(
            parsing_models=[HAIKU],
            answering_models=[SONNET],
            replicate_count=3,
        )
        templates = [_make_template()]

        tasks = generate_task_queue(templates, config)

        assert len(tasks) == 3


# =============================================================================
# parsing_only guard
# =============================================================================


@pytest.mark.unit
class TestParsingOnlyGuard:
    """Verify that parsing_only=True is rejected in the batch verification path."""

    def test_parsing_only_rejected_in_batch_runner(self) -> None:
        """run_verification_batch should raise ValueError for parsing_only=True."""
        config = VerificationConfig(
            parsing_models=[HAIKU],
            answering_models=[],
            parsing_only=True,
        )
        templates = [_make_template(raw_answer="The answer is 4.")]

        with pytest.raises(ValueError, match="parsing_only=True is not supported"):
            run_verification_batch(templates=templates, config=config)

    def test_parsing_only_produces_zero_tasks_in_queue(self) -> None:
        """generate_task_queue with empty answering_models produces 0 tasks."""
        config = VerificationConfig(
            parsing_models=[HAIKU],
            answering_models=[],
            parsing_only=True,
        )
        templates = [_make_template(raw_answer="The answer is 4.")]

        tasks = generate_task_queue(templates, config)

        # This confirms the underlying issue: the loop over answering_models
        # produces nothing. The guard in run_verification_batch prevents
        # users from hitting this silently.
        assert len(tasks) == 0

    def test_normal_mode_not_rejected(self) -> None:
        """Normal mode (parsing_only=False) should not raise."""
        config = VerificationConfig(
            parsing_models=[HAIKU],
            answering_models=[SONNET],
        )
        templates = [_make_template()]

        # Should not raise (we don't run the full pipeline, just check no guard fires)
        tasks = generate_task_queue(templates, config)
        assert len(tasks) == 1


# =============================================================================
# _apply_retry_config
# =============================================================================


@pytest.mark.unit
class TestApplyRetryConfig:
    """Tests for _apply_retry_config function."""

    def test_stamps_when_model_has_no_value(self) -> None:
        """Test that pipeline retry_policy is stamped when model has None."""
        from karenina.utils.retry_policy import RetryPolicy

        model = ModelConfig(id="test", model_name="test", model_provider="openai")
        policy = RetryPolicy()
        result = _apply_retry_config(model, retry_policy=policy)
        assert result.retry_policy is not None

    def test_preserves_existing_model_value(self) -> None:
        """Test that model-level retry_policy is not overwritten."""
        from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy

        model_policy = RetryPolicy(connection=CategoryRetryConfig(max_attempts=1))
        model = ModelConfig(
            id="test",
            model_name="test",
            model_provider="openai",
            retry_policy=model_policy,
        )
        pipeline_policy = RetryPolicy(connection=CategoryRetryConfig(max_attempts=99))
        result = _apply_retry_config(model, retry_policy=pipeline_policy)
        assert result.retry_policy.connection.max_attempts == 1

    def test_noop_when_pipeline_value_is_none(self) -> None:
        """Test that None pipeline value does not stamp."""
        model = ModelConfig(id="test", model_name="test", model_provider="openai")
        result = _apply_retry_config(model, retry_policy=None)
        assert result.retry_policy is None
        assert result is model  # Same object, no copy


# =============================================================================
# task_ordering resolution and dispatch
# =============================================================================


@pytest.mark.unit
class TestResolveTaskOrdering:
    """Tests for _resolve_task_ordering (auto resolution + passthrough)."""

    def _make_config(self, *, ordering: str, answerers: list[ModelConfig]) -> VerificationConfig:
        return VerificationConfig(
            answering_models=answerers,
            parsing_models=[HAIKU],
            task_ordering=ordering,
        )

    def test_auto_picks_prefix_cache_for_single_answerer(self) -> None:
        from karenina.benchmark.verification.batch_runner import _resolve_task_ordering

        config = self._make_config(ordering="auto", answerers=[HAIKU])
        assert _resolve_task_ordering(config) == "prefix_cache"

    def test_auto_picks_distribute_answerers_for_multiple_answerers(self) -> None:
        from karenina.benchmark.verification.batch_runner import _resolve_task_ordering

        config = self._make_config(ordering="auto", answerers=[HAIKU, SONNET])
        assert _resolve_task_ordering(config) == "distribute_answerers"

    def test_auto_picks_prefix_cache_when_duplicated_answerers_share_identity(self) -> None:
        """Duplicate ModelConfigs with the same canonical_key count as one group."""
        from karenina.benchmark.verification.batch_runner import _resolve_task_ordering

        dup = _make_model("haiku")
        config = self._make_config(ordering="auto", answerers=[HAIKU, dup])
        assert _resolve_task_ordering(config) == "prefix_cache"

    def test_passthrough_for_pinned_values(self) -> None:
        from karenina.benchmark.verification.batch_runner import _resolve_task_ordering

        for pinned in ("prefix_cache", "distribute_answerers", "generation_order", "random"):
            config = self._make_config(ordering=pinned, answerers=[HAIKU, SONNET])
            assert _resolve_task_ordering(config) == pinned


@pytest.mark.unit
class TestSortDispatch:
    """Tests for the dispatch that applies the resolved ordering to the queue."""

    def _tasks(self, answerers: list[str], questions: list[str]) -> list[dict]:
        out: list[dict] = []
        for ans in answerers:
            for q in questions:
                out.append(
                    {
                        "answering_model": _make_model(ans),
                        "parsing_model": HAIKU,
                        "question_id": q,
                        "replicate": 1,
                    }
                )
        return out

    def test_distribute_answerers_produces_round_robin_head(self) -> None:
        from karenina.benchmark.verification.batch_runner import _apply_task_ordering

        tasks = self._tasks(["a", "b", "c"], ["q1", "q2", "q3"])
        ordered = _apply_task_ordering(tasks, "distribute_answerers")

        head = [t["answering_model"].id for t in ordered[:3]]
        assert set(head) == {"a", "b", "c"}
        assert len(ordered) == len(tasks)

    def test_prefix_cache_groups_by_answerer(self) -> None:
        from karenina.benchmark.verification.batch_runner import _apply_task_ordering

        tasks = self._tasks(["b", "a"], ["q2", "q1"])
        ordered = _apply_task_ordering(tasks, "prefix_cache")

        answerer_run = [t["answering_model"].id for t in ordered]
        assert answerer_run == ["a", "a", "b", "b"]

    def test_generation_order_is_passthrough(self) -> None:
        from karenina.benchmark.verification.batch_runner import _apply_task_ordering

        tasks = self._tasks(["a", "b"], ["q1", "q2"])
        assert _apply_task_ordering(list(tasks), "generation_order") == tasks

    def test_random_preserves_task_count(self) -> None:
        from karenina.benchmark.verification.batch_runner import _apply_task_ordering

        tasks = self._tasks(["a", "b"], ["q1", "q2", "q3"])
        ordered = _apply_task_ordering(list(tasks), "random")
        assert len(ordered) == len(tasks)
        ids = sorted((t["answering_model"].id, t["question_id"]) for t in ordered)
        expected = sorted((t["answering_model"].id, t["question_id"]) for t in tasks)
        assert ids == expected


# =============================================================================
# _normalize_answerer_limits
# =============================================================================


@pytest.mark.unit
class TestNormalizeAnswererLimits:
    """Tests for _normalize_answerer_limits (int broadcast, dict passthrough, warnings)."""

    def test_none_returns_none(self) -> None:
        from karenina.benchmark.verification.batch_runner import _normalize_answerer_limits

        assert _normalize_answerer_limits(None, [HAIKU, SONNET]) is None

    def test_int_broadcast(self) -> None:
        from karenina.benchmark.verification.batch_runner import _normalize_answerer_limits

        assert _normalize_answerer_limits(16, [HAIKU, SONNET]) == {
            "haiku": 16,
            "sonnet": 16,
        }

    def test_int_broadcast_skips_models_without_id(self) -> None:
        from karenina.benchmark.verification.batch_runner import _normalize_answerer_limits

        class _NoId:
            id = None
            model_name = "x"

        assert _normalize_answerer_limits(5, [HAIKU, _NoId()]) == {"haiku": 5}

    def test_dict_passthrough(self) -> None:
        from karenina.benchmark.verification.batch_runner import _normalize_answerer_limits

        out = _normalize_answerer_limits({"haiku": 10, "sonnet": 20}, [HAIKU, SONNET])
        assert out == {"haiku": 10, "sonnet": 20}

    def test_dict_with_unknown_id_warns_and_returns_dict(self, caplog) -> None:
        from karenina.benchmark.verification.batch_runner import _normalize_answerer_limits

        with caplog.at_level("WARNING"):
            out = _normalize_answerer_limits({"ghost": 5}, [HAIKU])

        assert out == {"ghost": 5}
        assert any("ghost" in rec.message for rec in caplog.records)
