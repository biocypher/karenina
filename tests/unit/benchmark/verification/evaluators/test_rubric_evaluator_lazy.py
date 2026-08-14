"""Tests for lazy model adapter initialization in rubric evaluation."""

from __future__ import annotations

from typing import Any

import pytest

from karenina.benchmark.verification.evaluators.rubric.evaluator import RubricEvaluator
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import RegexRubricTrait, Rubric


def _model_config() -> ModelConfig:
    """Build a parsing model configuration without network access."""
    return ModelConfig(
        id="judge",
        model_provider="openai",
        model_name="judge-model",
        interface="openai_endpoint",
        endpoint_base_url="http://localhost:9999",
        endpoint_api_key="test-key",
    )


@pytest.mark.unit
class TestLazyModelInitialization:
    """Verify deterministic rubrics do not allocate model adapters."""

    def test_regex_only_evaluation_does_not_initialize_adapter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_get_llm(_config: ModelConfig) -> Any:
            raise AssertionError("regex-only evaluation must not initialize an adapter")

        monkeypatch.setattr(
            "karenina.benchmark.verification.evaluators.rubric.evaluator.get_llm",
            fail_get_llm,
        )
        evaluator = RubricEvaluator(_model_config())
        rubric = Rubric(
            regex_traits=[
                RegexRubricTrait(
                    name="mentions_tool",
                    description="Whether the trace contains a tool message.",
                    pattern=r"Tool Message",
                )
            ]
        )

        scores, labels, usage = evaluator.evaluate_rubric(
            question="",
            answer="--- Tool Message (call_id: 1) ---",
            rubric=rubric,
        )
        evaluator.close()

        assert scores == {"mentions_tool": True}
        assert labels is None
        assert usage == []

    def test_model_backed_access_initializes_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = object()
        calls = 0
        closed: list[object] = []

        def fake_get_llm(_config: ModelConfig) -> Any:
            nonlocal calls
            calls += 1
            return adapter

        monkeypatch.setattr(
            "karenina.benchmark.verification.evaluators.rubric.evaluator.get_llm",
            fake_get_llm,
        )
        monkeypatch.setattr(
            "karenina.benchmark.verification.evaluators.rubric.evaluator.close_adapter",
            closed.append,
        )
        evaluator = RubricEvaluator(_model_config())

        assert evaluator.llm is adapter
        assert evaluator.llm is adapter
        evaluator.close()

        assert calls == 1
        assert closed == [adapter]
