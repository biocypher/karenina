"""Tests for runtime callable registry overrides."""

from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.evaluators.rubric.evaluator import RubricEvaluator
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import CallableRubricTrait, Rubric


def _make_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    callable_registry: dict | None = None,
) -> RubricEvaluator:
    """Create a rubric evaluator without initializing a real LLM adapter."""
    monkeypatch.setattr(
        "karenina.benchmark.verification.evaluators.rubric.evaluator.get_llm",
        lambda _config: MagicMock(),
    )
    return RubricEvaluator(
        ModelConfig(id="test", model_name="test-model"),
        callable_registry=callable_registry,
    )


@pytest.mark.unit
class TestCallableRegistry:
    """Verify registered callables override embedded trait code by name."""

    def test_matching_callable_overrides_without_deserializing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trait = CallableRubricTrait.from_callable(
            name="short",
            func=lambda _text: False,
            kind="boolean",
        )
        evaluator = _make_evaluator(monkeypatch, {"short": lambda _text: True})

        with patch.object(
            CallableRubricTrait,
            "deserialize_callable",
            side_effect=AssertionError("embedded callable must not be deserialized"),
        ):
            results, _, _ = evaluator.evaluate_rubric(
                question="",
                answer="answer",
                rubric=Rubric(callable_traits=[trait]),
            )

        assert results == {"short": True}

    def test_unmatched_callable_falls_back_to_embedded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trait = CallableRubricTrait.from_callable(
            name="embedded",
            func=lambda text: text == "answer",
            kind="boolean",
        )
        evaluator = _make_evaluator(monkeypatch, {"other": lambda _text: False})

        with pytest.warns(UserWarning, match="Deserializing callable"):
            results, _, _ = evaluator.evaluate_rubric(
                question="",
                answer="answer",
                rubric=Rubric(callable_traits=[trait]),
            )

        assert results == {"embedded": True}

    @pytest.mark.parametrize(
        ("trait", "override", "expected"),
        [
            (
                CallableRubricTrait.from_callable(
                    name="inverted",
                    func=lambda _text: False,
                    kind="boolean",
                    invert_result=True,
                ),
                lambda _text: True,
                False,
            ),
            (
                CallableRubricTrait.from_callable(
                    name="score",
                    func=lambda _text: 1,
                    kind="score",
                    min_score=0,
                    max_score=10,
                ),
                lambda _text: 7.5,
                7.5,
            ),
            (
                CallableRubricTrait.from_callable(
                    name="tone",
                    func=lambda _text: "formal",
                    kind="literal",
                    classes={"formal": "Formal tone", "casual": "Casual tone"},
                ),
                lambda _text: "casual",
                1,
            ),
        ],
    )
    def test_override_uses_existing_trait_result_handling(
        self,
        trait: CallableRubricTrait,
        override,
        expected: bool | int | float,
    ) -> None:
        assert trait.evaluate("answer", callable_override=override) == expected

    def test_invalid_override_result_uses_existing_validation(self) -> None:
        trait = CallableRubricTrait.from_callable(
            name="bounded",
            func=lambda _text: 1,
            kind="score",
            min_score=0,
            max_score=5,
        )

        with pytest.raises(RuntimeError, match="above maximum"):
            trait.evaluate("answer", callable_override=lambda _text: 6)
