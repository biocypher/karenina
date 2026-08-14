"""Tests for token counting and judge-input budget truncation."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.utils.token_budget import (
    count_deep_judgment_reasoning_tokens,
    count_tokens,
    truncate_to_token_budget,
)
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import LLMRubricTrait


@pytest.mark.unit
class TestTokenBudget:
    def test_count_tokens_is_positive_and_monotonic(self) -> None:
        short = count_tokens("hello world")
        long = count_tokens("hello world " * 100)
        assert 0 < short < long

    def test_deep_judgment_count_includes_rendered_prompts(self) -> None:
        trait = LLMRubricTrait(
            name="Grounded",
            description="Check evidence.",
            kind="boolean",
            min_score=None,
            max_score=None,
            classes=None,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=False,
            deep_judgment_max_excerpts=None,
            deep_judgment_fuzzy_match_threshold=None,
            deep_judgment_excerpt_retry_attempts=None,
            deep_judgment_search_enabled=False,
        )
        answer = "A short trace."
        assert count_deep_judgment_reasoning_tokens(answer, trait) > count_tokens(answer)

    def test_deep_judgment_count_uses_endpoint_tokenizer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class Response:
            def __enter__(self) -> Response:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"count": 123}'

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda _request, **_kwargs: Response(),
        )
        model = ModelConfig(
            id="judge",
            model_provider="openai",
            model_name="judge-model",
            interface="openai_endpoint",
            endpoint_base_url="http://localhost:8000/v1",
            endpoint_api_key="EMPTY",
        )
        trait = LLMRubricTrait(
            name="Grounded",
            description="Check evidence.",
            kind="boolean",
            min_score=None,
            max_score=None,
            classes=None,
            deep_judgment_enabled=True,
            deep_judgment_excerpt_enabled=False,
            deep_judgment_max_excerpts=None,
            deep_judgment_fuzzy_match_threshold=None,
            deep_judgment_excerpt_retry_attempts=None,
            deep_judgment_search_enabled=False,
        )
        assert (
            count_deep_judgment_reasoning_tokens(
                "trace",
                trait,
                parsing_model=model,
            )
            == 123
        )

    def test_under_budget_text_is_unchanged(self) -> None:
        text = "hello world"
        result, truncated = truncate_to_token_budget(text, max_tokens=1000)
        assert result == text
        assert truncated is False

    def test_over_budget_text_is_truncated_within_budget(self) -> None:
        text = "word " * 5000
        result, truncated = truncate_to_token_budget(text, max_tokens=100)
        assert truncated is True
        assert "[truncated:" in result
        assert count_tokens(result) <= 100
        assert result.startswith("word")
        assert result.endswith("word ")

    def test_nonpositive_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            truncate_to_token_budget("text", max_tokens=0)
