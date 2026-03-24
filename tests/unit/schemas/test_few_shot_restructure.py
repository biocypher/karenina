"""Tests for FewShotConfig restructure with source/pool_mode/pool_k semantics."""

import pytest
from pydantic import ValidationError

from karenina.schemas.config.models import FewShotConfig, QuestionFewShotConfig


@pytest.mark.unit
class TestFewShotConfigNewFields:
    def test_default_source_is_both(self) -> None:
        assert FewShotConfig().source == "both"

    def test_default_pool_mode_is_all(self) -> None:
        assert FewShotConfig().pool_mode == "all"

    def test_default_pool_k_is_3(self) -> None:
        assert FewShotConfig().pool_k == 3

    def test_old_field_names_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FewShotConfig(**{"enabled": True})
        with pytest.raises(ValidationError):
            FewShotConfig(**{"global_mode": "all"})
        with pytest.raises(ValidationError):
            FewShotConfig(**{"global_k": 5})
        with pytest.raises(ValidationError):
            FewShotConfig(**{"global_external_examples": []})

    def test_question_config_no_external_examples(self) -> None:
        with pytest.raises(ValidationError):
            QuestionFewShotConfig(**{"external_examples": []})

    def test_question_config_no_mode_none(self) -> None:
        with pytest.raises(ValidationError):
            QuestionFewShotConfig(**{"mode": "none"})


@pytest.mark.unit
class TestFewShotSourceBehavior:
    def _pool_examples(self) -> list[dict[str, str]]:
        return [
            {"question": "What is X?", "answer": "X is A."},
            {"question": "What is Y?", "answer": "Y is B."},
        ]

    def test_disabled_returns_empty(self) -> None:
        assert FewShotConfig(source="disabled").resolve_examples_for_question("q1", self._pool_examples()) == []

    def test_question_pool_returns_pool_only(self) -> None:
        config = FewShotConfig(
            source="question_pool",
            pool_mode="all",
            global_examples=[{"question": "Global?", "answer": "Global."}],
        )
        result = config.resolve_examples_for_question("q1", self._pool_examples())
        assert len(result) == 2
        assert all(ex in self._pool_examples() for ex in result)

    def test_global_returns_global_only(self) -> None:
        global_ex = [{"question": "Global?", "answer": "Global."}]
        result = FewShotConfig(source="global", global_examples=global_ex).resolve_examples_for_question(
            "q1", self._pool_examples()
        )
        assert result == global_ex

    def test_both_returns_pool_and_global(self) -> None:
        global_ex = [{"question": "Global?", "answer": "Global."}]
        result = FewShotConfig(source="both", pool_mode="all", global_examples=global_ex).resolve_examples_for_question(
            "q1", self._pool_examples()
        )
        assert len(result) == 3
        assert result[:2] == self._pool_examples()
        assert result[2:] == global_ex

    def test_pool_mode_kshot(self) -> None:
        result = FewShotConfig(source="question_pool", pool_mode="k-shot", pool_k=1).resolve_examples_for_question(
            "q1", self._pool_examples()
        )
        assert len(result) == 1

    def test_pool_mode_custom(self) -> None:
        config = FewShotConfig(
            source="question_pool",
            pool_mode="custom",
            question_configs={"q1": QuestionFewShotConfig(mode="custom", selected_examples=[0])},
        )
        assert len(config.resolve_examples_for_question("q1", self._pool_examples())) == 1

    def test_inherit_uses_pool_mode(self) -> None:
        result = FewShotConfig(source="question_pool", pool_mode="k-shot", pool_k=1).resolve_examples_for_question(
            "q1", self._pool_examples()
        )
        assert len(result) == 1


@pytest.mark.unit
class TestFewShotFactoryMethods:
    def test_from_index_selections(self) -> None:
        config = FewShotConfig.from_index_selections({"q1": [0, 1]})
        assert config.pool_mode == "custom"

    def test_from_hash_selections(self) -> None:
        assert FewShotConfig.from_hash_selections({"q1": ["abc123"]}).pool_mode == "custom"

    def test_k_shot_for_questions(self) -> None:
        config = FewShotConfig.k_shot_for_questions({"q1": 5}, pool_k=3)
        assert config.pool_mode == "k-shot"
        assert config.pool_k == 3
