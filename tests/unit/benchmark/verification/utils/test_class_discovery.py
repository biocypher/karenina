"""Tests for find_answer_class utility."""

import pytest

from karenina.benchmark.verification.utils.class_discovery import find_answer_class
from karenina.schemas.entities.answer import BaseAnswer


@pytest.mark.unit
class TestFindAnswerClass:
    """Test find_answer_class() discovery logic."""

    def test_finds_single_subclass(self):
        class Answer(BaseAnswer):
            pass

        result = find_answer_class({"Answer": Answer, "BaseAnswer": BaseAnswer})
        assert result is Answer

    def test_finds_custom_named_class(self):
        class VenetoclaxAnswer(BaseAnswer):
            pass

        result = find_answer_class({"VenetoclaxAnswer": VenetoclaxAnswer})
        assert result is VenetoclaxAnswer

    def test_finds_leaf_in_hierarchy(self):
        class DrugBase(BaseAnswer):
            pass

        class Answer(DrugBase):
            pass

        ns = {"DrugBase": DrugBase, "Answer": Answer, "BaseAnswer": BaseAnswer}
        result = find_answer_class(ns)
        assert result is Answer

    def test_raises_on_no_subclass(self):
        with pytest.raises(ValueError, match="No BaseAnswer subclass found"):
            find_answer_class({"x": 42, "str": str})

    def test_raises_on_multiple_leaf_subclasses(self):
        class AnswerA(BaseAnswer):
            pass

        class AnswerB(BaseAnswer):
            pass

        with pytest.raises(ValueError, match="Multiple BaseAnswer subclasses"):
            find_answer_class({"AnswerA": AnswerA, "AnswerB": AnswerB})

    def test_ignores_base_answer_itself(self):
        with pytest.raises(ValueError, match="No BaseAnswer subclass found"):
            find_answer_class({"BaseAnswer": BaseAnswer})

    def test_ignores_non_class_entries(self):
        class Answer(BaseAnswer):
            pass

        result = find_answer_class(
            {
                "Answer": Answer,
                "some_func": lambda: None,
                "some_str": "hello",
                "some_int": 42,
            }
        )
        assert result is Answer
