"""Tests for RubricManager API improvements."""

import pytest

from karenina.benchmark import Benchmark
from karenina.schemas.entities.rubric import (
    LLMRubricTrait,
    RegexRubricTrait,
    Rubric,
)


def _create_benchmark():
    """Create a fresh Benchmark for testing."""
    return Benchmark.create(name="test_rubric_api")


def _create_benchmark_with_question():
    """Create a Benchmark with one question."""
    b = Benchmark.create(name="test_rubric_api")
    b.add_question(question="What is 2+2?", raw_answer="4", question_id="q1")
    return b, "q1"


@pytest.mark.unit
class TestRubricManagerSetGlobalRubric:
    """Test set_global_rubric accepts both Rubric and list."""

    def test_accepts_rubric_object(self):
        """set_global_rubric should accept a Rubric object."""
        b = _create_benchmark()
        rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="t1", description="d1", kind="boolean")],
            regex_traits=[RegexRubricTrait(name="t2", description="d2", pattern=r"\d+")],
        )
        b._rubric_manager.set_global_rubric(rubric)
        result = b._rubric_manager.get_global_rubric()
        assert result is not None
        assert len(result.llm_traits) == 1
        assert len(result.regex_traits) == 1

    def test_accepts_trait_list(self):
        """set_global_rubric should accept a flat list of traits."""
        b = _create_benchmark()
        traits = [
            LLMRubricTrait(name="t1", description="d1", kind="boolean"),
            RegexRubricTrait(name="t2", description="d2", pattern=r"\d+"),
        ]
        b._rubric_manager.set_global_rubric(traits)
        result = b._rubric_manager.get_global_rubric()
        assert result is not None
        assert len(result.llm_traits) == 1
        assert len(result.regex_traits) == 1


@pytest.mark.unit
class TestRubricManagerSetQuestionRubric:
    """Test set_question_rubric on RubricManager."""

    def test_set_question_rubric_with_rubric(self):
        """set_question_rubric should accept a Rubric object."""
        b, q_id = _create_benchmark_with_question()
        rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="t1", description="d1", kind="boolean")],
        )
        b._rubric_manager.set_question_rubric(q_id, rubric)
        raw = b._rubric_manager.get_question_rubric(q_id)
        assert raw is not None
        # get_question_rubric returns dict or list from cache
        if isinstance(raw, dict):
            all_traits = []
            for v in raw.values():
                if isinstance(v, list):
                    all_traits.extend(v)
            assert len(all_traits) == 1
        else:
            assert len(raw) == 1

    def test_set_question_rubric_replaces_existing(self):
        """set_question_rubric should clear existing then set new."""
        b, q_id = _create_benchmark_with_question()
        rubric1 = Rubric(
            llm_traits=[LLMRubricTrait(name="t1", description="d1", kind="boolean")],
        )
        rubric2 = Rubric(
            regex_traits=[RegexRubricTrait(name="t2", description="d2", pattern=r"\d+")],
        )
        b._rubric_manager.set_question_rubric(q_id, rubric1)
        b._rubric_manager.set_question_rubric(q_id, rubric2)
        raw = b._rubric_manager.get_question_rubric(q_id)
        # Collect all traits from whatever format is returned
        if isinstance(raw, dict):
            all_traits = []
            for v in raw.values():
                if isinstance(v, list):
                    all_traits.extend(v)
        else:
            all_traits = list(raw)
        assert len(all_traits) == 1
        assert all_traits[0].name == "t2"


@pytest.mark.unit
class TestRubricManagerRename:
    """Test get_questions_with_rubric rename."""

    def test_get_question_ids_with_rubric_exists(self):
        """Renamed method should exist and return list[str]."""
        b = _create_benchmark()
        result = b._rubric_manager.get_question_ids_with_rubric()
        assert isinstance(result, list)

    def test_old_name_does_not_exist(self):
        """Old method name should not exist."""
        b = _create_benchmark()
        assert not hasattr(b._rubric_manager, "get_questions_with_rubric")


@pytest.mark.unit
class TestValidateRubricsErrorShape:
    """Test validate_rubrics returns dict-based errors."""

    def test_returns_dict_errors(self):
        """validate_rubrics errors should be list[dict[str, str]]."""
        b, q_id = _create_benchmark_with_question()
        # Inject a trait with empty description into the cache directly
        # (bypasses Pydantic constructor validation)
        bad_trait = LLMRubricTrait.model_construct(name="bad_trait", description="", kind="boolean")
        b._questions_cache[q_id]["question_rubric"] = {
            "llm_traits": [bad_trait],
            "regex_traits": [],
            "callable_traits": [],
            "metric_traits": [],
            "agentic_traits": [],
        }
        valid, errors = b._rubric_manager.validate_rubrics()
        assert valid is False
        assert len(errors) > 0
        assert isinstance(errors[0], dict)
        assert "source" in errors[0]
        assert "error" in errors[0]
        assert errors[0]["source"] == f"question:{q_id}"
