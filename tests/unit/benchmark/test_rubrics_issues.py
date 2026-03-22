"""Tests for issue 108: validate_rubrics() crashes on dict-format question rubrics.

When question_rubric is stored as a dict (keyed by trait type) rather than a
flat list, validate_rubrics() iterates over the dict keys (strings) instead
of the trait objects within each value list, causing AttributeError.
"""

import pytest

from karenina import Benchmark
from karenina.schemas.entities import LLMRubricTrait


@pytest.mark.unit
class TestValidateRubricsDictFormat:
    """Tests for issue 108: validate_rubrics() with dict-format question rubrics."""

    def test_validate_rubrics_does_not_crash_on_dict_format(self) -> None:
        """validate_rubrics() should handle dict-format question rubrics without error.

        When question_rubric is a dict like {"llm_traits": [...], "regex_traits": [...]},
        the validator must iterate over the trait lists within the dict values,
        not over the dict keys (which are strings).
        """
        benchmark = Benchmark.create(name="test-dict-rubric")
        q_id = benchmark.add_question("What is 2+2?", "4")

        # Inject dict-format question rubric into the cache directly
        # (this simulates how rubrics can end up as dicts after deserialization)
        trait = LLMRubricTrait(name="accuracy", description="Is the answer correct?", kind="boolean")
        benchmark._questions_cache[q_id]["question_rubric"] = {
            "llm_traits": [trait],
            "regex_traits": [],
            "callable_traits": [],
            "metric_traits": [],
            "agentic_traits": [],
        }

        # This should NOT raise AttributeError
        valid, errors = benchmark.validate_rubrics()

        # Should validate successfully since the trait has name and description
        assert valid is True
        assert errors == []

    def test_validate_rubrics_reports_errors_for_dict_format_traits(self) -> None:
        """validate_rubrics() should correctly report errors for invalid traits in dict format."""
        benchmark = Benchmark.create(name="test-dict-rubric-errors")
        q_id = benchmark.add_question("What is 2+2?", "4")

        # Inject a trait with missing description
        bad_trait = LLMRubricTrait(name="accuracy", description="", kind="boolean")
        benchmark._questions_cache[q_id]["question_rubric"] = {
            "llm_traits": [bad_trait],
            "regex_traits": [],
            "callable_traits": [],
            "metric_traits": [],
            "agentic_traits": [],
        }

        valid, errors = benchmark.validate_rubrics()
        assert valid is False
        assert len(errors) > 0
