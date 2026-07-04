"""Tests for VerificationResultSet serialization and metric detection issues.

Covers:
- Issue 016: get_summary() tuple keys break JSON serialization
- Issue 118: _calculate_rubric_traits and get_trait_summary omit agentic traits
- Issue 119: _calculate_rubric_traits uses confusion_lists for metric detection
"""

import json

import pytest

from karenina.schemas.results.rubric import RubricResults
from karenina.schemas.results.verification_result_set import VerificationResultSet
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result import VerificationResult
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)


def _make_model_identity(interface: str = "langchain", model_name: str = "gpt-4") -> ModelIdentity:
    """Create a ModelIdentity for testing."""
    return ModelIdentity(interface=interface, model_name=model_name)


def _make_metadata(
    question_id: str = "q1",
    answering_model: str = "gpt-4",
    parsing_model: str = "gpt-4",
) -> VerificationResultMetadata:
    """Create a VerificationResultMetadata for testing."""
    answering = _make_model_identity(model_name=answering_model)
    parsing = _make_model_identity(model_name=parsing_model)
    return VerificationResultMetadata(
        question_id=question_id,
        template_id="tmpl_abc",
        failure=None,
        caveats=[],
        question_text="What is 2+2?",
        answering=answering,
        parsing=parsing,
        execution_time=1.0,
        timestamp="2026-01-01T00:00:00",
        result_id="abcdef1234567890",
    )


def _make_result(
    question_id: str = "q1",
    answering_model: str = "gpt-4",
    parsing_model: str = "gpt-4",
    template: VerificationResultTemplate | None = None,
    rubric: VerificationResultRubric | None = None,
) -> VerificationResult:
    """Create a minimal VerificationResult for testing."""
    return VerificationResult(
        metadata=_make_metadata(
            question_id=question_id,
            answering_model=answering_model,
            parsing_model=parsing_model,
        ),
        template=template,
        rubric=rubric,
    )


@pytest.mark.unit
class TestIssue016TupleKeysBreakJsonSerialization:
    """Issue 016: tokens_by_combo, completion_by_combo, template_pass_by_combo
    dicts use Python tuple keys, which cause json.dumps() TypeError."""

    def test_summary_tokens_by_combo_is_json_serializable(self) -> None:
        """tokens_by_combo keys must be strings, not tuples."""
        template = VerificationResultTemplate(
            raw_llm_response="answer",
            template_verification_performed=True,
            verify_result=True,
            usage_metadata={
                "total": {"input_tokens": 100, "output_tokens": 50},
            },
        )
        result = _make_result(template=template)
        result_set = VerificationResultSet(results=[result])

        summary = result_set.get_summary()

        # This must not raise TypeError: keys must be str, int, float, bool or None, not tuple
        serialized = json.dumps(summary)
        parsed = json.loads(serialized)

        # Verify the keys are strings in the format "model1 | model2"
        assert isinstance(parsed["tokens_by_combo"], dict)
        for key in parsed["tokens_by_combo"]:
            assert isinstance(key, str)
            assert " | " in key

    def test_summary_completion_by_combo_is_json_serializable(self) -> None:
        """completion_by_combo keys must be strings, not tuples."""
        result = _make_result()
        result_set = VerificationResultSet(results=[result])

        summary = result_set.get_summary()

        serialized = json.dumps(summary)
        parsed = json.loads(serialized)

        assert isinstance(parsed["completion_by_combo"], dict)
        for key in parsed["completion_by_combo"]:
            assert isinstance(key, str)

    def test_summary_template_pass_by_combo_is_json_serializable(self) -> None:
        """template_pass_by_combo keys must be strings, not tuples."""
        template = VerificationResultTemplate(
            raw_llm_response="answer",
            template_verification_performed=True,
            verify_result=True,
        )
        result = _make_result(template=template)
        result_set = VerificationResultSet(results=[result])

        summary = result_set.get_summary()

        serialized = json.dumps(summary)
        parsed = json.loads(serialized)

        assert isinstance(parsed["template_pass_by_combo"], dict)
        for key in parsed["template_pass_by_combo"]:
            assert isinstance(key, str)

    def test_combo_key_format_without_mcp(self) -> None:
        """Combo key should be 'answering_model | parsing_model' when no MCP servers."""
        template = VerificationResultTemplate(
            raw_llm_response="answer",
            template_verification_performed=True,
            verify_result=True,
            usage_metadata={
                "total": {"input_tokens": 100, "output_tokens": 50},
            },
        )
        result = _make_result(
            answering_model="gpt-4",
            parsing_model="gpt-3.5-turbo",
            template=template,
        )
        result_set = VerificationResultSet(results=[result])

        summary = result_set.get_summary()

        # Check that the key is a pipe-delimited string
        expected_key = "langchain:gpt-4 | langchain:gpt-3.5-turbo"
        assert expected_key in summary["tokens_by_combo"]
        assert expected_key in summary["completion_by_combo"]
        assert expected_key in summary["template_pass_by_combo"]

    def test_combo_key_format_with_mcp(self) -> None:
        """Combo key should include MCP servers when present."""
        template = VerificationResultTemplate(
            raw_llm_response="answer",
            template_verification_performed=True,
            verify_result=True,
            answering_mcp_servers=["brave_search", "filesystem"],
            usage_metadata={
                "total": {"input_tokens": 100, "output_tokens": 50},
            },
        )
        result = _make_result(template=template)
        result_set = VerificationResultSet(results=[result])

        summary = result_set.get_summary()

        # The key must be a string and include MCP server info
        for key in summary["tokens_by_combo"]:
            assert isinstance(key, str)

    def test_full_summary_json_round_trip(self) -> None:
        """The entire get_summary() output must survive json.dumps/loads."""
        template = VerificationResultTemplate(
            raw_llm_response="answer",
            template_verification_performed=True,
            verify_result=True,
            usage_metadata={
                "total": {"input_tokens": 100, "output_tokens": 50},
                "answer_generation": {"input_tokens": 60, "output_tokens": 30},
                "parsing": {"input_tokens": 40, "output_tokens": 20},
            },
        )
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4},
        )
        result = _make_result(template=template, rubric=rubric)
        result_set = VerificationResultSet(results=[result])

        summary = result_set.get_summary()

        # Must not raise
        serialized = json.dumps(summary)
        parsed = json.loads(serialized)
        assert parsed["num_results"] == 1


@pytest.mark.unit
class TestIssue119MetricTraitDetectionUsesConfusionLists:
    """Issue 119: _calculate_rubric_traits uses metric_trait_confusion_lists
    instead of metric_trait_scores, inconsistent with other trait types."""

    def test_metric_traits_detected_via_scores_not_confusion_lists(self) -> None:
        """Metric traits should be detected via metric_trait_scores, like other types."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            metric_trait_scores={
                "entity_detection": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
            },
            # No confusion lists set at all
            metric_trait_confusion_lists=None,
        )
        result = _make_result(rubric=rubric)
        result_set = VerificationResultSet(results=[result])

        summary = result_set.get_summary()

        # The metric trait must be detected even without confusion lists
        rubric_traits = summary["rubric_traits"]
        assert rubric_traits is not None

        global_metric_count = rubric_traits["global_traits"]["metric"]["count"]
        qs_metric_count = rubric_traits["question_specific_traits"]["metric"]["count"]
        total_metric = global_metric_count + qs_metric_count
        assert total_metric > 0, (
            "Metric trait 'entity_detection' was not detected. "
            "_calculate_rubric_traits should use metric_trait_scores, not metric_trait_confusion_lists."
        )

    def test_metric_traits_not_detected_when_only_confusion_lists_present(self) -> None:
        """If metric_trait_scores is None but confusion_lists has data,
        the old buggy code would detect it but the fix should not, since
        detection should be based on scores like all other trait types."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            metric_trait_scores=None,
            metric_trait_confusion_lists={
                "entity_detection": {"tp": ["a"], "tn": [], "fp": [], "fn": []},
            },
        )
        result = _make_result(rubric=rubric)
        result_set = VerificationResultSet(results=[result])

        summary = result_set.get_summary()

        rubric_traits = summary["rubric_traits"]
        assert rubric_traits is not None

        global_metric_count = rubric_traits["global_traits"]["metric"]["count"]
        qs_metric_count = rubric_traits["question_specific_traits"]["metric"]["count"]
        total_metric = global_metric_count + qs_metric_count
        assert total_metric == 0, (
            "Metric trait was detected from confusion_lists but should only be detected from metric_trait_scores."
        )


@pytest.mark.unit
class TestIssue118AgenticTraitsInRubricSummary:
    """Issue 118: _calculate_rubric_traits() and get_trait_summary() omit agentic traits."""

    def test_calculate_rubric_traits_includes_agentic(self) -> None:
        """Agentic traits appear in rubric_traits summary under 'agentic' key."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            agentic_trait_scores={"investigates_sources": True},
        )
        result = _make_result(rubric=rubric)
        result_set = VerificationResultSet(results=[result])

        summary = result_set.get_summary()

        rubric_traits = summary["rubric_traits"]
        assert rubric_traits is not None
        assert "agentic" in rubric_traits["global_traits"]
        assert rubric_traits["global_traits"]["agentic"]["count"] == 1

    def test_calculate_rubric_traits_agentic_question_specific(self) -> None:
        """Agentic traits on some questions appear as question-specific."""
        rubric_q1 = VerificationResultRubric(
            rubric_evaluation_performed=True,
            agentic_trait_scores={"depth": 4},
        )
        rubric_q2 = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 3},
        )
        r1 = _make_result(question_id="q1", rubric=rubric_q1)
        r2 = _make_result(question_id="q2", rubric=rubric_q2)
        result_set = VerificationResultSet(results=[r1, r2])

        summary = result_set.get_summary()
        rubric_traits = summary["rubric_traits"]
        assert rubric_traits["question_specific_traits"]["agentic"]["count"] == 1

    def test_calculate_rubric_traits_mixed_types(self) -> None:
        """Agentic traits coexist correctly with LLM traits."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 3},
            agentic_trait_scores={"depth": 4},
        )
        result = _make_result(rubric=rubric)
        result_set = VerificationResultSet(results=[result])

        summary = result_set.get_summary()
        rubric_traits = summary["rubric_traits"]
        assert rubric_traits["global_traits"]["llm"]["count"] == 1
        assert rubric_traits["global_traits"]["agentic"]["count"] == 1


@pytest.mark.unit
class TestIssue118RubricResultsGetTraitSummary:
    """Issue 118 (part 2): RubricResults.get_trait_summary() omits agentic traits."""

    def test_get_trait_summary_includes_agentic_traits(self) -> None:
        """get_trait_summary() returns agentic_traits key with trait names."""
        result = _make_result(
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                agentic_trait_scores={"investigates_sources": True, "depth": 4},
            )
        )
        rubric_results = RubricResults(results=[result])

        summary = rubric_results.get_trait_summary()
        assert "agentic_traits" in summary
        assert sorted(summary["agentic_traits"]) == ["depth", "investigates_sources"]

    def test_get_trait_summary_agentic_empty_when_none(self) -> None:
        """agentic_traits is empty list when no agentic scores present."""
        result = _make_result(
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores={"clarity": 3},
            )
        )
        rubric_results = RubricResults(results=[result])

        summary = rubric_results.get_trait_summary()
        assert "agentic_traits" in summary
        assert summary["agentic_traits"] == []
