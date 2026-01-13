"""Unit tests for VerificationResult and related component models.

Tests cover:
- VerificationResultMetadata fields and compute_result_id()
- VerificationResultTemplate fields
- VerificationResultRubric fields and get_all_trait_scores()
- VerificationResultDeepJudgment fields
- VerificationResult construction
- Backward compatibility properties
- Pass/fail determination via completed_without_errors
"""

import pytest

from karenina.schemas.workflow.verification.result import VerificationResult
from karenina.schemas.workflow.verification.result_components import (
    VerificationResultDeepJudgment,
    VerificationResultDeepJudgmentRubric,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)

# =============================================================================
# VerificationResultMetadata Tests
# =============================================================================


@pytest.mark.unit
def test_metadata_construction() -> None:
    """Test VerificationResultMetadata construction with all fields."""
    metadata = VerificationResultMetadata(
        question_id="q-123",
        template_id="t-456",
        completed_without_errors=True,
        question_text="What is 2+2?",
        answering_model="gpt-4",
        parsing_model="claude-haiku-4-5",
        execution_time=1.5,
        timestamp="2025-01-11T12:00:00Z",
        result_id="abc123",
    )

    assert metadata.question_id == "q-123"
    assert metadata.template_id == "t-456"
    assert metadata.completed_without_errors is True
    assert metadata.error is None
    assert metadata.question_text == "What is 2+2?"
    assert metadata.answering_model == "gpt-4"
    assert metadata.parsing_model == "claude-haiku-4-5"
    assert metadata.execution_time == 1.5
    assert metadata.timestamp == "2025-01-11T12:00:00Z"
    assert metadata.result_id == "abc123"


@pytest.mark.unit
def test_metadata_with_error() -> None:
    """Test VerificationResultMetadata with error state."""
    metadata = VerificationResultMetadata(
        question_id="q-123",
        template_id="t-456",
        completed_without_errors=False,
        error="API timeout",
        question_text="What is 2+2?",
        answering_model="gpt-4",
        parsing_model="claude-haiku-4-5",
        execution_time=0.5,
        timestamp="2025-01-11T12:00:00Z",
        result_id="abc123",
    )

    assert metadata.completed_without_errors is False
    assert metadata.error == "API timeout"


@pytest.mark.unit
def test_metadata_compute_result_id_deterministic() -> None:
    """Test that compute_result_id produces deterministic hash."""
    params = {
        "question_id": "q-123",
        "answering_model": "openai/gpt-4",
        "parsing_model": "anthropic/claude-haiku-4-5",
        "timestamp": "2025-01-11T12:00:00Z",
        "replicate": None,
        "answering_mcp_servers": None,
    }

    id1 = VerificationResultMetadata.compute_result_id(**params)
    id2 = VerificationResultMetadata.compute_result_id(**params)

    assert id1 == id2
    assert len(id1) == 16  # First 16 chars of SHA256


@pytest.mark.unit
def test_metadata_compute_result_id_includes_replicate() -> None:
    """Test that compute_result_id includes replicate in hash."""
    params1 = {
        "question_id": "q-123",
        "answering_model": "openai/gpt-4",
        "parsing_model": "anthropic/claude-haiku-4-5",
        "timestamp": "2025-01-11T12:00:00Z",
        "replicate": 1,
    }
    params2 = {
        "question_id": "q-123",
        "answering_model": "openai/gpt-4",
        "parsing_model": "anthropic/claude-haiku-4-5",
        "timestamp": "2025-01-11T12:00:00Z",
        "replicate": 2,
    }

    id1 = VerificationResultMetadata.compute_result_id(**params1)
    id2 = VerificationResultMetadata.compute_result_id(**params2)

    assert id1 != id2


@pytest.mark.unit
def test_metadata_compute_result_id_includes_mcp_servers() -> None:
    """Test that compute_result_id includes MCP servers in hash."""
    params1 = {
        "question_id": "q-123",
        "answering_model": "openai/gpt-4",
        "parsing_model": "anthropic/claude-haiku-4-5",
        "timestamp": "2025-01-11T12:00:00Z",
        "answering_mcp_servers": ["server1", "server2"],
    }
    params2 = {
        "question_id": "q-123",
        "answering_model": "openai/gpt-4",
        "parsing_model": "anthropic/claude-haiku-4-5",
        "timestamp": "2025-01-11T12:00:00Z",
        "answering_mcp_servers": ["server1"],
    }

    id1 = VerificationResultMetadata.compute_result_id(**params1)
    id2 = VerificationResultMetadata.compute_result_id(**params2)

    assert id1 != id2


@pytest.mark.unit
def test_metadata_compute_result_id_sorts_mcp_servers() -> None:
    """Test that MCP servers are sorted for deterministic hashing."""
    params = {
        "question_id": "q-123",
        "answering_model": "openai/gpt-4",
        "parsing_model": "anthropic/claude-haiku-4-5",
        "timestamp": "2025-01-11T12:00:00Z",
        "answering_mcp_servers": ["server2", "server1"],  # Unsorted
    }

    result_id = VerificationResultMetadata.compute_result_id(**params)

    # Same params with different order should produce same ID
    params_sorted = {
        "question_id": "q-123",
        "answering_model": "openai/gpt-4",
        "parsing_model": "anthropic/claude-haiku-4-5",
        "timestamp": "2025-01-11T12:00:00Z",
        "answering_mcp_servers": ["server1", "server2"],  # Sorted
    }

    result_id_sorted = VerificationResultMetadata.compute_result_id(**params_sorted)

    assert result_id == result_id_sorted


# =============================================================================
# VerificationResultTemplate Tests
# =============================================================================


@pytest.mark.unit
def test_template_construction() -> None:
    """Test VerificationResultTemplate construction with all fields."""
    template = VerificationResultTemplate(
        raw_llm_response="The answer is Paris.",
        parsed_gt_response={"value": "Paris"},
        parsed_llm_response={"value": "Paris"},
        template_verification_performed=True,
        verify_result=True,
        verify_granular_result={"value": True},
    )

    assert template.raw_llm_response == "The answer is Paris."
    assert template.parsed_gt_response == {"value": "Paris"}
    assert template.parsed_llm_response == {"value": "Paris"}
    assert template.template_verification_performed is True
    assert template.verify_result is True


@pytest.mark.unit
def test_template_with_embedding_check() -> None:
    """Test template with embedding check fields."""
    template = VerificationResultTemplate(
        raw_llm_response="Similar answer",
        parsed_llm_response={"value": "Paris"},
        embedding_check_performed=True,
        embedding_similarity_score=0.92,
        embedding_override_applied=False,
        embedding_model_used="all-MiniLM-L6-v2",
    )

    assert template.embedding_check_performed is True
    assert template.embedding_similarity_score == 0.92
    assert template.embedding_override_applied is False
    assert template.embedding_model_used == "all-MiniLM-L6-v2"


@pytest.mark.unit
def test_template_with_regex_validations() -> None:
    """Test template with regex validation fields."""
    template = VerificationResultTemplate(
        raw_llm_response="Answer: [PARIS]",
        parsed_llm_response={"value": "PARIS"},
        regex_validations_performed=True,
        regex_validation_results={"bracket_format": True, "uppercase": True},
        regex_overall_success=True,
        regex_extraction_results={"bracket_format": "PARIS", "uppercase": "PARIS"},
    )

    assert template.regex_validations_performed is True
    assert template.regex_validation_results == {"bracket_format": True, "uppercase": True}
    assert template.regex_overall_success is True


@pytest.mark.unit
def test_template_with_abstention() -> None:
    """Test template with abstention detection fields."""
    template = VerificationResultTemplate(
        raw_llm_response="I cannot answer this question.",
        parsed_llm_response=None,
        abstention_check_performed=True,
        abstention_detected=True,
        abstention_override_applied=False,
        abstention_reasoning="Model refused to answer due to policy",
    )

    assert template.abstention_check_performed is True
    assert template.abstention_detected is True
    assert template.abstention_reasoning == "Model refused to answer due to policy"


@pytest.mark.unit
def test_template_with_recursion_limit() -> None:
    """Test template with recursion limit metadata."""
    template = VerificationResultTemplate(
        raw_llm_response="Infinite loop",
        parsed_llm_response=None,
        recursion_limit_reached=True,
    )

    assert template.recursion_limit_reached is True


@pytest.mark.unit
def test_template_with_mcp_servers() -> None:
    """Test template with MCP server list."""
    template = VerificationResultTemplate(
        raw_llm_response="Answer",
        parsed_llm_response={"value": "Paris"},
        answering_mcp_servers=["brave-search", "read-resource"],
    )

    assert template.answering_mcp_servers == ["brave-search", "read-resource"]


@pytest.mark.unit
def test_template_with_usage_metadata() -> None:
    """Test template with usage metadata."""
    template = VerificationResultTemplate(
        raw_llm_response="Answer",
        parsed_llm_response={"value": "Paris"},
        usage_metadata={
            "answer_generation": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            "parsing": {"input_tokens": 200, "output_tokens": 30, "total_tokens": 230},
            "total": {"input_tokens": 300, "output_tokens": 80, "total_tokens": 380},
        },
    )

    assert template.usage_metadata["answer_generation"]["total_tokens"] == 150
    assert template.usage_metadata["parsing"]["total_tokens"] == 230
    assert template.usage_metadata["total"]["total_tokens"] == 380


@pytest.mark.unit
def test_template_with_agent_metrics() -> None:
    """Test template with agent metrics."""
    template = VerificationResultTemplate(
        raw_llm_response="Agent response",
        parsed_llm_response={"value": "Paris"},
        agent_metrics={
            "iterations": 3,
            "tool_calls": 5,
            "tools_used": ["brave-search", "read-resource"],
            "suspect_failed_tool_calls": 1,
            "suspect_failed_tools": ["brave-search"],
        },
    )

    assert template.agent_metrics["iterations"] == 3
    assert template.agent_metrics["tool_calls"] == 5
    assert template.agent_metrics["suspect_failed_tool_calls"] == 1


# =============================================================================
# VerificationResultRubric Tests
# =============================================================================


@pytest.mark.unit
def test_rubric_construction() -> None:
    """Test VerificationResultRubric construction."""
    rubric = VerificationResultRubric(
        rubric_evaluation_performed=True,
        rubric_evaluation_strategy="batch",
        llm_trait_scores={"clarity": 4, "completeness": 3},
        regex_trait_scores={"has_citation": True},
        callable_trait_scores={"is_concise": True},
        metric_trait_scores={"feature_identification": {"precision": 1.0, "recall": 0.8, "f1": 0.89}},
    )

    assert rubric.rubric_evaluation_performed is True
    assert rubric.rubric_evaluation_strategy == "batch"
    assert rubric.llm_trait_scores == {"clarity": 4, "completeness": 3}
    assert rubric.regex_trait_scores == {"has_citation": True}
    assert rubric.callable_trait_scores == {"is_concise": True}


@pytest.mark.unit
def test_rubric_get_all_trait_scores() -> None:
    """Test get_all_trait_scores combines all trait types."""
    rubric = VerificationResultRubric(
        rubric_evaluation_performed=True,
        llm_trait_scores={"clarity": 4},
        regex_trait_scores={"has_brackets": True},
        callable_trait_scores={"is_short": False},
        metric_trait_scores={"recall": {"precision": 0.9, "recall": 0.8, "f1": 0.85}},
    )

    all_scores = rubric.get_all_trait_scores()

    assert all_scores == {
        "clarity": 4,
        "has_brackets": True,
        "is_short": False,
        "recall": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
    }


@pytest.mark.unit
def test_rubric_get_all_trait_scores_empty() -> None:
    """Test get_all_trait_scores returns empty dict when no traits."""
    rubric = VerificationResultRubric(rubric_evaluation_performed=False)

    assert rubric.get_all_trait_scores() == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    "trait_type,scores_field,trait_name,expected_value,expected_type",
    [
        ("llm", "llm_trait_scores", "clarity", 4, "llm"),
        ("regex", "regex_trait_scores", "has_brackets", True, "regex"),
        ("callable", "callable_trait_scores", "is_short", False, "callable"),
        ("metric", "metric_trait_scores", "recall", {"precision": 0.9, "recall": 0.8, "f1": 0.85}, "metric"),
    ],
    ids=["llm_trait", "regex_trait", "callable_trait", "metric_trait"],
)
def test_rubric_get_trait_by_name(
    trait_type: str,
    scores_field: str,
    trait_name: str,
    expected_value: object,
    expected_type: str,
) -> None:
    """Test get_trait_by_name for various trait types."""
    rubric = VerificationResultRubric(**{scores_field: {trait_name: expected_value}})

    result = rubric.get_trait_by_name(trait_name)

    assert result is not None
    assert result[0] == expected_value
    assert result[1] == expected_type


@pytest.mark.unit
def test_rubric_get_trait_by_name_not_found() -> None:
    """Test get_trait_by_name returns None for unknown trait."""
    rubric = VerificationResultRubric(
        llm_trait_scores={"clarity": 4},
    )

    result = rubric.get_trait_by_name("unknown")

    assert result is None


# =============================================================================
# VerificationResultDeepJudgment Tests
# =============================================================================


@pytest.mark.unit
def test_deep_judgment_construction() -> None:
    """Test VerificationResultDeepJudgment construction."""
    deep_judgment = VerificationResultDeepJudgment(
        deep_judgment_performed=True,
        deep_judgment_enabled=True,
        extracted_excerpts={"value": [{"text": "excerpt1", "source": "doc1"}]},
        attribute_reasoning={"value": "The answer mentions Paris correctly"},
        deep_judgment_stages_completed=["extract_excerpts", "validate_excerpts"],
        deep_judgment_model_calls=4,
        deep_judgment_excerpt_retry_count=1,
        deep_judgment_search_enabled=False,
    )

    assert deep_judgment.deep_judgment_performed is True
    assert deep_judgment.deep_judgment_enabled is True
    assert "value" in deep_judgment.extracted_excerpts
    assert deep_judgment.attribute_reasoning["value"] == "The answer mentions Paris correctly"
    assert deep_judgment.deep_judgment_stages_completed == ["extract_excerpts", "validate_excerpts"]
    assert deep_judgment.deep_judgment_model_calls == 4


@pytest.mark.unit
def test_deep_judgment_default_values() -> None:
    """Test VerificationResultDeepJudgment default values."""
    deep_judgment = VerificationResultDeepJudgment()

    assert deep_judgment.deep_judgment_performed is False
    assert deep_judgment.deep_judgment_enabled is False
    assert deep_judgment.extracted_excerpts is None
    assert deep_judgment.attribute_reasoning is None
    assert deep_judgment.deep_judgment_stages_completed is None
    assert deep_judgment.deep_judgment_model_calls == 0
    assert deep_judgment.deep_judgment_excerpt_retry_count == 0
    assert deep_judgment.deep_judgment_search_enabled is False


@pytest.mark.unit
def test_deep_judgment_with_attributes_without_excerpts() -> None:
    """Test deep_judgment with attributes_without_excerpts list."""
    deep_judgment = VerificationResultDeepJudgment(
        deep_judgment_performed=True,
        deep_judgment_enabled=True,
        attributes_without_excerpts=["reasoning", "confidence"],
    )

    assert deep_judgment.attributes_without_excerpts == ["reasoning", "confidence"]


@pytest.mark.unit
def test_deep_judgment_with_hallucination_risk() -> None:
    """Test deep_judgment with hallucination risk assessment."""
    deep_judgment = VerificationResultDeepJudgment(
        deep_judgment_performed=True,
        deep_judgment_enabled=True,
        deep_judgment_search_enabled=True,
        hallucination_risk_assessment={
            "value": "low",
            "reasoning": "All excerpts validated against sources",
        },
    )

    assert deep_judgment.deep_judgment_search_enabled is True
    assert deep_judgment.hallucination_risk_assessment["value"] == "low"


# =============================================================================
# VerificationResultDeepJudgmentRubric Tests
# =============================================================================


@pytest.mark.unit
def test_deep_judgment_rubric_construction() -> None:
    """Test VerificationResultDeepJudgmentRubric construction."""
    deep_judgment_rubric = VerificationResultDeepJudgmentRubric(
        deep_judgment_rubric_performed=True,
        extracted_rubric_excerpts={"clarity": [{"text": "clear text"}]},
        rubric_trait_reasoning={"clarity": "Response is well structured"},
        deep_judgment_rubric_scores={"clarity": 5},
        total_deep_judgment_model_calls=6,
        total_excerpt_retries=0,
        rubric_hallucination_risk_assessment={"clarity": {"overall_risk": "low"}},
    )

    assert deep_judgment_rubric.deep_judgment_rubric_performed is True
    assert "clarity" in deep_judgment_rubric.extracted_rubric_excerpts
    assert deep_judgment_rubric.rubric_trait_reasoning["clarity"] == "Response is well structured"
    assert deep_judgment_rubric.total_deep_judgment_model_calls == 6


@pytest.mark.unit
def test_deep_judgment_rubric_defaults() -> None:
    """Test VerificationResultDeepJudgmentRubric default values."""
    deep_judgment_rubric = VerificationResultDeepJudgmentRubric()

    assert deep_judgment_rubric.deep_judgment_rubric_performed is False
    assert deep_judgment_rubric.extracted_rubric_excerpts is None
    assert deep_judgment_rubric.rubric_trait_reasoning is None
    assert deep_judgment_rubric.deep_judgment_rubric_scores is None
    assert deep_judgment_rubric.total_deep_judgment_model_calls == 0
    assert deep_judgment_rubric.total_excerpt_retries == 0
    assert deep_judgment_rubric.total_traits_evaluated == 0


# =============================================================================
# VerificationResult Construction Tests
# =============================================================================


@pytest.mark.unit
def test_verification_result_construction_minimal() -> None:
    """Test VerificationResult construction with minimal required fields."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=True,
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=1.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=VerificationResultTemplate(
            raw_llm_response="4",
            parsed_llm_response={"value": "4"},
            template_verification_performed=True,
            verify_result=True,
        ),
    )

    assert result.metadata.question_id == "q-123"
    assert result.template.verify_result is True
    assert result.rubric is None
    assert result.deep_judgment is None


@pytest.mark.unit
def test_verification_result_with_all_components() -> None:
    """Test VerificationResult with all components."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=True,
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=1.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=VerificationResultTemplate(
            raw_llm_response="4",
            parsed_llm_response={"value": "4"},
            template_verification_performed=True,
            verify_result=True,
        ),
        rubric=VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 5},
        ),
        deep_judgment=VerificationResultDeepJudgment(
            deep_judgment_performed=True,
            deep_judgment_enabled=True,
        ),
        deep_judgment_rubric=VerificationResultDeepJudgmentRubric(
            deep_judgment_rubric_performed=True,
            deep_judgment_rubric_enabled=True,
        ),
    )

    assert result.template is not None
    assert result.rubric is not None
    assert result.deep_judgment is not None
    assert result.deep_judgment_rubric is not None


@pytest.mark.unit
def test_verification_result_with_trace_filtering() -> None:
    """Test VerificationResult with trace filtering fields."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=True,
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=1.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=VerificationResultTemplate(
            raw_llm_response="Final answer",
            parsed_llm_response={"value": "4"},
        ),
        evaluation_input="Full trace with tool calls",
        used_full_trace=True,
        trace_extraction_error=None,
    )

    assert result.evaluation_input == "Full trace with tool calls"
    assert result.used_full_trace is True
    assert result.trace_extraction_error is None


@pytest.mark.unit
def test_verification_result_with_trace_extraction_error() -> None:
    """Test VerificationResult with trace extraction error."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=False,
            error="Could not extract final AI message",
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=0.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=None,
        trace_extraction_error="No AI message found at end of trace",
    )

    assert result.trace_extraction_error == "No AI message found at end of trace"
    assert result.metadata.completed_without_errors is False


# =============================================================================
# VerificationResult Backward Compatibility Tests
# =============================================================================


@pytest.mark.unit
def test_backward_compat_metadata_properties() -> None:
    """Test backward compatibility properties for metadata fields."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=True,
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=1.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
            run_name="test-run",
            keywords=["math", "simple"],
        ),
        template=VerificationResultTemplate(
            raw_llm_response="4",
            parsed_llm_response={"value": "4"},
        ),
    )

    assert result.question_id == "q-123"
    assert result.completed_without_errors is True
    assert result.error is None
    assert result.question_text == "What is 2+2?"
    assert result.answering_model == "gpt-4"
    assert result.parsing_model == "claude-haiku-4-5"
    assert result.run_name == "test-run"
    assert result.keywords == ["math", "simple"]
    assert result.timestamp == "2025-01-11T12:00:00Z"


@pytest.mark.unit
def test_backward_compat_template_properties() -> None:
    """Test backward compatibility properties for template fields."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=True,
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=1.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=VerificationResultTemplate(
            raw_llm_response="4",
            parsed_gt_response={"value": "4"},
            parsed_llm_response={"value": "4"},
            template_verification_performed=True,
            verify_result=True,
            verify_granular_result={"value": True},
            abstention_detected=False,
            recursion_limit_reached=False,
            answering_mcp_servers=["search"],
            usage_metadata={"total": {"total_tokens": 100}},
            agent_metrics={"iterations": 2},
            regex_validations_performed=False,
        ),
    )

    assert result.raw_llm_response == "4"
    assert result.parsed_gt_response == {"value": "4"}
    assert result.parsed_llm_response == {"value": "4"}
    assert result.verify_result is True
    assert result.verify_granular_result == {"value": True}
    assert result.abstention_detected is False
    assert result.recursion_limit_reached is False
    assert result.answering_mcp_servers == ["search"]
    assert result.usage_metadata["total"]["total_tokens"] == 100
    assert result.agent_metrics["iterations"] == 2
    assert result.regex_validations_performed is False


@pytest.mark.unit
def test_backward_compat_template_properties_return_none_when_no_template() -> None:
    """Test that template properties return None when template is None."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=False,
            error="Failed",
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=0.1,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=None,
    )

    assert result.raw_llm_response is None
    assert result.parsed_gt_response is None
    assert result.parsed_llm_response is None
    assert result.verify_result is None
    assert result.abstention_detected is None
    assert result.recursion_limit_reached is None


@pytest.mark.unit
def test_backward_compat_rubric_properties() -> None:
    """Test backward compatibility properties for rubric fields."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=True,
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=1.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=VerificationResultTemplate(
            raw_llm_response="4",
            parsed_llm_response={"value": "4"},
        ),
        rubric=VerificationResultRubric(
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="sequential",
            llm_trait_scores={"clarity": 5},
            metric_trait_scores={"recall": {"precision": 1.0, "recall": 0.9, "f1": 0.95}},
            metric_trait_confusion_lists={
                "recall": {
                    "tp": ["item1", "item2"],
                    "tn": ["item3"],
                    "fp": ["item4"],
                    "fn": ["item5"],
                }
            },
        ),
    )

    assert result.rubric_evaluation_performed is True
    assert result.verify_rubric == {"clarity": 5, "recall": {"precision": 1.0, "recall": 0.9, "f1": 0.95}}
    assert result.metric_trait_metrics == {"recall": {"precision": 1.0, "recall": 0.9, "f1": 0.95}}
    assert result.metric_trait_confusion_lists == {
        "recall": {
            "tp": ["item1", "item2"],
            "tn": ["item3"],
            "fp": ["item4"],
            "fn": ["item5"],
        }
    }


@pytest.mark.unit
def test_backward_compat_rubric_properties_return_defaults_when_no_rubric() -> None:
    """Test that rubric properties return defaults when rubric is None."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=True,
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=1.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=VerificationResultTemplate(
            raw_llm_response="4",
            parsed_llm_response={"value": "4"},
        ),
        rubric=None,
    )

    assert result.rubric_evaluation_performed is False
    assert result.verify_rubric is None
    assert result.metric_trait_metrics is None
    assert result.metric_trait_confusion_lists is None


@pytest.mark.unit
def test_backward_compat_deep_judgment_properties() -> None:
    """Test backward compatibility properties for deep judgment fields."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=True,
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=1.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=VerificationResultTemplate(
            raw_llm_response="4",
            parsed_llm_response={"value": "4"},
        ),
        deep_judgment=VerificationResultDeepJudgment(
            deep_judgment_performed=True,
            deep_judgment_enabled=True,
            deep_judgment_model_calls=3,
            deep_judgment_excerpt_retry_count=1,
        ),
    )

    assert result.deep_judgment_performed is True
    assert result.deep_judgment_enabled is True
    assert result.deep_judgment_model_calls == 3
    assert result.deep_judgment_excerpt_retry_count == 1


@pytest.mark.unit
def test_backward_compat_deep_judgment_properties_return_defaults_when_no_deep_judgment() -> None:
    """Test that deep judgment properties return defaults when deep_judgment is None."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=True,
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=1.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=VerificationResultTemplate(
            raw_llm_response="4",
            parsed_llm_response={"value": "4"},
        ),
        deep_judgment=None,
    )

    assert result.deep_judgment_performed is False
    assert result.deep_judgment_enabled is False
    assert result.deep_judgment_model_calls == 0
    assert result.deep_judgment_excerpt_retry_count == 0
    assert result.extracted_excerpts is None


# =============================================================================
# VerificationResult Pass/Fail Determination Tests
# =============================================================================


@pytest.mark.unit
def test_pass_determination_completed_without_errors() -> None:
    """Test pass determination via completed_without_errors=True."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=True,
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=1.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=VerificationResultTemplate(
            raw_llm_response="4",
            parsed_llm_response={"value": "4"},
            template_verification_performed=True,
            verify_result=True,
        ),
    )

    assert result.completed_without_errors is True


@pytest.mark.unit
def test_fail_determination_completed_with_errors() -> None:
    """Test fail determination via completed_without_errors=False."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=False,
            error="API timeout",
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=0.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=VerificationResultTemplate(
            raw_llm_response="Error",
            parsed_llm_response=None,
        ),
    )

    assert result.completed_without_errors is False


@pytest.mark.unit
def test_error_property_accessible() -> None:
    """Test that error property is accessible via backward compatibility."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=False,
            error="Parsing failed",
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=0.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=None,
    )

    assert result.error == "Parsing failed"


# =============================================================================
# VerificationResultTemplate Sufficiency Tests
# =============================================================================


@pytest.mark.unit
def test_template_with_sufficiency_check() -> None:
    """Test template with sufficiency check fields."""
    template = VerificationResultTemplate(
        raw_llm_response="The answer is Paris.",
        parsed_llm_response={"value": "Paris"},
        sufficiency_check_performed=True,
        sufficiency_detected=True,  # True = sufficient
        sufficiency_override_applied=False,
        sufficiency_reasoning="Response contains the requested information.",
    )

    assert template.sufficiency_check_performed is True
    assert template.sufficiency_detected is True
    assert template.sufficiency_override_applied is False
    assert template.sufficiency_reasoning == "Response contains the requested information."


@pytest.mark.unit
def test_template_with_insufficiency_detected() -> None:
    """Test template when response is insufficient."""
    template = VerificationResultTemplate(
        raw_llm_response="I don't know.",
        parsed_llm_response=None,
        sufficiency_check_performed=True,
        sufficiency_detected=False,  # False = insufficient
        sufficiency_override_applied=True,
        sufficiency_reasoning="Response lacks the required information.",
    )

    assert template.sufficiency_check_performed is True
    assert template.sufficiency_detected is False
    assert template.sufficiency_override_applied is True
    assert template.sufficiency_reasoning is not None


@pytest.mark.unit
def test_template_sufficiency_defaults() -> None:
    """Test template sufficiency field defaults."""
    template = VerificationResultTemplate(
        raw_llm_response="Some answer",
        parsed_llm_response={"value": "test"},
    )

    assert template.sufficiency_check_performed is False
    assert template.sufficiency_detected is None
    assert template.sufficiency_override_applied is False
    assert template.sufficiency_reasoning is None


# =============================================================================
# VerificationResult Backward Compatibility Sufficiency Tests
# =============================================================================


@pytest.mark.unit
def test_backward_compat_sufficiency_properties() -> None:
    """Test backward compatibility properties for sufficiency fields."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=True,
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=1.5,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=VerificationResultTemplate(
            raw_llm_response="4",
            parsed_llm_response={"value": "4"},
            sufficiency_check_performed=True,
            sufficiency_detected=True,
            sufficiency_override_applied=False,
            sufficiency_reasoning="Response contains the answer.",
        ),
    )

    assert result.sufficiency_check_performed is True
    assert result.sufficiency_detected is True
    assert result.sufficiency_override_applied is False
    assert result.sufficiency_reasoning == "Response contains the answer."


@pytest.mark.unit
def test_backward_compat_sufficiency_properties_return_defaults_when_no_template() -> None:
    """Test that sufficiency properties return defaults when template is None."""
    result = VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q-123",
            template_id="t-456",
            completed_without_errors=False,
            error="Failed",
            question_text="What is 2+2?",
            answering_model="gpt-4",
            parsing_model="claude-haiku-4-5",
            execution_time=0.1,
            timestamp="2025-01-11T12:00:00Z",
            result_id="abc123",
        ),
        template=None,
    )

    assert result.sufficiency_check_performed is False
    assert result.sufficiency_detected is None
    assert result.sufficiency_override_applied is False
    assert result.sufficiency_reasoning is None


@pytest.mark.unit
def test_sufficiency_and_abstention_can_coexist() -> None:
    """Test that sufficiency and abstention fields can coexist in template."""
    template = VerificationResultTemplate(
        raw_llm_response="I cannot provide that information.",
        parsed_llm_response=None,
        # Abstention check
        abstention_check_performed=True,
        abstention_detected=True,
        abstention_override_applied=True,
        abstention_reasoning="Model refused to answer.",
        # Sufficiency check (not performed if abstention detected first)
        sufficiency_check_performed=False,
        sufficiency_detected=None,
        sufficiency_override_applied=False,
        sufficiency_reasoning=None,
    )

    assert template.abstention_check_performed is True
    assert template.abstention_detected is True
    assert template.sufficiency_check_performed is False
    assert template.sufficiency_detected is None
