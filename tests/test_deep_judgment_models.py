"""Tests for deep-judgment data models and exceptions.

This module tests the deep-judgment feature's data models, configuration,
and exception handling.
"""

import pytest

from karenina.benchmark.verification.exceptions import ExcerptNotFoundError
from karenina.schemas import ModelConfig, VerificationConfig, VerificationResult


# Helper to create a minimal parsing model for tests
def _create_test_parsing_model():
    """Create a minimal parsing model for testing."""
    return ModelConfig(
        id="test-parser",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        temperature=0.0,
        system_prompt="You are a helpful assistant.",
    )


class TestExcerptNotFoundError:
    """Tests for ExcerptNotFoundError exception."""

    def test_exception_creation(self):
        """Test creating ExcerptNotFoundError with all attributes."""
        exc = ExcerptNotFoundError(
            excerpt="This is the problematic excerpt", attribute="drug_target", similarity_score=0.45
        )

        assert exc.excerpt == "This is the problematic excerpt"
        assert exc.attribute == "drug_target"
        assert exc.similarity_score == 0.45
        assert "drug_target" in str(exc)
        assert "0.45" in str(exc)

    def test_exception_message_format(self):
        """Test that exception message is well-formatted."""
        exc = ExcerptNotFoundError(excerpt="test excerpt", attribute="test_attr", similarity_score=0.32)

        message = str(exc)
        assert "test_attr" in message
        assert "0.32" in message
        assert "not found" in message.lower()


class TestVerificationConfigDeepJudgment:
    """Tests for VerificationConfig deep-judgment fields."""

    def test_default_deep_judgment_disabled(self):
        """Test that deep-judgment is disabled by default."""
        config = VerificationConfig(
            answering_models=[], parsing_models=[_create_test_parsing_model()], parsing_only=True
        )

        assert config.deep_judgment_enabled is False
        assert config.deep_judgment_max_excerpts_per_attribute == 3
        assert config.deep_judgment_fuzzy_match_threshold == 0.80
        assert config.deep_judgment_excerpt_retry_attempts == 2

    def test_enable_deep_judgment(self):
        """Test enabling deep-judgment with custom settings."""
        config = VerificationConfig(
            answering_models=[],
            parsing_models=[_create_test_parsing_model()],
            parsing_only=True,
            deep_judgment_enabled=True,
            deep_judgment_max_excerpts_per_attribute=5,
            deep_judgment_fuzzy_match_threshold=0.85,
            deep_judgment_excerpt_retry_attempts=3,
        )

        assert config.deep_judgment_enabled is True
        assert config.deep_judgment_max_excerpts_per_attribute == 5
        assert config.deep_judgment_fuzzy_match_threshold == 0.85
        assert config.deep_judgment_excerpt_retry_attempts == 3

    def test_deep_judgment_with_other_features(self):
        """Test deep-judgment can be enabled alongside other features."""
        config = VerificationConfig(
            answering_models=[],
            parsing_models=[_create_test_parsing_model()],
            parsing_only=True,
            deep_judgment_enabled=True,
            abstention_enabled=True,
            rubric_enabled=True,
            evaluation_mode="template_and_rubric",
        )

        assert config.deep_judgment_enabled is True
        assert config.abstention_enabled is True
        assert config.rubric_enabled is True


class TestVerificationResultDeepJudgment:
    """Tests for VerificationResult deep-judgment fields."""

    def test_default_deep_judgment_fields(self):
        """Test that deep-judgment fields have correct defaults."""
        result = VerificationResult(
            question_id="test_q1",
            template_id="test_t1",
            success=True,
            question_text="What is the drug target?",
            raw_llm_response="The drug target is BCL-2.",
            answering_model="gpt-4.1-mini",
            parsing_model="gpt-4.1-mini",
            execution_time=1.5,
            timestamp="2025-01-13T12:00:00",
        )

        # Deep-judgment disabled by default
        assert result.deep_judgment_enabled is False
        assert result.deep_judgment_performed is False
        assert result.extracted_excerpts is None
        assert result.attribute_reasoning is None
        assert result.deep_judgment_stages_completed is None
        assert result.deep_judgment_model_calls == 0
        assert result.deep_judgment_excerpt_retry_count == 0
        assert result.attributes_without_excerpts is None

    def test_deep_judgment_with_excerpts(self):
        """Test VerificationResult with extracted excerpts."""
        excerpts = {
            "drug_target": [
                {"text": "targets BCL-2", "confidence": "high", "similarity_score": 0.95},
                {"text": "inhibits BCL-2 protein", "confidence": "medium", "similarity_score": 0.88},
            ],
            "mechanism": [{"text": "apoptosis induction", "confidence": "low", "similarity_score": 0.76}],
        }

        reasoning = {
            "drug_target": "The excerpts clearly state that the drug targets BCL-2 protein.",
            "mechanism": "The text mentions apoptosis induction as the mechanism of action.",
        }

        result = VerificationResult(
            question_id="test_q1",
            template_id="test_t1",
            success=True,
            question_text="What is the drug target?",
            raw_llm_response="The drug targets BCL-2 and induces apoptosis.",
            answering_model="gpt-4.1-mini",
            parsing_model="gpt-4.1-mini",
            execution_time=3.2,
            timestamp="2025-01-13T12:00:00",
            deep_judgment_enabled=True,
            deep_judgment_performed=True,
            extracted_excerpts=excerpts,
            attribute_reasoning=reasoning,
            deep_judgment_stages_completed=["excerpts", "reasoning", "parameters"],
            deep_judgment_model_calls=3,
            deep_judgment_excerpt_retry_count=0,
        )

        assert result.deep_judgment_enabled is True
        assert result.deep_judgment_performed is True
        assert result.extracted_excerpts == excerpts
        assert result.attribute_reasoning == reasoning
        assert result.deep_judgment_stages_completed == ["excerpts", "reasoning", "parameters"]
        assert result.deep_judgment_model_calls == 3
        assert result.deep_judgment_excerpt_retry_count == 0

    def test_deep_judgment_with_missing_excerpts(self):
        """Test VerificationResult with missing excerpts (refusal scenario)."""
        excerpts = {
            "drug_target": [],  # No excerpts found
            "mechanism": [{"text": "some mechanism", "confidence": "medium", "similarity_score": 0.82}],
        }

        reasoning = {
            "drug_target": "The response contains a refusal and provides no information about the drug target.",
            "mechanism": "The mechanism is briefly mentioned.",
        }

        result = VerificationResult(
            question_id="test_q1",
            template_id="test_t1",
            success=True,
            question_text="What is the drug target?",
            raw_llm_response="I cannot provide information about that drug.",
            answering_model="gpt-4.1-mini",
            parsing_model="gpt-4.1-mini",
            execution_time=2.8,
            timestamp="2025-01-13T12:00:00",
            deep_judgment_enabled=True,
            deep_judgment_performed=True,
            extracted_excerpts=excerpts,
            attribute_reasoning=reasoning,
            deep_judgment_stages_completed=["excerpts", "reasoning", "parameters"],
            deep_judgment_model_calls=3,
            attributes_without_excerpts=["drug_target"],
        )

        assert result.deep_judgment_performed is True
        assert result.extracted_excerpts["drug_target"] == []  # Empty list is valid
        assert len(result.extracted_excerpts["mechanism"]) == 1
        assert result.attributes_without_excerpts == ["drug_target"]

    def test_deep_judgment_with_retries(self):
        """Test VerificationResult with excerpt validation retries."""
        result = VerificationResult(
            question_id="test_q1",
            template_id="test_t1",
            success=True,
            question_text="Test question",
            raw_llm_response="Test response",
            answering_model="gpt-4.1-mini",
            parsing_model="gpt-4.1-mini",
            execution_time=5.4,
            timestamp="2025-01-13T12:00:00",
            deep_judgment_enabled=True,
            deep_judgment_performed=True,
            deep_judgment_model_calls=5,  # 3 stages + 2 retries
            deep_judgment_excerpt_retry_count=2,
        )

        assert result.deep_judgment_model_calls == 5
        assert result.deep_judgment_excerpt_retry_count == 2

    def test_deep_judgment_enabled_but_not_performed(self):
        """Test scenario where deep-judgment is enabled but not performed (error case)."""
        result = VerificationResult(
            question_id="test_q1",
            template_id="test_t1",
            success=False,
            error="Deep-judgment parsing failed: JSON parsing error",
            question_text="Test question",
            raw_llm_response="Test response",
            answering_model="gpt-4.1-mini",
            parsing_model="gpt-4.1-mini",
            execution_time=1.2,
            timestamp="2025-01-13T12:00:00",
            deep_judgment_enabled=True,
            deep_judgment_performed=False,  # Failed to complete
        )

        assert result.deep_judgment_enabled is True
        assert result.deep_judgment_performed is False
        assert result.success is False

    def test_deep_judgment_fields_serialization(self):
        """Test that deep-judgment fields serialize correctly."""
        result = VerificationResult(
            question_id="test_q1",
            template_id="test_t1",
            success=True,
            question_text="Test",
            raw_llm_response="Response",
            answering_model="gpt-4.1-mini",
            parsing_model="gpt-4.1-mini",
            execution_time=1.0,
            timestamp="2025-01-13",
            deep_judgment_enabled=True,
            deep_judgment_performed=True,
            extracted_excerpts={"attr": [{"text": "test", "confidence": "high", "similarity_score": 0.9}]},
            attribute_reasoning={"attr": "reasoning"},
            deep_judgment_stages_completed=["excerpts", "reasoning"],
            deep_judgment_model_calls=2,
            attributes_without_excerpts=[],
        )

        # Test model_dump serialization
        dumped = result.model_dump()
        assert dumped["deep_judgment_enabled"] is True
        assert dumped["extracted_excerpts"]["attr"][0]["text"] == "test"
        assert dumped["attribute_reasoning"]["attr"] == "reasoning"

    def test_empty_excerpts_list_is_valid(self):
        """Test that empty excerpt lists are valid and distinct from None."""
        result = VerificationResult(
            question_id="test_q1",
            template_id="test_t1",
            success=True,
            question_text="Test",
            raw_llm_response="Response",
            answering_model="gpt-4.1-mini",
            parsing_model="gpt-4.1-mini",
            execution_time=1.0,
            timestamp="2025-01-13",
            deep_judgment_enabled=True,
            deep_judgment_performed=True,
            extracted_excerpts={"attr1": [], "attr2": []},  # All empty - valid for refusals
            attribute_reasoning={
                "attr1": "No excerpts found - response contains refusal",
                "attr2": "No corroborating evidence present",
            },
            attributes_without_excerpts=["attr1", "attr2"],
        )

        # Empty lists are valid and different from None
        assert result.extracted_excerpts is not None
        assert result.extracted_excerpts["attr1"] == []
        assert result.extracted_excerpts["attr2"] == []
        assert len(result.attributes_without_excerpts) == 2


class TestVerificationConfigSearchEnhancedDeepJudgment:
    """Tests for VerificationConfig search-enhanced deep-judgment fields and validation."""

    def test_default_search_disabled(self):
        """Test that search-enhanced deep-judgment is disabled by default."""
        config = VerificationConfig(
            answering_models=[], parsing_models=[_create_test_parsing_model()], parsing_only=True
        )

        assert config.deep_judgment_search_enabled is False
        assert config.deep_judgment_search_tool == "tavily"

    def test_enable_search_with_string_tool(self):
        """Test enabling search-enhanced deep-judgment with string tool name."""
        config = VerificationConfig(
            answering_models=[],
            parsing_models=[_create_test_parsing_model()],
            parsing_only=True,
            deep_judgment_search_enabled=True,
            deep_judgment_search_tool="tavily",
        )

        assert config.deep_judgment_search_enabled is True
        assert config.deep_judgment_search_tool == "tavily"

    def test_enable_search_with_callable_tool(self):
        """Test enabling search-enhanced deep-judgment with callable tool."""

        def my_search_tool(query: str | list[str]) -> str | list[str]:
            if isinstance(query, list):
                return [f"Results for {q}" for q in query]
            return f"Results for {query}"

        config = VerificationConfig(
            answering_models=[],
            parsing_models=[_create_test_parsing_model()],
            parsing_only=True,
            deep_judgment_search_enabled=True,
            deep_judgment_search_tool=my_search_tool,
        )

        assert config.deep_judgment_search_enabled is True
        assert callable(config.deep_judgment_search_tool)
        assert config.deep_judgment_search_tool("test") == "Results for test"

    def test_validation_fails_with_invalid_string_tool(self):
        """Test that validation fails with unsupported string tool name."""
        with pytest.raises(ValueError, match="Unknown search tool: 'invalid_tool'"):
            VerificationConfig(
                answering_models=[],
                parsing_models=[_create_test_parsing_model()],
                parsing_only=True,
                deep_judgment_search_enabled=True,
                deep_judgment_search_tool="invalid_tool",
            )

    def test_validation_fails_with_non_callable_non_string(self):
        """Test that validation fails with non-callable, non-string tool."""
        with pytest.raises(ValueError, match="Search tool must be either a supported tool name string or a callable"):
            VerificationConfig(
                answering_models=[],
                parsing_models=[_create_test_parsing_model()],
                parsing_only=True,
                deep_judgment_search_enabled=True,
                deep_judgment_search_tool=123,  # Invalid: not string or callable
            )

    def test_search_disabled_skips_validation(self):
        """Test that search tool is not validated when search is disabled."""
        # Should succeed even with invalid tool when search is disabled
        config = VerificationConfig(
            answering_models=[],
            parsing_models=[_create_test_parsing_model()],
            parsing_only=True,
            deep_judgment_search_enabled=False,
            deep_judgment_search_tool="invalid_tool",  # Invalid but not validated
        )

        assert config.deep_judgment_search_enabled is False
        # No validation error raised

    def test_search_with_mixed_features(self):
        """Test search-enhanced deep-judgment alongside other features."""
        config = VerificationConfig(
            answering_models=[],
            parsing_models=[_create_test_parsing_model()],
            parsing_only=True,
            deep_judgment_enabled=True,
            deep_judgment_search_enabled=True,
            abstention_enabled=True,
            rubric_enabled=True,
            evaluation_mode="template_and_rubric",
        )

        assert config.deep_judgment_enabled is True
        assert config.deep_judgment_search_enabled is True
        assert config.abstention_enabled is True
        assert config.rubric_enabled is True
