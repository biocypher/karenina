"""Tests for deep-judgment multi-stage parser.

This module tests the deep_judgment_parse function and its three stages:
1. Excerpt extraction
2. Reasoning generation
3. Parameter extraction
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import Field

from karenina.benchmark.models import ModelConfig, VerificationConfig
from karenina.benchmark.verification.deep_judgment import deep_judgment_parse
from karenina.schemas.answer_class import BaseAnswer


# Test answer template (use DrugAnswer to avoid pytest collection warning)
class DrugAnswer(BaseAnswer):
    """Simple test answer template."""

    drug_target: str = Field(description="The drug target")
    mechanism: str = Field(description="Mechanism of action")
    confidence: str = Field(description="Confidence level")


@pytest.fixture
def test_config():
    """Create test verification config."""
    return VerificationConfig(
        answering_models=[],
        parsing_models=[
            ModelConfig(
                id="test-parser",
                model_provider="openai",
                model_name="gpt-4.1-mini",
                temperature=0.0,
                system_prompt="You are a helpful assistant.",
            )
        ],
        parsing_only=True,
        deep_judgment_enabled=True,
        deep_judgment_max_excerpts_per_attribute=3,
        deep_judgment_fuzzy_match_threshold=0.80,
        deep_judgment_excerpt_retry_attempts=2,
    )


@pytest.fixture
def mock_parsing_llm():
    """Create mock parsing LLM."""
    return MagicMock()


class TestDeepJudgmentParseFullWorkflow:
    """Tests for complete deep-judgment workflow."""

    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_successful_full_workflow(self, mock_invoke, test_config, mock_parsing_llm):
        """Test successful execution of all three stages."""
        # Mock Stage 1: Excerpt extraction (return JSON string)
        excerpt_response = json.dumps(
            {
                "drug_target": [{"text": "targets BCL-2 protein", "confidence": "high"}],
                "mechanism": [{"text": "induces apoptosis", "confidence": "medium"}],
                "confidence": [{"text": "shown to be effective", "confidence": "low"}],
            }
        )

        # Mock Stage 2: Reasoning generation (return JSON string)
        reasoning_response = json.dumps(
            {
                "drug_target": "The excerpt clearly states BCL-2 as the target.",
                "mechanism": "The text indicates apoptosis induction as the mechanism.",
                "confidence": "The phrase 'shown to be effective' suggests some confidence.",
            }
        )

        # Mock Stage 3: Parameter extraction (return JSON string)
        parameter_response = json.dumps({"drug_target": "BCL-2", "mechanism": "apoptosis", "confidence": "medium"})

        # Set up mock to return string responses for each call
        mock_invoke.side_effect = [
            excerpt_response,  # Stage 1
            reasoning_response,  # Stage 2
            parameter_response,  # Stage 3
        ]

        # Execute
        raw_trace = "The drug targets BCL-2 protein and induces apoptosis. It has shown to be effective."
        question = "What is the drug target?"

        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response=raw_trace,
            RawAnswer=DrugAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text=question,
            config=test_config,
            format_instructions="test instructions",
            combined_system_prompt="test prompt",
        )

        # Verify parsed answer
        assert parsed.drug_target == "BCL-2"
        assert parsed.mechanism == "apoptosis"
        assert parsed.confidence == "medium"

        # Verify excerpts
        assert len(excerpts["drug_target"]) == 1
        assert excerpts["drug_target"][0]["text"] == "targets BCL-2 protein"
        assert excerpts["drug_target"][0]["confidence"] == "high"
        assert excerpts["drug_target"][0]["similarity_score"] >= 0.75

        # Verify reasoning
        assert "BCL-2" in reasoning["drug_target"]
        assert "apoptosis" in reasoning["mechanism"]

        # Verify metadata
        assert metadata["stages_completed"] == ["excerpts", "reasoning", "parameters"]
        assert metadata["model_calls"] == 3
        assert metadata["excerpt_retry_count"] == 0

    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_workflow_with_missing_excerpts(self, mock_invoke, test_config, mock_parsing_llm):
        """Test workflow when some attributes have no excerpts (refusal scenario)."""
        # Mock Stage 1: Some attributes have no excerpts (return JSON string)
        excerpt_response = json.dumps(
            {
                "drug_target": [],  # No excerpts found (refusal)
                "mechanism": [{"text": "some mechanism text", "confidence": "medium"}],
                "confidence": [],  # No excerpts found
            }
        )

        # Mock Stage 2: Reasoning explains why no excerpts (return JSON string)
        reasoning_response = json.dumps(
            {
                "drug_target": "The response contains a refusal and provides no drug target information.",
                "mechanism": "The mechanism is briefly mentioned.",
                "confidence": "No confidence information present in the response.",
            }
        )

        # Mock Stage 3: Parameter extraction (return JSON string)
        parameter_response = json.dumps({"drug_target": "unknown", "mechanism": "unclear", "confidence": "low"})

        mock_invoke.side_effect = [
            excerpt_response,
            reasoning_response,
            parameter_response,
        ]

        # Execute
        raw_trace = "I cannot provide information about that drug. Some mechanism text is mentioned."
        question = "What is the drug target?"

        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response=raw_trace,
            RawAnswer=DrugAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text=question,
            config=test_config,
            format_instructions="test instructions",
            combined_system_prompt="test prompt",
        )

        # Verify empty excerpt lists are preserved
        assert excerpts["drug_target"] == []
        assert excerpts["confidence"] == []
        assert len(excerpts["mechanism"]) == 1

        # Verify attributes_without_excerpts tracked
        assert "drug_target" in metadata["attributes_without_excerpts"]
        assert "confidence" in metadata["attributes_without_excerpts"]

        # Verify reasoning still generated even without excerpts
        assert "refusal" in reasoning["drug_target"].lower()

    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    @patch("karenina.benchmark.verification.deep_judgment.fuzzy_match_excerpt")
    def test_excerpt_validation_retry(self, mock_fuzzy_match, mock_invoke, test_config, mock_parsing_llm):
        """Test that invalid excerpts trigger retry with error feedback."""
        # First attempt: Invalid excerpt (low similarity) - return JSON string
        excerpt_response_1 = json.dumps(
            {
                "drug_target": [{"text": "hallucinated excerpt", "confidence": "high"}],
                "mechanism": [{"text": "valid excerpt", "confidence": "medium"}],
                "confidence": [],
            }
        )

        # Second attempt: Valid excerpt after retry - return JSON string
        excerpt_response_2 = json.dumps(
            {
                "drug_target": [{"text": "actual excerpt from trace", "confidence": "high"}],
                "mechanism": [{"text": "valid excerpt", "confidence": "medium"}],
                "confidence": [],
            }
        )

        # Mock other stages - return JSON strings
        reasoning_response = json.dumps(
            {"drug_target": "reasoning", "mechanism": "reasoning", "confidence": "reasoning"}
        )

        parameter_response = json.dumps({"drug_target": "test", "mechanism": "test", "confidence": "low"})

        mock_invoke.side_effect = [
            excerpt_response_1,  # Stage 1 attempt 1 (will fail validation)
            excerpt_response_2,  # Stage 1 attempt 2 (will succeed)
            reasoning_response,  # Stage 2
            parameter_response,  # Stage 3
        ]

        # Mock fuzzy matching: first excerpt fails, second succeeds
        def fuzzy_side_effect(excerpt, trace):
            if "hallucinated" in excerpt:
                return False, 0.3  # Low similarity (will trigger retry)
            else:
                return True, 0.95  # High similarity (will pass)

        mock_fuzzy_match.side_effect = fuzzy_side_effect

        # Execute
        raw_trace = "The actual excerpt from trace is present here. Valid excerpt is also here."
        question = "Test question"

        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response=raw_trace,
            RawAnswer=DrugAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text=question,
            config=test_config,
            format_instructions="test instructions",
            combined_system_prompt="test prompt",
        )

        # Verify retry occurred
        assert metadata["excerpt_retry_count"] == 1
        assert metadata["model_calls"] == 4  # 2 for stage 1 (with retry) + 1 for stage 2 + 1 for stage 3

        # Verify final excerpts are valid
        assert excerpts["drug_target"][0]["text"] == "actual excerpt from trace"

    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_invalid_confidence_level_defaults_to_medium(self, mock_invoke, test_config, mock_parsing_llm):
        """Test that invalid confidence levels are defaulted to 'medium'."""
        # Mock with invalid confidence level - return JSON strings
        excerpt_response = json.dumps(
            {
                "drug_target": [{"text": "excerpt text", "confidence": "invalid_level"}],
                "mechanism": [{"text": "another excerpt", "confidence": "high"}],
                "confidence": [],
            }
        )

        reasoning_response = json.dumps({"drug_target": "r", "mechanism": "r", "confidence": "r"})

        parameter_response = json.dumps({"drug_target": "t", "mechanism": "t", "confidence": "low"})

        mock_invoke.side_effect = [
            excerpt_response,
            reasoning_response,
            parameter_response,
        ]

        # Execute
        raw_trace = "excerpt text and another excerpt are here"

        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response=raw_trace,
            RawAnswer=DrugAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text="test",
            config=test_config,
            format_instructions="test",
            combined_system_prompt="test",
        )

        # Verify invalid confidence was defaulted to medium
        assert excerpts["drug_target"][0]["confidence"] == "medium"
        assert excerpts["mechanism"][0]["confidence"] == "high"  # Valid confidence unchanged

    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_json_parsing_failure_raises_error(self, mock_invoke, test_config, mock_parsing_llm):
        """Test that JSON parsing failures raise appropriate error after retries."""
        # Mock all attempts return invalid JSON - return string
        invalid_response = "This is not valid JSON at all"

        mock_invoke.side_effect = [
            invalid_response,  # Attempt 1
            invalid_response,  # Attempt 2 (first retry)
            invalid_response,  # Attempt 3 (second retry)
        ]

        # Execute - should raise ValueError after exhausting retries
        with pytest.raises(ValueError, match="Failed to parse excerpt JSON after"):
            deep_judgment_parse(
                raw_llm_response="test trace",
                RawAnswer=DrugAnswer,
                parsing_model=test_config.parsing_models[0],
                parsing_llm=mock_parsing_llm,
                question_text="test",
                config=test_config,
                format_instructions="test",
                combined_system_prompt="test",
            )

    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_reasoning_json_failure_handled_gracefully(self, mock_invoke, test_config, mock_parsing_llm):
        """Test that reasoning JSON failures are handled gracefully (empty dict)."""
        # Mock Stage 1: Valid excerpts - return JSON string
        excerpt_response = json.dumps(
            {"drug_target": [{"text": "test", "confidence": "high"}], "mechanism": [], "confidence": []}
        )

        # Mock Stage 2: Invalid JSON (should be handled gracefully) - return string
        reasoning_response = "Invalid JSON for reasoning"

        # Mock Stage 3: Valid parameters - return JSON string
        parameter_response = json.dumps({"drug_target": "t", "mechanism": "m", "confidence": "c"})

        mock_invoke.side_effect = [
            excerpt_response,
            reasoning_response,
            parameter_response,
        ]

        # Execute - should not raise error
        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response="test trace with test text",
            RawAnswer=DrugAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text="test",
            config=test_config,
            format_instructions="test",
            combined_system_prompt="test",
        )

        # Verify reasoning is empty dict (graceful failure)
        assert reasoning == {}

        # Verify other stages still completed
        assert "excerpts" in metadata["stages_completed"]
        assert "reasoning" in metadata["stages_completed"]
        assert "parameters" in metadata["stages_completed"]


class TestDeepJudgmentParseIntegration:
    """Integration-style tests using more realistic scenarios."""

    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_complex_answer_template(self, mock_invoke, test_config, mock_parsing_llm):
        """Test with complex answer template with many attributes."""

        class ComplexAnswer(BaseAnswer):
            drug_name: str
            drug_target: str
            mechanism: str
            indication: str
            phase: str
            confidence: str

        # Mock responses for all 6 attributes - return JSON strings
        excerpt_response = json.dumps(
            {
                "drug_name": [{"text": "venetoclax", "confidence": "high"}],
                "drug_target": [{"text": "BCL-2", "confidence": "high"}],
                "mechanism": [{"text": "apoptosis", "confidence": "medium"}],
                "indication": [{"text": "CLL treatment", "confidence": "medium"}],
                "phase": [],  # No excerpt
                "confidence": [{"text": "promising results", "confidence": "low"}],
            }
        )

        reasoning_response = json.dumps(
            {
                "drug_name": "Drug name is venetoclax",
                "drug_target": "Target is BCL-2",
                "mechanism": "Mechanism is apoptosis",
                "indication": "Used for CLL",
                "phase": "No phase information provided",
                "confidence": "Shows promise",
            }
        )

        parameter_response = json.dumps(
            {
                "drug_name": "venetoclax",
                "drug_target": "BCL-2",
                "mechanism": "apoptosis induction",
                "indication": "CLL",
                "phase": "unknown",
                "confidence": "medium",
            }
        )

        mock_invoke.side_effect = [excerpt_response, reasoning_response, parameter_response]

        # Execute
        raw_trace = "venetoclax targets BCL-2 and induces apoptosis for CLL treatment with promising results"

        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response=raw_trace,
            RawAnswer=ComplexAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text="What is the drug information?",
            config=test_config,
            format_instructions="test",
            combined_system_prompt="test",
        )

        # Verify all 6 attributes processed
        assert len(excerpts) == 6
        assert len(reasoning) == 6

        # Verify empty excerpt list for phase
        assert excerpts["phase"] == []
        assert "phase" in metadata["attributes_without_excerpts"]

        # Verify parsed answer has all attributes
        assert parsed.drug_name == "venetoclax"
        assert parsed.phase == "unknown"


class TestDeepJudgmentSearchEnhancement:
    """Tests for search-enhanced deep-judgment feature."""

    @patch("karenina.benchmark.verification.deep_judgment.create_search_tool")
    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_search_enhancement_enabled(self, mock_invoke, mock_create_search, test_config, mock_parsing_llm):
        """Test that search is performed when enabled and results added to excerpts."""
        # Enable search in config
        test_config.deep_judgment_search_enabled = True
        test_config.deep_judgment_search_tool = "tavily"

        # Create mock search tool
        mock_search_tool = MagicMock()
        mock_search_tool.return_value = [
            "Search result for BCL-2 target",
            "Search result for apoptosis mechanism",
        ]
        mock_create_search.return_value = mock_search_tool

        # Mock LLM responses (3 stages)
        excerpt_response = json.dumps(
            {
                "drug_target": [{"text": "targets BCL-2 protein", "confidence": "high"}],
                "mechanism": [{"text": "induces apoptosis", "confidence": "medium"}],
                "confidence": [],
            }
        )
        reasoning_response = json.dumps(
            {"drug_target": "Target is BCL-2", "mechanism": "Mechanism is apoptosis", "confidence": "No info"}
        )
        parameter_response = json.dumps({"drug_target": "BCL-2", "mechanism": "apoptosis", "confidence": "low"})

        mock_invoke.side_effect = [excerpt_response, reasoning_response, parameter_response]

        # Execute
        raw_trace = "The drug targets BCL-2 protein and induces apoptosis."
        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response=raw_trace,
            RawAnswer=DrugAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text="Test question",
            config=test_config,
            format_instructions="test",
            combined_system_prompt="test",
        )

        # Verify search tool was created with correct tool name
        mock_create_search.assert_called_once_with("tavily")

        # Verify search was called with both excerpts
        mock_search_tool.assert_called_once()
        call_args = mock_search_tool.call_args[0][0]
        assert len(call_args) == 2
        assert "targets BCL-2 protein" in call_args
        assert "induces apoptosis" in call_args

        # Verify search results were added to excerpts
        assert "search_results" in excerpts["drug_target"][0]
        assert excerpts["drug_target"][0]["search_results"] == "Search result for BCL-2 target"
        assert "search_results" in excerpts["mechanism"][0]
        assert excerpts["mechanism"][0]["search_results"] == "Search result for apoptosis mechanism"

    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_search_disabled_skips_search(self, mock_invoke, test_config, mock_parsing_llm):
        """Test that search is skipped when disabled."""
        # Ensure search is disabled (default)
        assert test_config.deep_judgment_search_enabled is False

        # Mock LLM responses
        excerpt_response = json.dumps(
            {"drug_target": [{"text": "excerpt", "confidence": "high"}], "mechanism": [], "confidence": []}
        )
        reasoning_response = json.dumps({"drug_target": "reasoning", "mechanism": "r", "confidence": "r"})
        parameter_response = json.dumps({"drug_target": "test", "mechanism": "m", "confidence": "c"})

        mock_invoke.side_effect = [excerpt_response, reasoning_response, parameter_response]

        # Execute
        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response="excerpt is here",
            RawAnswer=DrugAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text="test",
            config=test_config,
            format_instructions="test",
            combined_system_prompt="test",
        )

        # Verify excerpts do NOT have search_results field
        assert "search_results" not in excerpts["drug_target"][0]

    @patch("karenina.benchmark.verification.deep_judgment.create_search_tool")
    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_search_skips_empty_excerpts(self, mock_invoke, mock_create_search, test_config, mock_parsing_llm):
        """Test that search skips empty excerpts (confidence='none')."""
        # Enable search
        test_config.deep_judgment_search_enabled = True

        # Create mock search tool
        mock_search_tool = MagicMock()
        mock_search_tool.return_value = ["Search result for BCL-2"]
        mock_create_search.return_value = mock_search_tool

        # Mock responses with one empty excerpt
        excerpt_response = json.dumps(
            {
                "drug_target": [{"text": "targets BCL-2", "confidence": "high"}],
                "mechanism": [{"text": "", "confidence": "none", "explanation": "No mechanism mentioned"}],
                "confidence": [],
            }
        )
        reasoning_response = json.dumps({"drug_target": "r", "mechanism": "r", "confidence": "r"})
        parameter_response = json.dumps({"drug_target": "t", "mechanism": "m", "confidence": "c"})

        mock_invoke.side_effect = [excerpt_response, reasoning_response, parameter_response]

        # Execute
        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response="targets BCL-2 is mentioned",
            RawAnswer=DrugAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text="test",
            config=test_config,
            format_instructions="test",
            combined_system_prompt="test",
        )

        # Verify search was only called for non-empty excerpt
        mock_search_tool.assert_called_once()
        call_args = mock_search_tool.call_args[0][0]
        assert len(call_args) == 1  # Only one excerpt searched
        assert call_args[0] == "targets BCL-2"

        # Verify empty excerpt does NOT have search_results
        assert "search_results" in excerpts["drug_target"][0]
        assert "search_results" not in excerpts["mechanism"][0]

    @patch("karenina.benchmark.verification.deep_judgment.create_search_tool")
    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_search_failure_continues_pipeline(self, mock_invoke, mock_create_search, test_config, mock_parsing_llm):
        """Test that search failures don't break the pipeline."""
        # Enable search
        test_config.deep_judgment_search_enabled = True

        # Mock search tool that raises exception
        mock_search_tool = MagicMock()
        mock_search_tool.side_effect = Exception("Search API failure")
        mock_create_search.return_value = mock_search_tool

        # Mock LLM responses
        excerpt_response = json.dumps(
            {"drug_target": [{"text": "excerpt", "confidence": "high"}], "mechanism": [], "confidence": []}
        )
        reasoning_response = json.dumps({"drug_target": "r", "mechanism": "r", "confidence": "r"})
        parameter_response = json.dumps({"drug_target": "t", "mechanism": "m", "confidence": "c"})

        mock_invoke.side_effect = [excerpt_response, reasoning_response, parameter_response]

        # Execute - should not raise exception
        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response="excerpt is here",
            RawAnswer=DrugAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text="test",
            config=test_config,
            format_instructions="test",
            combined_system_prompt="test",
        )

        # Verify parsing completed successfully despite search failure
        assert parsed.drug_target == "t"
        assert metadata["stages_completed"] == ["excerpts", "reasoning", "parameters"]

        # Verify excerpts do NOT have search_results (due to failure)
        assert "search_results" not in excerpts["drug_target"][0]

    @patch("karenina.benchmark.verification.deep_judgment.create_search_tool")
    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_search_with_custom_callable(self, mock_invoke, mock_create_search, test_config, mock_parsing_llm):
        """Test search with custom callable tool."""

        # Custom search function
        def custom_search(query):
            return [f"Custom result for: {q}" for q in query]

        # Enable search with callable
        test_config.deep_judgment_search_enabled = True
        test_config.deep_judgment_search_tool = custom_search

        # Mock search tool creation
        mock_search_tool = MagicMock()
        mock_search_tool.return_value = ["Custom search result 1", "Custom search result 2"]
        mock_create_search.return_value = mock_search_tool

        # Mock LLM responses
        excerpt_response = json.dumps(
            {
                "drug_target": [{"text": "BCL-2", "confidence": "high"}],
                "mechanism": [{"text": "apoptosis", "confidence": "high"}],
                "confidence": [],
            }
        )
        reasoning_response = json.dumps({"drug_target": "r", "mechanism": "r", "confidence": "r"})
        parameter_response = json.dumps({"drug_target": "t", "mechanism": "m", "confidence": "c"})

        mock_invoke.side_effect = [excerpt_response, reasoning_response, parameter_response]

        # Execute
        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response="BCL-2 and apoptosis mentioned",
            RawAnswer=DrugAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text="test",
            config=test_config,
            format_instructions="test",
            combined_system_prompt="test",
        )

        # Verify custom callable was passed to factory
        mock_create_search.assert_called_once_with(custom_search)

        # Verify search results added
        assert "search_results" in excerpts["drug_target"][0]
        assert "search_results" in excerpts["mechanism"][0]

    @patch("karenina.benchmark.verification.deep_judgment.create_search_tool")
    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_stage2_reasoning_with_search_context(self, mock_invoke, mock_create_search, test_config, mock_parsing_llm):
        """Test Stage 2 reasoning includes hallucination assessment when search is enabled."""
        # Enable search
        test_config.deep_judgment_search_enabled = True

        # Mock search tool
        mock_search_tool = MagicMock()
        mock_search_tool.return_value = ["Search confirms BCL-2", "Search confirms apoptosis"]
        mock_create_search.return_value = mock_search_tool

        # Mock Stage 1: Excerpts
        excerpt_response = json.dumps(
            {
                "drug_target": [{"text": "targets BCL-2", "confidence": "high"}],
                "mechanism": [{"text": "apoptosis", "confidence": "medium"}],
                "confidence": [],
            }
        )

        # Mock Stage 2: Reasoning WITH hallucination confidence (nested format)
        reasoning_response = json.dumps(
            {
                "drug_target": {
                    "reasoning": "BCL-2 target confirmed by search results",
                    "hallucination_confidence": "high",
                },
                "mechanism": {
                    "reasoning": "Apoptosis mechanism supported by search",
                    "hallucination_confidence": "medium",
                },
                "confidence": {"reasoning": "No information provided", "hallucination_confidence": "none"},
            }
        )

        # Mock Stage 3: Parameters
        parameter_response = json.dumps({"drug_target": "BCL-2", "mechanism": "apoptosis", "confidence": "low"})

        mock_invoke.side_effect = [excerpt_response, reasoning_response, parameter_response]

        # Execute
        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response="targets BCL-2 and apoptosis",
            RawAnswer=DrugAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text="test",
            config=test_config,
            format_instructions="test",
            combined_system_prompt="test",
        )

        # Verify reasoning was extracted correctly
        assert reasoning["drug_target"] == "BCL-2 target confirmed by search results"
        assert reasoning["mechanism"] == "Apoptosis mechanism supported by search"

        # Verify hallucination confidence was extracted and stored in metadata
        assert "hallucination_confidence" in metadata
        assert metadata["hallucination_confidence"]["drug_target"] == "high"
        assert metadata["hallucination_confidence"]["mechanism"] == "medium"
        assert metadata["hallucination_confidence"]["confidence"] == "none"

    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_stage2_reasoning_without_search_backward_compatible(self, mock_invoke, test_config, mock_parsing_llm):
        """Test Stage 2 reasoning without search uses simple format (backward compatible)."""
        # Ensure search is disabled
        assert test_config.deep_judgment_search_enabled is False

        # Mock Stage 1: Excerpts (without search_results)
        excerpt_response = json.dumps(
            {"drug_target": [{"text": "excerpt", "confidence": "high"}], "mechanism": [], "confidence": []}
        )

        # Mock Stage 2: Simple string format (no nested structure)
        reasoning_response = json.dumps(
            {
                "drug_target": "Simple reasoning text",
                "mechanism": "Another reasoning",
                "confidence": "No info",
            }
        )

        # Mock Stage 3: Parameters
        parameter_response = json.dumps({"drug_target": "t", "mechanism": "m", "confidence": "c"})

        mock_invoke.side_effect = [excerpt_response, reasoning_response, parameter_response]

        # Execute
        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response="excerpt here",
            RawAnswer=DrugAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text="test",
            config=test_config,
            format_instructions="test",
            combined_system_prompt="test",
        )

        # Verify reasoning is simple strings
        assert reasoning["drug_target"] == "Simple reasoning text"
        assert reasoning["mechanism"] == "Another reasoning"

        # Verify NO hallucination confidence in metadata
        assert "hallucination_confidence" not in metadata

    @patch("karenina.benchmark.verification.deep_judgment.create_search_tool")
    @patch("karenina.benchmark.verification.deep_judgment._invoke_llm_with_retry")
    def test_stage2_reasoning_fallback_on_malformed_nested_format(
        self, mock_invoke, mock_create_search, test_config, mock_parsing_llm
    ):
        """Test Stage 2 reasoning falls back gracefully if LLM returns wrong format."""
        # Enable search
        test_config.deep_judgment_search_enabled = True

        # Mock search tool
        mock_search_tool = MagicMock()
        mock_search_tool.return_value = ["Search result"]
        mock_create_search.return_value = mock_search_tool

        # Mock Stage 1: Excerpts with search results
        excerpt_response = json.dumps(
            {"drug_target": [{"text": "excerpt", "confidence": "high"}], "mechanism": [], "confidence": []}
        )

        # Mock Stage 2: LLM returned STRING instead of nested dict (malformed)
        reasoning_response = json.dumps(
            {
                "drug_target": "This should be a dict but is a string",  # Wrong format!
                "mechanism": "Another string",
                "confidence": "String",
            }
        )

        # Mock Stage 3
        parameter_response = json.dumps({"drug_target": "t", "mechanism": "m", "confidence": "c"})

        mock_invoke.side_effect = [excerpt_response, reasoning_response, parameter_response]

        # Execute - should not crash
        parsed, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response="excerpt here",
            RawAnswer=DrugAnswer,
            parsing_model=test_config.parsing_models[0],
            parsing_llm=mock_parsing_llm,
            question_text="test",
            config=test_config,
            format_instructions="test",
            combined_system_prompt="test",
        )

        # Verify fallback: reasoning extracted as strings
        assert reasoning["drug_target"] == "This should be a dict but is a string"

        # Verify fallback: hallucination_confidence defaults to "none"
        assert metadata["hallucination_confidence"]["drug_target"] == "none"
