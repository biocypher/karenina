"""Integration tests for template parsing and verification.

These tests verify that TemplateEvaluator correctly:
- Builds prompts for LLM parsing
- Parses responses into Pydantic objects
- Verifies fields against ground truth
- Handles various template types

Test scenarios:
- Simple single-field extraction
- Multi-field extraction with nested structures
- Field verification (pass/fail)
- Prompt construction
- Error handling

The tests use the fixture-backed LLM client for deterministic results.
"""

import pytest

from karenina.benchmark.verification.evaluators import (
    FieldVerificationResult,
    ParseResult,
    TemplateEvaluator,
    TemplatePromptBuilder,
)
from karenina.schemas.domain import BaseAnswer
from karenina.schemas.workflow import ModelConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def parsing_model_config() -> ModelConfig:
    """Return a ModelConfig for parsing models."""
    return ModelConfig(
        id="test-parser",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.0,
        interface="langchain",
    )


# =============================================================================
# TemplateEvaluator Initialization Tests
# =============================================================================


@pytest.mark.integration
class TestTemplateEvaluatorInitialization:
    """Test TemplateEvaluator initialization and configuration."""

    def test_init_with_valid_config(self, parsing_model_config: ModelConfig, simple_answer: type[BaseAnswer]):
        """Verify evaluator initializes with valid configuration."""
        evaluator = TemplateEvaluator(
            model_config=parsing_model_config,
            answer_class=simple_answer,
        )

        assert evaluator.model_config == parsing_model_config
        assert evaluator.answer_class == simple_answer
        assert evaluator.model_str == "anthropic/claude-haiku-4-5"

    def test_init_requires_model_config(self, simple_answer: type[BaseAnswer]):
        """Verify evaluator raises error without model config."""
        with pytest.raises(ValueError, match="Model configuration is required"):
            TemplateEvaluator(
                model_config=None,
                answer_class=simple_answer,
            )

    def test_init_requires_model_name(self, simple_answer: type[BaseAnswer]):
        """Verify evaluator raises error without model name."""
        config = ModelConfig(
            id="test",
            model_provider="anthropic",
            model_name="",  # Empty model name
        )

        with pytest.raises(ValueError, match="Model name is required"):
            TemplateEvaluator(
                model_config=config,
                answer_class=simple_answer,
            )

    def test_model_string_formats(self, simple_answer: type[BaseAnswer]):
        """Verify model string formats correctly for different interfaces."""
        from unittest.mock import MagicMock, patch

        # Mock the LLM factory to avoid API key requirements
        with patch("karenina.benchmark.verification.evaluators.template.evaluator.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()

            # Standard langchain interface
            config = ModelConfig(
                id="test",
                model_provider="anthropic",
                model_name="claude-haiku-4-5",
                interface="langchain",
            )
            evaluator = TemplateEvaluator(model_config=config, answer_class=simple_answer)
            assert evaluator.model_str == "anthropic/claude-haiku-4-5"

            # OpenRouter interface (no provider in string)
            config_openrouter = ModelConfig(
                id="test",
                model_name="anthropic/claude-haiku-4-5",
                interface="openrouter",
            )
            evaluator_or = TemplateEvaluator(model_config=config_openrouter, answer_class=simple_answer)
            assert evaluator_or.model_str == "anthropic/claude-haiku-4-5"


# =============================================================================
# Prompt Construction Tests
# =============================================================================


@pytest.mark.integration
class TestPromptConstruction:
    """Test prompt construction methods via TemplatePromptBuilder."""

    def test_build_system_prompt_base(self, simple_answer: type[BaseAnswer]):
        """Verify base system prompt contains required elements."""
        builder = TemplatePromptBuilder(answer_class=simple_answer)

        system_prompt = builder.build_system_prompt(format_instructions="<format instructions here>")

        # Check base elements
        assert "evaluator that extracts structured information" in system_prompt
        assert "Extraction Protocol" in system_prompt
        assert "Critical Rules" in system_prompt
        assert "Output Format" in system_prompt
        assert "format_instructions" in system_prompt

        # Should NOT have tool trace section by default
        assert "Tool Trace Verification" not in system_prompt

    def test_build_system_prompt_with_tool_traces(self, simple_answer: type[BaseAnswer]):
        """Verify system prompt includes tool trace section when enabled."""
        builder = TemplatePromptBuilder(answer_class=simple_answer)

        system_prompt = builder.build_system_prompt(
            format_instructions="<format instructions here>",
            has_tool_traces=True,
        )

        # Should have tool trace section
        assert "Tool Trace Verification" in system_prompt
        assert "Verify Grounding in Tool Results" in system_prompt

    def test_build_system_prompt_with_user_prompt(self, simple_answer: type[BaseAnswer]):
        """Verify system prompt includes user customizations."""
        builder = TemplatePromptBuilder(answer_class=simple_answer)

        custom_prompt = "Always be concise and extract only the most relevant data."
        system_prompt = builder.build_system_prompt(
            format_instructions="<format instructions here>",
            user_system_prompt=custom_prompt,
        )

        assert "Additional Instructions" in system_prompt
        assert custom_prompt in system_prompt

    def test_build_system_prompt_with_ground_truth(self, simple_answer: type[BaseAnswer]):
        """Verify system prompt includes ground truth when provided."""
        builder = TemplatePromptBuilder(answer_class=simple_answer)

        ground_truth = {"value": "42", "confidence": 0.95}
        system_prompt = builder.build_system_prompt(
            format_instructions="<format instructions here>",
            ground_truth=ground_truth,
        )

        assert "Ground Truth Reference" in system_prompt
        assert '"value": "42"' in system_prompt
        assert '"confidence": 0.95' in system_prompt

    def test_build_user_prompt(self, simple_answer: type[BaseAnswer]):
        """Verify user prompt construction."""
        builder = TemplatePromptBuilder(answer_class=simple_answer)

        question = "What is the answer to life, the universe, and everything?"
        response = "The answer is 42."

        user_prompt = builder.build_user_prompt(
            question_text=question,
            response_to_parse=response,
        )

        assert "ORIGINAL QUESTION" in user_prompt
        assert question in user_prompt
        assert "RESPONSE TO PARSE" in user_prompt
        assert response in user_prompt
        assert "JSON SCHEMA" in user_prompt
        assert "YOUR JSON RESPONSE" in user_prompt

    def test_user_prompt_includes_schema(self, multi_field_answer: type[BaseAnswer]):
        """Verify user prompt includes the answer class schema."""
        builder = TemplatePromptBuilder(answer_class=multi_field_answer)

        user_prompt = builder.build_user_prompt(
            question_text="Test question",
            response_to_parse="Test response",
        )

        # Should include schema fields
        assert "main_answer" in user_prompt
        assert "confidence" in user_prompt
        assert "keywords" in user_prompt


# =============================================================================
# Field Verification Tests
# =============================================================================


@pytest.mark.integration
class TestFieldVerification:
    """Test field verification against ground truth."""

    def test_verify_fields_success(self, parsing_model_config: ModelConfig, simple_answer: type[BaseAnswer]):
        """Verify field verification succeeds when values match."""
        evaluator = TemplateEvaluator(
            model_config=parsing_model_config,
            answer_class=simple_answer,
        )

        # Create answer with correct value
        parsed_answer = simple_answer(value="42")

        result = evaluator.verify_fields(parsed_answer)

        assert isinstance(result, FieldVerificationResult)
        assert result.success is True
        assert result.error is None

    def test_verify_fields_failure(self, parsing_model_config: ModelConfig, simple_answer: type[BaseAnswer]):
        """Verify field verification fails when values don't match."""
        evaluator = TemplateEvaluator(
            model_config=parsing_model_config,
            answer_class=simple_answer,
        )

        # Create answer with wrong value
        parsed_answer = simple_answer(value="wrong")

        result = evaluator.verify_fields(parsed_answer)

        assert isinstance(result, FieldVerificationResult)
        assert result.success is False

    def test_verify_fields_with_multi_field(
        self, parsing_model_config: ModelConfig, multi_field_answer: type[BaseAnswer]
    ):
        """Verify field verification with complex multi-field template."""
        evaluator = TemplateEvaluator(
            model_config=parsing_model_config,
            answer_class=multi_field_answer,
        )

        # Create citation nested object
        from tests.fixtures.templates.multi_field import Citation

        citation = Citation(identifier="[1]", page=42, url=None)

        # Create answer matching ground truth
        parsed_answer = multi_field_answer(
            main_answer="The mitochondria is the powerhouse of the cell",
            confidence=0.95,
            keywords=["mitochondria", "cell", "powerhouse", "organelle"],
            entities=["cell", "mitochondria"],
            citation=citation,
            disclaimer=None,
        )

        result = evaluator.verify_fields(parsed_answer)

        assert result.success is True

    def test_verify_fields_multi_field_wrong_value(
        self, parsing_model_config: ModelConfig, multi_field_answer: type[BaseAnswer]
    ):
        """Verify field verification fails with wrong value in multi-field."""
        evaluator = TemplateEvaluator(
            model_config=parsing_model_config,
            answer_class=multi_field_answer,
        )

        # Create answer with wrong confidence
        parsed_answer = multi_field_answer(
            main_answer="The mitochondria is the powerhouse of the cell",
            confidence=0.5,  # Wrong value
            keywords=["mitochondria", "cell", "powerhouse", "organelle"],
            entities=["cell", "mitochondria"],
            citation=None,
            disclaimer=None,
        )

        result = evaluator.verify_fields(parsed_answer)

        assert result.success is False


# =============================================================================
# Answer Template Fixture Tests
# =============================================================================


@pytest.mark.integration
class TestAnswerTemplateFixtures:
    """Test that answer template fixtures work correctly."""

    def test_simple_answer_fixture(self, simple_answer: type[BaseAnswer]):
        """Verify simple_answer fixture has correct structure."""
        assert hasattr(simple_answer, "verify")
        assert hasattr(simple_answer, "model_fields")
        assert "value" in simple_answer.model_fields

        # Test instantiation
        instance = simple_answer(value="test")
        assert instance.value == "test"

    def test_multi_field_answer_fixture(self, multi_field_answer: type[BaseAnswer]):
        """Verify multi_field_answer fixture has correct structure."""
        assert hasattr(multi_field_answer, "verify")
        assert "main_answer" in multi_field_answer.model_fields
        assert "confidence" in multi_field_answer.model_fields
        assert "keywords" in multi_field_answer.model_fields
        assert "entities" in multi_field_answer.model_fields

    def test_answer_with_correct_dict_fixture(self, answer_with_correct_dict: type[BaseAnswer]):
        """Verify answer_with_correct_dict fixture has correct structure."""
        assert hasattr(answer_with_correct_dict, "verify")
        assert "gene_name" in answer_with_correct_dict.model_fields
        assert "chromosome" in answer_with_correct_dict.model_fields
        assert "function" in answer_with_correct_dict.model_fields

        # Test instantiation and ground truth
        instance = answer_with_correct_dict(
            gene_name="BCL2",
            chromosome="18q21.33",
            function="Apoptosis regulator that inhibits programmed cell death",
        )
        assert instance.correct["gene_name"] == "BCL2"

    def test_answer_templates_dict_fixture(self, answer_templates: dict[str, type[BaseAnswer]]):
        """Verify answer_templates dict contains all templates."""
        assert "simple_extraction" in answer_templates
        assert "multi_field" in answer_templates
        assert "with_correct_dict" in answer_templates
        assert len(answer_templates) == 3


# =============================================================================
# ParseResult and FieldVerificationResult Tests
# =============================================================================


@pytest.mark.integration
class TestResultDataclasses:
    """Test result dataclass structures."""

    def test_parse_result_defaults(self):
        """Verify ParseResult has correct defaults."""
        result = ParseResult()

        assert result.parsed_answer is None
        assert result.success is False
        assert result.error is None
        assert result.deep_judgment_performed is False
        assert result.usage_metadata_list == []

    def test_parse_result_with_values(self, simple_answer: type[BaseAnswer]):
        """Verify ParseResult can store values."""
        parsed = simple_answer(value="42")
        result = ParseResult(
            parsed_answer=parsed,
            success=True,
            usage_metadata_list=[{"input_tokens": 100, "output_tokens": 50}],
        )

        assert result.parsed_answer == parsed
        assert result.success is True
        assert len(result.usage_metadata_list) == 1

    def test_field_verification_result_defaults(self):
        """Verify FieldVerificationResult has correct defaults."""
        result = FieldVerificationResult()

        assert result.success is False
        assert result.error is None

    def test_field_verification_result_with_error(self):
        """Verify FieldVerificationResult can store error."""
        result = FieldVerificationResult(
            success=False,
            error="Field 'value' does not match ground truth",
        )

        assert result.success is False
        assert "value" in result.error


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_verify_fields_with_exception(self, parsing_model_config: ModelConfig, simple_answer: type[BaseAnswer]):
        """Verify field verification handles exceptions gracefully."""
        evaluator = TemplateEvaluator(
            model_config=parsing_model_config,
            answer_class=simple_answer,
        )

        # Create a mock object that raises on verify
        class BrokenAnswer:
            def verify(self):
                raise RuntimeError("Verification crashed!")

        result = evaluator.verify_fields(BrokenAnswer())

        assert result.success is False
        assert result.error is not None
        assert "verification failed" in result.error.lower()

    def test_evaluator_with_raw_answer_class(self, parsing_model_config: ModelConfig, simple_answer: type[BaseAnswer]):
        """Verify evaluator accepts separate raw_answer_class."""
        evaluator = TemplateEvaluator(
            model_config=parsing_model_config,
            answer_class=simple_answer,
            raw_answer_class=simple_answer,
        )

        assert evaluator.answer_class == simple_answer
        assert evaluator.raw_answer_class == simple_answer

    def test_build_user_prompt_with_long_response(self, simple_answer: type[BaseAnswer]):
        """Verify user prompt handles long responses."""
        builder = TemplatePromptBuilder(answer_class=simple_answer)

        long_response = "A" * 10000  # 10k character response

        user_prompt = builder.build_user_prompt(
            question_text="Test question",
            response_to_parse=long_response,
        )

        # Should contain the full response
        assert long_response in user_prompt

    def test_build_system_prompt_all_sections(self, simple_answer: type[BaseAnswer]):
        """Verify system prompt with all optional sections enabled."""
        builder = TemplatePromptBuilder(answer_class=simple_answer)

        system_prompt = builder.build_system_prompt(
            format_instructions="<format instructions here>",
            user_system_prompt="Custom instructions",
            has_tool_traces=True,
            ground_truth={"value": "42"},
        )

        # All sections should be present
        assert "Extraction Protocol" in system_prompt
        assert "Tool Trace Verification" in system_prompt
        assert "Additional Instructions" in system_prompt
        assert "Ground Truth Reference" in system_prompt
        assert "Output Format" in system_prompt


# =============================================================================
# Template Verification Integration Tests
# =============================================================================


@pytest.mark.integration
class TestTemplateVerificationIntegration:
    """Test full template verification flow."""

    def test_simple_answer_verification_flow(self, parsing_model_config: ModelConfig, simple_answer: type[BaseAnswer]):
        """Test complete verification flow with simple answer."""
        evaluator = TemplateEvaluator(
            model_config=parsing_model_config,
            answer_class=simple_answer,
        )

        # Create correct answer
        correct_answer = simple_answer(value="42")

        # Verify it passes
        result = evaluator.verify_fields(correct_answer)
        assert result.success is True

        # Create incorrect answer
        wrong_answer = simple_answer(value="24")

        # Verify it fails
        result = evaluator.verify_fields(wrong_answer)
        assert result.success is False

    def test_gene_template_verification(
        self, parsing_model_config: ModelConfig, answer_with_correct_dict: type[BaseAnswer]
    ):
        """Test verification with gene information template."""
        evaluator = TemplateEvaluator(
            model_config=parsing_model_config,
            answer_class=answer_with_correct_dict,
        )

        # Create answer matching ground truth
        correct_answer = answer_with_correct_dict(
            gene_name="BCL2",
            chromosome="18q21.33",
            function="Apoptosis regulator that inhibits programmed cell death",
            synonyms=["Bcl-2"],
            omim_id=151430,
        )

        result = evaluator.verify_fields(correct_answer)
        assert result.success is True

    def test_gene_template_case_insensitive(
        self, parsing_model_config: ModelConfig, answer_with_correct_dict: type[BaseAnswer]
    ):
        """Test that gene name verification is case-insensitive."""
        evaluator = TemplateEvaluator(
            model_config=parsing_model_config,
            answer_class=answer_with_correct_dict,
        )

        # Gene name in different case
        answer = answer_with_correct_dict(
            gene_name="bcl2",  # lowercase
            chromosome="18q21.33",
            function="Apoptosis regulator that inhibits programmed cell death",
        )

        result = evaluator.verify_fields(answer)
        assert result.success is True

    def test_gene_template_wrong_chromosome(
        self, parsing_model_config: ModelConfig, answer_with_correct_dict: type[BaseAnswer]
    ):
        """Test that wrong chromosome fails verification."""
        evaluator = TemplateEvaluator(
            model_config=parsing_model_config,
            answer_class=answer_with_correct_dict,
        )

        # Wrong chromosome
        answer = answer_with_correct_dict(
            gene_name="BCL2",
            chromosome="17p13.1",  # Wrong chromosome
            function="Apoptosis regulator that inhibits programmed cell death",
        )

        result = evaluator.verify_fields(answer)
        assert result.success is False
