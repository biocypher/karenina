"""Integration test for reasoning-only deep judgment with a real LLM call.

This test verifies that reasoning-only mode works end-to-end against an actual
Anthropic API, confirming that the prompts produce valid reasoning output and
that the metadata reflects the reasoning-only path correctly.

Tests are marked with:
- @pytest.mark.integration: Tests that combine multiple components
- @pytest.mark.deep_judgment: Tests specific to deep judgment

Run with: pytest tests/integration/test_deep_judgment_reasoning_only_api.py -v --timeout=120

Requires: ANTHROPIC_API_KEY environment variable to be set.
"""

import os

import pytest
from pydantic import Field

from karenina.adapters.factory import get_llm, get_parser
from karenina.benchmark.verification.evaluators.template.deep_judgment import (
    deep_judgment_parse,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import BaseAnswer
from karenina.schemas.verification import VerificationConfig

# Skip the entire module if no Anthropic API key is available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.deep_judgment,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set; skipping real API tests",
    ),
]

# Use a cheap, fast model for integration testing
MODEL_NAME = "claude-haiku-4-5-20251001"
MODEL_PROVIDER = "anthropic"
INTERFACE = "langchain"


# =============================================================================
# Answer template
# =============================================================================


class DrugAnswer(BaseAnswer):
    """Simple answer template for drug-related questions.

    Has two string fields and one boolean, providing enough structure
    to exercise reasoning without being expensive to parse.
    """

    id: str = ""
    drug_target: str = Field(default="", description="The protein target of the drug")
    mechanism: str = Field(default="", description="The mechanism of action of the drug")
    is_approved: bool = Field(default=False, description="Whether the drug is FDA-approved")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def parsing_model() -> ModelConfig:
    """Create a ModelConfig pointing at a cheap Anthropic model."""
    return ModelConfig(
        id=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        model_name=MODEL_NAME,
        temperature=0.0,
        interface=INTERFACE,
        system_prompt="You are an expert biomedical evaluator.",
    )


@pytest.fixture
def reasoning_only_config(parsing_model: ModelConfig) -> VerificationConfig:
    """Create a VerificationConfig with reasoning-only deep judgment enabled."""
    return VerificationConfig(
        answering_models=[parsing_model],
        parsing_models=[parsing_model],
        deep_judgment_mode="reasoning_only",
    )


SAMPLE_RESPONSE = (
    "Venetoclax (ABT-199) is a selective BCL-2 inhibitor. It works by binding to the "
    "BH3-binding groove of BCL-2, displacing pro-apoptotic proteins like BIM and thereby "
    "restoring apoptosis in cancer cells that overexpress BCL-2. Venetoclax received FDA "
    "approval in 2016 for the treatment of chronic lymphocytic leukemia (CLL) with 17p "
    "deletion. The drug specifically targets the anti-apoptotic protein BCL-2, which is "
    "overexpressed in many B-cell malignancies."
)

QUESTION_TEXT = "What is the drug target and mechanism of action of venetoclax? Is the drug FDA-approved?"


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.deep_judgment
class TestReasoningOnlyRealAPI:
    """End-to-end tests for reasoning-only deep judgment with a real LLM."""

    def test_reasoning_only_produces_valid_output(
        self,
        parsing_model: ModelConfig,
        reasoning_only_config: VerificationConfig,
    ):
        """Call deep_judgment_parse with reasoning_only=True against a real LLM.

        Verifies:
        - Excerpts dict is empty (excerpt extraction skipped)
        - Reasoning is populated for each template attribute
        - Each reasoning entry is non-trivial (len > 10)
        - "reasoning" is in metadata["stages_completed"]
        - "excerpts" is NOT in metadata["stages_completed"]
        - metadata["reasoning_only"] is True
        """
        llm = get_llm(parsing_model)
        parser = get_parser(parsing_model)

        parsed_answer, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response=SAMPLE_RESPONSE,
            RawAnswer=DrugAnswer,
            parsing_model=parsing_model,
            parsing_llm=llm,
            parser=parser,
            question_text=QUESTION_TEXT,
            config=reasoning_only_config,
            format_instructions="",
            combined_system_prompt="You are an expert biomedical evaluator.",
        )

        # 1. Excerpts dict must be empty (not extracted in reasoning-only mode)
        assert excerpts == {}, f"Excerpts should be empty in reasoning-only mode, got: {excerpts}"

        # 2. Reasoning is populated for each template attribute
        expected_attributes = ["drug_target", "mechanism", "is_approved"]
        for attr in expected_attributes:
            assert attr in reasoning, f"Reasoning missing for attribute '{attr}'. Got keys: {list(reasoning.keys())}"

        # 3. Each reasoning entry is non-trivial (len > 10)
        for attr in expected_attributes:
            reasoning_text = reasoning[attr]
            assert len(reasoning_text) > 10, (
                f"Reasoning for '{attr}' is too short (len={len(reasoning_text)}): {reasoning_text}"
            )

        # 4. "reasoning" is in metadata["stages_completed"]
        assert "reasoning" in metadata["stages_completed"], (
            f"Expected 'reasoning' in stages_completed, got: {metadata['stages_completed']}"
        )

        # 5. "excerpts" is NOT in metadata["stages_completed"]
        assert "excerpts" not in metadata["stages_completed"], (
            f"Expected 'excerpts' NOT in stages_completed for reasoning-only mode, got: {metadata['stages_completed']}"
        )

        # 6. metadata["reasoning_only"] is True
        assert metadata["reasoning_only"] is True, (
            f"Expected metadata['reasoning_only'] to be True, got: {metadata.get('reasoning_only')}"
        )

    def test_parsed_answer_has_populated_fields(
        self,
        parsing_model: ModelConfig,
        reasoning_only_config: VerificationConfig,
    ):
        """Verify the parsed answer has non-empty values for key fields.

        The sample response clearly mentions BCL-2, inhibition, and FDA approval,
        so the parser should extract meaningful values.
        """
        llm = get_llm(parsing_model)
        parser = get_parser(parsing_model)

        parsed_answer, _, _, _ = deep_judgment_parse(
            raw_llm_response=SAMPLE_RESPONSE,
            RawAnswer=DrugAnswer,
            parsing_model=parsing_model,
            parsing_llm=llm,
            parser=parser,
            question_text=QUESTION_TEXT,
            config=reasoning_only_config,
            format_instructions="",
            combined_system_prompt="You are an expert biomedical evaluator.",
        )

        # The answer should have populated fields
        assert parsed_answer.drug_target != "", "drug_target should be non-empty"
        assert parsed_answer.mechanism != "", "mechanism should be non-empty"

        # The response clearly states FDA approval
        assert parsed_answer.is_approved is True, "is_approved should be True (venetoclax is FDA-approved)"

    def test_model_calls_count(
        self,
        parsing_model: ModelConfig,
        reasoning_only_config: VerificationConfig,
    ):
        """Verify reasoning-only mode uses exactly 2 model calls.

        Reasoning-only should make:
        1. One call for reasoning generation
        2. One call for parameter extraction via ParserPort
        """
        llm = get_llm(parsing_model)
        parser = get_parser(parsing_model)

        _, _, _, metadata = deep_judgment_parse(
            raw_llm_response=SAMPLE_RESPONSE,
            RawAnswer=DrugAnswer,
            parsing_model=parsing_model,
            parsing_llm=llm,
            parser=parser,
            question_text=QUESTION_TEXT,
            config=reasoning_only_config,
            format_instructions="",
            combined_system_prompt="You are an expert biomedical evaluator.",
        )

        assert metadata["model_calls"] == 2, (
            f"Reasoning-only mode should use exactly 2 model calls, got: {metadata['model_calls']}"
        )

    def test_stages_completed_order(
        self,
        parsing_model: ModelConfig,
        reasoning_only_config: VerificationConfig,
    ):
        """Verify stages_completed contains the expected stages in order.

        Reasoning-only should produce: ["reasoning", "parameters"].
        """
        llm = get_llm(parsing_model)
        parser = get_parser(parsing_model)

        _, _, _, metadata = deep_judgment_parse(
            raw_llm_response=SAMPLE_RESPONSE,
            RawAnswer=DrugAnswer,
            parsing_model=parsing_model,
            parsing_llm=llm,
            parser=parser,
            question_text=QUESTION_TEXT,
            config=reasoning_only_config,
            format_instructions="",
            combined_system_prompt="You are an expert biomedical evaluator.",
        )

        assert metadata["stages_completed"] == ["reasoning", "parameters"], (
            f"Expected stages ['reasoning', 'parameters'], got: {metadata['stages_completed']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=120"])
