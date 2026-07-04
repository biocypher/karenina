"""Tests for the reasoning-only deep judgment prompt builders."""

import pytest

from karenina.benchmark.verification.prompts.deep_judgment.template.reasoning_only import (
    build_reasoning_only_system_prompt,
    build_reasoning_only_user_prompt,
)


@pytest.mark.unit
class TestBuildReasoningOnlySystemPrompt:
    """Tests for build_reasoning_only_system_prompt."""

    def test_contains_attribute_guidance(self):
        """System prompt includes the provided attribute guidance."""
        attr_guidance = "- drug_name: The name of the drug mentioned in the response"
        result = build_reasoning_only_system_prompt(
            generic_system_prompt="You are a helpful assistant.",
            attr_guidance=attr_guidance,
        )
        assert attr_guidance in result

    def test_does_not_mention_excerpts(self):
        """System prompt does not reference excerpts at all."""
        result = build_reasoning_only_system_prompt(
            generic_system_prompt="You are a helpful assistant.",
            attr_guidance="- attr: description",
        )
        assert "excerpt" not in result.lower()
        assert "extracted_excerpts" not in result

    def test_references_response_directly(self):
        """System prompt asks the LLM to reason from the response directly."""
        result = build_reasoning_only_system_prompt(
            generic_system_prompt="You are a helpful assistant.",
            attr_guidance="- attr: description",
        )
        assert "response" in result.lower()

    def test_contains_generic_system_prompt(self):
        """System prompt includes the provided generic system prompt."""
        generic = "You are an expert biomedical reasoning assistant."
        result = build_reasoning_only_system_prompt(
            generic_system_prompt=generic,
            attr_guidance="- attr: description",
        )
        assert generic in result

    def test_asks_for_per_attribute_reasoning(self):
        """System prompt requests reasoning about attribute values."""
        result = build_reasoning_only_system_prompt(
            generic_system_prompt="You are a helpful assistant.",
            attr_guidance="- attr: description",
        )
        assert "attribute" in result.lower()
        assert "reasoning" in result.lower()


@pytest.mark.unit
class TestBuildReasoningOnlyUserPrompt:
    """Tests for build_reasoning_only_user_prompt."""

    def test_contains_question_in_xml_tags(self):
        """User prompt wraps the question in <original_question> tags."""
        question = "What drug targets EGFR?"
        result = build_reasoning_only_user_prompt(
            question_text=question,
            raw_llm_response="Erlotinib targets EGFR.",
        )
        assert "<original_question>" in result
        assert question in result
        assert "</original_question>" in result

    def test_contains_response_in_xml_tags(self):
        """User prompt wraps the response in <response> tags."""
        response = "Erlotinib targets EGFR."
        result = build_reasoning_only_user_prompt(
            question_text="What drug targets EGFR?",
            raw_llm_response=response,
        )
        assert "<response>" in result
        assert response in result
        assert "</response>" in result

    def test_does_not_contain_extracted_excerpts_tags(self):
        """User prompt must NOT contain <extracted_excerpts> tags."""
        result = build_reasoning_only_user_prompt(
            question_text="What drug targets EGFR?",
            raw_llm_response="Erlotinib targets EGFR.",
        )
        assert "<extracted_excerpts>" not in result
        assert "</extracted_excerpts>" not in result

    def test_does_not_mention_excerpts(self):
        """User prompt does not mention excerpts at all."""
        result = build_reasoning_only_user_prompt(
            question_text="What drug targets EGFR?",
            raw_llm_response="Erlotinib targets EGFR.",
        )
        assert "excerpt" not in result.lower()

    def test_question_precedes_response(self):
        """Original question appears before the response in the user prompt."""
        result = build_reasoning_only_user_prompt(
            question_text="What drug targets EGFR?",
            raw_llm_response="Erlotinib targets EGFR.",
        )
        question_pos = result.index("<original_question>")
        response_pos = result.index("<response>")
        assert question_pos < response_pos
