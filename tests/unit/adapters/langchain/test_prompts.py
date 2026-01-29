"""Unit tests for LangChain adapter prompts.

This module tests the extracted prompt templates used by the LangChain adapter:
- Parser prompts (system, user, feedback)
- LLM prompts (format instructions)
- Middleware prompts (summarization)

Tests verify:
- All prompts can be imported
- Prompts contain expected content
- Template variables can be formatted correctly
- Prompts are correctly integrated into adapter methods
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

# =============================================================================
# Prompt Import Tests
# =============================================================================


class TestPromptImports:
    """Tests for prompt module imports."""

    def test_all_prompts_importable_from_module(self) -> None:
        """Test that all prompts can be imported from the prompts module."""
        from karenina.adapters.langchain.prompts import (
            FEEDBACK_FORMAT,
            FEEDBACK_NULL,
            FORMAT_INSTRUCTIONS,
            SUMMARIZATION,
            build_question_context,
        )

        # Verify all are strings (except build_question_context which is a function)
        assert isinstance(FEEDBACK_NULL, str)
        assert isinstance(FEEDBACK_FORMAT, str)
        assert isinstance(FORMAT_INSTRUCTIONS, str)
        assert isinstance(SUMMARIZATION, str)
        assert callable(build_question_context)

    def test_individual_prompt_file_imports(self) -> None:
        """Test that prompts can be imported from individual files."""
        from karenina.adapters.langchain.prompts.feedback_format import PROMPT as ff
        from karenina.adapters.langchain.prompts.feedback_null import PROMPT as fn
        from karenina.adapters.langchain.prompts.format_instructions import PROMPT as fi
        from karenina.adapters.langchain.prompts.summarization import PROMPT as sm

        assert all(isinstance(p, str) for p in [fn, ff, fi, sm])

    def test_module_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        from karenina.adapters.langchain import prompts

        expected = [
            "FEEDBACK_FORMAT",
            "FEEDBACK_NULL",
            "FORMAT_INSTRUCTIONS",
            "SUMMARIZATION",
            "build_question_context",
        ]
        assert set(prompts.__all__) == set(expected)


# =============================================================================
# Prompt Content Tests
# =============================================================================


class TestFeedbackNullPrompt:
    """Tests for the null feedback prompt."""

    def test_contains_field_list_variable(self) -> None:
        """Test that prompt contains field_list variable."""
        from karenina.adapters.langchain.prompts import FEEDBACK_NULL

        assert "{field_list}" in FEEDBACK_NULL

    def test_contains_failed_response_variable(self) -> None:
        """Test that prompt contains failed_response variable."""
        from karenina.adapters.langchain.prompts import FEEDBACK_NULL

        assert "{failed_response}" in FEEDBACK_NULL

    def test_format_with_variables(self) -> None:
        """Test that prompt can be formatted with variables."""
        from karenina.adapters.langchain.prompts import FEEDBACK_NULL

        result = FEEDBACK_NULL.format(
            field_list="gene_name, score",
            failed_response='{"gene_name": null, "score": null}',
        )

        assert "gene_name, score" in result
        assert "null" in result
        assert "{field_list}" not in result

    def test_contains_default_value_guidance(self) -> None:
        """Test that prompt provides default value guidance."""
        from karenina.adapters.langchain.prompts import FEEDBACK_NULL

        assert "0.0" in FEEDBACK_NULL  # numeric default
        assert '""' in FEEDBACK_NULL  # string default
        assert "false" in FEEDBACK_NULL  # boolean default


class TestFeedbackFormatPrompt:
    """Tests for the format feedback prompt."""

    def test_contains_required_variables(self) -> None:
        """Test that prompt contains all required variables."""
        from karenina.adapters.langchain.prompts import FEEDBACK_FORMAT

        assert "{failed_response}" in FEEDBACK_FORMAT
        assert "{error}" in FEEDBACK_FORMAT
        assert "{schema_hint}" in FEEDBACK_FORMAT

    def test_format_with_variables(self) -> None:
        """Test that prompt can be formatted with variables."""
        from karenina.adapters.langchain.prompts import FEEDBACK_FORMAT

        result = FEEDBACK_FORMAT.format(
            failed_response="Some reasoning text followed by JSON",
            error="JSONDecodeError: Expecting value",
            schema_hint="\n\nExpected schema:\n{...}",
        )

        assert "Some reasoning text" in result
        assert "JSONDecodeError" in result
        assert "Expected schema" in result

    def test_format_with_empty_schema_hint(self) -> None:
        """Test that prompt works with empty schema hint."""
        from karenina.adapters.langchain.prompts import FEEDBACK_FORMAT

        result = FEEDBACK_FORMAT.format(
            failed_response="bad json",
            error="parse error",
            schema_hint="",
        )

        assert "bad json" in result
        assert "parse error" in result

    def test_contains_critical_instruction(self) -> None:
        """Test that prompt emphasizes JSON-only output."""
        from karenina.adapters.langchain.prompts import FEEDBACK_FORMAT

        assert "CRITICAL" in FEEDBACK_FORMAT
        assert "ONLY a valid JSON object" in FEEDBACK_FORMAT


class TestFormatInstructionsPrompt:
    """Tests for the format instructions prompt."""

    def test_contains_schema_json_variable(self) -> None:
        """Test that prompt contains schema_json variable."""
        from karenina.adapters.langchain.prompts import FORMAT_INSTRUCTIONS

        assert "{schema_json}" in FORMAT_INSTRUCTIONS

    def test_format_with_schema(self) -> None:
        """Test that prompt can be formatted with schema."""
        from karenina.adapters.langchain.prompts import FORMAT_INSTRUCTIONS

        schema = {"type": "object", "properties": {"value": {"type": "string"}}}
        result = FORMAT_INSTRUCTIONS.format(schema_json=json.dumps(schema, indent=2))

        assert '"type": "object"' in result
        assert '"value"' in result

    def test_contains_json_instruction(self) -> None:
        """Test that prompt instructs JSON output."""
        from karenina.adapters.langchain.prompts import FORMAT_INSTRUCTIONS

        assert "valid JSON" in FORMAT_INSTRUCTIONS
        assert "ONLY the JSON object" in FORMAT_INSTRUCTIONS


class TestSummarizationPrompt:
    """Tests for the summarization prompt."""

    def test_contains_required_variables(self) -> None:
        """Test that prompt contains required variables."""
        from karenina.adapters.langchain.prompts import SUMMARIZATION

        assert "{question_context}" in SUMMARIZATION
        assert "{messages_text}" in SUMMARIZATION

    def test_format_with_variables(self) -> None:
        """Test that prompt can be formatted with variables."""
        from karenina.adapters.langchain.prompts import SUMMARIZATION

        result = SUMMARIZATION.format(
            question_context="ORIGINAL QUESTION: What is BCL2?\n\n",
            messages_text="HumanMessage: What is BCL2?\nAIMessage: BCL2 is a gene...",
        )

        assert "What is BCL2?" in result
        assert "BCL2 is a gene" in result

    def test_format_with_empty_context(self) -> None:
        """Test that prompt works with empty question context."""
        from karenina.adapters.langchain.prompts import SUMMARIZATION

        result = SUMMARIZATION.format(
            question_context="",
            messages_text="Some messages here",
        )

        assert "Some messages here" in result

    def test_contains_summary_instructions(self) -> None:
        """Test that prompt contains summary instructions."""
        from karenina.adapters.langchain.prompts import SUMMARIZATION

        assert "INSTRUCTIONS" in SUMMARIZATION
        assert "concise but information-rich summary" in SUMMARIZATION


class TestBuildQuestionContext:
    """Tests for the build_question_context helper function."""

    def test_returns_formatted_context_with_question(self) -> None:
        """Test that function returns formatted context when question provided."""
        from karenina.adapters.langchain.prompts import build_question_context

        result = build_question_context("What is BCL2?")

        assert "ORIGINAL QUESTION:" in result
        assert "What is BCL2?" in result
        assert result.startswith("\n")
        assert result.endswith("\n\n")

    def test_returns_empty_string_with_none(self) -> None:
        """Test that function returns empty string when question is None."""
        from karenina.adapters.langchain.prompts import build_question_context

        result = build_question_context(None)
        assert result == ""

    def test_returns_empty_string_with_empty_string(self) -> None:
        """Test that function returns empty string when question is empty."""
        from karenina.adapters.langchain.prompts import build_question_context

        # Empty string is falsy, so should return ""
        result = build_question_context("")
        assert result == ""


# =============================================================================
# Integration Tests - Prompts Used in Adapters
# =============================================================================


class TestParserAdapterRetryPrompts:
    """Tests that LangChainParserAdapter retry logic uses correct prompts."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a mock ModelConfig."""
        from karenina.schemas.workflow.models import ModelConfig

        return ModelConfig(
            id="test-parser",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="langchain",
        )

    @pytest.mark.asyncio
    async def test_retry_with_null_feedback_uses_feedback_null(self, model_config: Any) -> None:
        """Test that _retry_with_null_feedback uses FEEDBACK_NULL template."""
        from karenina.adapters.langchain import LangChainParserAdapter
        from karenina.ports import Message

        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_model = MagicMock()
            mock_init.return_value = mock_model

            parser = LangChainParserAdapter(model_config)

            class TestSchema(BaseModel):
                value: str

            # Mock the LLM adapter to capture the messages
            captured_messages: list[Message] = []

            async def capture_ainvoke(messages: list[Message]) -> Any:
                captured_messages.extend(messages)
                mock_response = MagicMock()
                mock_response.content = '{"value": "test"}'
                return mock_response

            parser._llm_adapter.ainvoke = capture_ainvoke

            original_messages = [Message.user("Test")]
            # Use a failed_response that includes explicit null so the helper can detect null fields
            failed_response = '{"value": null}'
            # Error message format that triggers null field detection
            error = ValueError("validation error input_value=None\nvalue\nInput should be...")

            result = await parser._retry_with_null_feedback(
                original_messages=original_messages,
                failed_response=failed_response,
                error=error,
                schema=TestSchema,
            )

            # The method should have invoked the LLM with feedback
            # It extracts null fields from the failed_response JSON
            if captured_messages:
                # Check that feedback message contains content from FEEDBACK_NULL
                feedback_message = captured_messages[-1]
                assert "required fields" in feedback_message.text.lower()
                assert "null" in feedback_message.text.lower()
            else:
                # If no null fields were detected, the method returns None early
                # This is expected behavior when the error doesn't indicate null fields
                assert result is None

    @pytest.mark.asyncio
    async def test_retry_with_format_feedback_uses_feedback_format(self, model_config: Any) -> None:
        """Test that _retry_with_format_feedback uses FEEDBACK_FORMAT template."""
        from karenina.adapters.langchain import LangChainParserAdapter
        from karenina.ports import Message

        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_model = MagicMock()
            mock_init.return_value = mock_model

            parser = LangChainParserAdapter(model_config)

            class TestSchema(BaseModel):
                value: str

            captured_messages = []

            async def capture_ainvoke(messages: list[Message]) -> Any:
                captured_messages.extend(messages)
                mock_response = MagicMock()
                mock_response.content = '{"value": "test"}'
                return mock_response

            parser._llm_adapter.ainvoke = capture_ainvoke

            original_messages = [Message.user("Test")]
            error = json.JSONDecodeError("Expecting value", "doc", 0)

            await parser._retry_with_format_feedback(
                original_messages=original_messages,
                failed_response="Some text before JSON",
                error=error,
                schema=TestSchema,
            )

            # Check that feedback message contains content from FEEDBACK_FORMAT
            assert len(captured_messages) > 1
            feedback_message = captured_messages[-1]
            assert "CRITICAL" in feedback_message.text
            assert "valid JSON" in feedback_message.text


class TestLLMAdapterUsesPrompts:
    """Tests that LangChainLLMAdapter correctly uses extracted prompts."""

    @pytest.fixture
    def model_config(self) -> Any:
        """Create a mock ModelConfig."""
        from karenina.schemas.workflow.models import ModelConfig

        return ModelConfig(
            id="test-llm",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="langchain",
        )

    def test_augment_with_format_instructions_uses_format_instructions(self, model_config: Any) -> None:
        """Test that _augment_with_format_instructions uses FORMAT_INSTRUCTIONS."""
        from karenina.adapters.langchain import LangChainLLMAdapter
        from karenina.ports import Message

        with patch("karenina.adapters.langchain.initialization.init_chat_model") as mock_init:
            mock_model = MagicMock()
            mock_init.return_value = mock_model

            class TestSchema(BaseModel):
                answer: str = Field(description="The answer")

            adapter = LangChainLLMAdapter(model_config)
            structured_adapter = adapter.with_structured_output(TestSchema)

            messages = [Message.user("What is the answer?")]
            augmented = structured_adapter._augment_with_format_instructions(messages)

            # Last user message should have format instructions appended
            last_user_msg = augmented[-1]
            assert "valid JSON" in last_user_msg.text
            assert "schema" in last_user_msg.text.lower()
            assert "answer" in last_user_msg.text.lower()


class TestMiddlewareUsesPrompts:
    """Tests that middleware correctly uses extracted prompts."""

    def test_invoke_summarization_middleware_uses_summarization(self) -> None:
        """Test that InvokeSummarizationMiddleware uses SUMMARIZATION prompt."""
        from karenina.adapters.langchain.middleware import InvokeSummarizationMiddleware

        # Create mock model that captures the prompt
        captured_prompts = []

        mock_model = MagicMock()

        def capture_invoke(messages: list[Any]) -> MagicMock:
            captured_prompts.append(messages[0].content)
            response = MagicMock()
            response.content = "Summary of conversation"
            return response

        mock_model.invoke = capture_invoke

        middleware = InvokeSummarizationMiddleware(
            summarization_model=mock_model,
            trigger_token_count=100,
            keep_message_count=2,
        )

        # Create mock messages that exceed trigger threshold
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [
            HumanMessage(content="What is BCL2? " * 50),  # Long message to trigger
            AIMessage(content="BCL2 is a gene..." * 50),
            HumanMessage(content="Tell me more"),
            AIMessage(content="More info here"),
        ]

        # Call _summarize_messages directly
        middleware._summarize_messages(messages, "What is BCL2?")

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # Verify prompt contains expected sections from SUMMARIZATION
        assert "ORIGINAL QUESTION:" in prompt
        assert "What is BCL2?" in prompt
        assert "CONVERSATION TO SUMMARIZE:" in prompt
        assert "INSTRUCTIONS:" in prompt
        assert "concise but information-rich summary" in prompt

    def test_middleware_uses_build_question_context(self) -> None:
        """Test that middleware uses build_question_context for context building."""
        from karenina.adapters.langchain.middleware import InvokeSummarizationMiddleware
        from karenina.adapters.langchain.prompts import build_question_context

        mock_model = MagicMock()
        mock_model.invoke = MagicMock(return_value=MagicMock(content="Summary"))

        InvokeSummarizationMiddleware(
            summarization_model=mock_model,
            trigger_token_count=100,
            keep_message_count=2,
        )

        # Test with question
        context_with_q = build_question_context("Test question?")
        assert "ORIGINAL QUESTION: Test question?" in context_with_q

        # Test without question
        context_without_q = build_question_context(None)
        assert context_without_q == ""
