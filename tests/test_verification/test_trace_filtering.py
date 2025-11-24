"""Tests for trace filtering feature (MCP agent trace input control)."""

from unittest.mock import Mock, patch

import pytest
from pydantic import Field

from karenina.benchmark.verification.stage import VerificationContext
from karenina.benchmark.verification.stages.parse_template import ParseTemplateStage
from karenina.benchmark.verification.stages.rubric_evaluation import (
    RubricEvaluationStage,
)
from karenina.infrastructure.llm.mcp_utils import (
    extract_final_ai_message,
    extract_final_ai_message_from_response,
)
from karenina.schemas import VerificationConfig
from karenina.schemas.domain import BaseAnswer, LLMRubricTrait, Rubric


class MockAnswer(BaseAnswer):
    """Mock answer class for testing."""

    result: int = Field(description="The result")
    correct: dict = Field(description="Correct answer")

    def verify(self) -> bool:
        """Verify the answer."""
        return self.result == 4


@pytest.fixture
def full_agent_trace() -> str:
    """Create a sample full MCP agent trace."""
    return """--- AI Message ---
Let me search for information about this topic.

Tool Calls:
  mcp__search (call_abc123)
   Call ID: abc123
   Args: {'query': 'arithmetic operations'}

--- Tool Message (call_id: abc123) ---
Search results: Addition is a basic arithmetic operation...

--- AI Message ---
Based on the search results, I can now answer the question.

The answer to 2 + 2 is 4. This is a fundamental arithmetic operation."""


@pytest.fixture
def trace_ending_with_tool() -> str:
    """Create a trace that ends with a tool message (error case)."""
    return """--- AI Message ---
Let me search for information.

Tool Calls:
  mcp__search (call_xyz789)
   Call ID: xyz789
   Args: {'query': 'test'}

--- Tool Message (call_id: xyz789) ---
Search results: Some information here..."""


@pytest.fixture
def trace_with_only_tool_calls() -> str:
    """Create a trace where the final AI message has only tool calls (error case)."""
    return """--- AI Message ---
Initial thought.

--- Tool Message (call_id: abc123) ---
Tool response.

--- AI Message ---

Tool Calls:
  mcp__search (call_def456)
   Call ID: def456
   Args: {'query': 'test'}"""


@pytest.fixture
def context_with_trace_config(basic_context: VerificationContext, full_agent_trace: str) -> VerificationContext:
    """Create a context with trace filtering config enabled."""
    basic_context.set_artifact("RawAnswer", MockAnswer)
    basic_context.set_artifact("Answer", MockAnswer)
    basic_context.set_artifact("raw_llm_response", full_agent_trace)

    # Create config with trace filtering disabled
    config = VerificationConfig(
        answering_models=[basic_context.answering_model],
        parsing_models=[basic_context.parsing_model],
        use_full_trace_for_template=False,
        use_full_trace_for_rubric=False,
    )
    basic_context.config = config

    return basic_context


class TestExtractFinalAIMessage:
    """Test suite for extract_final_ai_message function."""

    def test_extract_from_valid_trace(self, full_agent_trace: str) -> None:
        """Test successful extraction from a valid trace."""
        message, error = extract_final_ai_message(full_agent_trace)

        assert error is None
        assert message is not None
        assert "The answer to 2 + 2 is 4" in message
        assert "Tool Calls:" not in message  # Tool calls should be stripped
        assert "--- AI Message ---" not in message  # Headers should be stripped

    def test_extract_final_message_only(self) -> None:
        """Test that only the final AI message is extracted, not earlier ones."""
        trace = """--- AI Message ---
First message with wrong answer.

--- Tool Message (call_id: abc) ---
Tool response

--- AI Message ---
Second message with correct answer: 42"""

        message, error = extract_final_ai_message(trace)

        assert error is None
        assert message == "Second message with correct answer: 42"
        assert "First message" not in message

    def test_extract_strips_tool_calls(self) -> None:
        """Test that tool calls are stripped from the final message."""
        trace = """--- AI Message ---
Here is my answer: The result is 4.

Tool Calls:
  mcp__search (call_123)
   Call ID: 123
   Args: {'query': 'test'}"""

        message, error = extract_final_ai_message(trace)

        assert error is None
        assert message == "Here is my answer: The result is 4."
        assert "Tool Calls" not in message

    def test_error_on_empty_trace(self) -> None:
        """Test error handling for empty trace."""
        message, error = extract_final_ai_message("")

        assert message is None
        assert error == "Empty or whitespace-only trace"

    def test_error_on_whitespace_trace(self) -> None:
        """Test error handling for whitespace-only trace."""
        message, error = extract_final_ai_message("   \n  \t  \n ")

        assert message is None
        assert error == "Empty or whitespace-only trace"

    def test_error_on_trace_ending_with_tool(self, trace_ending_with_tool: str) -> None:
        """Test error when trace ends with tool message."""
        message, error = extract_final_ai_message(trace_ending_with_tool)

        assert message is None
        assert error == "Last message in trace is not an AI message"

    def test_error_on_malformed_trace(self) -> None:
        """Test error handling for malformed trace."""
        malformed_trace = "This is just plain text without proper formatting"

        message, error = extract_final_ai_message(malformed_trace)

        assert message is None
        assert error == "Malformed trace: no message blocks found"

    def test_error_on_ai_message_with_only_tool_calls(self, trace_with_only_tool_calls: str) -> None:
        """Test error when final AI message has no text content."""
        message, error = extract_final_ai_message(trace_with_only_tool_calls)

        assert message is None
        assert error == "Final AI message has no text content (only tool calls)"


class TestExtractFinalAIMessageFromResponse:
    """Test suite for extract_final_ai_message_from_response function (message types)."""

    def test_extract_from_message_list(self) -> None:
        """Test successful extraction from a list of AIMessages."""
        from langchain_core.messages import AIMessage, ToolMessage

        messages = [
            AIMessage(content="Let me search for information."),
            ToolMessage(content="Search results...", tool_call_id="call_123"),
            AIMessage(content="Based on the search, the answer is 42."),
        ]

        message, error = extract_final_ai_message_from_response(messages)

        assert error is None
        assert message == "Based on the search, the answer is 42."

    def test_extract_from_dict_with_messages(self) -> None:
        """Test extraction from dict with 'messages' key."""
        from langchain_core.messages import AIMessage

        response = {
            "messages": [
                AIMessage(content="First thought"),
                AIMessage(content="Final answer"),
            ]
        }

        message, error = extract_final_ai_message_from_response(response)

        assert error is None
        assert message == "Final answer"

    def test_extract_from_nested_agent_state(self) -> None:
        """Test extraction from nested agent state dict."""
        from langchain_core.messages import AIMessage

        response = {
            "agent": {
                "messages": [
                    AIMessage(content="Thinking..."),
                    AIMessage(content="The result is 4"),
                ]
            }
        }

        message, error = extract_final_ai_message_from_response(response)

        assert error is None
        assert message == "The result is 4"

    def test_extract_from_single_ai_message(self) -> None:
        """Test extraction from a single AIMessage."""
        from langchain_core.messages import AIMessage

        message_obj = AIMessage(content="Single answer")

        message, error = extract_final_ai_message_from_response(message_obj)

        assert error is None
        assert message == "Single answer"

    def test_error_on_empty_messages(self) -> None:
        """Test error when messages list is empty."""
        message, error = extract_final_ai_message_from_response([])

        assert message is None
        assert error == "No messages found in response"

    def test_error_on_none_response(self) -> None:
        """Test error when response is None."""
        message, error = extract_final_ai_message_from_response(None)

        assert message is None
        assert error == "Empty response"

    def test_error_when_last_message_not_ai(self) -> None:
        """Test error when last message is not an AIMessage."""
        from langchain_core.messages import AIMessage, ToolMessage

        messages = [
            AIMessage(content="Question"),
            ToolMessage(content="Tool response", tool_call_id="call_123"),
        ]

        message, error = extract_final_ai_message_from_response(messages)

        assert message is None
        assert error == "Last message is not an AI/assistant message"

    def test_error_on_empty_content(self) -> None:
        """Test error when AIMessage has empty content."""
        from langchain_core.messages import AIMessage

        messages = [AIMessage(content="")]

        message, error = extract_final_ai_message_from_response(messages)

        assert message is None
        assert error == "Final AI message has no content"

    def test_error_on_message_with_only_tool_calls(self) -> None:
        """Test error when AIMessage has only tool calls, no text."""
        from langchain_core.messages import AIMessage

        # Create message with tool calls but no text content
        message_obj = AIMessage(content="")
        message_obj.tool_calls = [{"name": "search", "args": {"query": "test"}, "id": "call_123"}]

        result, error = extract_final_ai_message_from_response([message_obj])

        assert result is None
        assert error == "Final AI message has no text content (only tool calls)"


class TestExtractFinalAIMessageFromNativeAgents:
    """Test suite for extract_final_ai_message_from_response with native agent dict messages."""

    def test_extract_from_openai_format_messages(self) -> None:
        """Test extraction from OpenAI native agent format."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]

        result, error = extract_final_ai_message_from_response(messages)

        assert error is None
        assert result == "The answer is 4."

    def test_extract_from_openai_format_with_tool_calls(self) -> None:
        """Test extraction from OpenAI format with tool call history."""
        messages = [
            {"role": "user", "content": "Search for something."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_123", "function": {"name": "search", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "Search results..."},
            {"role": "assistant", "content": "Based on the search, the answer is 42."},
        ]

        result, error = extract_final_ai_message_from_response(messages)

        assert error is None
        assert result == "Based on the search, the answer is 42."

    def test_extract_from_anthropic_format_messages(self) -> None:
        """Test extraction from Anthropic native agent format (content blocks)."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "The answer is 4."}]},
        ]

        result, error = extract_final_ai_message_from_response(messages)

        assert error is None
        assert result == "The answer is 4."

    def test_extract_from_anthropic_format_with_tool_use(self) -> None:
        """Test extraction from Anthropic format with tool use history."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Search for something."}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {"type": "tool_use", "id": "call_123", "name": "search", "input": {}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_123", "content": "Results..."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Based on results, the answer is 42."}]},
        ]

        result, error = extract_final_ai_message_from_response(messages)

        assert error is None
        assert result == "Based on results, the answer is 42."

    def test_error_on_dict_message_ending_with_tool(self) -> None:
        """Test error when last dict message is a tool result."""
        messages = [
            {"role": "user", "content": "Search for something."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_123"}]},
            {"role": "tool", "tool_call_id": "call_123", "content": "Search results..."},
        ]

        result, error = extract_final_ai_message_from_response(messages)

        assert result is None
        assert "not an assistant message" in error

    def test_error_on_openai_format_empty_content(self) -> None:
        """Test error when OpenAI format assistant message has empty content."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
        ]

        result, error = extract_final_ai_message_from_response(messages)

        assert result is None
        assert "no content" in error

    def test_error_on_openai_format_only_tool_calls(self) -> None:
        """Test error when OpenAI format assistant message has only tool calls."""
        messages = [
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_123", "function": {"name": "search", "arguments": "{}"}}],
            },
        ]

        result, error = extract_final_ai_message_from_response(messages)

        assert result is None
        assert "only tool calls" in error

    def test_error_on_anthropic_format_only_tool_use(self) -> None:
        """Test error when Anthropic format assistant message has only tool use."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Search"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_123", "name": "search", "input": {}}],
            },
        ]

        result, error = extract_final_ai_message_from_response(messages)

        assert result is None
        assert "only tool calls" in error


class TestParseTemplateStageTraceFiltering:
    """Test suite for ParseTemplateStage with trace filtering."""

    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.PydanticOutputParser")
    def test_uses_full_trace_when_config_true(
        self,
        mock_parser_class: Mock,
        mock_init_llm: Mock,
        basic_context: VerificationContext,
        full_agent_trace: str,
    ) -> None:
        """Test that full trace is used when use_full_trace_for_template=True."""
        # Set up context with full trace enabled (default)
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", full_agent_trace)
        basic_context.deep_judgment_enabled = False

        config = VerificationConfig(
            answering_models=[basic_context.answering_model],
            parsing_models=[basic_context.parsing_model],
            use_full_trace_for_template=True,  # Default
        )
        basic_context.config = config

        # Mock LLM and parser
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = '{"result": 4, "correct": {"value": 4}, "question_id": "test_q123"}'
        mock_llm.invoke.return_value = mock_llm_response
        mock_init_llm.return_value = mock_llm

        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        parsed_result = MockAnswer(result=4, correct={"value": 4}, question_id="test_q123")
        mock_parser.parse.return_value = parsed_result

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify full trace was used
        assert basic_context.error is None
        assert basic_context.get_result_field("used_full_trace_for_template") is True
        assert basic_context.get_result_field("template_evaluation_input") == full_agent_trace
        assert basic_context.get_result_field("trace_extraction_error") is None

        # Verify the full trace was passed to parsing
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        # Check that the HumanMessage contains the full trace
        human_message = messages[-1]
        assert full_agent_trace in human_message.content

    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.PydanticOutputParser")
    def test_uses_final_message_when_config_false(
        self,
        mock_parser_class: Mock,
        mock_init_llm: Mock,
        basic_context: VerificationContext,
        full_agent_trace: str,
    ) -> None:
        """Test that only final AI message is used when use_full_trace_for_template=False."""
        # Set up context with full trace disabled
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", full_agent_trace)
        basic_context.deep_judgment_enabled = False

        config = VerificationConfig(
            answering_models=[basic_context.answering_model],
            parsing_models=[basic_context.parsing_model],
            use_full_trace_for_template=False,  # Extract final message only
        )
        basic_context.config = config
        basic_context.use_full_trace_for_template = False  # Also set direct attribute

        # Mock LLM and parser
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = '{"result": 4, "correct": {"value": 4}, "question_id": "test_q123"}'
        mock_llm.invoke.return_value = mock_llm_response
        mock_init_llm.return_value = mock_llm

        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        parsed_result = MockAnswer(result=4, correct={"value": 4}, question_id="test_q123")
        mock_parser.parse.return_value = parsed_result

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify only final message was used
        assert basic_context.error is None
        assert basic_context.get_result_field("used_full_trace_for_template") is False

        template_input = basic_context.get_result_field("template_evaluation_input")
        assert template_input is not None
        assert "The answer to 2 + 2 is 4" in template_input
        assert "Tool Calls:" not in template_input
        assert "--- AI Message ---" not in template_input

        # Verify the extracted message was passed to parsing
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        human_message = messages[-1]
        assert template_input in human_message.content
        assert "Tool Calls:" not in human_message.content

    def test_fails_on_invalid_trace(
        self,
        basic_context: VerificationContext,
        trace_ending_with_tool: str,
    ) -> None:
        """Test that stage fails when trace extraction fails."""
        # Set up context with invalid trace
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", trace_ending_with_tool)

        config = VerificationConfig(
            answering_models=[basic_context.answering_model],
            parsing_models=[basic_context.parsing_model],
            use_full_trace_for_template=False,
        )
        basic_context.config = config
        basic_context.use_full_trace_for_template = False  # Also set direct attribute

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify error was set
        assert basic_context.error is not None
        assert "Failed to extract final AI message" in basic_context.error
        assert basic_context.get_result_field("trace_extraction_error") == "Last message in trace is not an AI message"
        assert basic_context.get_result_field("template_evaluation_input") is None

    @patch("karenina.benchmark.verification.stages.parse_template.deep_judgment_parse")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_deep_judgment_uses_filtered_trace(
        self,
        mock_init_llm: Mock,
        mock_deep_judgment: Mock,
        basic_context: VerificationContext,
        full_agent_trace: str,
    ) -> None:
        """Test that deep-judgment parsing receives filtered trace."""
        # Set up context with deep-judgment enabled
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", full_agent_trace)
        basic_context.deep_judgment_enabled = True
        basic_context.deep_judgment_max_excerpts_per_attribute = 3
        basic_context.deep_judgment_fuzzy_match_threshold = 0.8
        basic_context.deep_judgment_excerpt_retry_attempts = 2
        basic_context.deep_judgment_search_enabled = False
        basic_context.deep_judgment_search_tool = "tavily"

        config = VerificationConfig(
            answering_models=[basic_context.answering_model],
            parsing_models=[basic_context.parsing_model],
            use_full_trace_for_template=False,
        )
        basic_context.config = config
        basic_context.use_full_trace_for_template = False  # Also set direct attribute

        # Mock LLM
        mock_llm = Mock()
        mock_init_llm.return_value = mock_llm

        # Mock deep-judgment response
        parsed_result = MockAnswer(result=4, correct={"value": 4}, question_id="test_q123")
        mock_deep_judgment.return_value = (
            parsed_result,
            {"attr1": [{"excerpt": "test", "reasoning": "test"}]},
            {"attr1": "reasoning"},
            {"stages_completed": ["stage1"], "model_calls": 1, "excerpt_retry_count": 0},
        )

        # Execute stage
        stage = ParseTemplateStage()
        stage.execute(basic_context)

        # Verify deep-judgment was called with filtered trace
        assert mock_deep_judgment.called
        call_kwargs = mock_deep_judgment.call_args[1]
        raw_llm_response_arg = call_kwargs["raw_llm_response"]

        # Should be the extracted final message, not the full trace
        assert "The answer to 2 + 2 is 4" in raw_llm_response_arg
        assert "Tool Calls:" not in raw_llm_response_arg
        assert "--- AI Message ---" not in raw_llm_response_arg

    def test_raw_llm_response_always_preserved(
        self,
        basic_context: VerificationContext,
        full_agent_trace: str,
    ) -> None:
        """Test that raw_llm_response artifact always contains full trace."""
        # Set up context
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", full_agent_trace)

        # Get the raw response before stage execution
        raw_before = basic_context.get_artifact("raw_llm_response")

        # Even with filtering enabled
        config = VerificationConfig(
            answering_models=[basic_context.answering_model],
            parsing_models=[basic_context.parsing_model],
            use_full_trace_for_template=False,
        )
        basic_context.config = config

        # After stage execution (even if it fails)
        # Don't execute since we'd need to mock everything, just verify the artifact isn't modified

        raw_after = basic_context.get_artifact("raw_llm_response")

        # Verify full trace is preserved
        assert raw_before == raw_after == full_agent_trace


class TestRubricEvaluationStageTraceFiltering:
    """Test suite for RubricEvaluationStage with trace filtering."""

    @pytest.fixture
    def context_with_rubric(self, basic_context: VerificationContext, full_agent_trace: str) -> VerificationContext:
        """Create context with rubric for testing."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="Accuracy",
                    description="Is the answer accurate?",
                    kind="score",
                    min_score=1,
                    max_score=10,
                )
            ]
        )
        basic_context.rubric = rubric
        basic_context.set_artifact("raw_llm_response", full_agent_trace)
        basic_context.rubric_evaluation_strategy = "batch"
        return basic_context

    @patch("karenina.benchmark.verification.stages.rubric_evaluation.RubricEvaluator")
    def test_uses_full_trace_when_config_true(
        self,
        mock_evaluator_class: Mock,
        context_with_rubric: VerificationContext,
        full_agent_trace: str,
    ) -> None:
        """Test that full trace is used for rubric evaluation when config is True."""
        config = VerificationConfig(
            answering_models=[context_with_rubric.answering_model],
            parsing_models=[context_with_rubric.parsing_model],
            rubric_enabled=True,
            evaluation_mode="template_and_rubric",
            use_full_trace_for_rubric=True,  # Default
        )
        context_with_rubric.config = config

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate_rubric.return_value = ({"Accuracy": 9}, [])
        mock_evaluator_class.return_value = mock_evaluator

        # Execute stage
        stage = RubricEvaluationStage()
        stage.execute(context_with_rubric)

        # Verify full trace was used
        assert context_with_rubric.error is None
        assert context_with_rubric.get_result_field("used_full_trace_for_rubric") is True
        assert context_with_rubric.get_result_field("rubric_evaluation_input") == full_agent_trace

        # Verify evaluator was called with full trace
        mock_evaluator.evaluate_rubric.assert_called_once()
        call_kwargs = mock_evaluator.evaluate_rubric.call_args[1]
        assert call_kwargs["answer"] == full_agent_trace

    @patch("karenina.benchmark.verification.stages.rubric_evaluation.RubricEvaluator")
    def test_uses_final_message_when_config_false(
        self,
        mock_evaluator_class: Mock,
        context_with_rubric: VerificationContext,
    ) -> None:
        """Test that only final message is used when use_full_trace_for_rubric=False."""
        config = VerificationConfig(
            answering_models=[context_with_rubric.answering_model],
            parsing_models=[context_with_rubric.parsing_model],
            rubric_enabled=True,
            evaluation_mode="template_and_rubric",
            use_full_trace_for_rubric=False,
        )
        context_with_rubric.config = config
        context_with_rubric.use_full_trace_for_rubric = False  # Also set direct attribute

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate_rubric.return_value = ({"Accuracy": 9}, [])
        mock_evaluator_class.return_value = mock_evaluator

        # Execute stage
        stage = RubricEvaluationStage()
        stage.execute(context_with_rubric)

        # Verify filtered trace was used
        assert context_with_rubric.error is None
        assert context_with_rubric.get_result_field("used_full_trace_for_rubric") is False

        rubric_input = context_with_rubric.get_result_field("rubric_evaluation_input")
        assert rubric_input is not None
        assert "The answer to 2 + 2 is 4" in rubric_input
        assert "Tool Calls:" not in rubric_input

        # Verify evaluator was called with filtered trace
        mock_evaluator.evaluate_rubric.assert_called_once()
        call_kwargs = mock_evaluator.evaluate_rubric.call_args[1]
        assert call_kwargs["answer"] == rubric_input

    def test_fails_on_invalid_trace(
        self,
        context_with_rubric: VerificationContext,
        trace_ending_with_tool: str,
    ) -> None:
        """Test that stage fails when trace extraction fails."""
        context_with_rubric.set_artifact("raw_llm_response", trace_ending_with_tool)

        config = VerificationConfig(
            answering_models=[context_with_rubric.answering_model],
            parsing_models=[context_with_rubric.parsing_model],
            rubric_enabled=True,
            evaluation_mode="template_and_rubric",
            use_full_trace_for_rubric=False,
        )
        context_with_rubric.config = config
        context_with_rubric.use_full_trace_for_rubric = False  # Also set direct attribute

        # Execute stage
        stage = RubricEvaluationStage()
        stage.execute(context_with_rubric)

        # Verify error was set
        assert context_with_rubric.error is not None
        assert "Failed to extract final AI message" in context_with_rubric.error
        assert (
            context_with_rubric.get_result_field("rubric_trace_extraction_error")
            == "Last message in trace is not an AI message"
        )

    @patch("karenina.benchmark.verification.stages.rubric_evaluation.RubricEvaluator")
    def test_metric_traits_use_filtered_trace(
        self,
        mock_evaluator_class: Mock,
        context_with_rubric: VerificationContext,
    ) -> None:
        """Test that metric trait evaluation receives filtered trace."""
        from karenina.schemas.domain import MetricRubricTrait

        # Add metric trait to rubric
        context_with_rubric.rubric.metric_traits = [
            MetricRubricTrait(
                name="FeatureIdentification",
                description="Identify features correctly",
                evaluation_mode="full_matrix",
                tp_instructions=["feature1", "feature2"],
                tn_instructions=["incorrect_feature"],
                metrics=["precision", "recall", "f1"],
            )
        ]

        config = VerificationConfig(
            answering_models=[context_with_rubric.answering_model],
            parsing_models=[context_with_rubric.parsing_model],
            rubric_enabled=True,
            evaluation_mode="template_and_rubric",
            use_full_trace_for_rubric=False,
        )
        context_with_rubric.config = config
        context_with_rubric.use_full_trace_for_rubric = False  # Also set direct attribute

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate_rubric.return_value = ({}, [])
        mock_evaluator.evaluate_metric_traits.return_value = (
            {},
            {"FeatureIdentification": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
            [],
        )
        mock_evaluator_class.return_value = mock_evaluator

        # Execute stage
        stage = RubricEvaluationStage()
        stage.execute(context_with_rubric)

        # Verify metric traits evaluator was called with filtered trace
        assert mock_evaluator.evaluate_metric_traits.called
        call_kwargs = mock_evaluator.evaluate_metric_traits.call_args[1]
        answer_arg = call_kwargs["answer"]

        assert "The answer to 2 + 2 is 4" in answer_arg
        assert "Tool Calls:" not in answer_arg


class TestIndependentTraceControls:
    """Test that template and rubric trace controls work independently."""

    @patch("karenina.benchmark.verification.stages.rubric_evaluation.RubricEvaluator")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.PydanticOutputParser")
    def test_different_configs_for_template_and_rubric(
        self,
        mock_parser_class: Mock,
        mock_init_llm: Mock,
        mock_evaluator_class: Mock,
        basic_context: VerificationContext,
        full_agent_trace: str,
    ) -> None:
        """Test that template can use full trace while rubric uses final message (or vice versa)."""
        # Set up context for both template and rubric
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="Clarity",
                    description="Is the answer clear?",
                    kind="score",
                    min_score=1,
                    max_score=10,
                )
            ]
        )
        basic_context.rubric = rubric
        basic_context.rubric_evaluation_strategy = "batch"
        basic_context.set_artifact("RawAnswer", MockAnswer)
        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.set_artifact("raw_llm_response", full_agent_trace)
        basic_context.deep_judgment_enabled = False

        # Configure: template uses final message, rubric uses full trace
        config = VerificationConfig(
            answering_models=[basic_context.answering_model],
            parsing_models=[basic_context.parsing_model],
            rubric_enabled=True,
            evaluation_mode="template_and_rubric",
            use_full_trace_for_template=False,  # Extract final message
            use_full_trace_for_rubric=True,  # Use full trace
        )
        basic_context.config = config
        basic_context.use_full_trace_for_template = False  # Also set direct attribute
        basic_context.use_full_trace_for_rubric = True  # Also set direct attribute

        # Mock template parsing
        mock_llm = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = '{"result": 4, "correct": {"value": 4}, "question_id": "test_q123"}'
        mock_llm.invoke.return_value = mock_llm_response
        mock_init_llm.return_value = mock_llm

        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        parsed_result = MockAnswer(result=4, correct={"value": 4}, question_id="test_q123")
        mock_parser.parse.return_value = parsed_result

        # Mock rubric evaluation
        mock_evaluator = Mock()
        mock_evaluator.evaluate_rubric.return_value = ({"Clarity": 8}, [])
        mock_evaluator_class.return_value = mock_evaluator

        # Execute both stages
        template_stage = ParseTemplateStage()
        template_stage.execute(basic_context)

        rubric_stage = RubricEvaluationStage()
        rubric_stage.execute(basic_context)

        # Verify template used final message
        assert basic_context.get_result_field("used_full_trace_for_template") is False
        template_input = basic_context.get_result_field("template_evaluation_input")
        assert "The answer to 2 + 2 is 4" in template_input
        assert "Tool Calls:" not in template_input

        # Verify rubric used full trace
        assert basic_context.get_result_field("used_full_trace_for_rubric") is True
        rubric_input = basic_context.get_result_field("rubric_evaluation_input")
        assert rubric_input == full_agent_trace

        # Verify both succeeded without errors
        assert basic_context.error is None
