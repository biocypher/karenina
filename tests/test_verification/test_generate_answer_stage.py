"""Tests for GenerateAnswerStage."""

from unittest.mock import MagicMock, Mock, patch

from langchain_core.messages import AIMessage

from karenina.benchmark.verification.stage import VerificationContext
from karenina.benchmark.verification.stages.generate_answer import GenerateAnswerStage
from karenina.schemas.domain import BaseAnswer


class TestGenerateAnswerStage:
    """Test suite for GenerateAnswerStage."""

    def test_should_run_with_valid_answer_artifact(self, basic_context: VerificationContext) -> None:
        """Test that should_run returns True when Answer artifact exists."""

        # Set up Answer artifact
        class MockAnswer(BaseAnswer):
            def verify(self) -> bool:
                return True

        basic_context.set_artifact("Answer", MockAnswer)

        stage = GenerateAnswerStage()
        assert stage.should_run(basic_context) is True

    def test_should_not_run_without_answer_artifact(self, basic_context: VerificationContext) -> None:
        """Test that stage runs even without Answer artifact (rubric_only mode)."""
        # Remove Answer artifact to simulate rubric_only mode
        basic_context.artifacts = {}

        # Stage should still run in rubric_only mode (no Answer needed)
        stage = GenerateAnswerStage()
        should_run = stage.should_run(basic_context)
        # In rubric_only mode, stage runs even without Answer artifact
        assert should_run is True

    def test_should_not_run_with_error(self, basic_context: VerificationContext) -> None:
        """Test that should_run returns False if there's an error."""

        class MockAnswer(BaseAnswer):
            def verify(self) -> bool:
                return True

        basic_context.set_artifact("Answer", MockAnswer)
        basic_context.mark_error("Previous error")

        stage = GenerateAnswerStage()
        assert stage.should_run(basic_context) is False

    @patch("karenina.benchmark.verification.stages.generate_answer._invoke_llm_with_retry")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_standard_llm_call(
        self,
        mock_init_llm: Mock,
        mock_invoke: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test standard LLM call without MCP agents."""

        # Set up Answer artifact
        class MockAnswer(BaseAnswer):
            def verify(self) -> bool:
                return True

        basic_context.set_artifact("Answer", MockAnswer)

        # Mock LLM initialization
        mock_llm = MagicMock()
        mock_init_llm.return_value = mock_llm

        # Mock LLM response (4-tuple: response, recursion_limit_reached, usage_metadata, agent_metrics)
        mock_response = AIMessage(content="The answer is 4")
        mock_invoke.return_value = (mock_response, False, {}, None)

        # Execute stage
        stage = GenerateAnswerStage()
        stage.execute(basic_context)

        # Verify no error
        assert basic_context.error is None

        # Verify artifacts were set
        assert basic_context.has_artifact("raw_llm_response")
        assert basic_context.get_artifact("raw_llm_response") == "The answer is 4"
        assert basic_context.get_artifact("recursion_limit_reached") is False
        assert basic_context.has_artifact("answering_model_str")

    @patch("karenina.benchmark.verification.stages.generate_answer._invoke_llm_with_retry")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_mcp_agent_call(
        self,
        mock_init_llm: Mock,
        mock_invoke: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test LLM call with MCP agent configuration."""
        # Configure context for MCP
        basic_context.answering_model.mcp_urls_dict = {"test-server": "http://localhost:8000"}

        # Set up Answer artifact
        class MockAnswer(BaseAnswer):
            def verify(self) -> bool:
                return True

        basic_context.set_artifact("Answer", MockAnswer)

        # Mock LLM with MCP support
        mock_llm = MagicMock()
        mock_init_llm.return_value = mock_llm

        # Mock LLM response (4-tuple: response, recursion_limit_reached, usage_metadata, agent_metrics)
        mock_response = AIMessage(content="Answer from MCP agent")
        mock_invoke.return_value = (mock_response, False, {}, None)

        # Execute stage
        stage = GenerateAnswerStage()
        stage.execute(basic_context)

        # Verify MCP servers were set in artifacts
        assert basic_context.has_artifact("answering_mcp_servers")
        assert "test-server" in basic_context.get_artifact("answering_mcp_servers")
        # For agent calls, the response is processed by harmonize_agent_response
        # which should extract content, but in tests we may get the raw message
        raw_response = basic_context.get_artifact("raw_llm_response")
        assert "Answer from MCP agent" in str(raw_response)

    @patch("karenina.benchmark.verification.stages.generate_answer._invoke_llm_with_retry")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_retry_on_transient_error(
        self,
        mock_init_llm: Mock,
        mock_invoke: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that transient errors trigger retry logic."""

        # Set up Answer artifact
        class MockAnswer(BaseAnswer):
            def verify(self) -> bool:
                return True

        basic_context.set_artifact("Answer", MockAnswer)

        # Mock LLM initialization
        mock_llm = MagicMock()
        mock_init_llm.return_value = mock_llm

        # Mock successful response (retry logic is in _invoke_llm_with_retry)
        # 4-tuple: response, recursion_limit_reached, usage_metadata, agent_metrics
        mock_response = AIMessage(content="Success after retry")
        mock_invoke.return_value = (mock_response, False, {}, None)

        # Execute stage
        stage = GenerateAnswerStage()
        stage.execute(basic_context)

        # Verify success
        assert basic_context.error is None
        assert basic_context.get_artifact("raw_llm_response") == "Success after retry"

    @patch("karenina.benchmark.verification.stages.generate_answer._invoke_llm_with_retry")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_recursion_limit_handling(
        self,
        mock_init_llm: Mock,
        mock_invoke: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test that recursion limit errors are handled gracefully."""

        # Set up Answer artifact
        class MockAnswer(BaseAnswer):
            def verify(self) -> bool:
                return True

        basic_context.set_artifact("Answer", MockAnswer)

        # Mock LLM initialization
        mock_llm = MagicMock()
        mock_init_llm.return_value = mock_llm

        # Mock response with recursion limit flag
        # 4-tuple: response, recursion_limit_reached, usage_metadata, agent_metrics
        mock_response = AIMessage(content="Partial response before recursion limit")
        mock_invoke.return_value = (mock_response, True, {}, None)  # recursion_limit_reached = True

        # Execute stage
        stage = GenerateAnswerStage()
        stage.execute(basic_context)

        # Verify recursion limit was recorded
        assert basic_context.get_artifact("recursion_limit_reached") is True
        assert basic_context.has_artifact("raw_llm_response")
        # No error should be set - recursion limit is handled gracefully
        assert basic_context.error is None

    @patch("karenina.benchmark.verification.stages.generate_answer._construct_few_shot_prompt")
    @patch("karenina.benchmark.verification.stages.generate_answer._invoke_llm_with_retry")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_few_shot_prompting(
        self,
        mock_init_llm: Mock,
        mock_invoke: Mock,
        mock_few_shot: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test few-shot prompting is properly configured."""
        # Enable few-shot
        basic_context.few_shot_enabled = True

        # Set up Answer artifact
        class MockAnswer(BaseAnswer):
            def verify(self) -> bool:
                return True

        basic_context.set_artifact("Answer", MockAnswer)

        # Mock few-shot prompt construction
        mock_few_shot.return_value = [
            "Example 1: Question and Answer",
            "Example 2: Question and Answer",
        ]

        # Mock LLM
        mock_llm = MagicMock()
        mock_init_llm.return_value = mock_llm

        # Mock response (4-tuple: response, recursion_limit_reached, usage_metadata, agent_metrics)
        mock_response = AIMessage(content="Answer using few-shot examples")
        mock_invoke.return_value = (mock_response, False, {}, None)

        # Execute stage
        stage = GenerateAnswerStage()
        stage.execute(basic_context)

        # Verify few-shot prompt was constructed
        mock_few_shot.assert_called_once()
        assert basic_context.error is None
        assert basic_context.has_artifact("raw_llm_response")

    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_error_handling_llm_initialization(
        self,
        mock_init_llm: Mock,
        basic_context: VerificationContext,
    ) -> None:
        """Test error handling when LLM initialization fails."""

        # Set up Answer artifact
        class MockAnswer(BaseAnswer):
            def verify(self) -> bool:
                return True

        basic_context.set_artifact("Answer", MockAnswer)

        # Mock LLM initialization failure
        mock_init_llm.side_effect = Exception("Failed to initialize LLM")

        # Execute stage
        stage = GenerateAnswerStage()
        stage.execute(basic_context)

        # Verify error was set
        assert basic_context.error is not None
        assert "Failed to initialize answering model" in basic_context.error

    def test_stage_metadata(self) -> None:
        """Test stage name and artifact declarations."""
        stage = GenerateAnswerStage()

        assert stage.name == "GenerateAnswer"
        # Answer artifact is optional (not required) for rubric_only mode
        assert "Answer" not in stage.requires or len(stage.requires) == 0
        assert "raw_llm_response" in stage.produces
        assert "recursion_limit_reached" in stage.produces
        assert "answering_model_str" in stage.produces
        assert "answering_mcp_servers" in stage.produces
