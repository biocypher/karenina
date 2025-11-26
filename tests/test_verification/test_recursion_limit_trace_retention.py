"""Tests for recursion limit trace retention.

These tests verify that when a GraphRecursionError is raised, the accumulated
trace (messages from tool calls) is properly retained rather than lost.
"""

from unittest.mock import MagicMock, Mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from karenina.benchmark.verification.verification_utils import _invoke_llm_with_retry


class MockGraphRecursionError(Exception):
    """Mock GraphRecursionError that can optionally carry state."""

    def __init__(self, message: str, state: dict | None = None, messages: list | None = None):
        super().__init__(message)
        self.state = state
        self.messages = messages


class TestRecursionLimitTraceRetention:
    """Tests for trace retention when recursion limit is hit.

    Note: _invoke_llm_with_retry returns a harmonized STRING (via harmonize_agent_response),
    NOT a dict. When recursion limit is hit, the accumulated messages are formatted into
    a readable trace string containing tool calls and AI messages.
    """

    def test_trace_retained_via_exception_state(self):
        """Test that trace is retained when exception carries state (Method 1)."""
        # Setup: Create accumulated messages that should be preserved
        input_message = HumanMessage(content="What is the answer?")
        tool_call_message = AIMessage(
            content="Let me search for that.",
            tool_calls=[{"id": "call_1", "name": "search", "args": {"query": "answer"}}],
        )
        tool_result_message = ToolMessage(content="The answer is 42", tool_call_id="call_1")
        partial_ai_message = AIMessage(content="Based on my search, I found...")

        accumulated_messages = [input_message, tool_call_message, tool_result_message, partial_ai_message]

        # Create exception with state containing accumulated messages
        error = MockGraphRecursionError(
            "GraphRecursionError: recursion_limit of 25 exceeded",
            state={"messages": accumulated_messages},
        )

        # Mock LLM agent
        mock_llm = MagicMock()

        # Create async mock for ainvoke that raises error
        async def raise_error(*args, **kwargs):
            raise error

        mock_llm.ainvoke = raise_error

        # Execute
        response, recursion_limit_reached, usage_metadata, agent_metrics = _invoke_llm_with_retry(
            llm=mock_llm,
            messages=[input_message],
            is_agent=True,
            timeout=30,
        )

        # Verify
        assert recursion_limit_reached is True
        # Response is a harmonized string containing the trace
        assert isinstance(response, str)
        # Verify tool calls are preserved in the formatted trace
        assert "search" in response
        assert "The answer is 42" in response
        assert "Based on my search, I found..." in response

    def test_trace_retained_via_checkpointer(self):
        """Test that trace is retained via checkpointer.get_state (Method 2)."""
        # Setup: Create accumulated messages
        input_message = HumanMessage(content="What is the answer?")
        tool_call_message = AIMessage(
            content="I'll use my tools.",
            tool_calls=[{"id": "call_1", "name": "lookup", "args": {}}],
        )
        tool_result_message = ToolMessage(content="Result: 42", tool_call_id="call_1")

        accumulated_messages = [input_message, tool_call_message, tool_result_message]

        # Create exception WITHOUT state (simulates case where e.state is None)
        error = MockGraphRecursionError(
            "GraphRecursionError: recursion_limit exceeded",
            state=None,  # No state on exception
        )

        # Mock checkpointer state
        mock_state = Mock()
        mock_state.values = {"messages": accumulated_messages}

        # Mock LLM with checkpointer
        mock_llm = MagicMock()

        async def raise_error(*args, **kwargs):
            raise error

        mock_llm.ainvoke = raise_error
        mock_llm.checkpointer = Mock()  # Has checkpointer
        mock_llm.get_state = Mock(return_value=mock_state)

        # Execute
        response, recursion_limit_reached, usage_metadata, agent_metrics = _invoke_llm_with_retry(
            llm=mock_llm,
            messages=[input_message],
            is_agent=True,
            timeout=30,
        )

        # Verify
        assert recursion_limit_reached is True
        # Response is a harmonized string
        assert isinstance(response, str)
        # Verify tool calls are preserved
        assert "lookup" in response
        assert "Result: 42" in response
        # Verify get_state was called with correct config
        mock_llm.get_state.assert_called_once()
        call_args = mock_llm.get_state.call_args
        assert call_args[0][0]["configurable"]["thread_id"] == "default"

    def test_trace_retained_via_exception_messages(self):
        """Test that trace is retained via exception.messages (Method 3)."""
        # Setup: Create accumulated messages
        input_message = HumanMessage(content="Calculate something")
        ai_message = AIMessage(content="Calculating the result now...")

        accumulated_messages = [input_message, ai_message]

        # Create exception with messages attribute but no state
        error = MockGraphRecursionError(
            "GraphRecursionError: recursion_limit exceeded",
            state=None,
            messages=accumulated_messages,
        )

        # Mock LLM without checkpointer
        mock_llm = MagicMock()

        async def raise_error(*args, **kwargs):
            raise error

        mock_llm.ainvoke = raise_error
        # Remove checkpointer attribute to force Method 3
        del mock_llm.checkpointer

        # Execute
        response, recursion_limit_reached, usage_metadata, agent_metrics = _invoke_llm_with_retry(
            llm=mock_llm,
            messages=[input_message],
            is_agent=True,
            timeout=30,
        )

        # Verify
        assert recursion_limit_reached is True
        assert isinstance(response, str)
        # Verify AI message content is preserved
        assert "Calculating the result now..." in response

    def test_fallback_returns_input_only_when_no_state_available(self):
        """Test that fallback returns only input messages when no state can be extracted."""
        # Setup
        input_message = HumanMessage(content="What is the answer?")

        # Create exception with no state and no messages
        error = MockGraphRecursionError(
            "GraphRecursionError: recursion_limit exceeded",
            state=None,
            messages=None,
        )

        # Mock LLM without checkpointer or get_state
        mock_llm = MagicMock()

        async def raise_error(*args, **kwargs):
            raise error

        mock_llm.ainvoke = raise_error
        # Remove checkpointer to force fallback
        del mock_llm.checkpointer

        # Execute
        response, recursion_limit_reached, usage_metadata, agent_metrics = _invoke_llm_with_retry(
            llm=mock_llm,
            messages=[input_message],
            is_agent=True,
            timeout=30,
        )

        # Verify - fallback returns harmonized input messages only
        assert recursion_limit_reached is True
        assert isinstance(response, str)
        # In fallback case, only the input message is returned - but it's harmonized
        # The harmonize function filters out human messages, so the result may be empty or minimal
        # The key is that recursion_limit_reached is True

    def test_accumulated_trace_includes_multiple_tool_calls(self):
        """Test that multiple tool calls are preserved in the trace."""
        # Setup: Simulate a complex multi-step tool interaction
        input_message = HumanMessage(content="Find proteins related to colorectal cancer")

        messages = [
            input_message,
            AIMessage(
                content="I'll search for proteins.",
                tool_calls=[
                    {"id": "call_1", "name": "bc_get_uniprot_id_by_protein_symbol", "args": {"symbol": "TP53"}}
                ],
            ),
            ToolMessage(content='{"uniprot_id": "P04637"}', tool_call_id="call_1"),
            AIMessage(
                content="Found TP53. Let me get more info.",
                tool_calls=[{"id": "call_2", "name": "bc_get_uniprot_protein_info", "args": {"uniprot_id": "P04637"}}],
            ),
            ToolMessage(content='{"name": "Cellular tumor antigen p53", "function": "..."}', tool_call_id="call_2"),
            AIMessage(
                content="Now let me search for another protein.",
                tool_calls=[
                    {"id": "call_3", "name": "bc_get_uniprot_id_by_protein_symbol", "args": {"symbol": "KRAS"}}
                ],
            ),
            ToolMessage(content='{"uniprot_id": "P01116"}', tool_call_id="call_3"),
        ]

        # Create exception with accumulated state
        error = MockGraphRecursionError(
            "GraphRecursionError: recursion_limit of 25 exceeded",
            state={"messages": messages},
        )

        mock_llm = MagicMock()

        async def raise_error(*args, **kwargs):
            raise error

        mock_llm.ainvoke = raise_error

        # Execute
        response, recursion_limit_reached, usage_metadata, agent_metrics = _invoke_llm_with_retry(
            llm=mock_llm,
            messages=[input_message],
            is_agent=True,
            timeout=30,
        )

        # Verify all tool calls are preserved in the formatted trace
        assert recursion_limit_reached is True
        assert isinstance(response, str)
        assert "bc_get_uniprot_id_by_protein_symbol" in response
        assert "bc_get_uniprot_protein_info" in response
        assert "P04637" in response  # UniProt ID
        assert "P01116" in response  # Second UniProt ID
        assert "TP53" in response or "Cellular tumor antigen p53" in response

    def test_method_priority_state_over_checkpointer(self):
        """Test that exception.state takes priority over checkpointer."""
        # Setup: Both state sources available
        input_message = HumanMessage(content="Test")

        # Exception state has distinctive message
        exception_messages = [
            input_message,
            AIMessage(content="FROM_EXCEPTION_STATE_UNIQUE_MARKER"),
            AIMessage(content="More from exception"),
        ]

        # Checkpointer state has different distinctive message (should NOT be used)
        checkpointer_messages = [
            input_message,
            AIMessage(content="FROM_CHECKPOINTER_UNIQUE_MARKER"),
        ]

        error = MockGraphRecursionError(
            "GraphRecursionError",
            state={"messages": exception_messages},
        )

        mock_state = Mock()
        mock_state.values = {"messages": checkpointer_messages}

        mock_llm = MagicMock()

        async def raise_error(*args, **kwargs):
            raise error

        mock_llm.ainvoke = raise_error
        mock_llm.checkpointer = Mock()
        mock_llm.get_state = Mock(return_value=mock_state)

        # Execute
        response, recursion_limit_reached, usage_metadata, agent_metrics = _invoke_llm_with_retry(
            llm=mock_llm,
            messages=[input_message],
            is_agent=True,
            timeout=30,
        )

        # Verify exception.state was used (contains exception marker, not checkpointer marker)
        assert "FROM_EXCEPTION_STATE_UNIQUE_MARKER" in response
        assert "FROM_CHECKPOINTER_UNIQUE_MARKER" not in response
        # get_state should NOT have been called since exception.state was available
        mock_llm.get_state.assert_not_called()


class TestRecursionLimitStageSkipping:
    """Tests for verifying that ParseTemplate and VerifyTemplate stages are skipped."""

    def _create_minimal_context(self, recursion_limit_reached: bool = False):
        """Create a minimal VerificationContext for testing."""
        from karenina.benchmark.verification.stage import VerificationContext
        from karenina.schemas.workflow import ModelConfig

        model = ModelConfig(
            id="test",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
        )

        context = VerificationContext(
            question_id="q1",
            template_id="test_template_id",
            question_text="Test question",
            template_code="class Answer(BaseAnswer): pass",
            answering_model=model,
            parsing_model=model,
        )

        if recursion_limit_reached:
            context.set_artifact("recursion_limit_reached", True)

        return context

    def test_parse_template_skipped_when_recursion_limit_reached(self):
        """Test that ParseTemplateStage.should_run returns False when recursion limit reached."""
        from karenina.benchmark.verification.stages.parse_template import ParseTemplateStage

        context = self._create_minimal_context(recursion_limit_reached=True)
        context.set_artifact("raw_llm_response", "Some response")
        context.set_artifact("Answer", Mock())

        stage = ParseTemplateStage()
        assert stage.should_run(context) is False

    def test_parse_template_runs_when_no_recursion_limit(self):
        """Test that ParseTemplateStage.should_run returns True normally."""
        from karenina.benchmark.verification.stages.parse_template import ParseTemplateStage

        context = self._create_minimal_context(recursion_limit_reached=False)
        context.set_artifact("raw_llm_response", "Some response")
        context.set_artifact("Answer", Mock())

        stage = ParseTemplateStage()
        assert stage.should_run(context) is True

    def test_verify_template_skipped_when_recursion_limit_reached(self):
        """Test that VerifyTemplateStage.should_run returns False when recursion limit reached."""
        from karenina.benchmark.verification.stages.verify_template import VerifyTemplateStage

        context = self._create_minimal_context(recursion_limit_reached=True)
        context.set_artifact("parsed_answer", Mock())
        context.set_artifact("raw_llm_response", "Some response")

        stage = VerifyTemplateStage()
        assert stage.should_run(context) is False

    def test_verify_template_runs_when_no_recursion_limit(self):
        """Test that VerifyTemplateStage.should_run returns True normally."""
        from karenina.benchmark.verification.stages.verify_template import VerifyTemplateStage

        context = self._create_minimal_context(recursion_limit_reached=False)
        context.set_artifact("parsed_answer", Mock())
        context.set_artifact("raw_llm_response", "Some response")

        stage = VerifyTemplateStage()
        assert stage.should_run(context) is True

    def test_recursion_limit_false_by_default(self):
        """Test that recursion_limit_reached defaults to False when not set."""
        from karenina.benchmark.verification.stages.parse_template import ParseTemplateStage

        context = self._create_minimal_context(recursion_limit_reached=False)
        context.set_artifact("raw_llm_response", "Some response")
        context.set_artifact("Answer", Mock())
        # Don't explicitly set recursion_limit_reached - it should default to False

        stage = ParseTemplateStage()
        # Should run because recursion_limit_reached is False by default
        assert stage.should_run(context) is True
