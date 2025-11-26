"""Integration tests for recursion limit trace retention with real LangGraph components.

These tests verify that when a GraphRecursionError is raised by a LangGraph agent,
the accumulated trace (tool calls, AI messages) is properly retained using the
MemorySaver checkpointer we added.
"""

from unittest.mock import MagicMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from karenina.benchmark.verification.stage import VerificationContext
from karenina.benchmark.verification.stages.generate_answer import GenerateAnswerStage
from karenina.schemas.workflow import ModelConfig


class TestRecursionLimitIntegration:
    """Integration tests for recursion limit handling in GenerateAnswerStage."""

    def _create_context(self, with_mcp: bool = True):
        """Create a verification context for testing.

        Args:
            with_mcp: If True, configure model with MCP (triggers agent path).
        """
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
            # MCP configuration triggers the agent path where recursion limit can occur
            mcp_urls_dict={"test": "http://localhost:8000"} if with_mcp else None,
        )

        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        return VerificationContext(
            question_id="test-q1",
            template_id="test_template_id",
            question_text="What is the capital of France?",
            template_code="class Answer(BaseAnswer): pass",
            answering_model=answering_model,
            parsing_model=parsing_model,
        )

    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_trace_preserved_when_recursion_limit_hit_via_exception_state(self, mock_init_llm):
        """Test that trace is preserved when exception carries state (Method 1)."""
        # Create accumulated messages that represent a partial agent execution
        input_msg = HumanMessage(content="What is the capital of France?")
        ai_msg1 = AIMessage(
            content="Let me search for that information.",
            tool_calls=[{"id": "call_1", "name": "search", "args": {"query": "capital of France"}}],
        )
        tool_msg1 = ToolMessage(content="Paris is the capital of France.", tool_call_id="call_1")
        ai_msg2 = AIMessage(
            content="Based on my search, I found that Paris is the capital.",
            tool_calls=[{"id": "call_2", "name": "verify", "args": {"fact": "Paris is capital"}}],
        )
        tool_msg2 = ToolMessage(content="Verified: Paris is indeed the capital.", tool_call_id="call_2")

        accumulated_messages = [input_msg, ai_msg1, tool_msg1, ai_msg2, tool_msg2]

        # Create a mock GraphRecursionError with state
        class MockGraphRecursionError(Exception):
            def __init__(self, message, state):
                super().__init__(message)
                self.state = state

        error = MockGraphRecursionError(
            "GraphRecursionError: Recursion limit of 25 reached",
            state={"messages": accumulated_messages},
        )

        # Mock the LLM to raise the error on ainvoke
        mock_llm = MagicMock()

        async def raise_error(*args, **kwargs):
            raise error

        mock_llm.ainvoke = raise_error
        mock_init_llm.return_value = mock_llm

        # Run the stage
        context = self._create_context()
        stage = GenerateAnswerStage()
        stage.execute(context)

        # Verify trace was preserved
        raw_llm_response = context.get_artifact("raw_llm_response")
        recursion_limit_reached = context.get_artifact("recursion_limit_reached")

        assert recursion_limit_reached is True, "recursion_limit_reached should be True"
        assert raw_llm_response is not None, "raw_llm_response should not be None"

        # The trace should contain the tool calls and responses
        assert "search" in raw_llm_response, f"Trace should contain tool name 'search'. Got: {raw_llm_response}"
        assert "Paris" in raw_llm_response, f"Trace should contain 'Paris'. Got: {raw_llm_response}"
        assert "verify" in raw_llm_response, f"Trace should contain tool name 'verify'. Got: {raw_llm_response}"

        # Should NOT just be the truncation note
        assert (
            raw_llm_response
            != "[Note: Recursion limit reached before completion. Error: GraphRecursionError: Recursion limit of 25 reached]"
        )

    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_trace_preserved_via_checkpointer(self, mock_init_llm):
        """Test that trace is preserved via checkpointer.get_state (Method 2)."""
        # Create accumulated messages
        input_msg = HumanMessage(content="What is 2+2?")
        ai_msg = AIMessage(
            content="Let me calculate that.",
            tool_calls=[{"id": "call_1", "name": "calculator", "args": {"expression": "2+2"}}],
        )
        tool_msg = ToolMessage(content="4", tool_call_id="call_1")

        accumulated_messages = [input_msg, ai_msg, tool_msg]

        # Create a mock GraphRecursionError WITHOUT state (to force Method 2)
        class MockGraphRecursionError(Exception):
            state = None  # No state on exception

        error = MockGraphRecursionError("GraphRecursionError: recursion_limit exceeded")

        # Mock checkpointer state
        mock_state = Mock()
        mock_state.values = {"messages": accumulated_messages}

        # Mock the LLM with checkpointer
        mock_llm = MagicMock()

        async def raise_error(*args, **kwargs):
            raise error

        mock_llm.ainvoke = raise_error
        mock_llm.checkpointer = Mock()  # Has checkpointer
        mock_llm.get_state = Mock(return_value=mock_state)

        mock_init_llm.return_value = mock_llm

        # Run the stage
        context = self._create_context()
        stage = GenerateAnswerStage()
        stage.execute(context)

        # Verify trace was preserved
        raw_llm_response = context.get_artifact("raw_llm_response")
        recursion_limit_reached = context.get_artifact("recursion_limit_reached")

        assert recursion_limit_reached is True
        assert raw_llm_response is not None

        # The trace should contain the tool call
        assert "calculator" in raw_llm_response, f"Trace should contain 'calculator'. Got: {raw_llm_response}"
        assert "4" in raw_llm_response or "2+2" in raw_llm_response, (
            f"Trace should contain result or expression. Got: {raw_llm_response}"
        )

    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_complex_biocontext_trace_preserved(self, mock_init_llm):
        """Test that complex multi-step traces (like biocontext MCP) are preserved."""
        # Simulate a complex biocontext interaction
        input_msg = HumanMessage(content="Find proteins related to colorectal cancer")

        messages = [
            input_msg,
            AIMessage(
                content="I'll search for proteins associated with colorectal cancer.",
                tool_calls=[
                    {"id": "call_1", "name": "bc_get_uniprot_id_by_protein_symbol", "args": {"symbol": "TP53"}}
                ],
            ),
            ToolMessage(content='{"uniprot_id": "P04637", "gene": "TP53"}', tool_call_id="call_1"),
            AIMessage(
                content="Found TP53 (P04637). Let me get more details.",
                tool_calls=[{"id": "call_2", "name": "bc_get_uniprot_protein_info", "args": {"uniprot_id": "P04637"}}],
            ),
            ToolMessage(
                content='{"name": "Cellular tumor antigen p53", "function": "Tumor suppressor", "diseases": ["colorectal cancer"]}',
                tool_call_id="call_2",
            ),
            AIMessage(
                content="Now searching for KRAS...",
                tool_calls=[
                    {"id": "call_3", "name": "bc_get_uniprot_id_by_protein_symbol", "args": {"symbol": "KRAS"}}
                ],
            ),
            ToolMessage(content='{"uniprot_id": "P01116", "gene": "KRAS"}', tool_call_id="call_3"),
            AIMessage(
                content="Let me get AlphaFold structure...",
                tool_calls=[
                    {"id": "call_4", "name": "bc_get_alphafold_info_by_protein_symbol", "args": {"symbol": "KRAS"}}
                ],
            ),
            ToolMessage(
                content='{"pdb_id": "AF-P01116-F1", "confidence": 0.95}',
                tool_call_id="call_4",
            ),
            # More calls would continue until recursion limit...
        ]

        class MockGraphRecursionError(Exception):
            def __init__(self, message, state):
                super().__init__(message)
                self.state = state

        error = MockGraphRecursionError(
            "GraphRecursionError: recursion_limit of 25 exceeded",
            state={"messages": messages},
        )

        mock_llm = MagicMock()

        async def raise_error(*args, **kwargs):
            raise error

        mock_llm.ainvoke = raise_error
        mock_init_llm.return_value = mock_llm

        # Run the stage
        context = self._create_context()
        context.question_text = "Find proteins related to colorectal cancer"
        stage = GenerateAnswerStage()
        stage.execute(context)

        # Verify trace was preserved with all tool calls
        raw_llm_response = context.get_artifact("raw_llm_response")
        recursion_limit_reached = context.get_artifact("recursion_limit_reached")

        assert recursion_limit_reached is True
        assert raw_llm_response is not None

        # Verify all tool names are in the trace
        assert "bc_get_uniprot_id_by_protein_symbol" in raw_llm_response
        assert "bc_get_uniprot_protein_info" in raw_llm_response
        assert "bc_get_alphafold_info_by_protein_symbol" in raw_llm_response

        # Verify key data is in the trace
        assert "P04637" in raw_llm_response  # TP53 UniProt ID
        assert "P01116" in raw_llm_response  # KRAS UniProt ID
        assert "TP53" in raw_llm_response or "Cellular tumor antigen p53" in raw_llm_response

    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    def test_fallback_case_still_indicates_recursion_limit(self, mock_init_llm):
        """Test that when all extraction methods fail, we still mark recursion_limit_reached."""

        # Create exception with no state and no messages
        class MockGraphRecursionError(Exception):
            state = None
            messages = None

        error = MockGraphRecursionError("GraphRecursionError: recursion_limit exceeded")

        # Mock LLM without checkpointer (to force fallback)
        mock_llm = MagicMock()

        async def raise_error(*args, **kwargs):
            raise error

        mock_llm.ainvoke = raise_error
        # Remove checkpointer to force fallback
        del mock_llm.checkpointer

        mock_init_llm.return_value = mock_llm

        # Run the stage
        context = self._create_context()
        stage = GenerateAnswerStage()
        stage.execute(context)

        # Even in fallback, recursion_limit_reached should be True
        recursion_limit_reached = context.get_artifact("recursion_limit_reached")
        assert recursion_limit_reached is True

        # Pipeline should still complete (graceful handling)
        assert context.error is None
