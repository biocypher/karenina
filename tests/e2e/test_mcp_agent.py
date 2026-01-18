"""E2E tests for MCP-enabled agent verification.

These tests verify that LangGraph agents with MCP tools and middleware
can be created and invoked successfully. They use fixture-backed LLM
responses for deterministic testing.

Run with: pytest tests/e2e/test_mcp_agent.py -v
"""

from pathlib import Path

import pytest

from karenina.utils.testing import FixtureBackedLLMClient

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "llm_responses"


def test_mcp_agent_middleware_signature() -> None:
    """Test that MCP agent with middleware can be created and invoked.

    This test verifies that the InvokeSummarizationMiddleware.before_model()
    method has the correct signature (runtime parameter) to work with
    LangGraph's agent invocation.

    Uses fixture-backed LLM responses captured from:
    tests/fixtures/llm_responses/claude-haiku-4-5/mcp_agent/

    Regression test for: TypeError: before_model() missing 1 required
    positional argument '_runtime'
    """
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from karenina.infrastructure.llm.interface import _build_agent_middleware
    from karenina.infrastructure.llm.mcp_utils import sync_create_mcp_client_and_tools
    from karenina.schemas.workflow.models import AgentMiddlewareConfig

    # Create fixture-backed LLM client
    llm = FixtureBackedLLMClient(FIXTURE_DIR)

    # Connect to Open Targets MCP server to get tools
    # (tools come from MCP server, LLM responses come from fixtures)
    mcp_urls = {"opentargets": "https://mcp.platform.opentargets.org/mcp"}
    try:
        _, tools = sync_create_mcp_client_and_tools(mcp_urls, None, None)
    except (TimeoutError, ConnectionError, OSError) as e:
        pytest.skip(f"MCP server unavailable: {e}")

    assert len(tools) > 0, "Should retrieve tools from MCP server"

    # Build middleware - this includes InvokeSummarizationMiddleware
    middleware_config = AgentMiddlewareConfig()
    middleware = _build_agent_middleware(
        middleware_config,
        max_context_tokens=8000,
        base_model=llm,
    )

    assert len(middleware) > 0, "Should build middleware components"

    # Create agent with middleware
    memory = InMemorySaver()
    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        middleware=middleware,
    )

    # Invoke agent - this is where the middleware signature issue would surface
    # The TypeError would occur here if before_model() has wrong parameter names
    question = "What is the most severe consequence of 19_44908822_C_T?"

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": "test-mcp-middleware"}},
        )
    except TypeError as e:
        if "missing 1 required positional argument" in str(e) and "runtime" in str(e):
            pytest.fail(
                f"Middleware signature error - before_model() has wrong parameter name: {e}\n"
                "Fix: Ensure before_model() uses 'runtime' (not '_runtime') as parameter name"
            )
        raise

    # Verify we got a response
    messages = result.get("messages", [])
    assert len(messages) > 0, "Agent should return messages"

    final_message = messages[-1]
    assert hasattr(final_message, "content"), "Final message should have content"
    assert len(str(final_message.content)) > 0, "Response content should not be empty"
