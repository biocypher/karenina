"""Tests that adapter invoke() methods have the with_llm_semaphore decorator."""

import pytest


@pytest.mark.unit
class TestInvokeSemaphoreWiring:
    """Verify that invoke() on each adapter is wrapped with with_llm_semaphore."""

    def test_langchain_invoke_is_wrapped(self):
        from karenina.adapters.langchain.llm import LangChainLLMAdapter

        assert hasattr(LangChainLLMAdapter.invoke, "__wrapped__"), (
            "LangChainLLMAdapter.invoke should be decorated with @with_llm_semaphore"
        )

    def test_claude_tool_invoke_is_wrapped(self):
        from karenina.adapters.claude_tool.llm import ClaudeToolLLMAdapter

        assert hasattr(ClaudeToolLLMAdapter.invoke, "__wrapped__"), (
            "ClaudeToolLLMAdapter.invoke should be decorated with @with_llm_semaphore"
        )

    def test_claude_sdk_invoke_is_wrapped(self):
        from karenina.adapters.claude_agent_sdk.llm import ClaudeSDKLLMAdapter

        assert hasattr(ClaudeSDKLLMAdapter.invoke, "__wrapped__"), (
            "ClaudeSDKLLMAdapter.invoke should be decorated with @with_llm_semaphore"
        )

    def test_deep_agents_invoke_is_wrapped(self):
        from karenina.adapters.langchain_deep_agents.llm import DeepAgentsLLMAdapter

        assert hasattr(DeepAgentsLLMAdapter.invoke, "__wrapped__"), (
            "DeepAgentsLLMAdapter.invoke should be decorated with @with_llm_semaphore"
        )

    def test_manual_invoke_is_not_wrapped(self):
        from karenina.adapters.manual import ManualLLMAdapter

        assert not hasattr(ManualLLMAdapter.invoke, "__wrapped__"), (
            "ManualLLMAdapter.invoke should NOT be wrapped (it never makes LLM calls)"
        )
