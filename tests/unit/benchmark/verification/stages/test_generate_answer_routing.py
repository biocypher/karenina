"""Tests for GenerateAnswerStage routing logic.

Verifies that the generate_answer stage correctly routes to AgentPort
or LLMPort based on the model configuration, particularly for manual
interface models.
"""

import contextlib
import inspect
from types import SimpleNamespace

import pytest

import karenina.benchmark.verification.stages.pipeline.generate_answer as gen_mod
from karenina.adapters.registry import AdapterRegistry, AdapterSpec


def _make_context(interface, model_provider=None, model_name="test"):
    """Build a minimal VerificationContext stand-in via SimpleNamespace."""
    model = SimpleNamespace(
        id=model_name,
        interface=interface,
        mcp_urls_dict=None,
        system_prompt=None,
        agent_middleware=None,
        agent_timeout=180,
        model_provider=model_provider,
        model_name=model_name,
    )
    artifacts = {}
    context = SimpleNamespace(
        answering_model=model,
        question_text="What is 2+2?",
        few_shot_examples=[],
        few_shot_enabled=False,
        workspace_path=None,
        agentic_parsing=False,
        cached_answer_data=None,
        question_id="q1",
        error=None,
        completed_without_errors=True,
    )
    context.get_artifact = lambda key: artifacts.get(key)
    context.set_artifact = lambda key, value: artifacts.__setitem__(key, value)
    context.mark_error = lambda msg: setattr(context, "error", msg)
    return context


@pytest.mark.unit
class TestGenerateAnswerRouting:
    """Tests for the use_agent routing decision in GenerateAnswerStage."""

    def test_source_routes_manual_to_agent(self):
        """The execute() source must contain a manual interface check.

        Inspects the source of GenerateAnswerStage.execute() to verify that
        manual interface routing is present. The routing expression must
        include a check for interface == "manual" so that manual models use
        AgentPort (ManualAgentAdapter) instead of LLMPort
        (ManualLLMAdapter, which raises ManualInterfaceError).
        """
        source = inspect.getsource(gen_mod.GenerateAnswerStage.execute)

        assert 'interface == "manual"' in source, (
            "GenerateAnswerStage.execute() must contain a check for "
            'interface == "manual" in the use_agent routing logic. '
            "Without this, manual models are sent to ManualLLMAdapter "
            "which raises ManualInterfaceError."
        )

    def test_manual_get_agent_called(self, monkeypatch):
        """Verify GenerateAnswerStage calls get_agent (not get_llm) for manual.

        Patches get_agent and get_llm at the module level and runs execute()
        to confirm that the manual interface routes through the agent path.
        """
        agent_called = False
        llm_called = False

        class FakeAgent:
            """Stub AgentPort for tracking calls."""

            async def run(self, *_args, **_kwargs):
                from karenina.ports import Message

                return [Message.ai("fake")]

        class FakeLLM:
            """Stub LLMPort for tracking calls."""

            async def ainvoke(self, *_args, **_kwargs):
                from karenina.ports import Message

                return Message.ai("fake")

        def fake_get_agent(_model_config, auto_fallback=False):
            nonlocal agent_called
            agent_called = True
            return FakeAgent()

        def fake_get_llm(_model_config, auto_fallback=False):
            nonlocal llm_called
            llm_called = True
            return FakeLLM()

        monkeypatch.setattr(gen_mod, "get_agent", fake_get_agent)
        monkeypatch.setattr(gen_mod, "get_llm", fake_get_llm)

        manual_spec = AdapterSpec(
            interface="manual",
            description="Manual interface for pre-recorded traces",
            supports_mcp=False,
            supports_tools=False,
            requires_provider=False,
            agent_tier="tool_loop",
        )
        monkeypatch.setattr(
            AdapterRegistry,
            "get_spec",
            staticmethod(lambda _iface: manual_spec),
        )

        context = _make_context("manual")
        stage = gen_mod.GenerateAnswerStage()

        # execute() will call get_agent then fail at the stub invocation.
        # We only care which factory was called.
        with contextlib.suppress(Exception):
            stage.execute(context)

        assert agent_called is True, (
            "Expected get_agent() to be called for manual interface, "
            "but it was not called. The routing logic did not recognize "
            "manual as requiring AgentPort."
        )
        assert llm_called is False, (
            "get_llm() should not be called for manual interface. ManualLLMAdapter raises ManualInterfaceError."
        )

    def test_langchain_without_mcp_routes_to_llm(self, monkeypatch):
        """Standard LangChain model without MCP should route to LLMPort."""
        agent_called = False
        llm_called = False

        class FakeLLM:
            """Stub LLMPort."""

            async def ainvoke(self, *_args, **_kwargs):
                from karenina.ports import Message

                return Message.ai("fake")

        def fake_get_agent(_model_config, auto_fallback=False):
            nonlocal agent_called
            agent_called = True

        def fake_get_llm(_model_config, auto_fallback=False):
            nonlocal llm_called
            llm_called = True
            return FakeLLM()

        monkeypatch.setattr(gen_mod, "get_agent", fake_get_agent)
        monkeypatch.setattr(gen_mod, "get_llm", fake_get_llm)

        langchain_spec = AdapterSpec(
            interface="langchain",
            description="LangChain adapter",
            agent_tier="tool_loop",
        )
        monkeypatch.setattr(
            AdapterRegistry,
            "get_spec",
            staticmethod(lambda _iface: langchain_spec),
        )

        context = _make_context("langchain", model_provider="anthropic", model_name="claude-sonnet-4-20250514")
        stage = gen_mod.GenerateAnswerStage()

        with contextlib.suppress(Exception):
            stage.execute(context)

        assert llm_called is True, "Standard langchain model without MCP should route to LLMPort."
        assert agent_called is False, "get_agent() should not be called for standard langchain model."
