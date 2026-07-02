"""Tests for GenerateAnswerStage routing logic.

Verifies that the generate_answer stage routes to AgentPort or LLMPort
based on the model configuration. Routing is driven by the real
``use_agent`` decision in ``GenerateAnswerStage.execute``:

    use_agent = bool(mcp_urls) or spec.agent_tier == "deep_agent" or interface == "manual"

These tests exercise that decision against the real ``VerificationContext``
by patching ``get_agent`` / ``get_llm`` at the module where the stage looks
them up, then asserting which factory was called. The manual path is
especially load-bearing: ``ManualLLMAdapter`` raises
``ManualInterfaceError``, so a routing regression would silently send
manual runs to the wrong adapter.
"""

from unittest.mock import MagicMock

import pytest

import karenina.benchmark.verification.stages.pipeline.generate_answer as gen_mod
from karenina.adapters.registry import AdapterRegistry, AdapterSpec
from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.benchmark.verification.stages.pipeline.generate_answer import GenerateAnswerStage
from karenina.schemas.config import ModelConfig


def _make_context(
    *,
    interface: str,
    mcp_urls_dict: dict | None = None,
) -> VerificationContext:
    """Build a minimal real VerificationContext for routing tests.

    Uses the real context class (not a SimpleNamespace stand-in) so that
    earlier sections of execute() — replay lookup, workspace resolution,
    cache check — do not short-circuit or raise before the routing branch
    under test is reached.
    """
    extra: dict = {}
    if interface == "manual":
        # ModelConfig validates that interface="manual" carries a non-None,
        # non-bool manual_traces. A sentinel object satisfies the validator
        # without forcing us to build a real ManualTraces (which needs a
        # Benchmark). The routing branch under test never inspects it.
        extra["manual_traces"] = object()
    model = ModelConfig(
        id="m1",
        model_name="test-model",
        model_provider="openai" if interface != "manual" else None,
        interface=interface,
        system_prompt="You are helpful.",
        request_timeout=30.0,
        mcp_urls_dict=mcp_urls_dict,
        **extra,
    )
    return VerificationContext(
        question_id="q1",
        template_id="t1",
        question_text="What is 2+2?",
        template_code="class Answer(BaseAnswer): value: str",
        answering_model=model,
        parsing_model=model,
    )


def _install_factories(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Patch get_agent/get_llm with call-tracking mocks and return them.

    Both mocks return a sentinel adapter so the routing branch is observable
    via ``call_count`` regardless of what execute() does afterwards. The
    downstream invocation may fail; we wrap execute() in ``suppress`` so the
    test only asserts on which factory was selected.
    """
    agent_factory = MagicMock(return_value=MagicMock(name="FakeAgent"))
    llm_factory = MagicMock(return_value=MagicMock(name="FakeLLM"))
    monkeypatch.setattr(gen_mod, "get_agent", agent_factory)
    monkeypatch.setattr(gen_mod, "get_llm", llm_factory)
    return agent_factory, llm_factory


def _force_spec(monkeypatch: pytest.MonkeyPatch, spec: AdapterSpec) -> None:
    monkeypatch.setattr(AdapterRegistry, "get_spec", staticmethod(lambda _iface: spec))


@pytest.mark.unit
class TestGenerateAnswerRouting:
    """Tests for the use_agent routing decision in GenerateAnswerStage."""

    def test_manual_interface_routes_to_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Manual interface must route to AgentPort (ManualAgentAdapter).

        ManualLLMAdapter raises ManualInterfaceError, so a regression that
        drops the ``interface == "manual"`` clause would silently break every
        manual run. The manual spec here uses agent_tier="tool_loop" to prove
        it is the manual-interface clause — not the agent_tier clause — that
        forces the agent path.
        """
        agent_factory, llm_factory = _install_factories(monkeypatch)
        _force_spec(
            monkeypatch,
            AdapterSpec(
                interface="manual",
                description="Manual interface for pre-recorded traces",
                supports_mcp=False,
                supports_tools=False,
                requires_provider=False,
                agent_tier="tool_loop",
            ),
        )

        stage = GenerateAnswerStage()
        stage.execute(_make_context(interface="manual"))

        assert agent_factory.call_count == 1
        assert llm_factory.call_count == 0

    def test_langchain_without_mcp_routes_to_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Standard LangChain model without MCP routes to LLMPort.

        agent_tier is "tool_loop" (not "deep_agent"), no MCP URLs, and the
        interface is not manual — so none of the use_agent clauses fire and
        the stage must use LLMPort.
        """
        agent_factory, llm_factory = _install_factories(monkeypatch)
        _force_spec(
            monkeypatch,
            AdapterSpec(
                interface="langchain",
                description="LangChain adapter",
                agent_tier="tool_loop",
            ),
        )

        stage = GenerateAnswerStage()
        stage.execute(_make_context(interface="langchain"))

        assert llm_factory.call_count == 1
        assert agent_factory.call_count == 0

    def test_deep_agent_tier_routes_to_agent_without_mcp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deep-agent spec (e.g. Claude Code) uses AgentPort even with no MCP.

        Deep-agent runtimes execute tools internally; the LLMPort path would
        lose the tool-call trace. This pins the agent_tier clause of use_agent
        independently of the manual clause.
        """
        agent_factory, llm_factory = _install_factories(monkeypatch)
        _force_spec(
            monkeypatch,
            AdapterSpec(
                interface="langchain",
                description="Deep agent runtime",
                agent_tier="deep_agent",
            ),
        )

        stage = GenerateAnswerStage()
        stage.execute(_make_context(interface="langchain"))

        assert agent_factory.call_count == 1
        assert llm_factory.call_count == 0

    def test_mcp_urls_force_agent_regardless_of_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MCP-attached models route to AgentPort even with a tool_loop tier."""
        agent_factory, llm_factory = _install_factories(monkeypatch)
        _force_spec(
            monkeypatch,
            AdapterSpec(
                interface="langchain",
                description="LangChain adapter",
                agent_tier="tool_loop",
            ),
        )

        stage = GenerateAnswerStage()
        stage.execute(_make_context(interface="langchain", mcp_urls_dict={"brave-search": "http://x/mcp"}))

        assert agent_factory.call_count == 1
        assert llm_factory.call_count == 0
