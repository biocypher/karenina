"""Runtime conformance of every shipped adapter to the port Protocols.

The ``LLMPort`` / ``AgentPort`` / ``ParserPort`` Protocols are declared
``runtime_checkable``, so ``isinstance(adapter, Protocol)`` performs a
structural check against the method names the protocol requires. This is
the only layer that catches the regression where someone renames or
removes a protocol method (e.g. ``ainvoke`` -> ``invoke_async``) without
updating every adapter — the type checker will not see it because the
adapters are not declared as protocol subclasses.

The earlier revision of this file defined its own ``MockLLMPort`` /
``MockAgentPort`` / ``MockParserPort`` classes inside the test module and
then asserted ``isinstance(mock, Protocol)``. Those tests were circular:
the mock was written by the same hand that wrote the assertion, so they
encoded the protocol shape twice and never exercised a real adapter. They
provided zero regression signal — renaming ``ainvoke`` on the real
``LLMPort`` would have left every mock-based assertion still green while
every shipped adapter broke.

The tests below instantiate the real adapter classes that
``AdapterFactory`` hands out (langchain, claude_tool, claude_agent_sdk,
langchain_deep_agents, manual) and check Protocol membership, signature
shape, and capabilities exposure against the actual objects. A small
``MissingX`` stub class is still used in exactly one place to demonstrate
that the ``runtime_checkable`` check rejects incomplete implementations —
that is the only meaningful use of a test-local class here.

The LangChain chat model is constructed eagerly inside
``LangChainLLMAdapter.__init__`` / ``LangChainParserAdapter.__init__``;
the autouse fixture below sets placeholder provider env vars so those
constructors return without making a network call. No requests are sent.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from karenina.ports import (
    AgentPort,
    LLMPort,
    ParserPort,
)
from karenina.ports.capabilities import PortCapabilities
from karenina.schemas.config import ModelConfig


@pytest.fixture(autouse=True)
def _stub_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let LangChain adapter constructors return without a network call.

    The tests below only inspect structural conformance — no adapter
    method is invoked — so placeholder credentials are sufficient.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _config(interface: str) -> ModelConfig:
    return ModelConfig(
        id="protocol-conformance",
        model_name="claude-sonnet-4-20250514",
        model_provider="anthropic",
        interface=interface,
    )


# ---------------------------------------------------------------------------
# Real adapter construction. Each factory returns a fully-initialised
# instance of a class the AdapterFactory hands out in production. Importing
# inside the fixture keeps the test module importable when an optional
# backend dependency is missing.
# ---------------------------------------------------------------------------


def _make_llm_adapters() -> list[tuple[str, LLMPort]]:
    from karenina.adapters.claude_agent_sdk.llm import ClaudeSDKLLMAdapter
    from karenina.adapters.claude_tool.llm import ClaudeToolLLMAdapter
    from karenina.adapters.langchain.llm import LangChainLLMAdapter
    from karenina.adapters.langchain_deep_agents.llm import DeepAgentsLLMAdapter
    from karenina.adapters.manual import ManualLLMAdapter

    return [
        ("langchain", LangChainLLMAdapter(_config("langchain"), _base_model=object())),
        ("claude_tool", ClaudeToolLLMAdapter(_config("claude_tool"))),
        ("claude_agent_sdk", ClaudeSDKLLMAdapter(_config("claude_agent_sdk"))),
        ("langchain_deep_agents", DeepAgentsLLMAdapter(_config("langchain_deep_agents"))),
        ("manual", ManualLLMAdapter(model_config=None)),
    ]


def _make_agent_adapters() -> list[tuple[str, AgentPort]]:
    from karenina.adapters.claude_agent_sdk.agent import ClaudeSDKAgentAdapter
    from karenina.adapters.claude_tool.agent import ClaudeToolAgentAdapter
    from karenina.adapters.langchain.agent import LangChainAgentAdapter
    from karenina.adapters.langchain_deep_agents.agent import DeepAgentsAgentAdapter
    from karenina.adapters.manual import ManualAgentAdapter

    return [
        ("langchain", LangChainAgentAdapter(_config("langchain"))),
        ("claude_tool", ClaudeToolAgentAdapter(_config("claude_tool"))),
        ("claude_agent_sdk", ClaudeSDKAgentAdapter(_config("claude_agent_sdk"))),
        ("langchain_deep_agents", DeepAgentsAgentAdapter(_config("langchain_deep_agents"))),
        ("manual", ManualAgentAdapter(model_config=None)),
    ]


def _make_parser_adapters() -> list[tuple[str, ParserPort]]:
    from karenina.adapters.claude_agent_sdk.parser import ClaudeSDKParserAdapter
    from karenina.adapters.claude_tool.parser import ClaudeToolParserAdapter
    from karenina.adapters.langchain.parser import LangChainParserAdapter
    from karenina.adapters.langchain_deep_agents.parser import DeepAgentsParserAdapter
    from karenina.adapters.manual import ManualParserAdapter

    return [
        ("langchain", LangChainParserAdapter(_config("langchain"))),
        ("claude_tool", ClaudeToolParserAdapter(_config("claude_tool"))),
        ("claude_agent_sdk", ClaudeSDKParserAdapter(_config("claude_agent_sdk"))),
        ("langchain_deep_agents", DeepAgentsParserAdapter(_config("langchain_deep_agents"))),
        ("manual", ManualParserAdapter(model_config=None)),
    ]


# ---------------------------------------------------------------------------
# Protocol membership for real adapters
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRealAdaptersSatisfyProtocols:
    """Every shipped adapter must satisfy its declared Protocol.

    If a Protocol method is renamed, removed, or has its signature
    changed in a way that breaks the runtime_checkable structural check,
    these assertions fail for the affected adapter(s). That is the
    regression signal the Protocol layer exists to provide.
    """

    def test_all_llm_adapters_satisfy_llm_port(self) -> None:
        adapters = _make_llm_adapters()
        # Guard against silently dropping an adapter from the test list.
        assert {name for name, _ in adapters} == {
            "langchain",
            "claude_tool",
            "claude_agent_sdk",
            "langchain_deep_agents",
            "manual",
        }
        for name, adapter in adapters:
            assert isinstance(adapter, LLMPort), (
                f"{name} LLM adapter no longer satisfies LLMPort; check for a renamed/removed method on LLMPort"
            )

    def test_all_agent_adapters_satisfy_agent_port(self) -> None:
        adapters = _make_agent_adapters()
        assert {name for name, _ in adapters} == {
            "langchain",
            "claude_tool",
            "claude_agent_sdk",
            "langchain_deep_agents",
            "manual",
        }
        for name, adapter in adapters:
            assert isinstance(adapter, AgentPort), f"{name} agent adapter no longer satisfies AgentPort"

    def test_all_parser_adapters_satisfy_parser_port(self) -> None:
        adapters = _make_parser_adapters()
        assert {name for name, _ in adapters} == {
            "langchain",
            "claude_tool",
            "claude_agent_sdk",
            "langchain_deep_agents",
            "manual",
        }
        for name, adapter in adapters:
            assert isinstance(adapter, ParserPort), f"{name} parser adapter no longer satisfies ParserPort"

    def test_protocols_are_runtime_checkable(self) -> None:
        """The runtime_checkable decorator must actually be in effect.

        A class that is missing required methods must fail the isinstance
        check. This is the mechanism the conformance assertions above
        depend on; if the decorator were removed, every check would
        silently pass for any object.
        """

        class _MissingEverything:
            pass

        assert not isinstance(_MissingEverything(), LLMPort)
        assert not isinstance(_MissingEverything(), AgentPort)
        assert not isinstance(_MissingEverything(), ParserPort)
        # Sanity: non-objects also rejected.
        assert not isinstance(None, LLMPort)
        assert not isinstance({}, AgentPort)


# ---------------------------------------------------------------------------
# Signature contracts callers rely on
# ---------------------------------------------------------------------------

# The pipeline invokes adapter methods by keyword argument (e.g.
# ``await adapter.arun(messages=..., tools=..., mcp_servers=..., config=...)``).
# If a parameter is renamed, every call site breaks at runtime even though
# the structural isinstance check still passes (Protocol checks only look
# at method names, not parameter names). These tests pin the parameter
# names that real call sites depend on.

_LLM_REQUIRED_METHODS = ("ainvoke", "invoke", "with_structured_output", "astream", "stream_invoke", "aclose")
_AGENT_RUN_REQUIRED_PARAMS = ("messages", "tools", "mcp_servers", "config")
_LLM_INVOKE_REQUIRED_PARAMS = ("messages",)
_PARSER_PARSE_REQUIRED_PARAMS = ("messages", "schema")


@pytest.mark.unit
class TestProtocolSignatureContracts:
    """Pin the parameter names that real call sites depend on."""

    @pytest.mark.parametrize("name,adapter", _make_llm_adapters())
    def test_llm_adapters_expose_required_methods(self, name: str, adapter: LLMPort) -> None:
        for method_name in _LLM_REQUIRED_METHODS:
            assert hasattr(adapter, method_name), (
                f"{name} LLM adapter is missing required LLMPort method: {method_name}"
            )
            assert callable(getattr(adapter, method_name))

    @pytest.mark.parametrize("name,adapter", _make_llm_adapters())
    def test_llm_ainvoke_invoke_accept_messages_kwarg(self, name: str, adapter: LLMPort) -> None:
        for method_name in ("ainvoke", "invoke"):
            params = inspect.signature(getattr(adapter, method_name)).parameters
            assert "messages" in params, (
                f"{name}.LLM.{method_name} must accept a 'messages' keyword "
                f"(call sites invoke it as {method_name}(messages=...))"
            )

    @pytest.mark.parametrize("name,adapter", _make_llm_adapters())
    def test_llm_stream_invoke_accepts_timeout_kwarg(self, name: str, adapter: LLMPort) -> None:
        """stream_invoke must accept a 'timeout' keyword.

        generate_answer and the per-call timeout middleware call this as
        ``stream_invoke(messages, timeout=...)``; renaming the kwarg would
        silently drop the per-call timeout.
        """
        params = inspect.signature(adapter.stream_invoke).parameters
        assert "timeout" in params, f"{name}.LLM.stream_invoke must accept a 'timeout' keyword argument"

    @pytest.mark.parametrize("name,adapter", _make_llm_adapters())
    def test_llm_with_structured_output_returns_llm_port(self, name: str, adapter: LLMPort) -> None:
        """with_structured_output() must return another LLMPort.

        Pipeline code chains .ainvoke() on the returned object, so the
        returned adapter must still satisfy the LLMPort contract.

        The manual adapter is excluded: its safety contract is to raise
        ManualInterfaceError whenever a method is actually called (see
        test_factory.py). The structural isinstance check above already
        verifies it satisfies LLMPort.
        """
        if name == "manual":
            pytest.skip("manual adapter raises ManualInterfaceError on call by design")

        from pydantic import BaseModel

        class Schema(BaseModel):
            value: str = "x"

        result = adapter.with_structured_output(Schema)
        assert isinstance(result, LLMPort), (
            f"{name}.LLM.with_structured_output() must return an LLMPort, got {type(result).__name__}"
        )

    @pytest.mark.parametrize("name,adapter", _make_agent_adapters())
    def test_agent_run_arun_expose_call_site_kwargs(self, name: str, adapter: AgentPort) -> None:
        for method_name in ("run", "arun"):
            params = inspect.signature(getattr(adapter, method_name)).parameters
            missing = [p for p in _AGENT_RUN_REQUIRED_PARAMS if p not in params]
            assert not missing, (
                f"{name}.Agent.{method_name} is missing parameters {missing}; "
                f"pipeline calls {method_name}(messages=, tools=, mcp_servers=, config=)"
            )

    @pytest.mark.parametrize("name,adapter", _make_parser_adapters())
    def test_parser_parse_methods_expose_messages_and_schema(
        self,
        name: str,
        adapter: ParserPort,
    ) -> None:
        for method_name in ("parse_to_pydantic", "aparse_to_pydantic"):
            params = inspect.signature(getattr(adapter, method_name)).parameters
            missing = [p for p in _PARSER_PARSE_REQUIRED_PARAMS if p not in params]
            assert not missing, f"{name}.Parser.{method_name} is missing parameters {missing}"


# ---------------------------------------------------------------------------
# Capabilities contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCapabilitiesContract:
    """Every adapter must expose a PortCapabilities instance.

    Pipeline stages (system prompt injection, structured output selection,
    trace sufficiency checks) branch on ``adapter.capabilities``. If an
    adapter returns ``None`` or omits the property, those branches
    silently take the wrong path.
    """

    @pytest.mark.parametrize("name,adapter", _make_llm_adapters())
    def test_llm_adapters_expose_capabilities(self, name: str, adapter: LLMPort) -> None:
        caps = adapter.capabilities
        assert isinstance(caps, PortCapabilities), (
            f"{name}.LLM.capabilities must return PortCapabilities, got {type(caps).__name__}"
        )
        # Capability flags are typed bool; pipeline code does truthiness checks.
        for flag in ("supports_system_prompt", "supports_structured_output", "supports_streaming"):
            assert isinstance(getattr(caps, flag), bool), (
                f"{name}.LLM.capabilities.{flag} must be bool (call sites use truthiness)"
            )

    @pytest.mark.parametrize("name,adapter", _make_agent_adapters())
    def test_agent_adapters_expose_capabilities(self, name: str, adapter: AgentPort) -> None:
        caps = adapter.capabilities
        assert isinstance(caps, PortCapabilities), (
            f"{name}.Agent.capabilities must return PortCapabilities, got {type(caps).__name__}"
        )

    @pytest.mark.parametrize("name,adapter", _make_parser_adapters())
    def test_parser_adapters_expose_capabilities(self, name: str, adapter: ParserPort) -> None:
        caps = adapter.capabilities
        assert isinstance(caps, PortCapabilities), (
            f"{name}.Parser.capabilities must return PortCapabilities, got {type(caps).__name__}"
        )


# ---------------------------------------------------------------------------
# Cross-protocol discrimination
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProtocolDiscrimination:
    """An LLMPort is not an AgentPort, etc.

    Real adapters specialise: an LLM adapter does not implement ``arun``.
    This guards against an accidental merge of the protocols (or an
    adapter that happens to grow the other protocol's methods) by
    checking that no real LLM adapter satisfies AgentPort and vice versa.
    """

    def test_no_llm_adapter_satisfies_agent_port(self) -> None:
        for name, adapter in _make_llm_adapters():
            assert not isinstance(adapter, AgentPort), (
                f"{name} LLM adapter unexpectedly satisfies AgentPort — "
                f"the protocols may have been merged or the adapter grew agent methods"
            )
            assert not isinstance(adapter, ParserPort), f"{name} LLM adapter unexpectedly satisfies ParserPort"

    def test_no_agent_adapter_satisfies_llm_port(self) -> None:
        for name, adapter in _make_agent_adapters():
            assert not isinstance(adapter, LLMPort), f"{name} agent adapter unexpectedly satisfies LLMPort"
            assert not isinstance(adapter, ParserPort), f"{name} agent adapter unexpectedly satisfies ParserPort"

    def test_no_parser_adapter_satisfies_llm_or_agent_port(self) -> None:
        for name, adapter in _make_parser_adapters():
            assert not isinstance(adapter, LLMPort), f"{name} parser adapter unexpectedly satisfies LLMPort"
            assert not isinstance(adapter, AgentPort), f"{name} parser adapter unexpectedly satisfies AgentPort"


# ---------------------------------------------------------------------------
# Lifecycle contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAcloseContract:
    """Every adapter must expose aclose().

    AdapterRegistry.cleanup_all_adapters() iterates active adapters and
    calls ``aclose()`` on each to release HTTP/MCP resources; an adapter
    without the method would AttributeError during pipeline teardown.
    """

    @pytest.mark.parametrize("name,adapter", _make_llm_adapters() + _make_agent_adapters() + _make_parser_adapters())
    def test_adapter_exposes_aclose(self, name: str, adapter: Any) -> None:
        assert hasattr(adapter, "aclose"), f"{name} adapter is missing aclose()"
        assert callable(adapter.aclose)
