"""Manual interface adapters for pre-recorded traces.

The manual interface is used for pre-recorded traces where no LLM calls are needed.
ManualAgentAdapter reads from the ManualTraceManager and returns an AgentResult,
enabling generate_answer.py to use a single unified code path for ALL interfaces.

ManualLLMAdapter and ManualParserAdapter remain no-op implementations that raise
errors if invoked, as these adapters are only used by call sites that need
live LLM calls (template/rubric evaluation).

This module also exports all manual trace management functionality:
- ManualTraceManager: Session-based thread-safe storage
- ManualTraces: High-level API for benchmark-level trace management
- Helper functions: load_manual_traces, get_manual_trace, etc.

Usage Pattern:
    >>> model_config = ModelConfig(interface="manual", ...)
    >>> agent = get_agent(model_config)  # Returns ManualAgentAdapter
    >>>
    >>> # Use same pattern as other adapters
    >>> result = agent.run_sync(
    ...     messages=[...],
    ...     config=AgentConfig(question_hash="abc123...")
    ... )
    >>> print(result.raw_trace)  # Pre-recorded trace

Note:
    ManualAgentAdapter requires question_hash in AgentConfig. This hash is used
    to look up the pre-recorded trace in ManualTraceManager. If the trace is not
    found, ManualTraceNotFoundError is raised with a helpful message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from karenina.ports import (
    AgentConfig,
    AgentPort,
    AgentResult,
    LLMPort,
    LLMResponse,
    MCPServerConfig,
    Message,
    ParserPort,
    Tool,
    UsageMetadata,
)
from karenina.ports.capabilities import PortCapabilities

# Import exceptions from dedicated module to avoid circular imports
from .exceptions import ManualInterfaceError, ManualTraceError, ManualTraceNotFoundError
from .helpers import (
    clear_manual_traces,
    get_manual_trace,
    get_manual_trace_count,
    get_manual_trace_manager,
    get_manual_trace_with_metrics,
    get_memory_usage_info,
    has_manual_trace,
    load_manual_traces,
    set_manual_trace,
)
from .manager import ManualTraceManager, get_trace_manager
from .message_utils import (
    convert_langchain_messages,
    extract_agent_metrics,
    extract_agent_metrics_from_langchain,
    harmonize_messages,
    is_langchain_message_list,
    is_port_message_list,
    preprocess_message_list,
)
from .traces import ManualTraces

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Adapters
# =============================================================================


class ManualAgentAdapter(AgentPort):
    """AgentPort implementation for manual interface using pre-recorded traces.

    Reads traces from ManualTraceManager and returns them as AgentResult.
    This enables generate_answer.py to use a single unified code path for
    ALL interfaces including manual.

    Requires AgentConfig.question_hash to look up the trace.

    Example:
        >>> # Load traces first
        >>> from karenina.adapters.manual import load_manual_traces
        >>> load_manual_traces({"abc123...": "The answer is 42."})
        >>>
        >>> # Then use the adapter
        >>> adapter = ManualAgentAdapter(model_config)
        >>> result = adapter.run_sync(
        ...     messages=[Message.user("What is the answer?")],
        ...     config=AgentConfig(question_hash="abc123...")
        ... )
        >>> print(result.raw_trace)  # "The answer is 42."
    """

    def __init__(self, model_config: Any) -> None:
        """Initialize the manual agent adapter.

        Args:
            model_config: ModelConfig (stored for reference).
        """
        self._model_config = model_config

    async def run(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Async implementation delegates to run_sync.

        Args:
            messages: Input messages (ignored for manual interface).
            tools: Tools (ignored for manual interface).
            mcp_servers: MCP servers (ignored for manual interface).
            config: Must contain question_hash for trace lookup.

        Returns:
            AgentResult with pre-recorded trace.

        Raises:
            ManualInterfaceError: If question_hash not provided in config.
            ManualTraceNotFoundError: If trace not found for hash.
        """
        return self.run_sync(messages, tools, mcp_servers, config)

    def run_sync(
        self,
        messages: list[Message],  # noqa: ARG002
        tools: list[Tool] | None = None,  # noqa: ARG002
        mcp_servers: dict[str, MCPServerConfig] | None = None,  # noqa: ARG002
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Look up pre-recorded trace and return as AgentResult.

        Args:
            messages: Input messages (ignored for manual interface).
            tools: Tools (ignored for manual interface).
            mcp_servers: MCP servers (ignored for manual interface).
            config: Must contain question_hash for trace lookup.

        Returns:
            AgentResult with pre-recorded trace from ManualTraceManager.

        Raises:
            ManualInterfaceError: If question_hash not provided in config.
            ManualTraceNotFoundError: If trace not found for hash.
        """
        config = config or AgentConfig()

        # Get question hash from config
        question_hash = config.question_hash
        if not question_hash:
            raise ManualInterfaceError("agent.run_sync() requires question_hash in AgentConfig for manual interface")

        # Look up trace from manager (use local imports)
        from .helpers import get_manual_trace_count, get_manual_trace_with_metrics

        trace, agent_metrics = get_manual_trace_with_metrics(question_hash)

        if trace is None:
            raise ManualTraceNotFoundError(
                question_hash=question_hash,
                loaded_count=get_manual_trace_count(),
            )

        # Build AgentResult from trace and metrics
        return AgentResult(
            final_response=trace,
            raw_trace=trace,
            trace_messages=[],  # Empty list - no structured messages for manual traces
            usage=UsageMetadata(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                model="manual",
            ),
            turns=agent_metrics.get("iterations", 1) if agent_metrics else 1,
            limit_reached=agent_metrics.get("limit_reached", False) if agent_metrics else False,
            session_id=None,
            actual_model="manual",
        )


class ManualLLMAdapter(LLMPort):
    """No-op LLMPort implementation for manual interface.

    Exists to satisfy the type system (factories return Port, not None).
    Raises ManualInterfaceError if any method is invoked.

    Call sites should check `model_config.interface != "manual"` before
    attempting to use the adapter.
    """

    def __init__(self, model_config: Any) -> None:
        """Initialize the manual LLM adapter.

        Args:
            model_config: ModelConfig (stored but not used since this is a no-op).
        """
        self._model_config = model_config

    @property
    def capabilities(self) -> PortCapabilities:
        """Default capabilities for the manual adapter (no-op)."""
        return PortCapabilities()

    async def ainvoke(self, messages: list[Message]) -> LLMResponse:  # noqa: ARG002
        """Raises ManualInterfaceError - manual interface cannot invoke LLM."""
        raise ManualInterfaceError("llm.ainvoke()")

    def invoke(self, messages: list[Message]) -> LLMResponse:  # noqa: ARG002
        """Raises ManualInterfaceError - manual interface cannot invoke LLM."""
        raise ManualInterfaceError("llm.invoke()")

    def with_structured_output(
        self,
        schema: type[BaseModel],  # noqa: ARG002
        *,
        max_retries: int | None = None,  # noqa: ARG002
    ) -> LLMPort:
        """Raises ManualInterfaceError - manual interface cannot invoke LLM."""
        raise ManualInterfaceError("llm.with_structured_output()")


class ManualParserAdapter(ParserPort):
    """No-op ParserPort implementation for manual interface.

    Exists to satisfy the type system (factories return Port, not None).
    Raises ManualInterfaceError if any method is invoked.

    Call sites should check `model_config.interface != "manual"` before
    attempting to use the adapter.
    """

    def __init__(self, model_config: Any) -> None:
        """Initialize the manual parser adapter.

        Args:
            model_config: ModelConfig (stored but not used since this is a no-op).
        """
        self._model_config = model_config

    @property
    def capabilities(self) -> PortCapabilities:
        """Default capabilities for the manual parser adapter (no-op)."""
        return PortCapabilities()

    async def aparse_to_pydantic(self, messages: list[Message], schema: type[T]) -> T:  # noqa: ARG002
        """Raises ManualInterfaceError - manual interface cannot parse via LLM."""
        raise ManualInterfaceError("parser.aparse_to_pydantic()")

    def parse_to_pydantic(self, messages: list[Message], schema: type[T]) -> T:  # noqa: ARG002
        """Raises ManualInterfaceError - manual interface cannot parse via LLM."""
        raise ManualInterfaceError("parser.parse_to_pydantic()")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Exceptions
    "ManualTraceError",
    "ManualInterfaceError",
    "ManualTraceNotFoundError",
    # Adapters
    "ManualAgentAdapter",
    "ManualLLMAdapter",
    "ManualParserAdapter",
    # Manager
    "ManualTraceManager",
    "get_trace_manager",
    # Traces
    "ManualTraces",
    # Helper functions
    "load_manual_traces",
    "get_manual_trace",
    "has_manual_trace",
    "clear_manual_traces",
    "get_manual_trace_count",
    "get_memory_usage_info",
    "set_manual_trace",
    "get_manual_trace_with_metrics",
    "get_manual_trace_manager",
    # Message utilities
    "convert_langchain_messages",
    "is_langchain_message_list",
    "is_port_message_list",
    "harmonize_messages",
    "extract_agent_metrics",
    "extract_agent_metrics_from_langchain",
    "preprocess_message_list",
]
