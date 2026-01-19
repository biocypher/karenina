"""Manual interface adapters for pre-recorded traces.

The manual interface is used for pre-recorded traces where no LLM calls are needed.
ManualAgentAdapter reads from the ManualTraceManager and returns an AgentResult,
enabling generate_answer.py to use a single unified code path for ALL interfaces.

ManualLLMAdapter and ManualParserAdapter remain no-op implementations that raise
errors if invoked, as these adapters are only used by call sites that need
live LLM calls (template/rubric evaluation).

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
    PortError,
    Tool,
    UsageMetadata,
)

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=BaseModel)


class ManualInterfaceError(PortError):
    """Raised when attempting LLM operations with manual interface.

    Manual interface is for pre-recorded traces only. If this error is raised,
    it means code is incorrectly trying to invoke an LLM when it should be
    using pre-recorded trace data instead.

    This is a safety net - properly written call sites should check
    `interface != "manual"` before attempting adapter operations.

    Attributes:
        operation: The operation that was attempted (e.g., "agent.run_sync()").

    Example:
        >>> try:
        ...     agent.run_sync(messages)  # Manual interface
        ... except ManualInterfaceError as e:
        ...     print(f"Cannot {e.operation}: manual interface uses pre-recorded traces")
    """

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(
            f"Cannot {operation} with manual interface. "
            f"Manual interface uses pre-recorded traces, not live LLM calls. "
            f"Check that interface != 'manual' before calling adapter methods, "
            f"or use model_config.manual_traces for pre-recorded data."
        )


class ManualTraceNotFoundError(PortError):
    """Raised when a manual trace is not found for a question hash.

    This error indicates that the ManualTraceManager does not contain a
    pre-recorded trace for the requested question. This typically happens when:
    - Traces were not loaded before running verification
    - The question hash doesn't match any loaded traces
    - The trace file has different question hashes than expected

    Attributes:
        question_hash: The hash that was not found.
        loaded_count: Number of traces currently loaded.

    Example:
        >>> try:
        ...     result = agent.run_sync(
        ...         messages=[...],
        ...         config=AgentConfig(question_hash="abc123...")
        ...     )
        ... except ManualTraceNotFoundError as e:
        ...     print(f"Missing trace for hash: {e.question_hash}")
    """

    def __init__(self, question_hash: str, loaded_count: int) -> None:
        self.question_hash = question_hash
        self.loaded_count = loaded_count
        super().__init__(
            f"No manual trace found for hash: '{question_hash}'. "
            f"Loaded {loaded_count} trace(s). "
            "Ensure traces are loaded via ManualTraceManager before verification."
        )


class ManualAgentAdapter(AgentPort):
    """AgentPort implementation for manual interface using pre-recorded traces.

    Reads traces from ManualTraceManager and returns them as AgentResult.
    This enables generate_answer.py to use a single unified code path for
    ALL interfaces including manual.

    Requires AgentConfig.question_hash to look up the trace.

    Example:
        >>> # Load traces first
        >>> from karenina.infrastructure.llm.manual_traces import load_manual_traces
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

        # Look up trace from manager
        from karenina.infrastructure.llm.manual_traces import (
            get_manual_trace_count,
            get_manual_trace_with_metrics,
        )

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

    async def aparse_to_pydantic(self, response: str, schema: type[T]) -> T:  # noqa: ARG002
        """Raises ManualInterfaceError - manual interface cannot parse via LLM."""
        raise ManualInterfaceError("parser.aparse_to_pydantic()")

    def parse_to_pydantic(self, response: str, schema: type[T]) -> T:  # noqa: ARG002
        """Raises ManualInterfaceError - manual interface cannot parse via LLM."""
        raise ManualInterfaceError("parser.parse_to_pydantic()")


__all__ = [
    "ManualInterfaceError",
    "ManualTraceNotFoundError",
    "ManualAgentAdapter",
    "ManualLLMAdapter",
    "ManualParserAdapter",
]
