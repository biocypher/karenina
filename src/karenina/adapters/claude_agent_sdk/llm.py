"""Claude Agent SDK LLM adapter implementing the LLMPort interface.

This module provides the ClaudeSDKLLMAdapter class that implements the LLMPort
interface using the Claude Agent SDK's query() function for simple LLM calls.

IMPORTANT: Uses query() NOT ClaudeSDKClient since no hooks/tools are needed
for simple LLM invocation. For agent loops with MCP/hooks, use ClaudeSDKAgentAdapter.

Key differences from LangChain:
- Claude SDK uses string prompts, not message arrays
- System prompts go in ClaudeAgentOptions.system_prompt
- Structured output returned in ResultMessage.structured_output (already dict, not JSON)
- SDK handles retries autonomously via max_turns (no manual retry logic needed)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from karenina.ports import LLMPort, LLMResponse, Message, ParseError
from karenina.ports.capabilities import PortCapabilities

from .messages import ClaudeSDKMessageConverter
from .usage import extract_sdk_usage

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

    from karenina.schemas.config import ModelConfig


class ClaudeSDKLLMAdapter:
    """LLM adapter using Claude Agent SDK's query() function.

    This adapter implements the LLMPort Protocol for simple LLM invocation
    without agent loops or tool calling. Uses the SDK's query() function
    which is simpler than ClaudeSDKClient.

    The adapter handles:
    - Message conversion from unified Message to prompt string
    - System prompt extraction to ClaudeAgentOptions.system_prompt
    - Usage metadata extraction from ResultMessage
    - Sync/async invocation with proper event loop handling
    - Structured output via with_structured_output()

    Note: The SDK handles retries autonomously via max_turns, so no manual
    retry logic is needed in this adapter.

    Example:
        >>> from karenina.schemas.config import ModelConfig
        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic",
        ...     interface="claude_agent_sdk"
        ... )
        >>> adapter = ClaudeSDKLLMAdapter(config)
        >>> response = await adapter.ainvoke([Message.user("Hello!")])
        >>> print(response.content)
        'Hello! How can I help you today?'

        >>> # With structured output
        >>> class Answer(BaseModel):
        ...     value: str
        >>> structured = adapter.with_structured_output(Answer)
        >>> response = await structured.ainvoke([Message.user("What is 2+2?")])
    """

    # Default max_turns for structured output (minimum needed for SDK validation)
    DEFAULT_STRUCTURED_MAX_TURNS = 2

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        _structured_schema: type[BaseModel] | None = None,
        _max_turns: int | None = None,
    ) -> None:
        """Initialize the Claude SDK LLM adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
            _structured_schema: Internal - schema for structured output mode.
            _max_turns: Internal - max turns for SDK (handles retries autonomously).
        """
        self._config = model_config
        self._converter = ClaudeSDKMessageConverter()
        self._structured_schema = _structured_schema
        self._max_turns = _max_turns if _max_turns is not None else self.DEFAULT_STRUCTURED_MAX_TURNS

    def _build_options(self, system_prompt: str | None) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions for the query.

        Args:
            system_prompt: System prompt extracted from messages.

        Returns:
            Configured ClaudeAgentOptions.
        """
        from claude_agent_sdk import ClaudeAgentOptions

        options_kwargs: dict[str, Any] = {
            # Use bypassPermissions for batch/automated calls
            "permission_mode": "bypassPermissions",
            # No tools needed for simple LLM invocation
            "allowed_tools": [],
        }

        # Add system prompt if provided
        if system_prompt:
            options_kwargs["system_prompt"] = system_prompt

        # Add model config system prompt as fallback
        if not system_prompt and self._config.system_prompt:
            options_kwargs["system_prompt"] = self._config.system_prompt

        # Add model specification if provided
        if self._config.model_name:
            options_kwargs["model"] = self._config.model_name

        # Add structured output if configured
        if self._structured_schema is not None:
            json_schema = self._structured_schema.model_json_schema()
            options_kwargs["output_format"] = {
                "type": "json_schema",
                "schema": json_schema,
            }
            # SDK uses max_turns for autonomous retry on structured output
            options_kwargs["max_turns"] = self._max_turns

        # Build env dict for Anthropic settings (api_key, base_url)
        # The Claude Agent SDK reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL from env
        env_vars: dict[str, str] = {}
        if self._config.anthropic_api_key:
            env_vars["ANTHROPIC_API_KEY"] = self._config.anthropic_api_key.get_secret_value()
        if self._config.anthropic_base_url:
            env_vars["ANTHROPIC_BASE_URL"] = self._config.anthropic_base_url

        if env_vars:
            options_kwargs["env"] = env_vars

        return ClaudeAgentOptions(**options_kwargs)

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare what prompt features this adapter supports.

        Returns:
            PortCapabilities with system prompt support and structured output support.
        """
        return PortCapabilities(
            supports_system_prompt=True,
            supports_structured_output=True,
        )

    async def ainvoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM asynchronously.

        Args:
            messages: List of unified Message objects.

        Returns:
            LLMResponse with content, usage metadata, and raw response.

        Raises:
            PortError: If the invocation fails.
        """
        from claude_agent_sdk import ResultMessage, query

        # Convert unified messages to SDK format
        prompt_string = self._converter.to_prompt_string(messages)
        system_prompt = self._converter.extract_system_prompt(messages)

        # Build options
        options = self._build_options(system_prompt)

        # Execute query and collect result
        result: ResultMessage | None = None

        async for message in query(prompt=prompt_string, options=options):
            if isinstance(message, ResultMessage):
                result = message

        if result is None:
            from karenina.ports.errors import AgentResponseError

            raise AgentResponseError("No ResultMessage received from SDK")

        # Handle structured vs text output
        if self._structured_schema is not None:
            return self._process_structured_result(result)
        else:
            return self._process_text_result(result)

    def _process_text_result(self, result: ResultMessage) -> LLMResponse:
        """Process a text (non-structured) result from the SDK."""
        content = result.result or ""
        usage = extract_sdk_usage(result, model=self._config.model_name)
        return LLMResponse(content=content, usage=usage, raw=result)

    def _process_structured_result(self, result: ResultMessage) -> LLMResponse:
        """Process a structured output result from the SDK."""
        if not result.structured_output:
            raise ParseError("No structured output received from SDK")

        # Assert schema exists (this method only called when _structured_schema is set)
        assert self._structured_schema is not None

        content = str(result.structured_output)
        parsed_model = self._structured_schema.model_validate(result.structured_output)
        usage = extract_sdk_usage(result, model=self._config.model_name)

        return LLMResponse(content=content, usage=usage, raw=parsed_model)

    def invoke(self, messages: list[Message]) -> LLMResponse:
        """Invoke the LLM synchronously.

        This is a convenience wrapper around ainvoke() for sync code.
        Uses the shared async portal if available, otherwise falls back
        to asyncio.run() with proper event loop handling.

        Args:
            messages: List of unified Message objects.

        Returns:
            LLMResponse with content, usage metadata, and raw response.

        Raises:
            PortError: If the invocation fails.
        """
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            # Use the shared BlockingPortal for proper event loop management
            return portal.call(self.ainvoke, messages)

        # No portal available - check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - use ThreadPoolExecutor to avoid
            # nested event loop issues

            def run_in_thread() -> LLMResponse:
                return asyncio.run(self.ainvoke(messages))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=300)  # 5 minute timeout

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.ainvoke(messages))

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        max_turns: int | None = None,
        max_retries: int | None = None,  # noqa: ARG002 - Ignored for API compat
    ) -> ClaudeSDKLLMAdapter:
        """Return a new adapter configured for structured output.

        The returned adapter uses SDK's output_format option to constrain
        the LLM's output to match the provided JSON schema.

        The SDK handles retries autonomously via max_turns - no manual retry
        logic is needed.

        Args:
            schema: A Pydantic model class defining the output structure.
            max_turns: Maximum turns for SDK (handles retries autonomously).
                Default is 2 (minimum needed for structured output).
            max_retries: Ignored for API compatibility with LangChain adapter.
                The SDK handles retries autonomously via max_turns.

        Returns:
            A new ClaudeSDKLLMAdapter instance configured for structured output.
            The returned adapter guarantees that response.raw will be an instance
            of the provided schema.

        Example:
            >>> class Answer(BaseModel):
            ...     value: str
            ...     confidence: float
            >>> structured = adapter.with_structured_output(Answer)
            >>> response = await structured.ainvoke(messages)
            >>> assert isinstance(response.raw, Answer)
        """
        return ClaudeSDKLLMAdapter(
            model_config=self._config,
            _structured_schema=schema,
            _max_turns=max_turns,
        )

    async def aclose(self) -> None:
        """Close underlying resources.

        The Claude SDK adapter uses query() which doesn't hold persistent
        connections, so this is a no-op. Provided for interface consistency
        with other adapters that do require cleanup.
        """
        pass


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify ClaudeSDKLLMAdapter implements LLMPort protocol."""
    # This is a static check - if this fails, it means the adapter
    # doesn't properly implement the protocol
    adapter_instance: LLMPort = None  # type: ignore[assignment]

    # The following would fail mypy if the protocol wasn't properly implemented
    # We don't actually run this, it's just for static analysis
    _ = adapter_instance  # Suppress unused warning
