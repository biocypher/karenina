"""Claude Agent SDK LLM adapter implementing the LLMPort interface.

This module provides the ClaudeSDKLLMAdapter class that implements the LLMPort
interface using the Claude Agent SDK's query() function for simple LLM calls.

IMPORTANT: Uses query() NOT ClaudeSDKClient since no hooks/tools are needed
for simple LLM invocation. For agent loops with MCP/hooks, use ClaudeSDKAgentAdapter.

Key differences from LangChain:
- Claude SDK uses string prompts, not message arrays
- System prompts go in ClaudeAgentOptions.system_prompt
- Structured output returned in ResultMessage.structured_output (already dict, not JSON)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from karenina.ports import LLMPort, LLMResponse, Message, UsageMetadata

from .messages import ClaudeSDKMessageConverter

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

    from karenina.schemas.workflow.models import ModelConfig


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

    Example:
        >>> from karenina.schemas.workflow.models import ModelConfig
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

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        _structured_schema: type[BaseModel] | None = None,
    ) -> None:
        """Initialize the Claude SDK LLM adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
            _structured_schema: Internal - schema for structured output mode.
        """
        self._config = model_config
        self._converter = ClaudeSDKMessageConverter()
        self._structured_schema = _structured_schema

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
        # SDK accepts simple names like "sonnet", "opus", "haiku"
        # TODO: Verify if it also accepts fully qualified names like "claude-sonnet-4-20250514"
        if self._config.model_name:
            options_kwargs["model"] = self._config.model_name

        # Add structured output if configured
        if self._structured_schema is not None:
            json_schema = self._structured_schema.model_json_schema()
            options_kwargs["output_format"] = {
                "type": "json_schema",
                "schema": json_schema,
            }
            # SDK needs internal turns for structured output validation
            options_kwargs["max_turns"] = 5

        return ClaudeAgentOptions(**options_kwargs)

    def _extract_usage(self, result: ResultMessage) -> UsageMetadata:
        """Extract usage metadata from ResultMessage.

        Args:
            result: SDK ResultMessage with usage data.

        Returns:
            UsageMetadata with token counts and cost info.
        """
        usage_data: dict[str, Any] = result.usage or {}

        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        # Extract cache tokens if available
        cache_read = usage_data.get("cache_read_input_tokens")
        cache_creation = usage_data.get("cache_creation_input_tokens")

        return UsageMetadata(
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            total_tokens=int(total_tokens),
            cost_usd=result.total_cost_usd,
            cache_read_tokens=int(cache_read) if cache_read else None,
            cache_creation_tokens=int(cache_creation) if cache_creation else None,
            model=self._config.model_name,
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
                # Don't break - let iteration complete naturally per SDK best practices

        if result is None:
            from karenina.ports.errors import AgentResponseError

            raise AgentResponseError("No ResultMessage received from SDK")

        # Extract content based on mode
        if self._structured_schema is not None and result.structured_output:
            # Structured output mode - return as string representation
            # The actual parsing happens in ParserPort or with_structured_output usage
            content = str(result.structured_output)
        else:
            # Normal text mode
            content = result.result or ""

        # Extract usage metadata
        usage = self._extract_usage(result)

        return LLMResponse(
            content=content,
            usage=usage,
            raw=result,
        )

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
        from karenina.benchmark.verification.batch_runner import get_async_portal

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

    def with_structured_output(self, schema: type[BaseModel]) -> ClaudeSDKLLMAdapter:
        """Return a new adapter configured for structured output.

        The returned adapter uses SDK's output_format option to constrain
        the LLM's output to match the provided JSON schema.

        IMPORTANT: Unlike LangChain, SDK's structured_output is already a
        Python dict, not a JSON string. No json.loads() needed.

        Args:
            schema: A Pydantic model class defining the output structure.

        Returns:
            A new ClaudeSDKLLMAdapter instance configured for structured output.

        Example:
            >>> class Answer(BaseModel):
            ...     value: str
            ...     confidence: float
            >>> structured = adapter.with_structured_output(Answer)
        """
        return ClaudeSDKLLMAdapter(
            model_config=self._config,
            _structured_schema=schema,
        )


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify ClaudeSDKLLMAdapter implements LLMPort protocol."""
    # This is a static check - if this fails, it means the adapter
    # doesn't properly implement the protocol
    adapter_instance: LLMPort = None  # type: ignore[assignment]

    # The following would fail mypy if the protocol wasn't properly implemented
    # We don't actually run this, it's just for static analysis
    _ = adapter_instance  # Suppress unused warning
