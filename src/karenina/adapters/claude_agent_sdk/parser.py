"""Claude Agent SDK Parser adapter implementing the ParserPort interface.

This module provides the ClaudeSDKParserAdapter class that implements the ParserPort
interface using the Claude Agent SDK's query() function with structured output.

IMPORTANT: This is NOT just JSON parsing. It invokes an LLM to interpret
natural language responses and extract structured data according to a schema.

Key differences from LangChain:
- Uses query() with output_format={'type': 'json_schema', 'schema': ...}
- SDK's structured_output is already a Python dict, NOT a JSON string
- Use schema.model_validate(result.structured_output) NOT json.loads()
- SDK needs max_turns>=2 for internal structured output validation
- System prompt passed via ClaudeAgentOptions.system_prompt (same as LLM adapter)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from karenina.ports import Message, ParseError, ParsePortResult, ParserPort, UsageMetadata
from karenina.ports.capabilities import PortCapabilities

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

    from karenina.schemas.workflow.models import ModelConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ClaudeSDKParserAdapter:
    """Parser adapter using Claude Agent SDK's structured output capabilities.

    This adapter implements the ParserPort Protocol and provides LLM-based
    structured output parsing. It invokes an LLM (the "judge" model) to
    interpret natural language responses and extract structured data.

    The parsing flow:
    1. Build a prompt with the response text and instruction to extract data
    2. Invoke the SDK's query() with output_format set to the JSON schema
    3. The SDK constrains the LLM to output valid JSON matching the schema
    4. Validate the result using Pydantic and return the model instance

    IMPORTANT: Unlike LangChain, the SDK's structured_output is already a
    Python dict. No json.loads() is needed - use model_validate() directly.

    Example:
        >>> from karenina.schemas.workflow.models import ModelConfig
        >>> from pydantic import BaseModel, Field

        >>> class Answer(BaseModel):
        ...     gene_name: str = Field(description="The gene mentioned")
        ...     is_oncogene: bool = Field(description="Whether it's an oncogene")

        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic",
        ...     interface="claude_agent_sdk"
        ... )
        >>> parser = ClaudeSDKParserAdapter(config)
        >>> trace = "Based on my analysis, BCL2 is an anti-apoptotic gene..."
        >>> answer = await parser.aparse_to_pydantic(trace, Answer)
        >>> print(answer.gene_name)
        'BCL2'
    """

    # Default max_turns for structured output (minimum needed for SDK validation)
    DEFAULT_STRUCTURED_MAX_TURNS = 2

    def __init__(self, model_config: ModelConfig, *, max_turns: int | None = None) -> None:
        """Initialize the Claude SDK parser adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
            max_turns: Maximum turns for SDK structured output validation.
                Default is 2 (minimum needed for structured output).
        """
        self._config = model_config
        self._max_turns = max_turns if max_turns is not None else self.DEFAULT_STRUCTURED_MAX_TURNS

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare what prompt features this parser adapter supports.

        Claude Agent SDK supports native structured output via output_format
        and system prompts via ClaudeAgentOptions.system_prompt.

        Returns:
            PortCapabilities with system prompt and structured output support.
        """
        return PortCapabilities(supports_system_prompt=True, supports_structured_output=True)

    def _build_options(self, schema: type[BaseModel], system_prompt: str) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions for structured output parsing.

        Args:
            schema: The Pydantic schema to use for structured output.
            system_prompt: System prompt text extracted from messages.

        Returns:
            Configured ClaudeAgentOptions with output_format and system_prompt set.
        """
        from claude_agent_sdk import ClaudeAgentOptions

        # Generate JSON schema from the Pydantic model
        json_schema = schema.model_json_schema()

        options_kwargs: dict[str, Any] = {
            # Use bypassPermissions for batch/automated calls
            "permission_mode": "bypassPermissions",
            # No tools needed for parsing
            "allowed_tools": [],
            # Set structured output format
            "output_format": {
                "type": "json_schema",
                "schema": json_schema,
            },
            # SDK needs internal turns for structured output validation
            "max_turns": self._max_turns,
            # System prompt from pre-assembled messages
            "system_prompt": system_prompt,
        }

        # Add model specification if provided
        if self._config.model_name:
            options_kwargs["model"] = self._config.model_name

        return ClaudeAgentOptions(**options_kwargs)

    @staticmethod
    def _extract_from_messages(messages: list[Message]) -> tuple[str, str]:
        """Extract system and user text from pre-assembled messages.

        Args:
            messages: Pre-assembled prompt messages.

        Returns:
            Tuple of (system_text, user_text).
        """
        system_text = ""
        user_parts: list[str] = []
        for msg in messages:
            if msg.role == "system":
                system_text = msg.text
            elif msg.role == "user":
                user_parts.append(msg.text)
        return system_text, "\n\n".join(user_parts)

    async def aparse_to_pydantic(self, messages: list[Message], schema: type[T]) -> ParsePortResult[T]:
        """Parse using pre-assembled prompt messages into a structured Pydantic model.

        Args:
            messages: Pre-assembled prompt messages (system + user).
            schema: A Pydantic model class defining the expected structure.
                    Field descriptions guide the LLM on what to extract.

        Returns:
            ParsePortResult containing the parsed model and usage metadata.

        Raises:
            ParseError: If the LLM fails to extract valid structured data.
            PortError: If the underlying LLM invocation fails.
        """
        from claude_agent_sdk import ResultMessage, query
        from pydantic import ValidationError

        # Extract system and user text from pre-assembled messages
        system_text, user_text = self._extract_from_messages(messages)

        # Build prompt and options
        prompt = user_text
        options = self._build_options(schema, system_text)

        # Execute query and collect result
        result: ResultMessage | None = None

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result = message
                    # Don't break - let iteration complete naturally per SDK best practices

        except Exception as e:
            raise ParseError(f"SDK query failed during parsing: {e}") from e

        if result is None:
            raise ParseError("No ResultMessage received from SDK")

        # Extract usage from ResultMessage if available
        usage = UsageMetadata()
        if hasattr(result, "usage") and result.usage:
            ru = result.usage
            usage = UsageMetadata(
                input_tokens=getattr(ru, "input_tokens", 0),
                output_tokens=getattr(ru, "output_tokens", 0),
                total_tokens=getattr(ru, "input_tokens", 0) + getattr(ru, "output_tokens", 0),
                model=self._config.model_name,
            )

        # Check for structured output failure
        # SDK returns subtype='error_max_structured_output_retries' on validation failure
        if result.subtype and "error" in result.subtype.lower():
            error_msg = f"Structured output extraction failed: {result.subtype}"
            if result.result:
                error_msg += f" - {result.result}"
            raise ParseError(error_msg)

        # Extract structured output
        # IMPORTANT: SDK's structured_output is already a Python dict, NOT JSON string
        if result.structured_output is None:
            # Fallback: try to parse result.result as JSON if no structured output
            if result.result:
                try:
                    data = json.loads(result.result)
                    return ParsePortResult(parsed=schema.model_validate(data), usage=usage)
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.debug(f"Fallback JSON parsing failed: {e}")

            raise ParseError(
                f"No structured output in SDK response. "
                f"Subtype: {result.subtype}, Result: {result.result[:100] if result.result else 'None'}"
            )

        # Validate and return the structured output
        try:
            return ParsePortResult(parsed=schema.model_validate(result.structured_output), usage=usage)
        except ValidationError as e:
            raise ParseError(f"Structured output validation failed: {e}. Raw output: {result.structured_output}") from e

    def parse_to_pydantic(self, messages: list[Message], schema: type[T]) -> ParsePortResult[T]:
        """Parse using pre-assembled prompt messages (sync).

        This is a convenience wrapper around aparse_to_pydantic() for sync code.
        Uses the shared async portal if available, otherwise falls back to
        asyncio.run() with proper event loop handling.

        Args:
            messages: Pre-assembled prompt messages (system + user).
            schema: A Pydantic model class defining the expected structure.

        Returns:
            ParsePortResult containing the parsed model and usage metadata.

        Raises:
            ParseError: If the LLM fails to extract valid structured data.
            PortError: If the underlying LLM invocation fails.
        """
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            return portal.call(self.aparse_to_pydantic, messages, schema)

        # No portal available - check if we're already in an async context
        try:
            asyncio.get_running_loop()

            def run_in_thread() -> ParsePortResult[T]:
                return asyncio.run(self.aparse_to_pydantic(messages, schema))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=300)  # 5 minute timeout

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.aparse_to_pydantic(messages, schema))

    async def aclose(self) -> None:
        """Close underlying resources.

        The Claude SDK parser adapter uses query() which doesn't hold persistent
        connections, so this is a no-op. Provided for interface consistency
        with other adapters that do require cleanup.
        """
        pass


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify ClaudeSDKParserAdapter implements ParserPort protocol."""
    # This is a static check - if this fails, it means the adapter
    # doesn't properly implement the protocol
    adapter_instance: ParserPort = None  # type: ignore[assignment]

    # The following would fail mypy if the protocol wasn't properly implemented
    # We don't actually run this, it's just for static analysis
    _ = adapter_instance  # Suppress unused warning
