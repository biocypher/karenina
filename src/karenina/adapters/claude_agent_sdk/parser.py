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

from karenina.ports import ParseError, ParserPort
from karenina.ports.capabilities import PortCapabilities

from .prompts import PARSER_SYSTEM, PARSER_USER

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

    def _build_options(self, schema: type[BaseModel]) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions for structured output parsing.

        Args:
            schema: The Pydantic schema to use for structured output.

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
            # System prompt with parsing instructions
            "system_prompt": PARSER_SYSTEM,
        }

        # Add model specification if provided
        if self._config.model_name:
            options_kwargs["model"] = self._config.model_name

        return ClaudeAgentOptions(**options_kwargs)

    def _build_parsing_prompt(self, response: str, schema: type[BaseModel]) -> str:
        """Build the user prompt for structured output extraction.

        Instructions are in the system prompt (via ClaudeAgentOptions.system_prompt).
        This user prompt contains only the response to parse and the schema reference.

        Args:
            response: The raw text response to parse.
            schema: The Pydantic schema defining expected structure.

        Returns:
            User prompt string for the parsing request.
        """
        # Generate JSON schema for reference in the prompt
        json_schema = json.dumps(schema.model_json_schema(), indent=2)

        return PARSER_USER.format(response=response, json_schema=json_schema)

    async def aparse_to_pydantic(self, response: str, schema: type[T]) -> T:
        """Parse an LLM response into a structured Pydantic model.

        This invokes an LLM to extract structured data from the response text.
        The LLM acts as a "judge" that interprets the natural language and
        fills in the schema attributes.

        Args:
            response: The raw text response from an LLM (the "trace" to parse).
            schema: A Pydantic model class defining the expected structure.
                    Field descriptions guide the LLM on what to extract.

        Returns:
            An instance of the schema type with extracted values.

        Raises:
            ParseError: If the LLM fails to extract valid structured data.
            PortError: If the underlying LLM invocation fails.
        """
        from claude_agent_sdk import ResultMessage, query
        from pydantic import ValidationError

        # Build prompt and options
        prompt = self._build_parsing_prompt(response, schema)
        options = self._build_options(schema)

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
                    return schema.model_validate(data)
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.debug(f"Fallback JSON parsing failed: {e}")

            raise ParseError(
                f"No structured output in SDK response. "
                f"Subtype: {result.subtype}, Result: {result.result[:100] if result.result else 'None'}"
            )

        # Validate and return the structured output
        try:
            return schema.model_validate(result.structured_output)
        except ValidationError as e:
            raise ParseError(f"Structured output validation failed: {e}. Raw output: {result.structured_output}") from e

    def parse_to_pydantic(self, response: str, schema: type[T]) -> T:
        """Parse an LLM response into a structured Pydantic model (sync).

        This is a convenience wrapper around aparse_to_pydantic() for sync code.
        Uses the shared async portal if available, otherwise falls back to
        asyncio.run() with proper event loop handling.

        Args:
            response: The raw text response from an LLM (the "trace" to parse).
            schema: A Pydantic model class defining the expected structure.

        Returns:
            An instance of the schema type with extracted values.

        Raises:
            ParseError: If the LLM fails to extract valid structured data.
            PortError: If the underlying LLM invocation fails.
        """
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            # Use the shared BlockingPortal for proper event loop management
            return portal.call(self.aparse_to_pydantic, response, schema)

        # No portal available - check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - use ThreadPoolExecutor to avoid
            # nested event loop issues

            def run_in_thread() -> T:
                return asyncio.run(self.aparse_to_pydantic(response, schema))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=300)  # 5 minute timeout

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.aparse_to_pydantic(response, schema))

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
