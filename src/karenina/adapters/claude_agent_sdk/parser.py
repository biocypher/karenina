"""Claude Agent SDK Parser adapter implementing the ParserPort interface.

This module provides the ClaudeSDKParserAdapter that uses direct API calls
for structured output parsing, bypassing the Agent SDK's subprocess transport.

The adapter detects the endpoint type from the model configuration:

- **Anthropic API** (no custom base URL): Uses ``anthropic.AsyncAnthropic``
  with ``output_config`` for native constrained decoding.
- **Custom endpoint** (vLLM, sglang): Uses ``openai.AsyncOpenAI`` with
  ``response_format`` on the OpenAI-compatible endpoint (auto-derived from
  the same host). Logs a warning on first use.

This hybrid approach gives guaranteed schema enforcement on both endpoint
types while keeping the Agent SDK's subprocess transport for agent and LLM
operations (where it works correctly).

Retry Logic:
    API calls route through RetryExecutor with per-category retry budgets
    for transient errors (connection errors, timeouts, rate limits, 5xx
    errors). The SDK clients run with max_retries=0 so RetryExecutor is
    the sole retry layer and retry telemetry via track_retries() is
    accurate.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from karenina.adapters._parallel_base import run_coro_in_thread
from karenina.ports import Message, ParseError, ParsePortResult, ParserPort, UsageMetadata
from karenina.ports.capabilities import PortCapabilities
from karenina.utils.errors import ErrorRegistry
from karenina.utils.json_extraction import extract_json_from_response
from karenina.utils.retry_policy import RetryExecutor, RetryPolicy

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _enforce_no_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively set ``additionalProperties: false`` on all object types.

    Anthropic's ``output_config`` requires this; Pydantic's
    ``model_json_schema()`` omits it (defaulting to true per JSON Schema spec).

    Args:
        schema: A JSON schema dict (mutated in place and returned).

    Returns:
        The same schema dict with ``additionalProperties`` set on all objects.
    """
    if schema.get("type") == "object":
        schema["additionalProperties"] = False

    # Walk nested schemas: properties, items, $defs, allOf/anyOf/oneOf
    for prop in schema.get("properties", {}).values():
        _enforce_no_additional_properties(prop)
    if "items" in schema and isinstance(schema["items"], dict):
        _enforce_no_additional_properties(schema["items"])
    for defn in schema.get("$defs", {}).values():
        _enforce_no_additional_properties(defn)
    for keyword in ("allOf", "anyOf", "oneOf"):
        for sub in schema.get(keyword, []):
            if isinstance(sub, dict):
                _enforce_no_additional_properties(sub)

    return schema


def _derive_openai_base_url(anthropic_base_url: str) -> str:
    """Derive an OpenAI-compatible base URL from an Anthropic base URL.

    Both vLLM and sglang expose OpenAI and Anthropic endpoints on the same
    host and port. This strips any path from the Anthropic URL and appends
    ``/v1`` for the OpenAI endpoint.

    Args:
        anthropic_base_url: The custom Anthropic endpoint URL.

    Returns:
        OpenAI-compatible base URL (e.g. ``http://host:8000/v1``).
    """
    parsed = urlparse(anthropic_base_url)
    return f"{parsed.scheme}://{parsed.netloc}/v1"


def _configured_openai_base_url(model_config: ModelConfig) -> str | None:
    """Return an explicit OpenAI-compatible parser endpoint override, if configured."""

    extra_kwargs = getattr(model_config, "extra_kwargs", None) or {}
    value = extra_kwargs.get("claude_sdk_parser_openai_base_url")
    return str(value) if value else None


class ClaudeSDKParserAdapter:
    """Parser adapter using direct API calls for structured output.

    Bypasses the Agent SDK subprocess (which hangs on custom endpoints with
    ``--json-schema``) by calling the API directly. Uses constrained decoding
    for schema enforcement on both Anthropic API and custom endpoints.

    Example:
        >>> from karenina.schemas.config import ModelConfig
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
        >>> answer = await parser.aparse_to_pydantic(messages, Answer)
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize the Claude SDK parser adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
        """
        self._config = model_config
        self._is_custom_endpoint = bool(model_config.anthropic_base_url)
        self._anthropic_client: Any | None = None
        self._openai_client: Any | None = None
        self._warned_custom_endpoint = False

        retry_policy = model_config.retry_policy or RetryPolicy()
        self._retry_executor = RetryExecutor(retry_policy, ErrorRegistry())

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare what prompt features this parser adapter supports.

        Returns:
            PortCapabilities with system prompt and structured output support.
        """
        return PortCapabilities(supports_system_prompt=True, supports_structured_output=True)

    def _get_anthropic_client(self) -> Any:
        """Get or create the async Anthropic client (for Anthropic API)."""
        if self._anthropic_client is None:
            import anthropic

            kwargs: dict[str, Any] = {}
            if self._config.anthropic_api_key:
                kwargs["api_key"] = self._config.anthropic_api_key.get_secret_value()
            if self._config.request_timeout is not None:
                kwargs["timeout"] = self._config.request_timeout

            # Suppress SDK-level retries. RetryExecutor is the sole retry
            # layer (design decision D1). Known cost: the SDK's honoring of
            # server retry-after headers is dropped. Deliberate deferred
            # follow-up for first-party Anthropic endpoints.
            kwargs["max_retries"] = 0

            self._anthropic_client = anthropic.AsyncAnthropic(**kwargs)
        return self._anthropic_client

    def _get_openai_client(self) -> Any:
        """Get or create the async OpenAI client (for custom endpoints)."""
        if self._openai_client is None:
            from openai import AsyncOpenAI

            base_url = _configured_openai_base_url(self._config) or _derive_openai_base_url(
                self._config.anthropic_base_url  # type: ignore[arg-type]
            )
            api_key = self._config.anthropic_api_key.get_secret_value() if self._config.anthropic_api_key else "EMPTY"
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "base_url": base_url,
            }
            if self._config.request_timeout is not None:
                kwargs["timeout"] = self._config.request_timeout

            # Suppress SDK-level retries. RetryExecutor is the sole retry
            # layer (design decision D1, see _get_anthropic_client for the
            # deferred retry-after follow-up).
            kwargs["max_retries"] = 0

            self._openai_client = AsyncOpenAI(**kwargs)

            if not self._warned_custom_endpoint:
                logger.warning(
                    "Custom Anthropic endpoint detected (%s). "
                    "Parsing will use the OpenAI-compatible endpoint (%s) "
                    "for schema enforcement via response_format. "
                    "This is compatible with vLLM and sglang.",
                    self._config.anthropic_base_url,
                    base_url,
                )
                self._warned_custom_endpoint = True
        return self._openai_client

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

    async def _parse_via_anthropic(
        self,
        system_text: str,
        user_text: str,
        schema: type[T],
    ) -> ParsePortResult[T]:
        """Parse via Anthropic API with output_config (constrained decoding).

        Args:
            system_text: System prompt text.
            user_text: User prompt text.
            schema: Pydantic model class for the expected structure.

        Returns:
            ParsePortResult with the validated model instance.
        """
        client = self._get_anthropic_client()
        json_schema = _enforce_no_additional_properties(schema.model_json_schema())

        kwargs: dict[str, Any] = {
            "model": self._config.model_name,
            "max_tokens": self._config.max_tokens or 4096,
            "messages": [{"role": "user", "content": user_text}],
            "output_config": {
                "format": {"type": "json_schema", "schema": json_schema},
            },
        }
        if system_text:
            kwargs["system"] = system_text
        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature

        try:
            response = await self._retry_executor.aexecute_with_timeout(
                self._acall_with_timeout,
                client.messages.create,
                timeout=self._config.request_timeout,
                **kwargs,
            )
        except Exception as e:
            raise ParseError(f"Anthropic API call failed during parsing: {e}") from e

        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        raw_text = "\n".join(text_parts)

        usage = UsageMetadata(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", None),
            cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", None),
            model=getattr(response, "model", None) or self._config.model_name,
        )

        return self._validate_json_response(raw_text, schema, usage)

    async def _parse_via_openai(
        self,
        system_text: str,
        user_text: str,
        schema: type[T],
    ) -> ParsePortResult[T]:
        """Parse via OpenAI-compatible endpoint with response_format (constrained decoding).

        Args:
            system_text: System prompt text.
            user_text: User prompt text.
            schema: Pydantic model class for the expected structure.

        Returns:
            ParsePortResult with the validated model instance.
        """
        client = self._get_openai_client()
        json_schema = _enforce_no_additional_properties(schema.model_json_schema())

        messages: list[dict[str, str]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})

        kwargs: dict[str, Any] = {
            "model": self._config.model_name,
            "max_tokens": self._config.max_tokens or 4096,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": json_schema,
                    "strict": True,
                },
            },
        }
        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature

        try:
            response = await self._retry_executor.aexecute_with_timeout(
                self._acall_with_timeout,
                client.chat.completions.create,
                timeout=self._config.request_timeout,
                **kwargs,
            )
        except Exception as e:
            raise ParseError(f"OpenAI API call failed during parsing: {e}") from e

        raw_text = response.choices[0].message.content or ""

        usage = UsageMetadata(
            input_tokens=getattr(response.usage, "prompt_tokens", 0),
            output_tokens=getattr(response.usage, "completion_tokens", 0),
            total_tokens=getattr(response.usage, "total_tokens", 0),
            model=getattr(response, "model", None) or self._config.model_name,
        )

        return self._validate_json_response(raw_text, schema, usage)

    async def _acall_with_timeout(
        self,
        call: Any,
        *,
        timeout: float | None = None,
        **call_kwargs: Any,
    ) -> Any:
        """Run an async SDK call under a wall-clock timeout as a guardrail.

        The SDK client already carries an httpx-level timeout from
        ``request_timeout``, but that does not catch every stall, so this
        karenina-layer ``asyncio.wait_for`` enforces a hard per-attempt
        wall-clock budget (mirrors the langchain adapter's helper).

        A fired timeout raises a stock ``asyncio.TimeoutError``, which
        ``ErrorRegistry`` classifies as ``TIMEOUT`` via the built-in MRO
        check, so the timeout retry budget applies inside
        ``RetryExecutor.aexecute_with_timeout``. Note that
        ``RetryPolicy.timeout_escalation`` only extends this guard on
        TIMEOUT retries. The SDK client's own timeout stays pinned at
        ``request_timeout`` from construction, so the SDK may cut the
        request before an escalated guard fires.

        Args:
            call: The bound async SDK method to invoke (for example
                ``client.messages.create``).
            timeout: Optional per-call wall-clock timeout in seconds.
                When None, falls back to ``self._config.request_timeout``.
                When that is also None, the call is made without a wrapper.
            **call_kwargs: Keyword arguments forwarded to ``call``.

        Returns:
            The raw SDK response object.

        Raises:
            asyncio.TimeoutError: If the call exceeds the effective timeout.
        """
        if timeout is None:
            timeout = self._config.request_timeout
        if timeout is None:
            return await call(**call_kwargs)
        return await asyncio.wait_for(call(**call_kwargs), timeout=timeout)

    def _validate_json_response(
        self,
        raw_text: str,
        schema: type[T],
        usage: UsageMetadata,
    ) -> ParsePortResult[T]:
        """Extract JSON from response text and validate against the schema.

        Tries direct ``json.loads`` first (expected when constrained decoding
        is active), then falls back to ``extract_json_from_response`` for
        responses wrapped in code fences or mixed with reasoning text.

        Args:
            raw_text: Raw text from the API response.
            schema: Pydantic model class to validate against.
            usage: Pre-built usage metadata.

        Returns:
            ParsePortResult with the validated model instance.
        """
        if not raw_text.strip():
            raise ParseError("Empty response from API during parsing")

        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            try:
                json_str = extract_json_from_response(raw_text)
                data = json.loads(json_str)
            except (ValueError, json.JSONDecodeError) as e:
                raise ParseError(f"Failed to extract JSON from response: {e}. Raw text: {raw_text[:300]}") from e

        try:
            return ParsePortResult(parsed=schema.model_validate(data), usage=usage)
        except ValidationError as e:
            raise ParseError(
                f"Schema validation failed for {schema.__name__}: {e}. Extracted data: {json.dumps(data)[:300]}"
            ) from e

    async def aparse_to_pydantic(self, messages: list[Message], schema: type[T]) -> ParsePortResult[T]:
        """Parse pre-assembled prompt messages into a structured Pydantic model.

        Routes to the Anthropic API (with ``output_config``) or the
        OpenAI-compatible endpoint (with ``response_format``) based on
        whether a custom base URL is configured.

        Args:
            messages: Pre-assembled prompt messages (system + user).
            schema: A Pydantic model class defining the expected structure.

        Returns:
            ParsePortResult containing the parsed model and usage metadata.

        Raises:
            ParseError: If the LLM fails to produce valid structured data.
        """
        system_text, user_text = self._extract_from_messages(messages)

        if self._is_custom_endpoint:
            return await self._parse_via_openai(system_text, user_text, schema)
        return await self._parse_via_anthropic(system_text, user_text, schema)

    def parse_to_pydantic(self, messages: list[Message], schema: type[T]) -> ParsePortResult[T]:
        """Parse using pre-assembled prompt messages (sync).

        Uses the shared async portal if available, otherwise falls back to
        asyncio.run() with proper event loop handling.

        Args:
            messages: Pre-assembled prompt messages (system + user).
            schema: A Pydantic model class defining the expected structure.

        Returns:
            ParsePortResult containing the parsed model and usage metadata.

        Raises:
            ParseError: If parsing fails.
        """
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            return portal.call(self.aparse_to_pydantic, messages, schema)

        try:
            asyncio.get_running_loop()
            # Fresh thread with the caller's context propagated, so
            # track_retries telemetry survives the dispatch.
            return run_coro_in_thread(self.aparse_to_pydantic, messages, schema, timeout=300)

        except RuntimeError:
            return asyncio.run(self.aparse_to_pydantic(messages, schema))

    async def aclose(self) -> None:
        """Close underlying HTTP clients."""
        if self._anthropic_client is not None:
            await self._anthropic_client.close()
            self._anthropic_client = None
        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify ClaudeSDKParserAdapter implements ParserPort protocol."""
    adapter_instance: ParserPort = None  # type: ignore[assignment]
    _ = adapter_instance
