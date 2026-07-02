"""LangChain Deep Agents Parser adapter implementing the ParserPort interface.

This module provides the DeepAgentsParserAdapter class that implements
ParserPort using LangChain's with_structured_output() for extracting
structured data from LLM responses.

Retry Logic:
    aparse_to_pydantic() routes its LLM calls (structured and text fallback)
    through RetryExecutor with per-category retry budgets for transient
    errors. The underlying SDK clients run with max_retries=0 (see
    initialization.create_chat_model), so RetryExecutor is the sole retry
    layer and retry telemetry via track_retries() is accurate.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from karenina.adapters._parallel_base import run_coro_in_thread
from karenina.adapters._timeouts import (
    DEEP_AGENTS_SYNC_WRAPPER_FLOOR,
    PARSE_INTERNAL_CALL_SEQUENCES,
    compute_sync_wrapper_timeout,
)
from karenina.benchmark.verification.async_lifecycle import get_global_llm_limiter
from karenina.ports import ParseError
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.parser import ParsePortResult
from karenina.ports.usage import UsageMetadata
from karenina.utils.errors import ErrorRegistry
from karenina.utils.json_extraction import extract_json_from_response
from karenina.utils.retry_policy import RetryExecutor, RetryPolicy

from .initialization import create_chat_model
from .messages import DeepAgentsMessageConverter

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class DeepAgentsParserAdapter:
    """Parser adapter using LangChain's structured output for data extraction.

    Implements the ParserPort Protocol by using with_structured_output()
    on the LangChain model. Falls back to JSON extraction from text if
    structured output is not available.

    Example:
        >>> from pydantic import BaseModel, Field
        >>> class Answer(BaseModel):
        ...     gene: str = Field(description="Gene name")
        >>> parser = DeepAgentsParserAdapter(config)
        >>> result = await parser.aparse_to_pydantic(messages, Answer)
        >>> print(result.parsed.gene)
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize the Deep Agents Parser adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
        """
        self._config = model_config
        self._converter = DeepAgentsMessageConverter()

        retry_policy = model_config.retry_policy or RetryPolicy()
        self._retry_executor = RetryExecutor(retry_policy, ErrorRegistry())

    @property
    def capabilities(self) -> PortCapabilities:
        """Declare adapter capabilities.

        Returns:
            PortCapabilities with system_prompt=True and structured_output=True.
        """
        return PortCapabilities(
            supports_system_prompt=True,
            supports_structured_output=True,
        )

    async def aparse_to_pydantic(
        self,
        messages: list[Any],
        schema: type[T],
    ) -> ParsePortResult[T]:
        """Parse pre-assembled prompt messages into a Pydantic model.

        Uses LangChain's with_structured_output() to constrain the LLM.
        Falls back to JSON extraction if structured output fails.

        Args:
            messages: Pre-assembled prompt messages (system + user).
            schema: A Pydantic model class defining the expected structure.

        Returns:
            ParsePortResult containing the parsed model and usage metadata.

        Raises:
            ParseError: If the LLM fails to produce valid structured data.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        # Build LangChain message list
        lc_messages: list[Any] = []
        for msg in messages:
            text = msg.text or ""
            if msg.role.value == "system":
                lc_messages.append(SystemMessage(content=text))
            elif msg.role.value == "user":
                lc_messages.append(HumanMessage(content=text))
            elif msg.role.value == "assistant":
                lc_messages.append(AIMessage(content=text))

        # Create the model once, outside the retried callable, so that
        # retries repeat the API call rather than model construction.
        chat_model = create_chat_model(self._config)

        try:
            # Use include_raw=True to get the AIMessage alongside the parsed output.
            # This is critical for usage tracking: without it, with_structured_output
            # returns only the parsed dict/model, losing the AIMessage.response_metadata
            # where token counts live.
            structured_model = chat_model.with_structured_output(schema, include_raw=True)
            # This parser calls the raw LangChain model directly (it does
            # not delegate to an LLM adapter ainvoke), so it borrows its
            # own GlobalLLMLimiter permit per attempt inside
            # _ainvoke_with_timeout.
            raw_response = await self._retry_executor.aexecute_with_timeout(
                self._ainvoke_with_timeout,
                structured_model,
                lc_messages,
                timeout=self._config.request_timeout,
            )
        except Exception as e:
            if ErrorRegistry().classify(e).is_retryable():
                # A transient error here means the structured attempt
                # already exhausted its retry budget inside RetryExecutor.
                # Re-raise instead of paying a second budget on the text
                # fallback: a transient outage is not a missing
                # structured-output capability.
                raise
            logger.warning("Structured output failed, falling back to text extraction: %s", e)
            response = await self._retry_executor.aexecute_with_timeout(
                self._ainvoke_with_timeout,
                chat_model,
                lc_messages,
                timeout=self._config.request_timeout,
            )
            return self._extract_from_text(response, schema)

        # include_raw=True returns {"raw": AIMessage, "parsed": dict/model, "parsing_error": ...}
        parsed_output = raw_response.get("parsed") if isinstance(raw_response, dict) else raw_response
        raw_msg = raw_response.get("raw") if isinstance(raw_response, dict) else None
        usage = self._extract_usage_from_response(raw_msg) if raw_msg else UsageMetadata(model=self._config.model_name)

        if isinstance(parsed_output, schema):
            return ParsePortResult(parsed=parsed_output, usage=usage)

        if isinstance(parsed_output, dict):
            try:
                parsed = schema.model_validate(parsed_output)
                return ParsePortResult(parsed=parsed, usage=usage)
            except Exception as e:
                raise ParseError(f"Failed to validate structured output: {e}") from e

        if isinstance(parsed_output, BaseModel):
            try:
                parsed = schema.model_validate(parsed_output.model_dump())
                return ParsePortResult(parsed=parsed, usage=usage)
            except Exception as e:
                raise ParseError(f"Failed to convert structured output to target schema: {e}") from e

        raise ParseError(f"Unexpected response type from structured output: {type(parsed_output).__name__}")

    async def _ainvoke_with_timeout(
        self,
        model: Any,
        lc_messages: list[Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        """Call ``model.ainvoke`` under a wall-clock timeout as a guardrail.

        Mirrors the langchain adapter's helper: a fired timeout raises a
        stock ``asyncio.TimeoutError``, which ``ErrorRegistry`` classifies
        as ``TIMEOUT``, so the timeout retry budget applies inside
        ``RetryExecutor.aexecute_with_timeout``. Note that
        ``RetryPolicy.timeout_escalation`` only extends this guard on
        TIMEOUT retries. The underlying SDK client's own timeout stays
        pinned at ``request_timeout`` from construction, so the SDK may
        cut the request before an escalated guard fires.

        Each attempt holds one GlobalLLMLimiter permit for the wire call
        only (uniform per-attempt policy), so retry backoff sleeps never
        hold a permit. The permit wait itself is NOT bounded by the
        attempt timeout: a saturated limiter delays the attempt rather
        than timing it out, matching the legacy semaphore semantics.

        Args:
            model: The LangChain model (plain or structured) exposing
                ``ainvoke``.
            lc_messages: Provider-formatted messages to send.
            timeout: Optional per-call wall-clock timeout in seconds.
                When None, falls back to ``self._config.request_timeout``.
                When that is also None, the call is made without a wrapper.

        Returns:
            The raw model response object.

        Raises:
            asyncio.TimeoutError: If the call exceeds the effective timeout.
        """
        async with get_global_llm_limiter().borrow():
            if timeout is None:
                timeout = self._config.request_timeout
            if timeout is None:
                return await model.ainvoke(lc_messages)
            return await asyncio.wait_for(model.ainvoke(lc_messages), timeout=timeout)

    def _extract_from_text(self, response: Any, schema: type[T]) -> ParsePortResult[T]:
        """Extract structured data from a text response (fallback path).

        Args:
            response: The AIMessage response from the LLM.
            schema: The target Pydantic schema.

        Returns:
            ParsePortResult with the parsed model.

        Raises:
            ParseError: If JSON extraction or validation fails.
        """
        from langchain_core.messages import AIMessage

        content = ""
        if isinstance(response, AIMessage):
            content = response.content if isinstance(response.content, str) else str(response.content)
        else:
            content = str(response)

        usage = self._extract_usage_from_response(response)

        # Try to extract JSON from the text
        try:
            json_str = extract_json_from_response(content)
            data = json.loads(json_str)
            parsed = schema.model_validate(data)
            return ParsePortResult(parsed=parsed, usage=usage)
        except (json.JSONDecodeError, Exception) as e:
            raise ParseError(f"Failed to extract structured data from text response: {e}") from e

    def _extract_usage_from_response(self, response: Any) -> UsageMetadata:
        """Extract usage metadata from a LangChain response.

        Args:
            response: The response object (AIMessage or other).

        Returns:
            UsageMetadata with token counts if available.
        """
        from langchain_core.messages import AIMessage

        if isinstance(response, AIMessage):
            usage_meta = getattr(response, "usage_metadata", None)
            if usage_meta and isinstance(usage_meta, dict):
                return UsageMetadata(
                    input_tokens=usage_meta.get("input_tokens", 0),
                    output_tokens=usage_meta.get("output_tokens", 0),
                    total_tokens=usage_meta.get("input_tokens", 0) + usage_meta.get("output_tokens", 0),
                    model=self._config.model_name,
                )
        return UsageMetadata(model=self._config.model_name)

    def parse_to_pydantic(
        self,
        messages: list[Any],
        schema: type[T],
    ) -> ParsePortResult[T]:
        """Parse pre-assembled prompt messages (sync wrapper).

        Args:
            messages: Pre-assembled prompt messages.
            schema: A Pydantic model class defining the expected structure.

        Returns:
            ParsePortResult containing the parsed model and usage metadata.
        """
        from karenina.benchmark.verification.executor import get_async_portal

        portal = get_async_portal()
        if portal is not None:
            return portal.call(self.aparse_to_pydantic, messages, schema)

        try:
            asyncio.get_running_loop()
            # Fresh thread with the caller's context propagated, so
            # track_retries telemetry survives the dispatch.
            thread_timeout = compute_sync_wrapper_timeout(
                self._config.request_timeout,
                floor=DEEP_AGENTS_SYNC_WRAPPER_FLOOR,
                retry_policy=self._config.retry_policy,
                internal_call_sequences=PARSE_INTERNAL_CALL_SEQUENCES,
            )
            return run_coro_in_thread(self.aparse_to_pydantic, messages, schema, timeout=thread_timeout)

        except RuntimeError:
            return asyncio.run(self.aparse_to_pydantic(messages, schema))

    async def aclose(self) -> None:
        """Close underlying resources (no-op for this adapter)."""
