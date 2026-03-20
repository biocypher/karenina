"""LangChain Deep Agents Parser adapter implementing the ParserPort interface.

This module provides the DeepAgentsParserAdapter class that implements
ParserPort using LangChain's with_structured_output() for extracting
structured data from LLM responses.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from karenina.ports import ParseError
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.parser import ParsePortResult
from karenina.ports.usage import UsageMetadata

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

        # Create model with structured output
        chat_model = create_chat_model(self._config)

        try:
            # Use include_raw=True to get the AIMessage alongside the parsed output.
            # This is critical for usage tracking: without it, with_structured_output
            # returns only the parsed dict/model, losing the AIMessage.response_metadata
            # where token counts live.
            structured_model = chat_model.with_structured_output(schema, include_raw=True)
            raw_response = await structured_model.ainvoke(lc_messages)
        except Exception as e:
            logger.warning("Structured output failed, falling back to text extraction: %s", e)
            response = await chat_model.ainvoke(lc_messages)
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
            # Look for JSON in code blocks or raw JSON
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()

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

            def run_in_thread() -> ParsePortResult[T]:
                return asyncio.run(self.aparse_to_pydantic(messages, schema))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=600)

        except RuntimeError:
            return asyncio.run(self.aparse_to_pydantic(messages, schema))
