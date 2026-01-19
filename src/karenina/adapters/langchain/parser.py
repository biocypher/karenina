"""LangChain Parser adapter implementing the ParserPort interface.

This module provides the LangChainParserAdapter class that wraps LangChain's
structured output capabilities behind the unified ParserPort interface.

IMPORTANT: This is NOT just JSON parsing. It invokes an LLM to interpret
natural language responses and extract structured data according to a schema.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from karenina.ports import Message, ParseError, ParserPort

from .llm import LangChainLLMAdapter
from .messages import LangChainMessageConverter

if TYPE_CHECKING:
    from karenina.schemas.workflow.models import ModelConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LangChainParserAdapter:
    """Parser adapter using LangChain's structured output capabilities.

    This adapter implements the ParserPort Protocol and provides LLM-based
    structured output parsing. It invokes an LLM (the "judge" model) to
    interpret natural language responses and extract structured data.

    The parsing flow:
    1. Build a prompt with the response text and schema description
    2. Invoke the LLM with structured output enabled
    3. Parse and validate the LLM output into a Pydantic model
    4. Return the validated model instance

    Example:
        >>> from karenina.schemas.workflow.models import ModelConfig
        >>> from pydantic import BaseModel, Field

        >>> class Answer(BaseModel):
        ...     gene_name: str = Field(description="The gene mentioned")
        ...     is_oncogene: bool = Field(description="Whether it's an oncogene")

        >>> config = ModelConfig(
        ...     id="claude-sonnet",
        ...     model_name="claude-sonnet-4-20250514",
        ...     model_provider="anthropic"
        ... )
        >>> parser = LangChainParserAdapter(config)
        >>> trace = "Based on my analysis, BCL2 is an anti-apoptotic gene..."
        >>> answer = await parser.aparse_to_pydantic(trace, Answer)
        >>> print(answer.gene_name)
        'BCL2'
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize the LangChain parser adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
        """
        self._config = model_config
        self._converter = LangChainMessageConverter()
        self._llm_adapter = LangChainLLMAdapter(model_config)

    def _build_parsing_prompt(self, response: str, schema: type[BaseModel]) -> list[Message]:
        """Build the parsing prompt with response and schema.

        Args:
            response: The raw text response to parse.
            schema: The Pydantic schema defining expected structure.

        Returns:
            List of Message objects for the parsing request.
        """
        # Generate JSON schema from the Pydantic model
        json_schema = json.dumps(schema.model_json_schema(), indent=2)

        system_content = """You are an evaluator that extracts structured information from responses.

You will receive:
1. A response to parse (from an LLM or other source)
2. A JSON schema with descriptive fields indicating what information to extract

# Extraction Protocol

## 1. Extract According to Schema
- Each field description specifies WHAT to extract from the response
- Follow field descriptions precisely
- Use `null` for information not present (if field allows null)

## 2. Validate Structure
- Return valid JSON matching the provided schema exactly
- Use correct data types for each field

# Critical Rules

**Fidelity**: Extract only what's actually stated. Don't infer or add information not present.

**JSON Only**: Return ONLY the JSON object - no explanations, no markdown fences, no surrounding text."""

        user_content = f"""Parse the following response and extract structured information.

**RESPONSE TO PARSE:**
{response}

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**PARSING NOTES:**
- Extract values for each field based on its description in the schema
- If information for a field is not present, use null (if field allows null) or your best inference
- Return ONLY the JSON object - no surrounding text

**YOUR JSON RESPONSE:**"""

        return [
            Message.system(system_content),
            Message.user(user_content),
        ]

    def _parse_response_content(self, content: str, schema: type[T]) -> T:
        """Parse the LLM response content into the schema.

        Uses multiple strategies to extract valid JSON from the response:
        1. Direct JSON parsing
        2. Strip markdown fences and retry
        3. Extract JSON object from mixed text
        4. JSON repair for malformed JSON

        Args:
            content: The raw LLM response content.
            schema: The Pydantic schema to validate against.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ParseError: If all parsing strategies fail.
        """
        from pydantic import ValidationError

        # Strategy 1: Direct JSON parsing
        try:
            data = json.loads(content.strip())
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Direct JSON parse failed: {e}")

        # Strategy 2: Strip markdown fences
        cleaned = self._strip_markdown_fences(content)
        if cleaned and cleaned != content:
            try:
                data = json.loads(cleaned)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.debug(f"Cleaned text parse failed: {e}")

        # Strategy 3: Extract JSON object from text
        json_str = self._extract_json_from_text(content)
        if json_str:
            try:
                data = json.loads(json_str)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.debug(f"Extracted JSON parse failed: {e}")

        # Strategy 4: JSON repair
        try:
            from json_repair import repair_json

            repaired = repair_json(content)
            data = json.loads(repaired)
            return schema.model_validate(data)
        except ImportError:
            logger.debug("json-repair not installed, skipping repair strategy")
        except Exception as e:
            logger.debug(f"JSON repair failed: {e}")

        # All strategies failed
        preview = content[:200] if len(content) > 200 else content
        raise ParseError(f"Could not parse response into {schema.__name__}: {preview}")

    def _strip_markdown_fences(self, text: str) -> str | None:
        """Strip markdown code fences from text.

        Args:
            text: Text that may contain markdown fences.

        Returns:
            Text with fences removed, or None if empty.
        """
        import re

        # Pattern for ```json ... ``` or ``` ... ```
        pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return text.strip() if text else None

    def _extract_json_from_text(self, text: str) -> str | None:
        """Extract JSON object from mixed text.

        Finds the first { ... } block that looks like valid JSON.

        Args:
            text: Text containing embedded JSON.

        Returns:
            Extracted JSON string, or None if not found.
        """
        # Find first { and last } to extract JSON object
        start = text.find("{")
        if start == -1:
            return None

        # Count braces to find matching }
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None

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
        # First, try using native structured output for more reliable parsing
        try:
            structured_adapter = self._llm_adapter.with_structured_output(schema)
            messages = self._build_parsing_prompt(response, schema)

            llm_response = await structured_adapter.ainvoke(messages)

            # The raw response should be the parsed schema instance
            if isinstance(llm_response.raw, schema):
                return llm_response.raw

            # Otherwise, try to parse the content
            return self._parse_response_content(llm_response.content, schema)

        except Exception as structured_error:
            logger.debug(f"Structured output parsing failed: {structured_error}")

        # Fallback: Use regular LLM invocation with manual parsing
        try:
            messages = self._build_parsing_prompt(response, schema)
            llm_response = await self._llm_adapter.ainvoke(messages)
            return self._parse_response_content(llm_response.content, schema)
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"Failed to parse response: {e}") from e

    def parse_to_pydantic(self, response: str, schema: type[T]) -> T:
        """Parse an LLM response into a structured Pydantic model (sync).

        This is a convenience wrapper around aparse_to_pydantic() for sync code.

        Args:
            response: The raw text response from an LLM (the "trace" to parse).
            schema: A Pydantic model class defining the expected structure.

        Returns:
            An instance of the schema type with extracted values.

        Raises:
            ParseError: If the LLM fails to extract valid structured data.
            PortError: If the underlying LLM invocation fails.
        """
        from karenina.benchmark.verification.batch_runner import get_async_portal

        portal = get_async_portal()

        if portal is not None:
            return portal.call(self.aparse_to_pydantic, response, schema)

        # Check if we're in an async context
        try:
            asyncio.get_running_loop()

            # Use ThreadPoolExecutor to avoid nested event loop issues
            def run_in_thread() -> T:
                return asyncio.run(self.aparse_to_pydantic(response, schema))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=300)

        except RuntimeError:
            # No event loop running
            return asyncio.run(self.aparse_to_pydantic(response, schema))


# Verify protocol compliance at import time
def _verify_protocol_compliance() -> None:
    """Verify LangChainParserAdapter implements ParserPort protocol."""
    adapter_instance: ParserPort = None  # type: ignore[assignment]
    _ = adapter_instance
