"""LangChain Parser adapter implementing the ParserPort interface.

This module provides the LangChainParserAdapter class that wraps LangChain's
structured output capabilities behind the unified ParserPort interface.

IMPORTANT: This is NOT just JSON parsing. It invokes an LLM to interpret
natural language responses and extract structured data according to a schema.

Retry Logic:
    This adapter includes retry-with-feedback strategies for parsing failures:
    - Null-value feedback: When required fields are null, prompts LLM to provide defaults
    - Format feedback: When JSON format is invalid, prompts LLM for clean JSON only

    These strategies are applied automatically when initial parsing fails, giving
    the LLM a chance to correct common mistakes before raising ParseError.

Module Organization:
    - Helper Functions: _extract_null_fields_from_error (parser-specific)
    - Main Adapter Class: LangChainParserAdapter
    - Protocol Verification: Runtime check for ParserPort compliance
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from karenina.ports import Message, ParseError, ParserPort
from karenina.utils.json_extraction import extract_json_from_response, is_invalid_json_error

from .llm import LangChainLLMAdapter
from .messages import LangChainMessageConverter

if TYPE_CHECKING:
    from karenina.schemas.workflow.models import ModelConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_null_fields_from_error(
    error_str: str,
    failed_json: str | None = None,
) -> list[str]:
    """Extract field names that had null values from a parsing error.

    Used by retry strategies to identify which fields need actual values.

    Args:
        error_str: Error message string.
        failed_json: Optional JSON string that failed to parse.

    Returns:
        List of field names that had null/None values.
    """
    null_fields: list[str] = []

    # Approach 1: Try to extract JSON and find null fields
    if failed_json:
        try:
            data = json.loads(failed_json)
            null_fields = [k for k, v in data.items() if v is None]
            if null_fields:
                return null_fields
        except json.JSONDecodeError:
            pass

    # Approach 2: Parse Pydantic validation error
    lines = error_str.split("\n")
    for i, line in enumerate(lines):
        if "input_value=None" in line or "input_type=NoneType" in line:
            for j in range(i - 1, max(i - 3, -1), -1):
                potential_field = lines[j].strip()
                if (
                    potential_field
                    and " " not in potential_field
                    and potential_field not in ["Answer", "Input", "For", "Got:", "validation", "error"]
                ):
                    null_fields.append(potential_field)
                    break

    return list(set(null_fields))


# =============================================================================
# Main Adapter Class
# =============================================================================


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

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self, model_config: ModelConfig, *, max_retries: int = 2) -> None:
        """Initialize the LangChain parser adapter.

        Args:
            model_config: Configuration specifying model, provider, and interface.
            max_retries: Maximum number of retry attempts with feedback on parsing failure.
                Default is 2 retries (null-value feedback, then format feedback).
        """
        self._config = model_config
        self._converter = LangChainMessageConverter()
        self._llm_adapter = LangChainLLMAdapter(model_config)
        self._max_retries = max_retries

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def aparse_to_pydantic(self, response: str, schema: type[T]) -> T:
        """Parse an LLM response into a structured Pydantic model.

        This invokes an LLM to extract structured data from the response text.
        The LLM acts as a "judge" that interprets the natural language and
        fills in the schema attributes.

        Retry Strategy:
            On parsing failure, the adapter attempts recovery with feedback:
            1. Native structured output (if supported)
            2. Manual JSON parsing with json-repair
            3. Retry with null-value feedback (if error is null-related)
            4. Retry with format feedback (if error is JSON-format related)

        Args:
            response: The raw text response from an LLM (the "trace" to parse).
            schema: A Pydantic model class defining the expected structure.
                    Field descriptions guide the LLM on what to extract.

        Returns:
            An instance of the schema type with extracted values.

        Raises:
            ParseError: If the LLM fails to extract valid structured data after all retries.
            PortError: If the underlying LLM invocation fails.
        """
        messages = self._build_parsing_prompt(response, schema)

        # Strategy 1: Try native structured output for more reliable parsing
        try:
            structured_adapter = self._llm_adapter.with_structured_output(schema)
            llm_response = await structured_adapter.ainvoke(messages)

            # The raw response should be the parsed schema instance
            if isinstance(llm_response.raw, schema):
                logger.debug("Template parsing succeeded via native structured output")
                return llm_response.raw

            # Otherwise, try to parse the content
            result = self._parse_response_content(llm_response.content, schema)
            logger.debug("Template parsing succeeded via structured output content parsing")
            return result

        except Exception as structured_error:
            logger.debug(f"Structured output parsing failed: {structured_error}")

        # Strategy 2: Fallback to regular LLM invocation with manual JSON parsing
        llm_response = await self._llm_adapter.ainvoke(messages)
        raw_content = llm_response.content

        try:
            result = self._parse_response_content(raw_content, schema)
            logger.debug("Template parsing succeeded via fallback JSON parsing")
            return result
        except Exception as parse_error:
            logger.debug(f"Fallback JSON parsing failed: {parse_error}")

            # Check if retries are disabled
            if self._max_retries <= 0:
                raise ParseError(f"Failed to parse response: {parse_error}") from parse_error

            # Strategy 3: Try null-value feedback retry
            null_retry_result = await self._retry_with_null_feedback(
                original_messages=messages,
                failed_response=raw_content,
                error=parse_error,
                schema=schema,
            )

            if null_retry_result is not None:
                return null_retry_result

            # Strategy 4: Try format feedback retry
            format_retry_result = await self._retry_with_format_feedback(
                original_messages=messages,
                failed_response=raw_content,
                error=parse_error,
                schema=schema,
            )

            if format_retry_result is not None:
                return format_retry_result

            # All strategies failed
            preview = raw_content[:200] if len(raw_content) > 200 else raw_content
            raise ParseError(
                f"Failed to parse response into {schema.__name__} after all retry strategies: {preview}"
            ) from parse_error

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

    # -------------------------------------------------------------------------
    # Parsing Logic
    # -------------------------------------------------------------------------

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

        Uses extract_json_from_response from utils for JSON extraction,
        with json_repair as a fallback for malformed JSON.

        Args:
            content: The raw LLM response content.
            schema: The Pydantic schema to validate against.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ParseError: If all parsing strategies fail.
        """
        from pydantic import ValidationError

        # Strategy 1: Use shared JSON extraction utility
        try:
            json_str = extract_json_from_response(content)
            data = json.loads(json_str)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.debug(f"JSON extraction failed: {e}")

        # Strategy 2: JSON repair for malformed JSON
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

    # -------------------------------------------------------------------------
    # Retry Strategies
    # -------------------------------------------------------------------------

    async def _retry_with_null_feedback(
        self,
        original_messages: list[Message],
        failed_response: str,
        error: Exception,
        schema: type[T],
    ) -> T | None:
        """Retry parsing with feedback about null values in required fields.

        When parsing fails due to null values, this method:
        1. Extracts which fields had null values
        2. Sends feedback to LLM asking for actual values instead of nulls
        3. Retries parsing once

        Args:
            original_messages: Original messages that produced failed_response.
            failed_response: The response that failed to parse.
            error: The validation error from first parse attempt.
            schema: The Pydantic schema to parse into.

        Returns:
            Parsed answer if retry succeeds, None if retry also fails.
        """
        # Try to extract JSON from error message
        failed_json = None
        error_str = str(error)
        if "from completion" in error_str:
            try:
                json_start = error_str.index("{")
                json_end = error_str.index("}.", json_start) + 1
                failed_json = error_str[json_start:json_end]
            except (ValueError, IndexError):
                pass

        # Extract null fields
        null_fields = _extract_null_fields_from_error(error_str, failed_json)

        if not null_fields:
            logger.debug("Parsing error is not null-related, skipping null-value retry")
            return None

        logger.info(f"Detected null values in required fields: {null_fields}. Retrying with feedback...")

        # Build feedback message
        field_list = ", ".join(null_fields)
        feedback_prompt = f"""The previous response contained null values for required fields: [{field_list}].

Required fields cannot be null. Please provide actual values instead:
- If the information is not available in the source, provide an appropriate default value:
  * 0.0 for numeric fields (float/int)
  * Empty string "" for text fields
  * false for boolean fields
- If the field represents "unknown" or "not applicable", use a sensible placeholder
- **Never use null/None for required fields**

Previous response that failed:
{failed_response}

Please provide a corrected response with all required fields populated."""

        # Create retry messages
        retry_messages = list(original_messages)
        retry_messages.append(Message.user(feedback_prompt))

        try:
            llm_response = await self._llm_adapter.ainvoke(retry_messages)
            result = self._parse_response_content(llm_response.content, schema)
            logger.info(f"Successfully parsed after null-value retry. Fixed fields: {field_list}")
            return result

        except Exception as e:
            logger.warning(f"Retry parsing failed after null-value feedback: {e}")
            return None

    async def _retry_with_format_feedback(
        self,
        original_messages: list[Message],
        failed_response: str,
        error: Exception,
        schema: type[T],
    ) -> T | None:
        """Retry parsing with feedback about JSON format requirements.

        When parsing fails due to invalid JSON (e.g., reasoning text mixed with JSON),
        this method:
        1. Detects if the error is JSON-format related
        2. Sends clear feedback to LLM asking for clean JSON only
        3. Retries parsing once

        Args:
            original_messages: Original messages that produced failed_response.
            failed_response: The response that failed to parse.
            error: The validation error from first parse attempt.
            schema: The Pydantic schema to parse into.

        Returns:
            Parsed answer if retry succeeds, None if retry also fails.
        """
        # Only handle JSON format errors
        if not is_invalid_json_error(error):
            logger.debug("Error is not JSON-format related, skipping format feedback retry")
            return None

        logger.info("Detected invalid JSON output. Retrying with format feedback...")

        # Get schema hint
        schema_hint = ""
        try:
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            schema_hint = f"\n\nExpected schema:\n{schema_json}"
        except Exception:
            pass

        # Build feedback message
        feedback_prompt = f"""Your previous response could not be parsed as valid JSON.

**CRITICAL**: You must output ONLY a valid JSON object. Do not include:
- Any reasoning, explanation, or thinking
- Any text before or after the JSON
- Any markdown formatting (no ``` blocks)
- Any comments

**Your previous response that failed to parse:**
{failed_response[:1000]}{"..." if len(failed_response) > 1000 else ""}

**Error message:**
{str(error)[:500]}
{schema_hint}

Please respond with ONLY the JSON object, nothing else."""

        # Create retry messages
        retry_messages = list(original_messages)
        retry_messages.append(Message.user(feedback_prompt))

        try:
            llm_response = await self._llm_adapter.ainvoke(retry_messages)
            result = self._parse_response_content(llm_response.content, schema)
            logger.info("Successfully parsed after format feedback retry")
            return result

        except Exception as e:
            logger.warning(f"Retry parsing failed after format feedback: {e}")
            return None


# =============================================================================
# Protocol Verification
# =============================================================================


def _verify_protocol_compliance() -> None:
    """Verify LangChainParserAdapter implements ParserPort protocol."""
    adapter_instance: ParserPort = None  # type: ignore[assignment]
    _ = adapter_instance
