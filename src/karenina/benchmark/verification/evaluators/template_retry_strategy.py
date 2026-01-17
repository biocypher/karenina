"""Template parsing retry strategies.

This module provides retry strategies for template parsing failures.
These strategies handle common failure modes like null values in required fields
and malformed JSON output.

These functions are designed to be used by TemplateEvaluator when initial parsing fails.
"""

import json
import logging
import re
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import BaseMessage, HumanMessage

logger = logging.getLogger(__name__)


class TemplateRetryHandler:
    """
    Handles retry strategies for template parsing failures.

    This class encapsulates retry logic for common parsing failure modes:
    - Null values in required fields
    - Invalid JSON format (mixed text, markdown fences, etc.)

    Example:
        handler = TemplateRetryHandler(llm=parsing_llm, parser=pydantic_parser)

        # Try null-value retry
        result, usage = handler.retry_with_null_feedback(
            original_messages=messages,
            failed_response=response_text,
            error=parse_error,
        )

        # Try format feedback retry
        result, usage = handler.retry_with_format_feedback(
            original_messages=messages,
            failed_response=response_text,
            error=parse_error,
        )
    """

    def __init__(self, llm: Any, parser: Any):
        """
        Initialize the retry handler.

        Args:
            llm: LangChain chat model for retry invocations
            parser: PydanticOutputParser instance for parsing responses
        """
        self.llm = llm
        self.parser = parser

    def retry_with_null_feedback(
        self,
        original_messages: list[BaseMessage],
        failed_response: str,
        error: Exception,
    ) -> tuple[Any | None, dict[str, Any]]:
        """
        Retry parsing with feedback about null values in required fields.

        When parsing fails due to null values, this method:
        1. Extracts which fields had null values
        2. Sends feedback to LLM asking for actual values instead of nulls
        3. Retries parsing once

        Args:
            original_messages: Original messages that produced failed_response
            failed_response: The response that failed to parse
            error: The validation error from first parse attempt

        Returns:
            Tuple of (parsed_answer, usage_metadata)
            parsed_answer is None if retry also fails
        """
        from ..utils.json_helpers import strip_markdown_fences as _strip_markdown_fences

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
        null_fields = self._extract_null_fields_from_error(error_str, failed_json)

        if not null_fields:
            logger.debug("Parsing error is not null-related, skipping retry")
            return None, {}

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
        retry_messages.append(HumanMessage(content=feedback_prompt))

        try:
            with get_usage_metadata_callback() as cb:
                response = self.llm.invoke(retry_messages)

            usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}

            raw_response = response.content if hasattr(response, "content") else str(response)
            cleaned = _strip_markdown_fences(raw_response)
            parsed = self.parser.parse(cleaned)

            logger.info(f"Successfully parsed after null-value retry. Fixed fields: {field_list}")
            return parsed, usage_metadata

        except Exception as e:
            logger.warning(f"Retry parsing failed after null-value feedback: {e}")
            return None, {}

    def retry_with_format_feedback(
        self,
        original_messages: list[BaseMessage],
        failed_response: str,
        error: Exception,
    ) -> tuple[Any | None, dict[str, Any]]:
        """
        Retry parsing with feedback about JSON format requirements.

        When parsing fails due to invalid JSON (e.g., reasoning text mixed with JSON),
        this method:
        1. Detects if the error is JSON-format related
        2. Sends clear feedback to LLM asking for clean JSON only
        3. Retries parsing once

        Args:
            original_messages: Original messages that produced failed_response
            failed_response: The response that failed to parse
            error: The validation error from first parse attempt

        Returns:
            Tuple of (parsed_answer, usage_metadata)
            parsed_answer is None if retry also fails
        """
        from ..utils.json_helpers import strip_markdown_fences as _strip_markdown_fences

        # Only handle JSON format errors
        if not self._is_invalid_json_error(error):
            logger.debug("Error is not JSON-format related, skipping format feedback retry")
            return None, {}

        logger.info("Detected invalid JSON output. Retrying with format feedback...")

        # Get schema hint
        try:
            format_instructions = self.parser.get_format_instructions()
            schema_hint = ""
            if "```" in format_instructions:
                schema_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", format_instructions, re.DOTALL)
                if schema_match:
                    schema_hint = f"\n\nExpected schema:\n{schema_match.group(1).strip()}"
        except Exception:
            schema_hint = ""

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
        retry_messages.append(HumanMessage(content=feedback_prompt))

        try:
            with get_usage_metadata_callback() as cb:
                response = self.llm.invoke(retry_messages)

            usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}

            raw_response = response.content if hasattr(response, "content") else str(response)
            cleaned = _strip_markdown_fences(raw_response)
            parsed = self.parser.parse(cleaned)

            logger.info("Successfully parsed after format feedback retry")
            return parsed, usage_metadata

        except Exception as e:
            logger.warning(f"Retry parsing failed after format feedback: {e}")
            return None, {}

    def _extract_null_fields_from_error(
        self,
        error_str: str,
        failed_json: str | None = None,
    ) -> list[str]:
        """
        Extract field names that had null values from parsing error.

        Args:
            error_str: Error message string
            failed_json: Optional JSON string that failed to parse

        Returns:
            List of field names that had null/None values
        """
        null_fields = []

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

    def _is_invalid_json_error(self, error: Exception) -> bool:
        """Check if an error is related to invalid JSON output.

        Args:
            error: The exception from parsing attempt

        Returns:
            True if this is an invalid JSON error
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        json_error_patterns = [
            "invalid json",
            "json decode",
            "jsondecodeerror",
            "expecting value",
            "expecting property name",
            "unterminated string",
            "extra data",
            "invalid control character",
            "invalid \\escape",
            "invalid literal",
            "no json object could be decoded",
            "output_parsing_failure",
        ]

        if any(pattern in error_str for pattern in json_error_patterns):
            return True

        return error_type in ["JSONDecodeError", "OutputParserException"]
