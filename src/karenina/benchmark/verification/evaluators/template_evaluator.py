"""Template evaluation for parsing and verifying LLM responses.

This module provides the TemplateEvaluator class which encapsulates all template
parsing and verification logic, following the same pattern as RubricEvaluator.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from ....infrastructure.llm.interface import init_chat_model_unified
from ....infrastructure.llm.mcp_utils import extract_final_ai_message
from ....schemas.domain import BaseAnswer
from ....schemas.workflow import INTERFACES_NO_PROVIDER_REQUIRED, ModelConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Result Dataclasses
# ============================================================================


@dataclass
class ParseResult:
    """Result of template parsing operation."""

    parsed_answer: Any | None = None
    """Parsed Pydantic object (Answer instance)."""

    success: bool = False
    """Whether parsing succeeded."""

    error: str | None = None
    """Error message if parsing failed."""

    # Deep judgment metadata
    deep_judgment_performed: bool = False
    """Whether deep judgment was used for parsing."""

    extracted_excerpts: dict[str, list[dict[str, Any]]] | None = None
    """Excerpts extracted per attribute (if deep judgment enabled)."""

    attribute_reasoning: dict[str, str] | None = None
    """Reasoning traces per attribute (if deep judgment enabled)."""

    deep_judgment_stages_completed: list[str] | None = None
    """List of completed deep judgment stages."""

    deep_judgment_model_calls: int = 0
    """Number of LLM calls made during deep judgment."""

    deep_judgment_excerpt_retry_count: int = 0
    """Number of excerpt extraction retries."""

    attributes_without_excerpts: list[str] | None = None
    """Attributes that failed excerpt extraction."""

    hallucination_risk_assessment: dict[str, Any] | None = None
    """Hallucination risk per attribute (if search enabled)."""

    usage_metadata_list: list[dict[str, Any]] = field(default_factory=list)
    """Usage metadata from LLM calls."""


@dataclass
class FieldVerificationResult:
    """Result of field verification."""

    success: bool = False
    """Whether all fields verified successfully."""

    error: str | None = None
    """Error message if verification failed."""


@dataclass
class RegexVerificationResult:
    """Result of regex verification."""

    success: bool = False
    """Whether all regex patterns matched."""

    results: dict[str, bool] = field(default_factory=dict)
    """Per-field regex results."""

    details: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Detailed regex match information per field."""

    extraction_results: dict[str, list[str]] = field(default_factory=dict)
    """Actual regex matches extracted per field."""

    error: str | None = None
    """Error message if verification failed."""


# ============================================================================
# TemplateEvaluator Class
# ============================================================================


class TemplateEvaluator:
    """
    Evaluates LLM responses by parsing them into Pydantic objects and verifying templates.

    This class encapsulates all template parsing and verification logic,
    following the same pattern as RubricEvaluator for architectural consistency.

    The evaluator supports:
    - Standard parsing with multiple fallback strategies
    - Deep judgment multi-stage parsing with excerpt extraction
    - Field verification (via Answer.verify())
    - Regex verification (via Answer.verify_regex())

    Example:
        evaluator = TemplateEvaluator(
            model_config=parsing_model,
            answer_class=Answer,
        )

        # Parse response
        parse_result = evaluator.parse_response(
            raw_response=raw_llm_response,
            question_text=question_text,
            config=parsing_config,
        )

        # Verify parsed answer
        if parse_result.success:
            field_result = evaluator.verify_fields(parse_result.parsed_answer)
            regex_result = evaluator.verify_regex(
                parse_result.parsed_answer,
                raw_llm_response,
            )
    """

    def __init__(
        self,
        model_config: ModelConfig,
        answer_class: type[BaseAnswer],
        raw_answer_class: type[BaseAnswer] | None = None,
    ):
        """
        Initialize the template evaluator.

        Args:
            model_config: Configuration for the parsing model
            answer_class: The Answer class (with question ID injected) for parsing
            raw_answer_class: The RawAnswer class (before ID injection) for ground truth extraction

        Raises:
            ValueError: If model configuration is invalid
            RuntimeError: If LLM initialization fails
        """
        if not model_config:
            raise ValueError("Model configuration is required")

        if not model_config.model_name:
            raise ValueError("Model name is required in model configuration")

        # Model provider is optional for OpenRouter and manual interfaces
        if model_config.interface not in INTERFACES_NO_PROVIDER_REQUIRED and not model_config.model_provider:
            raise ValueError(
                f"Model provider is required for model {model_config.id} "
                f"(interface: {model_config.interface}). Only {INTERFACES_NO_PROVIDER_REQUIRED} "
                f"interfaces allow empty providers."
            )

        self.model_config = model_config
        self.answer_class: type[BaseAnswer] = answer_class
        self.raw_answer_class: type[BaseAnswer] = raw_answer_class or answer_class

        # Initialize LLM
        try:
            model_kwargs: dict[str, Any] = {
                "model": model_config.model_name,
                "provider": model_config.model_provider,
                "temperature": model_config.temperature,
                "interface": model_config.interface,
            }

            # Add interface-specific parameters
            if model_config.interface == "openai_endpoint":
                if not model_config.endpoint_base_url:
                    raise ValueError("endpoint_base_url is required for openai_endpoint interface")
                if not model_config.endpoint_api_key:
                    raise ValueError("endpoint_api_key is required for openai_endpoint interface")

                model_kwargs["endpoint_base_url"] = model_config.endpoint_base_url
                model_kwargs["endpoint_api_key"] = model_config.endpoint_api_key.get_secret_value()

            # Add extra kwargs if provided
            if model_config.extra_kwargs:
                model_kwargs.update(model_config.extra_kwargs)

            self.llm = init_chat_model_unified(**model_kwargs)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM for template evaluation: {e}") from e

        # Build model string for tracking
        if model_config.interface == "openrouter":
            self.model_str = model_config.model_name
        elif model_config.interface == "openai_endpoint":
            self.model_str = f"endpoint/{model_config.model_name}"
        else:
            self.model_str = f"{model_config.model_provider}/{model_config.model_name}"

        # Create parser
        self.parser: Any = PydanticOutputParser(pydantic_object=answer_class)

    # ========================================================================
    # Public API
    # ========================================================================

    def parse_response(
        self,
        raw_response: str | dict[str, Any],
        question_text: str,
        deep_judgment_enabled: bool = False,
        deep_judgment_config: dict[str, Any] | None = None,
        use_full_trace: bool = True,
        usage_tracker: Any | None = None,
    ) -> ParseResult:
        """
        Parse raw LLM response into structured Answer object.

        This method orchestrates the parsing process, choosing between:
        - Standard parsing with native structured output + fallbacks
        - Deep judgment multi-stage parsing with excerpt extraction

        Args:
            raw_response: Raw LLM response (string or agent dict with messages)
            question_text: The original question text
            deep_judgment_enabled: Whether to use deep judgment parsing
            deep_judgment_config: Configuration for deep judgment (if enabled)
            use_full_trace: Whether to parse full trace or extract final AI message
            usage_tracker: Optional usage tracker for token counting

        Returns:
            ParseResult with parsed answer and metadata
        """
        result = ParseResult()

        # Harmonize raw response to string if needed
        from ....infrastructure.llm.mcp_utils import harmonize_agent_response

        # Pass question_text to enable reliable summary detection
        harmonized_response = harmonize_agent_response(raw_response, original_question=question_text)

        # Determine template evaluation input
        template_input: str = harmonized_response
        if not use_full_trace:
            extracted_message, error = extract_final_ai_message(harmonized_response)
            if error is not None:
                result.error = f"Failed to extract final AI message: {error}"
                return result
            if extracted_message is None:
                result.error = "Failed to extract final AI message: no message found"
                return result
            template_input = extracted_message

        # Extract ground truth if enabled
        ground_truth = None
        if self._should_expose_ground_truth():
            try:
                from ..utils.parsing import create_test_instance_from_answer_class

                _, ground_truth = create_test_instance_from_answer_class(self.raw_answer_class)
            except Exception as e:
                logger.warning(f"Could not extract ground truth: {e}")

        # Detect tool traces
        agent_metrics = None
        if isinstance(raw_response, dict):
            agent_metrics = self._extract_agent_metrics(raw_response)
        has_tool_traces = agent_metrics is not None and agent_metrics.get("tool_calls", 0) > 0

        # Build prompts
        format_instructions = self.parser.get_format_instructions()
        system_prompt = self._build_system_prompt(
            format_instructions=format_instructions,
            user_system_prompt=self.model_config.system_prompt,
            has_tool_traces=has_tool_traces,
            ground_truth=ground_truth,
        )
        user_prompt = self._build_user_prompt(
            question_text=question_text,
            response_to_parse=template_input,
        )

        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        try:
            if deep_judgment_enabled:
                result = self._parse_with_deep_judgment(
                    template_input=template_input,
                    question_text=question_text,
                    format_instructions=format_instructions,
                    combined_system_prompt=system_prompt,
                    deep_judgment_config=deep_judgment_config or {},
                    usage_tracker=usage_tracker,
                )
            else:
                result = self._parse_standard(
                    messages=messages,
                    usage_tracker=usage_tracker,
                )
        except Exception as e:
            result.error = f"Parsing failed: {e}"
            logger.error(result.error)

        return result

    def verify_fields(self, parsed_answer: Any) -> FieldVerificationResult:
        """
        Verify parsed answer fields against template constraints.

        Calls the user-defined verify() method on the parsed Answer object.

        Args:
            parsed_answer: Parsed Pydantic Answer object

        Returns:
            FieldVerificationResult with success status
        """
        result = FieldVerificationResult()

        try:
            result.success = parsed_answer.verify()
        except Exception as e:
            result.error = f"Field verification failed: {e}"
            logger.error(result.error)

        return result

    def verify_regex(
        self,
        parsed_answer: Any,
        raw_response: str | dict[str, Any],
    ) -> RegexVerificationResult:
        """
        Verify parsed answer against regex patterns defined in template.

        Calls the user-defined verify_regex() method on the parsed Answer object.

        Args:
            parsed_answer: Parsed Pydantic Answer object
            raw_response: Raw LLM response for regex matching

        Returns:
            RegexVerificationResult with match details
        """
        result = RegexVerificationResult()

        try:
            regex_results = parsed_answer.verify_regex(raw_response)

            result.success = regex_results["success"]
            result.results = regex_results["results"]
            result.details = regex_results["details"]

            # Extract actual matches for display
            if regex_results["details"]:
                for field_name, details in regex_results["details"].items():
                    result.extraction_results[field_name] = details.get("matches_found", [])

        except Exception as e:
            result.error = f"Regex verification failed: {e}"
            logger.error(result.error)

        return result

    # ========================================================================
    # Prompt Construction (moved from verification_utils.py)
    # ========================================================================

    def _build_system_prompt(
        self,
        format_instructions: str,
        user_system_prompt: str | None = None,
        has_tool_traces: bool = False,
        ground_truth: dict[str, Any] | None = None,
    ) -> str:
        """Build enhanced composable system prompt for template parsing.

        This function creates a composable system prompt with:
        1. Base guidelines (always included) - extraction protocol and critical rules
        2. Tool trace verification section (conditional - only when MCP/tools present)
        3. User customizations (merged as "Additional Instructions")
        4. Ground truth section (optional - for semantic matching assistance)
        5. Output format section with format instructions

        Args:
            format_instructions: Pydantic format instructions from PydanticOutputParser
            user_system_prompt: Optional user-provided system prompt to merge
            has_tool_traces: Whether the response includes tool call traces (MCP context)
            ground_truth: Optional ground truth for disambiguation assistance

        Returns:
            Composed system prompt string
        """
        # === BASE GUIDELINES (always included) ===
        base_guidelines = """You are an evaluator that extracts structured information from responses.

You will receive:
1. A response to parse (either a final answer or a complete trace)
2. A JSON schema with descriptive fields indicating what information to extract

# Extraction Protocol

## 1. Focus on the Final Answer
- Your primary extraction source is the **final answer** given to the user
- Extract information from this answer according to the schema field descriptions

## 2. Extract According to Schema
- Each field description specifies WHAT to extract from the answer and HOW
- Follow field descriptions precisely
- Use `null` for information not present in the final answer (if field allows null)

## 3. Validate Structure
- Return valid JSON matching the provided schema exactly
- Use correct data types for each field

# Critical Rules

**Answer-First**: Extract primarily from the final answer content.

**Description Adherence**: Each field's description is authoritative for what and how to extract.

**Fidelity**: Extract only what's actually stated. Don't infer or add information not present.

**JSON Only**: Return ONLY the JSON object - no explanations, no markdown fences, no surrounding text."""

        # === TOOL TRACE SECTION (conditional - only when MCP/tools present) ===
        tool_trace_section = """

# Tool Trace Verification (when traces are present)

When the response includes tool calls and results:

## Verify Grounding in Tool Results
- Cross-reference claims in the final answer against tool results in the trace
- Check that factual statements are supported by retrieved data
- Note if the answer includes information not present in tool results

**Grounding Check**: Use the trace to verify the answer's claims are supported by tool calls/results."""

        # === USER CUSTOMIZATIONS SECTION ===
        user_section = ""
        if user_system_prompt:
            user_section = f"""

# Additional Instructions

{user_system_prompt}"""

        # === GROUND TRUTH SECTION (optional) ===
        ground_truth_section = ""
        if ground_truth is not None:
            ground_truth_str = json.dumps(ground_truth, indent=2, default=str)
            ground_truth_section = f"""

# Ground Truth Reference

The following ground truth information is provided as reference to help with semantic matching and disambiguation.
Use this information carefully - do not blindly copy it, but it may help resolve ambiguities when the trace
and template are semantically close but differ in exact wording.

Ground Truth:
{ground_truth_str}"""

        # === OUTPUT FORMAT ===
        output_section = f"""

# Output Format

Return only the completed JSON object - no surrounding text, no markdown fences:

<format_instructions>
{format_instructions}
</format_instructions>"""

        # Compose final prompt
        sections = [base_guidelines]
        if has_tool_traces:
            sections.append(tool_trace_section)
        if user_section:
            sections.append(user_section)
        if ground_truth_section:
            sections.append(ground_truth_section)
        sections.append(output_section)

        return "".join(sections)

    def _build_user_prompt(
        self,
        question_text: str,
        response_to_parse: str | Any,
    ) -> str:
        """Build enhanced user prompt for template parsing.

        Includes the original question, response to parse, and JSON schema
        to help the LLM understand the expected output structure.

        Args:
            question_text: The original question that was asked
            response_to_parse: The LLM response to parse into structured format

        Returns:
            Formatted user prompt string
        """
        # Generate JSON schema from the Answer class
        json_schema = json.dumps(self.answer_class.model_json_schema(), indent=2)

        return f"""Parse the following response and extract structured information.

**ORIGINAL QUESTION:**
{question_text}

**RESPONSE TO PARSE:**
{response_to_parse}

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**PARSING NOTES:**
- Extract values for each field based on its description in the schema
- If information for a field is not present, use null (if field allows null) or your best inference
- Return ONLY the JSON object - no surrounding text

**YOUR JSON RESPONSE:**"""

    # ========================================================================
    # Standard Parsing
    # ========================================================================

    def _parse_standard(
        self,
        messages: list[BaseMessage],
        usage_tracker: Any | None = None,
    ) -> ParseResult:
        """
        Standard parsing with native structured output + fallback strategies.

        Strategy order:
        1. Native structured output (method="json_schema")
        2. Manual parsing with json-repair
        3. Null-value feedback retry
        4. Format feedback retry

        Args:
            messages: Prepared parsing messages
            usage_tracker: Optional usage tracker

        Returns:
            ParseResult with parsed answer
        """
        from ..utils.parsing import _strip_markdown_fences
        from .template_parsing import (
            invoke_with_structured_output_for_template,
            parse_template_response,
        )

        result = ParseResult()

        # Strategy 1: Try native structured output
        structured_result, struct_usage, used_structured = invoke_with_structured_output_for_template(
            llm=self.llm,
            messages=messages,
            answer_class=self.answer_class,
        )

        if struct_usage:
            result.usage_metadata_list.append(struct_usage)
            if usage_tracker:
                usage_tracker.track_call("parsing", self.model_str, struct_usage)

        if structured_result is not None and used_structured:
            result.parsed_answer = structured_result
            result.success = True
            logger.debug("Template parsing succeeded via native structured output")
            return result

        # Strategy 2: Fall back to manual parsing with json-repair
        logger.debug("Native structured output failed, falling back to manual parsing")

        with get_usage_metadata_callback() as cb:
            parsing_response = self.llm.invoke(messages)

        usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}
        if usage_metadata:
            result.usage_metadata_list.append(usage_metadata)
            if usage_tracker:
                usage_tracker.track_call("parsing", self.model_str, usage_metadata)

        raw_parsing_response = (
            parsing_response.content if hasattr(parsing_response, "content") else str(parsing_response)
        )

        # Try multi-strategy parsing with json-repair
        try:
            result.parsed_answer = parse_template_response(raw_parsing_response, self.answer_class)
            result.success = True
            logger.debug("Template parsing succeeded via manual parsing with json-repair")
            return result
        except Exception as parse_error:
            logger.warning(f"Initial parsing failed: {parse_error}")

            # Get cleaned response for retry strategies
            cleaned_response = _strip_markdown_fences(raw_parsing_response)
            if cleaned_response is None:
                result.error = "Empty response from parsing model after markdown fence removal"
                return result

            # Strategy 3: Try null-value feedback
            retried_answer, retry_usage = self._retry_parse_with_null_feedback(
                original_messages=messages,
                failed_response=cleaned_response,
                error=parse_error,
            )

            if retry_usage:
                result.usage_metadata_list.append(retry_usage)
                if usage_tracker:
                    usage_tracker.track_call("parsing_null_retry", self.model_str, retry_usage)

            if retried_answer is not None:
                result.parsed_answer = retried_answer
                result.success = True
                return result

            # Strategy 4: Try format feedback
            logger.info("Null-value retry did not succeed, trying format feedback...")
            retried_answer, retry_usage = self._retry_parse_with_format_feedback(
                original_messages=messages,
                failed_response=cleaned_response,
                error=parse_error,
            )

            if retry_usage:
                result.usage_metadata_list.append(retry_usage)
                if usage_tracker:
                    usage_tracker.track_call("parsing_format_retry", self.model_str, retry_usage)

            if retried_answer is not None:
                result.parsed_answer = retried_answer
                result.success = True
                return result

            # All strategies failed
            result.error = f"Parsing failed: {parse_error}"
            return result

    # ========================================================================
    # Deep Judgment Parsing (via composition)
    # ========================================================================

    def _parse_with_deep_judgment(
        self,
        template_input: str | Any,
        question_text: str,
        format_instructions: str,
        combined_system_prompt: str,
        deep_judgment_config: dict[str, Any],
        usage_tracker: Any | None = None,
    ) -> ParseResult:
        """
        Deep judgment multi-stage parsing with excerpt extraction.

        Delegates to the deep_judgment module for the actual implementation.

        Args:
            template_input: Response to parse
            question_text: Original question
            format_instructions: Parser format instructions
            combined_system_prompt: Built system prompt
            deep_judgment_config: Deep judgment configuration
            usage_tracker: Optional usage tracker

        Returns:
            ParseResult with deep judgment metadata
        """
        from ....schemas.workflow import VerificationConfig
        from ..evaluators.deep_judgment import deep_judgment_parse

        result = ParseResult()
        result.deep_judgment_performed = True

        # Create minimal config for deep-judgment
        dj_config = VerificationConfig(
            answering_models=[],
            parsing_models=[self.model_config],
            parsing_only=True,
            deep_judgment_enabled=True,
            deep_judgment_max_excerpts_per_attribute=deep_judgment_config.get("max_excerpts_per_attribute", 3),
            deep_judgment_fuzzy_match_threshold=deep_judgment_config.get("fuzzy_match_threshold", 0.8),
            deep_judgment_excerpt_retry_attempts=deep_judgment_config.get("excerpt_retry_attempts", 2),
            deep_judgment_search_enabled=deep_judgment_config.get("search_enabled", False),
            deep_judgment_search_tool=deep_judgment_config.get("search_tool", "wikipedia"),
        )

        try:
            parsed_answer, extracted_excerpts, attribute_reasoning, dj_metadata = deep_judgment_parse(
                raw_llm_response=template_input,
                RawAnswer=self.raw_answer_class,
                parsing_model=self.model_config,
                parsing_llm=self.llm,
                question_text=question_text,
                config=dj_config,
                format_instructions=format_instructions,
                combined_system_prompt=combined_system_prompt,
                usage_tracker=usage_tracker,
                parsing_model_str=self.model_str,
            )

            result.parsed_answer = parsed_answer
            result.success = True
            result.extracted_excerpts = extracted_excerpts
            result.attribute_reasoning = attribute_reasoning
            result.deep_judgment_stages_completed = dj_metadata.get("stages_completed", [])
            result.deep_judgment_model_calls = dj_metadata.get("model_calls", 0)
            result.deep_judgment_excerpt_retry_count = dj_metadata.get("excerpt_retry_count", 0)
            result.attributes_without_excerpts = dj_metadata.get("attributes_without_excerpts", None)
            result.hallucination_risk_assessment = dj_metadata.get("hallucination_risk", None)

        except Exception as e:
            result.error = f"Deep judgment parsing failed: {e}"
            logger.error(result.error)

        return result

    # ========================================================================
    # Retry Strategies (moved from verification_utils.py)
    # ========================================================================

    def _retry_parse_with_null_feedback(
        self,
        original_messages: list[BaseMessage],
        failed_response: str,
        error: Exception,
    ) -> tuple[Any | None, dict[str, Any]]:
        """
        Retry parsing with feedback about null values in required fields.

        When parsing fails due to null values, this function:
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
        from ..utils.parsing import _strip_markdown_fences

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

    def _retry_parse_with_format_feedback(
        self,
        original_messages: list[BaseMessage],
        failed_response: str,
        error: Exception,
    ) -> tuple[Any | None, dict[str, Any]]:
        """
        Retry parsing with feedback about JSON format requirements.

        When parsing fails due to invalid JSON (e.g., reasoning text mixed with JSON),
        this function:
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
        from ..utils.parsing import _strip_markdown_fences

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

    # ========================================================================
    # Utility Methods (moved from verification_utils.py)
    # ========================================================================

    def _should_expose_ground_truth(self) -> bool:
        """
        Check if ground truth should be exposed to the parser model.

        Reads from the KARENINA_EXPOSE_GROUND_TRUTH environment variable.
        Defaults to False for backward compatibility.

        Returns:
            True if ground truth should be exposed, False otherwise
        """
        return os.getenv("KARENINA_EXPOSE_GROUND_TRUTH", "false").lower() in ("true", "1", "yes", "on")

    def _extract_agent_metrics(self, response: dict[str, Any]) -> dict[str, Any] | None:
        """
        Extract agent execution metrics from LangGraph agent response.

        This function analyzes agent messages to track:
        - Iterations (AI message cycles)
        - Tool calls (successful tool invocations)
        - Tools used (unique tool names)
        - Suspected failed tool calls (tools with error-like output patterns)

        Args:
            response: Agent response object from LangGraph (dict with "messages" key)

        Returns:
            Dict with agent metrics or None if extraction fails
        """
        if not response or not isinstance(response, dict):
            return None

        messages = response.get("messages", [])
        if not messages:
            return None

        iterations = 0
        tool_calls = 0
        tools_used: set[str] = set()
        suspect_failed_tool_calls = 0
        suspect_failed_tools: set[str] = set()

        # Regex patterns to detect suspected tool failures
        tool_failure_patterns = [
            re.compile(r"\berror\b", re.IGNORECASE),
            re.compile(r"\bfailed\b", re.IGNORECASE),
            re.compile(r"\bexception\b", re.IGNORECASE),
            re.compile(r"\btraceback\b", re.IGNORECASE),
            re.compile(r"\b404\b", re.IGNORECASE),
            re.compile(r"\b500\b", re.IGNORECASE),
            re.compile(r"\btimeout\b", re.IGNORECASE),
        ]

        for msg in messages:
            msg_type = getattr(msg, "__class__", None)
            if msg_type:
                type_name = msg_type.__name__

                if type_name == "AIMessage":
                    iterations += 1

                elif type_name == "ToolMessage":
                    tool_calls += 1
                    tool_name = getattr(msg, "name", None)
                    if tool_name:
                        tools_used.add(tool_name)

                    # Check for suspected failures
                    is_suspect_failure = False
                    content = getattr(msg, "content", None)
                    if content and isinstance(content, str):
                        for pattern in tool_failure_patterns:
                            if pattern.search(content):
                                is_suspect_failure = True
                                break

                    if is_suspect_failure:
                        suspect_failed_tool_calls += 1
                        if tool_name:
                            suspect_failed_tools.add(tool_name)

        return {
            "iterations": iterations,
            "tool_calls": tool_calls,
            "tools_used": sorted(tools_used),
            "suspect_failed_tool_calls": suspect_failed_tool_calls,
            "suspect_failed_tools": sorted(suspect_failed_tools),
        }

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
