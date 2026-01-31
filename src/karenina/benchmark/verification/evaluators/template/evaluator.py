"""Template evaluation for parsing and verifying LLM responses.

This module provides the TemplateEvaluator class which encapsulates all template
parsing and verification logic, following the same pattern as RubricEvaluator.

The evaluator uses adapter-based parsing via ParserPort, which handles all
retry/fallback logic internally (including json-repair, null-value feedback,
and format feedback retries).
"""

import logging
import os
from typing import TYPE_CHECKING, Any

from .....adapters import format_model_string, get_llm, get_parser
from .....ports import LLMPort
from .....schemas.domain import BaseAnswer
from .....schemas.workflow import ModelConfig
from ...prompts import PromptAssembler, PromptTask
from ...prompts.parsing.parsing_instructions import TemplatePromptBuilder
from ...utils import prepare_evaluation_input
from .results import FieldVerificationResult, ParseResult, RegexVerificationResult

if TYPE_CHECKING:
    from .....ports import ParserPort
    from .....schemas.verification import PromptConfig

logger = logging.getLogger(__name__)


class TemplateEvaluator:
    """
    Evaluates LLM responses by parsing them into Pydantic objects and verifying templates.

    This class encapsulates all template parsing and verification logic,
    following the same pattern as RubricEvaluator for architectural consistency.

    The evaluator supports:
    - Standard parsing via ParserPort (adapter handles all fallback/retry logic)
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
        prompt_config: "PromptConfig | None" = None,
    ):
        """
        Initialize the template evaluator.

        Args:
            model_config: Configuration for the parsing model
            answer_class: The Answer class (with question ID injected) for parsing
            raw_answer_class: The RawAnswer class (before ID injection) for ground truth extraction
            prompt_config: Optional per-task-type user instructions for prompt assembly

        Raises:
            ValueError: If model configuration is invalid (validated by adapter factory)
            RuntimeError: If adapter initialization fails
        """
        self.model_config = model_config
        self._prompt_config = prompt_config
        self.answer_class: type[BaseAnswer] = answer_class
        self.raw_answer_class: type[BaseAnswer] = raw_answer_class or answer_class

        # Initialize LLM via factory (for deep judgment)
        # Note: ValueError from validate_model_config propagates directly;
        # only runtime errors (adapter unavailable, etc.) are wrapped.
        try:
            self._llm: LLMPort = get_llm(model_config)
        except ValueError:
            raise  # Let validation errors propagate
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM for template evaluation: {e}") from e

        # Build model string for tracking (centralized via adapter registry)
        self.model_str = format_model_string(model_config)

        # Initialize adapter-based parser via the registry
        # Factory always returns a ParserPort (LangChainParserAdapter, ClaudeSDKParserAdapter,
        # or ManualParserAdapter) or raises AdapterUnavailableError
        self._parser: ParserPort = get_parser(model_config)
        logger.debug(f"Initialized ParserPort adapter for interface={model_config.interface}")

        # Initialize prompt builder
        self._prompt_builder = TemplatePromptBuilder(answer_class=answer_class)

    # ========================================================================
    # Public API
    # ========================================================================

    def parse_response(
        self,
        raw_response: str,
        question_text: str,
        deep_judgment_enabled: bool = False,
        deep_judgment_config: dict[str, Any] | None = None,
        use_full_trace: bool = False,
        usage_tracker: Any | None = None,
    ) -> ParseResult:
        """
        Parse raw LLM response into structured Answer object.

        This method orchestrates the parsing process, choosing between:
        - Standard parsing via ParserPort adapter (handles all fallbacks internally)
        - Deep judgment multi-stage parsing with excerpt extraction

        Args:
            raw_response: Raw LLM response trace string (already harmonized by adapter)
            question_text: The original question text
            deep_judgment_enabled: Whether to use deep judgment parsing
            deep_judgment_config: Configuration for deep judgment (if enabled)
            use_full_trace: Whether to parse full trace or extract final AI message
            usage_tracker: Optional usage tracker for token counting

        Returns:
            ParseResult with parsed answer and metadata
        """
        result = ParseResult()

        # Input is already a harmonized trace string from AgentResult.raw_trace
        # Both LangChain and Claude SDK adapters produce the same string format
        harmonized_response = raw_response

        # Determine template evaluation input (optionally extract final AI message)
        template_input, extraction_error = prepare_evaluation_input(harmonized_response, use_full_trace)
        if extraction_error is not None:
            result.error = extraction_error
            return result

        try:
            if deep_judgment_enabled:
                result = self._parse_with_deep_judgment(
                    template_input=template_input,
                    question_text=question_text,
                    deep_judgment_config=deep_judgment_config or {},
                    usage_tracker=usage_tracker,
                )
            else:
                result = self._parse_standard(
                    trace_text=template_input,
                    question_text=question_text,
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
    # Standard Parsing (via ParserPort)
    # ========================================================================

    def _parse_standard(
        self,
        trace_text: str,
        question_text: str,
        usage_tracker: Any | None = None,
    ) -> ParseResult:
        """
        Standard parsing via ParserPort adapter with tri-section prompt assembly.

        Builds prompt messages using PromptAssembler (task + adapter + user
        instructions), then delegates to ParserPort for LLM invocation and
        retry/fallback logic.

        Args:
            trace_text: Raw trace text to parse
            question_text: The original question text
            usage_tracker: Optional UsageTracker for tracking parsing token usage

        Returns:
            ParseResult with parsed answer
        """
        result = ParseResult()

        try:
            # 1. Build task instructions (format-agnostic base prompts)
            system_text = self._prompt_builder.build_system_prompt(
                has_tool_traces=False,
                ground_truth=self._get_ground_truth(),
            )
            user_text = self._prompt_builder.build_user_prompt(
                question_text=question_text,
                response_to_parse=trace_text,
            )

            # 2. Assemble (adapter instructions + user instructions)
            assembler = PromptAssembler(
                task=PromptTask.PARSING,
                interface=self.model_config.interface,
                capabilities=self._parser.capabilities,
            )
            user_instructions = (
                self._prompt_config.get_for_task(PromptTask.PARSING.value) if self._prompt_config else None
            )
            messages = assembler.assemble(
                system_text=system_text,
                user_text=user_text,
                user_instructions=user_instructions,
                instruction_context={
                    "json_schema": self.answer_class.model_json_schema(),
                    "format_instructions": "",
                },
            )

            # 3. Parse via adapter (returns ParsePortResult with usage)
            parse_port_result = self._parser.parse_to_pydantic(messages, self.answer_class)
            parsed = parse_port_result.parsed

            if isinstance(parsed, self.answer_class):
                result.parsed_answer = parsed
                result.success = True
                logger.debug("Template parsing succeeded via ParserPort adapter")

                # Track parsing token usage if tracker provided
                if usage_tracker is not None and parse_port_result.usage:
                    usage_dict = parse_port_result.usage.to_dict()
                    if usage_dict.get("input_tokens", 0) > 0 or usage_dict.get("output_tokens", 0) > 0:
                        usage_tracker.track_call("parsing", self.model_str, usage_dict)
            else:
                result.error = f"Unexpected parse result type: {type(parsed)}"
                logger.error(result.error)

        except Exception as e:
            result.error = f"Parsing failed: {e}"
            logger.debug(f"ParserPort parsing failed: {e}")

        return result

    def _get_ground_truth(self) -> dict[str, Any] | None:
        """Extract ground truth if enabled via environment variable."""
        if not self._should_expose_ground_truth():
            return None
        try:
            from ...utils.template_parsing_helpers import create_test_instance_from_answer_class

            _, ground_truth = create_test_instance_from_answer_class(self.raw_answer_class)
            return ground_truth
        except Exception as e:
            logger.warning(f"Could not extract ground truth: {e}")
            return None

    # ========================================================================
    # Deep Judgment Parsing (via composition)
    # ========================================================================

    def _parse_with_deep_judgment(
        self,
        template_input: str | Any,
        question_text: str,
        deep_judgment_config: dict[str, Any],
        usage_tracker: Any | None = None,
    ) -> ParseResult:
        """
        Deep judgment multi-stage parsing with excerpt extraction.

        Delegates to the deep_judgment module for the actual implementation.

        Args:
            template_input: Response to parse
            question_text: Original question
            deep_judgment_config: Deep judgment configuration
            usage_tracker: Optional usage tracker

        Returns:
            ParseResult with deep judgment metadata
        """
        from langchain_core.output_parsers import PydanticOutputParser

        from .....schemas.workflow import VerificationConfig
        from .deep_judgment import deep_judgment_parse

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

        # Build prompts for deep judgment
        parser = PydanticOutputParser(pydantic_object=self.answer_class)
        format_instructions = parser.get_format_instructions()

        # Extract ground truth if enabled
        ground_truth = None
        if self._should_expose_ground_truth():
            try:
                from ...utils.template_parsing_helpers import create_test_instance_from_answer_class

                _, ground_truth = create_test_instance_from_answer_class(self.raw_answer_class)
            except Exception as e:
                logger.warning(f"Could not extract ground truth: {e}")

        combined_system_prompt = self._prompt_builder.build_system_prompt(
            has_tool_traces=False,
            ground_truth=ground_truth,
        )

        # Apply adapter + user instructions via PromptAssembler
        user_instructions = self._prompt_config.get_for_task(PromptTask.PARSING.value) if self._prompt_config else None
        assembler = PromptAssembler(
            task=PromptTask.PARSING,
            interface=self.model_config.interface,
            capabilities=self._parser.capabilities,
        )
        combined_system_prompt, _ = assembler.assemble_text(
            system_text=combined_system_prompt,
            user_text="",
            user_instructions=user_instructions,
            instruction_context={
                "json_schema": self.answer_class.model_json_schema(),
                "format_instructions": format_instructions,
            },
        )

        try:
            parsed_answer, extracted_excerpts, attribute_reasoning, dj_metadata = deep_judgment_parse(
                raw_llm_response=template_input,
                RawAnswer=self.raw_answer_class,
                parsing_model=self.model_config,
                parsing_llm=self._llm,
                parser=self._parser,
                question_text=question_text,
                config=dj_config,
                format_instructions=format_instructions,
                combined_system_prompt=combined_system_prompt,
                usage_tracker=usage_tracker,
                parsing_model_str=self.model_str,
                prompt_config=self._prompt_config,
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
    # Utility Methods
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
