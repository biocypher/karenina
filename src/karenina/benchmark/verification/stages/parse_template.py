"""Template parsing stage.

Parses LLM responses into Pydantic objects using standard or deep-judgment parsing.
"""

import logging
from typing import Any

from ..evaluators.template_evaluator import TemplateEvaluator
from ..stage import BaseVerificationStage, VerificationContext
from ..utils import UsageTracker

# Set up logger
logger = logging.getLogger(__name__)


class ParseTemplateStage(BaseVerificationStage):
    """
    Parses LLM response into Pydantic object.

    This stage:
    1. Creates a TemplateEvaluator with the parsing model
    2. Delegates parsing to the evaluator (standard or deep-judgment)
    3. Stores parsing results and metadata

    Requires:
        - "RawAnswer": Validated Answer class (before question ID injection)
        - "Answer": Answer class with question ID injected
        - "raw_llm_response": Raw LLM response text

    Produces:
        - "parsed_answer": Parsed Pydantic object
        - "parsing_model_str": Model string for result
        - "template_evaluator": TemplateEvaluator instance for reuse
        - "deep_judgment_performed": Whether deep-judgment was used (bool)
        - "extracted_excerpts": Dict of excerpts per attribute (if deep-judgment)
        - "attribute_reasoning": Dict of reasoning per attribute (if deep-judgment)
        - "deep_judgment_stages_completed": List of completed stages (if deep-judgment)
        - "deep_judgment_model_calls": Number of LLM calls (if deep-judgment)
        - "deep_judgment_excerpt_retry_count": Retry count (if deep-judgment)
        - "attributes_without_excerpts": Attributes missing excerpts (if deep-judgment)
        - "hallucination_risk_assessment": Risk per attribute (if deep-judgment with search)

    Error Handling:
        If parsing fails (PydanticOutputParser creation or parsing),
        marks context.error and sets completed_without_errors=False.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "ParseTemplate"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        return ["RawAnswer", "Answer", "raw_llm_response"]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            "parsed_answer",
            "parsing_model_str",
            "template_evaluator",
            "deep_judgment_performed",
            "extracted_excerpts",
            "attribute_reasoning",
            "deep_judgment_stages_completed",
            "deep_judgment_model_calls",
            "deep_judgment_excerpt_retry_count",
            "attributes_without_excerpts",
            "hallucination_risk_assessment",
            "template_evaluation_input",
            "used_full_trace_for_template",
            "trace_extraction_error",
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """Run if we have raw LLM response, no errors, no recursion limit, and no trace validation failure."""
        # Skip parsing if recursion limit was reached (response is truncated/unreliable)
        if context.get_artifact("recursion_limit_reached", False):
            return False
        # Skip parsing if trace validation failed (trace doesn't end with AI message)
        if context.get_artifact("trace_validation_failed", False):
            return False
        return context.has_artifact("raw_llm_response") and context.has_artifact("Answer") and not context.error

    def execute(self, context: VerificationContext) -> None:
        """
        Parse LLM response into Pydantic object.

        Args:
            context: Verification context

        Side Effects:
            - Sets context.artifacts["parsed_answer"]
            - Sets context.artifacts["parsing_model_str"]
            - Sets context.artifacts["template_evaluator"]
            - Sets deep-judgment artifacts if enabled
            - Sets context.error if parsing fails
        """
        parsing_model = context.parsing_model
        raw_llm_response = context.get_artifact("raw_llm_response")
        Answer = context.get_artifact("Answer")
        RawAnswer = context.get_artifact("RawAnswer")

        # Retrieve usage tracker from previous stage or initialize new one
        usage_tracker = context.get_artifact("usage_tracker")
        if usage_tracker is None:
            usage_tracker = UsageTracker()
            logger.warning("No usage tracker found in context, initializing new one")

        # Determine what input to pass to template parsing based on config
        use_full_trace = context.use_full_trace_for_template

        # Step 1: Create TemplateEvaluator
        try:
            evaluator = TemplateEvaluator(
                model_config=parsing_model,
                answer_class=Answer,
                raw_answer_class=RawAnswer,
            )
        except Exception as e:
            error_msg = f"Failed to create TemplateEvaluator: {type(e).__name__}: {e}"
            logger.error(error_msg)
            context.mark_error(error_msg)
            return

        # Store evaluator for reuse by verify stage
        context.set_artifact("template_evaluator", evaluator)
        context.set_artifact("parsing_model_str", evaluator.model_str)

        # Build deep judgment config
        deep_judgment_config: dict[str, Any] = {}
        if context.deep_judgment_enabled:
            deep_judgment_config = {
                "max_excerpts_per_attribute": context.deep_judgment_max_excerpts_per_attribute,
                "fuzzy_match_threshold": context.deep_judgment_fuzzy_match_threshold,
                "excerpt_retry_attempts": context.deep_judgment_excerpt_retry_attempts,
                "search_enabled": context.deep_judgment_search_enabled,
                "search_tool": context.deep_judgment_search_tool,
            }

        # Step 2: Parse response using evaluator
        parse_result = evaluator.parse_response(
            raw_response=raw_llm_response,
            question_text=context.question_text,
            deep_judgment_enabled=context.deep_judgment_enabled,
            deep_judgment_config=deep_judgment_config,
            use_full_trace=use_full_trace,
            usage_tracker=usage_tracker,
        )

        # Step 3: Handle parse result
        if not parse_result.success:
            context.mark_error(parse_result.error or "Parsing failed")
            # Store metadata even on failure
            context.set_artifact("used_full_trace_for_template", use_full_trace)
            context.set_artifact("template_evaluation_input", None)
            context.set_artifact("trace_extraction_error", parse_result.error)
            context.set_result_field("used_full_trace", use_full_trace)
            context.set_result_field("used_full_trace_for_template", use_full_trace)
            context.set_result_field("trace_extraction_error", parse_result.error)
            return

        # Step 4: Store results
        context.set_artifact("parsed_answer", parse_result.parsed_answer)
        context.set_artifact("deep_judgment_performed", parse_result.deep_judgment_performed)
        context.set_artifact("used_full_trace_for_template", use_full_trace)
        context.set_artifact("template_evaluation_input", raw_llm_response if use_full_trace else None)
        context.set_artifact("trace_extraction_error", None)

        # Store deep judgment metadata
        context.set_artifact("extracted_excerpts", parse_result.extracted_excerpts)
        context.set_artifact("attribute_reasoning", parse_result.attribute_reasoning)
        context.set_artifact("deep_judgment_stages_completed", parse_result.deep_judgment_stages_completed)
        context.set_artifact("deep_judgment_model_calls", parse_result.deep_judgment_model_calls)
        context.set_artifact("deep_judgment_excerpt_retry_count", parse_result.deep_judgment_excerpt_retry_count)
        context.set_artifact("attributes_without_excerpts", parse_result.attributes_without_excerpts)
        context.set_artifact("hallucination_risk_assessment", parse_result.hallucination_risk_assessment)

        # Update usage tracker with any usage from parsing
        for usage_meta in parse_result.usage_metadata_list:
            if usage_meta:
                usage_tracker.track_call("parsing", evaluator.model_str, usage_meta)
        context.set_artifact("usage_tracker", usage_tracker)

        # Store in result builder
        context.set_result_field("deep_judgment_enabled", context.deep_judgment_enabled)
        context.set_result_field("deep_judgment_performed", parse_result.deep_judgment_performed)
        context.set_result_field("extracted_excerpts", parse_result.extracted_excerpts)
        context.set_result_field("attribute_reasoning", parse_result.attribute_reasoning)
        context.set_result_field("deep_judgment_stages_completed", parse_result.deep_judgment_stages_completed)
        context.set_result_field("deep_judgment_model_calls", parse_result.deep_judgment_model_calls)
        context.set_result_field("deep_judgment_excerpt_retry_count", parse_result.deep_judgment_excerpt_retry_count)
        context.set_result_field("attributes_without_excerpts", parse_result.attributes_without_excerpts)
        context.set_result_field("deep_judgment_search_enabled", context.deep_judgment_search_enabled)
        context.set_result_field("hallucination_risk_assessment", parse_result.hallucination_risk_assessment)

        # Store trace filtering metadata in result builder
        context.set_result_field("used_full_trace", use_full_trace)
        context.set_result_field("used_full_trace_for_template", use_full_trace)
        context.set_result_field("evaluation_input", raw_llm_response if use_full_trace else None)
        context.set_result_field("template_evaluation_input", raw_llm_response if use_full_trace else None)
        context.set_result_field("trace_extraction_error", None)
