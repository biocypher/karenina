"""Template parsing stage.

Parses LLM responses into Pydantic objects using standard or deep-judgment parsing.
"""

import logging
from typing import Any

from ...evaluators import TemplateEvaluator
from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext

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
        return [
            ArtifactKeys.RAW_ANSWER,
            ArtifactKeys.ANSWER,
            ArtifactKeys.RAW_LLM_RESPONSE,
        ]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            ArtifactKeys.PARSED_ANSWER,
            ArtifactKeys.PARSING_MODEL_STR,
            ArtifactKeys.TEMPLATE_EVALUATOR,
            ArtifactKeys.DEEP_JUDGMENT_PERFORMED,
            ArtifactKeys.EXTRACTED_EXCERPTS,
            ArtifactKeys.ATTRIBUTE_REASONING,
            ArtifactKeys.DEEP_JUDGMENT_STAGES_COMPLETED,
            ArtifactKeys.DEEP_JUDGMENT_MODEL_CALLS,
            ArtifactKeys.DEEP_JUDGMENT_EXCERPT_RETRY_COUNT,
            ArtifactKeys.ATTRIBUTES_WITHOUT_EXCERPTS,
            ArtifactKeys.HALLUCINATION_RISK_ASSESSMENT,
            # Note: Also sets root-level trace filtering result fields:
            # used_full_trace, evaluation_input, trace_extraction_error
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run if we have raw LLM response, no errors, and various conditions are met.

        Inherits error-checking from BaseVerificationStage.
        """
        if not super().should_run(context):
            return False
        # Skip parsing if recursion limit was reached (response is truncated/unreliable)
        if context.get_artifact(ArtifactKeys.RECURSION_LIMIT_REACHED, False):
            return False
        # Skip parsing if trace validation failed (trace doesn't end with AI message)
        if context.get_artifact(ArtifactKeys.TRACE_VALIDATION_FAILED, False):
            return False
        # Skip parsing if abstention was detected (model refused to answer)
        if context.get_artifact(ArtifactKeys.ABSTENTION_DETECTED, False):
            return False
        # Skip parsing if sufficiency check ran and detected insufficient info
        # Note: sufficiency_detected=True means sufficient, False means insufficient
        if context.get_artifact(ArtifactKeys.SUFFICIENCY_DETECTED) is False:
            return False
        return context.has_artifact(ArtifactKeys.RAW_LLM_RESPONSE) and context.has_artifact(ArtifactKeys.ANSWER)

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
        raw_llm_response = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
        Answer = context.get_artifact(ArtifactKeys.ANSWER)
        RawAnswer = context.get_artifact(ArtifactKeys.RAW_ANSWER)

        # Retrieve usage tracker from previous stage or create new one
        usage_tracker = self.get_or_create_usage_tracker(context)

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
        context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, evaluator)
        context.set_artifact(ArtifactKeys.PARSING_MODEL_STR, evaluator.model_str)

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
            # Store trace extraction error for diagnostics
            context.set_artifact(ArtifactKeys.TRACE_EXTRACTION_ERROR, parse_result.error)
            context.set_result_field(ArtifactKeys.USED_FULL_TRACE, use_full_trace)
            context.set_result_field(ArtifactKeys.TRACE_EXTRACTION_ERROR, parse_result.error)
            return

        # Step 4: Store results
        context.set_artifact(ArtifactKeys.PARSED_ANSWER, parse_result.parsed_answer)
        context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, parse_result.deep_judgment_performed)
        context.set_artifact(ArtifactKeys.TRACE_EXTRACTION_ERROR, None)

        # Store deep judgment metadata
        context.set_artifact(ArtifactKeys.EXTRACTED_EXCERPTS, parse_result.extracted_excerpts)
        context.set_artifact(ArtifactKeys.ATTRIBUTE_REASONING, parse_result.attribute_reasoning)
        context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_STAGES_COMPLETED, parse_result.deep_judgment_stages_completed)
        context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_MODEL_CALLS, parse_result.deep_judgment_model_calls)
        context.set_artifact(
            ArtifactKeys.DEEP_JUDGMENT_EXCERPT_RETRY_COUNT, parse_result.deep_judgment_excerpt_retry_count
        )
        context.set_artifact(ArtifactKeys.ATTRIBUTES_WITHOUT_EXCERPTS, parse_result.attributes_without_excerpts)
        context.set_artifact(ArtifactKeys.HALLUCINATION_RISK_ASSESSMENT, parse_result.hallucination_risk_assessment)

        # Update usage tracker with any usage from parsing
        for usage_meta in parse_result.usage_metadata_list:
            if usage_meta:
                usage_tracker.track_call("parsing", evaluator.model_str, usage_meta)
        context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)

        # Store in result builder
        context.set_result_field(ArtifactKeys.DEEP_JUDGMENT_ENABLED, context.deep_judgment_enabled)
        context.set_result_field(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, parse_result.deep_judgment_performed)
        context.set_result_field(ArtifactKeys.EXTRACTED_EXCERPTS, parse_result.extracted_excerpts)
        context.set_result_field(ArtifactKeys.ATTRIBUTE_REASONING, parse_result.attribute_reasoning)
        context.set_result_field(
            ArtifactKeys.DEEP_JUDGMENT_STAGES_COMPLETED, parse_result.deep_judgment_stages_completed
        )
        context.set_result_field(ArtifactKeys.DEEP_JUDGMENT_MODEL_CALLS, parse_result.deep_judgment_model_calls)
        context.set_result_field(
            ArtifactKeys.DEEP_JUDGMENT_EXCERPT_RETRY_COUNT, parse_result.deep_judgment_excerpt_retry_count
        )
        context.set_result_field(ArtifactKeys.ATTRIBUTES_WITHOUT_EXCERPTS, parse_result.attributes_without_excerpts)
        context.set_result_field(ArtifactKeys.DEEP_JUDGMENT_SEARCH_ENABLED, context.deep_judgment_search_enabled)
        context.set_result_field(ArtifactKeys.HALLUCINATION_RISK_ASSESSMENT, parse_result.hallucination_risk_assessment)

        # Store trace filtering metadata in result builder (root-level fields only)
        context.set_result_field(ArtifactKeys.USED_FULL_TRACE, use_full_trace)
        context.set_result_field(ArtifactKeys.EVALUATION_INPUT, raw_llm_response if use_full_trace else None)
        context.set_result_field(ArtifactKeys.TRACE_EXTRACTION_ERROR, None)
