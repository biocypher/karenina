"""Result finalization stage.

Builds the final VerificationResult from accumulated context.
"""

import logging

from ....schemas.workflow import VerificationResult
from ..utils.llm_invocation import _split_parsed_response
from .base import BaseVerificationStage, VerificationContext

# Set up logger
logger = logging.getLogger(__name__)


class FinalizeResultStage(BaseVerificationStage):
    """
    Constructs final VerificationResult from context.

    This stage:
    1. Always runs (last stage in pipeline)
    2. Collects all artifacts from context
    3. Handles both success and error cases
    4. Constructs complete VerificationResult object
    5. Extracts parsed ground truth and LLM response

    This is the only stage that produces the final VerificationResult.

    Requires:
        - Various artifacts depending on which stages ran successfully

    Produces:
        - VerificationResult object (returned, not stored in context)

    Note:
        This stage must handle both complete success and partial failure cases.
        It uses context.result_builder which has been populated by previous stages.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "FinalizeResult"

    @property
    def requires(self) -> list[str]:
        """No hard requirements - works with whatever artifacts are available."""
        return []

    @property
    def produces(self) -> list[str]:
        """Produces the final VerificationResult (not stored in artifacts)."""
        return ["final_result"]

    def should_run(self, context: VerificationContext) -> bool:  # noqa: ARG002
        """Always run - this is the final stage."""
        return True

    def execute(self, context: VerificationContext) -> None:
        """
        Build final VerificationResult from context.

        Args:
            context: Verification context with accumulated artifacts

        Side Effects:
            - Sets context.artifacts["final_result"] with VerificationResult
        """
        # Calculate execution time
        execution_time = context.get_result_field("execution_time", 0.0)
        timestamp = context.get_result_field("timestamp", "")

        # Extract model strings
        answering_model_str = context.get_artifact("answering_model_str", "")
        parsing_model_str = context.get_artifact("parsing_model_str", "")

        # Extract parsed responses if available
        parsed_gt_response = None
        parsed_llm_response = None
        parsed_answer = context.get_artifact("parsed_answer")
        if parsed_answer is not None:
            try:
                parsed_gt_response, parsed_llm_response = _split_parsed_response(parsed_answer)
            except Exception as e:
                logger.warning(f"Failed to split parsed response: {e}")

        # Determine which verification types were performed
        # Template verification was performed if VerifyTemplateStage ran and set field_verification_result
        template_verification_performed = context.get_artifact("field_verification_result") is not None

        # Rubric evaluation was performed if RubricEvaluationStage ran and set verify_rubric
        rubric_evaluation_performed = context.get_result_field("verify_rubric") is not None

        # Aggregate usage tracking metadata
        usage_tracker = context.get_artifact("usage_tracker")
        usage_metadata = None
        agent_metrics = None

        if usage_tracker is not None:
            # Get aggregated usage summary across all stages
            usage_metadata = usage_tracker.get_total_summary()

            # Get agent metrics if available (from answer generation with MCP agents)
            agent_metrics = usage_tracker.get_agent_metrics()

            # Log usage summary
            if usage_metadata:
                total_info = usage_metadata.get("total", {})
                logger.info(
                    f"Usage tracking summary - Total tokens: {total_info.get('total_tokens', 0)} "
                    f"(input: {total_info.get('input_tokens', 0)}, output: {total_info.get('output_tokens', 0)})"
                )
            if agent_metrics:
                logger.info(
                    f"Agent metrics - Iterations: {agent_metrics.get('iterations', 0)}, "
                    f"Tool calls: {agent_metrics.get('tool_calls', 0)}"
                )

        # Build VerificationResult from context with nested structure
        from karenina.schemas.workflow import (
            VerificationResultDeepJudgment,
            VerificationResultMetadata,
            VerificationResultRubric,
            VerificationResultTemplate,
        )

        # Get MCP servers for result_id computation (also used in template below)
        answering_mcp_servers = context.get_result_field("answering_mcp_servers")

        # Compute deterministic result_id
        result_id = VerificationResultMetadata.compute_result_id(
            question_id=context.question_id,
            answering_model=answering_model_str,
            parsing_model=parsing_model_str,
            timestamp=timestamp,
            replicate=context.replicate,
            answering_mcp_servers=answering_mcp_servers,
        )

        # Create metadata subclass
        metadata = VerificationResultMetadata(
            question_id=context.question_id,
            template_id=context.template_id,
            completed_without_errors=context.completed_without_errors,
            error=context.error,
            keywords=context.keywords,
            question_text=context.question_text,
            raw_answer=context.raw_answer,
            answering_model=answering_model_str,
            parsing_model=parsing_model_str,
            answering_system_prompt=context.answering_model.system_prompt,
            parsing_system_prompt=context.parsing_model.system_prompt,
            execution_time=execution_time,
            timestamp=timestamp,
            result_id=result_id,
            run_name=context.run_name,
            replicate=context.replicate,
        )

        # Create template subclass
        # Note: trace filtering fields (evaluation_input, used_full_trace, trace_extraction_error)
        # are now stored at the root level of VerificationResult, not in template
        template = VerificationResultTemplate(
            raw_llm_response=context.get_result_field("raw_llm_response", ""),
            parsed_gt_response=parsed_gt_response,
            parsed_llm_response=parsed_llm_response,
            template_verification_performed=template_verification_performed,
            verify_result=context.get_result_field("verify_result"),
            verify_granular_result=context.get_result_field("verify_granular_result"),
            embedding_check_performed=context.get_result_field("embedding_check_performed", False),
            embedding_similarity_score=context.get_result_field("embedding_similarity_score"),
            embedding_override_applied=context.get_result_field("embedding_override_applied", False),
            embedding_model_used=context.get_result_field("embedding_model_used"),
            regex_validations_performed=context.get_result_field("regex_validations_performed", False),
            regex_validation_results=context.get_result_field("regex_validation_results"),
            regex_validation_details=context.get_result_field("regex_validation_details"),
            regex_overall_success=context.get_result_field("regex_overall_success"),
            regex_extraction_results=context.get_result_field("regex_extraction_results"),
            recursion_limit_reached=context.get_result_field("recursion_limit_reached", False),
            abstention_check_performed=context.get_result_field("abstention_check_performed", False),
            abstention_detected=context.get_result_field("abstention_detected"),
            abstention_override_applied=context.get_result_field("abstention_override_applied", False),
            abstention_reasoning=context.get_result_field("abstention_reasoning"),
            sufficiency_check_performed=context.get_result_field("sufficiency_check_performed", False),
            sufficiency_detected=context.get_result_field("sufficiency_detected"),
            sufficiency_override_applied=context.get_result_field("sufficiency_override_applied", False),
            sufficiency_reasoning=context.get_result_field("sufficiency_reasoning"),
            answering_mcp_servers=answering_mcp_servers,
            usage_metadata=usage_metadata,
            agent_metrics=agent_metrics,
        )

        # Create rubric subclass (if rubric evaluation was performed)
        rubric_result = None
        if rubric_evaluation_performed:
            # Split verify_rubric into separate trait score dicts
            verify_rubric = context.get_result_field("verify_rubric")
            # Get rubric definition from context directly (not from result_field)
            # Note: evaluation_rubric is no longer stored per-result, it goes in shared_data at export
            evaluation_rubric = context.rubric

            llm_trait_scores: dict[str, int] | None = None
            regex_trait_scores: dict[str, bool] | None = None
            callable_trait_scores: dict[str, bool | int] | None = None
            metric_trait_scores_dict: dict[str, dict[str, float]] | None = None

            if verify_rubric and evaluation_rubric and isinstance(verify_rubric, dict):
                # Get trait names from evaluation_rubric (Rubric object)
                llm_trait_names = {trait.name for trait in (evaluation_rubric.llm_traits or [])}
                regex_trait_names = {trait.name for trait in (evaluation_rubric.regex_traits or [])}
                callable_trait_names = {trait.name for trait in (evaluation_rubric.callable_traits or [])}

                # Split verify_rubric by trait type
                llm_results: dict[str, int] = {}
                regex_results: dict[str, bool] = {}
                callable_results: dict[str, bool | int] = {}

                for trait_name, trait_value in verify_rubric.items():
                    # Skip failed trait evaluations (None values)
                    if trait_value is None:
                        continue
                    if trait_name in llm_trait_names:
                        llm_results[trait_name] = trait_value
                    elif trait_name in regex_trait_names:
                        regex_results[trait_name] = trait_value
                    elif trait_name in callable_trait_names:
                        callable_results[trait_name] = trait_value
                    # Note: metric traits are stored separately in metric_trait_metrics

                llm_trait_scores = llm_results if llm_results else None
                regex_trait_scores = regex_results if regex_results else None
                callable_trait_scores = callable_results if callable_results else None

            # Get metric trait scores from context (already in the right format)
            metric_trait_scores_dict = context.get_result_field("metric_trait_metrics")

            # Note: trace filtering fields and evaluation_rubric are no longer stored per-result
            # - trace filtering fields are at root level of VerificationResult
            # - evaluation_rubric goes in shared_data at export time
            # Get llm_trait_labels for literal-kind LLM traits (maps trait name to class name)
            llm_trait_labels = context.get_result_field("llm_trait_labels")

            rubric_result = VerificationResultRubric(
                rubric_evaluation_performed=rubric_evaluation_performed,
                rubric_evaluation_strategy=context.get_result_field("rubric_evaluation_strategy"),
                llm_trait_scores=llm_trait_scores,
                llm_trait_labels=llm_trait_labels,
                regex_trait_scores=regex_trait_scores,
                callable_trait_scores=callable_trait_scores,
                metric_trait_scores=metric_trait_scores_dict,
                metric_trait_confusion_lists=context.get_result_field("metric_trait_confusion_lists"),
            )

        # Create deep-judgment subclass (if enabled)
        deep_judgment = None
        if context.get_result_field("deep_judgment_enabled", False):
            deep_judgment = VerificationResultDeepJudgment(
                deep_judgment_enabled=context.get_result_field("deep_judgment_enabled", False),
                deep_judgment_performed=context.get_result_field("deep_judgment_performed", False),
                extracted_excerpts=context.get_result_field("extracted_excerpts"),
                attribute_reasoning=context.get_result_field("attribute_reasoning"),
                deep_judgment_stages_completed=context.get_result_field("deep_judgment_stages_completed"),
                deep_judgment_model_calls=context.get_result_field("deep_judgment_model_calls", 0),
                deep_judgment_excerpt_retry_count=context.get_result_field("deep_judgment_excerpt_retry_count", 0),
                attributes_without_excerpts=context.get_result_field("attributes_without_excerpts"),
                deep_judgment_search_enabled=context.get_result_field("deep_judgment_search_enabled", False),
                hallucination_risk_assessment=context.get_result_field("hallucination_risk_assessment"),
            )

        # Create deep-judgment rubric subclass (if performed)
        deep_judgment_rubric = None
        if context.get_result_field("deep_judgment_rubric_performed", False):
            from karenina.schemas.workflow import VerificationResultDeepJudgmentRubric

            deep_judgment_rubric = VerificationResultDeepJudgmentRubric(
                deep_judgment_rubric_performed=context.get_result_field("deep_judgment_rubric_performed", False),
                extracted_rubric_excerpts=context.get_result_field("extracted_rubric_excerpts"),
                rubric_trait_reasoning=context.get_result_field("rubric_trait_reasoning"),
                deep_judgment_rubric_scores=context.get_result_field("deep_judgment_rubric_scores"),
                standard_rubric_scores=context.get_result_field("standard_rubric_scores"),
                trait_metadata=context.get_result_field("trait_metadata"),
                traits_without_valid_excerpts=context.get_result_field("traits_without_valid_excerpts"),
                rubric_hallucination_risk_assessment=context.get_result_field("rubric_hallucination_risk_assessment"),
                total_deep_judgment_model_calls=context.get_result_field("total_deep_judgment_model_calls", 0),
                total_traits_evaluated=context.get_result_field("total_traits_evaluated", 0),
                total_excerpt_retries=context.get_result_field("total_excerpt_retries", 0),
            )

        # Create final VerificationResult with nested composition
        # Note: trace filtering fields are now at the root level (shared by template and rubric)
        result = VerificationResult(
            metadata=metadata,
            template=template,
            rubric=rubric_result,
            deep_judgment=deep_judgment,
            deep_judgment_rubric=deep_judgment_rubric,
            # Root-level trace filtering fields (shared by template and rubric evaluation)
            evaluation_input=context.get_result_field("evaluation_input"),
            used_full_trace=context.get_result_field("used_full_trace", True),
            trace_extraction_error=context.get_result_field("trace_extraction_error"),
        )

        # Store final result
        context.set_artifact("final_result", result)
