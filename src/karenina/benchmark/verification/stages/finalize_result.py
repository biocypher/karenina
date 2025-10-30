"""Result finalization stage.

Builds the final VerificationResult from accumulated context.
"""

import logging

from ....schemas.workflow import VerificationResult
from ..stage import BaseVerificationStage, VerificationContext
from ..verification_utils import _split_parsed_response

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

        # Build VerificationResult from context
        result = VerificationResult(
            # Identity & Metadata
            question_id=context.question_id,
            template_id=context.template_id,
            completed_without_errors=context.completed_without_errors,
            error=context.error,
            keywords=context.keywords,
            question_text=context.question_text,
            # Core Results
            template_verification_performed=template_verification_performed,
            verify_result=context.get_result_field("verify_result"),
            verify_granular_result=context.get_result_field("verify_granular_result"),
            rubric_evaluation_performed=rubric_evaluation_performed,
            verify_rubric=context.get_result_field("verify_rubric"),
            evaluation_rubric=context.get_result_field("evaluation_rubric"),
            # Metric Traits
            metric_trait_confusion_lists=context.get_result_field("metric_trait_confusion_lists"),
            metric_trait_metrics=context.get_result_field("metric_trait_metrics"),
            # Raw & Parsed Data
            raw_llm_response=context.get_result_field("raw_llm_response", ""),
            parsed_gt_response=parsed_gt_response,
            parsed_llm_response=parsed_llm_response,
            # Model Configuration
            answering_model=answering_model_str,
            parsing_model=parsing_model_str,
            answering_system_prompt=context.answering_model.system_prompt,
            parsing_system_prompt=context.parsing_model.system_prompt,
            # Timing & Tracking
            execution_time=execution_time,
            timestamp=timestamp,
            run_name=context.run_name,
            job_id=context.job_id,
            answering_replicate=context.answering_replicate,
            parsing_replicate=context.parsing_replicate,
            # Embedding Check Metadata
            embedding_check_performed=context.get_result_field("embedding_check_performed", False),
            embedding_similarity_score=context.get_result_field("embedding_similarity_score"),
            embedding_override_applied=context.get_result_field("embedding_override_applied", False),
            embedding_model_used=context.get_result_field("embedding_model_used"),
            # Regex Validation Metadata
            regex_validations_performed=context.get_result_field("regex_validations_performed", False),
            regex_validation_results=context.get_result_field("regex_validation_results"),
            regex_validation_details=context.get_result_field("regex_validation_details"),
            regex_overall_success=context.get_result_field("regex_overall_success"),
            regex_extraction_results=context.get_result_field("regex_extraction_results"),
            # Recursion Limit Metadata
            recursion_limit_reached=context.get_result_field("recursion_limit_reached", False),
            # Abstention Detection Metadata
            abstention_check_performed=context.get_result_field("abstention_check_performed", False),
            abstention_detected=context.get_result_field("abstention_detected"),
            abstention_override_applied=context.get_result_field("abstention_override_applied", False),
            abstention_reasoning=context.get_result_field("abstention_reasoning"),
            # MCP Server Metadata
            answering_mcp_servers=context.get_result_field("answering_mcp_servers"),
            # Deep-Judgment Metadata
            deep_judgment_enabled=context.get_result_field("deep_judgment_enabled", False),
            deep_judgment_performed=context.get_result_field("deep_judgment_performed", False),
            extracted_excerpts=context.get_result_field("extracted_excerpts"),
            attribute_reasoning=context.get_result_field("attribute_reasoning"),
            deep_judgment_stages_completed=context.get_result_field("deep_judgment_stages_completed"),
            deep_judgment_model_calls=context.get_result_field("deep_judgment_model_calls", 0),
            deep_judgment_excerpt_retry_count=context.get_result_field("deep_judgment_excerpt_retry_count", 0),
            attributes_without_excerpts=context.get_result_field("attributes_without_excerpts"),
            # Search-Enhanced Deep-Judgment Metadata
            deep_judgment_search_enabled=context.get_result_field("deep_judgment_search_enabled", False),
            hallucination_risk_assessment=context.get_result_field("hallucination_risk_assessment"),
            # LLM Usage Tracking Metadata
            usage_metadata=usage_metadata,
            agent_metrics=agent_metrics,
        )

        # Store final result
        context.set_artifact("final_result", result)
