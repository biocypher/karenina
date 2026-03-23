"""Result finalization stage.

Builds the final VerificationResult from accumulated context.
"""

import logging
import shutil
from typing import Any

from karenina.benchmark.verification.utils.llm_invocation import _split_parsed_response
from karenina.schemas.verification import VerificationResult

from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext

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
        return [ArtifactKeys.FINAL_RESULT]

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
        execution_time = context.get_result_field(ArtifactKeys.EXECUTION_TIME, 0.0)
        timestamp = context.get_result_field(ArtifactKeys.TIMESTAMP, "")

        # Read ModelIdentity objects from pipeline artifacts (set by runner.py)
        from karenina.schemas.verification.model_identity import ModelIdentity

        answering_identity = context.get_artifact(ArtifactKeys.ANSWERING_MODEL_IDENTITY)
        parsing_identity = context.get_artifact(ArtifactKeys.PARSING_MODEL_IDENTITY)

        # Fallback: construct from model configs if artifacts not set (e.g., direct context usage)
        if answering_identity is None:
            answering_identity = ModelIdentity.from_model_config(context.answering_model, role="answering")
        if parsing_identity is None:
            parsing_identity = ModelIdentity.from_model_config(context.parsing_model, role="parsing")

        # Get MCP servers for template (still stored as a separate field on VerificationResultTemplate)
        answering_mcp_servers = context.get_result_field(ArtifactKeys.ANSWERING_MCP_SERVERS)

        # Extract parsed responses if available
        parsed_gt_response = None
        parsed_llm_response = None
        field_results_dict = None
        composition_strategy_str = None
        parsed_answer = context.get_artifact(ArtifactKeys.PARSED_ANSWER)
        if parsed_answer is not None:
            try:
                parsed_gt_response, parsed_llm_response = _split_parsed_response(parsed_answer)
            except Exception as e:
                logger.warning("Failed to split parsed response: %s", e)

            # Compute per-field primitive verification results (issue 150)
            if hasattr(parsed_answer, "_compute_field_results"):
                try:
                    computed = parsed_answer._compute_field_results()
                    if computed:
                        field_results_dict = computed
                except Exception as e:
                    logger.warning("Failed to compute field results: %s", e)

            # Extract composition strategy from template class (issue 151)
            strategy_cls = getattr(parsed_answer.__class__, "VerificationStrategy", None)
            if strategy_cls is not None:
                strategy = getattr(strategy_cls, "verify_strategy", None)
                if strategy is not None:
                    composition_strategy_str = self._format_strategy(strategy)

        # Determine which verification types were performed
        # Template verification was performed if VerifyTemplateStage ran and set field_verification_result
        template_verification_performed = context.get_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT) is not None

        # Rubric evaluation was performed if RubricEvaluationStage ran and set verify_rubric
        rubric_evaluation_performed = context.get_result_field(ArtifactKeys.VERIFY_RUBRIC) is not None

        # Aggregate usage tracking metadata
        usage_tracker = context.get_artifact(ArtifactKeys.USAGE_TRACKER)
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
        from karenina.schemas.verification import (
            VerificationResultDeepJudgment,
            VerificationResultMetadata,
            VerificationResultRubric,
            VerificationResultTemplate,
        )

        # Compute deterministic result_id
        result_id = VerificationResultMetadata.compute_result_id(
            question_id=context.question_id,
            answering=answering_identity,
            parsing=parsing_identity,
            timestamp=timestamp,
            replicate=context.replicate,
        )

        # Create metadata subclass
        metadata = VerificationResultMetadata(
            question_id=context.question_id,
            template_id=context.template_id,
            completed_without_errors=context.completed_without_errors,
            error=context.error,
            failed_stage=context.get_result_field(ArtifactKeys.FAILED_STAGE),
            keywords=context.keywords,
            question_text=context.question_text,
            raw_answer=context.raw_answer,
            answering=answering_identity,
            parsing=parsing_identity,
            answering_system_prompt=context.answering_model.system_prompt,
            parsing_system_prompt=context.parsing_model.system_prompt,
            execution_time=execution_time,
            timestamp=timestamp,
            result_id=result_id,
            run_name=context.run_name,
            replicate=context.replicate,
            few_shot_enabled=context.few_shot_enabled,
            few_shot_example_count=len(context.few_shot_examples) if context.few_shot_examples else 0,
            evaluation_mode=context.get_result_field(ArtifactKeys.EVALUATION_MODE),
            scenario_id=context.scenario_id,
            scenario_node=context.scenario_node,
            scenario_turn=context.scenario_turn,
            scenario_path=context.scenario_path,
        )

        # Build structured trace_messages for storage
        trace_messages_dicts: list[dict[str, Any]] = []
        trace_messages_raw = context.get_artifact(ArtifactKeys.TRACE_MESSAGES)
        if trace_messages_raw:
            # trace_messages contains list[Message] — convert to list[dict]
            for block_index, msg in enumerate(trace_messages_raw):
                d = msg.to_dict()
                d["block_index"] = block_index
                trace_messages_dicts.append(d)

        # Compute raw_llm_response from trace_messages if available,
        # otherwise fall back to the string stored by generate_answer
        raw_llm_response = context.get_result_field(ArtifactKeys.RAW_LLM_RESPONSE, "")
        if trace_messages_raw and not raw_llm_response:
            from karenina.benchmark.verification.utils.trace_formatting import messages_to_raw_trace

            raw_llm_response = messages_to_raw_trace(trace_messages_raw)

        # Create template subclass
        # Note: trace filtering fields (evaluation_input, used_full_trace, trace_extraction_error)
        # are now stored at the root level of VerificationResult, not in template
        template = VerificationResultTemplate(
            raw_llm_response=raw_llm_response,
            trace_messages=trace_messages_dicts,
            parsed_gt_response=parsed_gt_response,
            parsed_llm_response=parsed_llm_response,
            template_verification_performed=template_verification_performed,
            verify_result=context.get_result_field(ArtifactKeys.VERIFY_RESULT),
            verify_granular_result=context.get_result_field(ArtifactKeys.VERIFY_GRANULAR_RESULT),
            field_verification_error=context.get_result_field(ArtifactKeys.FIELD_VERIFICATION_ERROR),
            field_results=field_results_dict,
            composition_strategy=composition_strategy_str,
            embedding_check_performed=context.get_result_field(ArtifactKeys.EMBEDDING_CHECK_PERFORMED, False),
            embedding_similarity_score=context.get_result_field(ArtifactKeys.EMBEDDING_SIMILARITY_SCORE),
            embedding_override_applied=context.get_result_field(ArtifactKeys.EMBEDDING_OVERRIDE_APPLIED, False),
            embedding_model_used=context.get_result_field(ArtifactKeys.EMBEDDING_MODEL_USED),
            regex_validations_performed=context.get_result_field(ArtifactKeys.REGEX_VALIDATIONS_PERFORMED, False),
            regex_validation_results=context.get_result_field(ArtifactKeys.REGEX_VALIDATION_RESULTS),
            regex_validation_details=context.get_result_field(ArtifactKeys.REGEX_VALIDATION_DETAILS),
            regex_overall_success=context.get_result_field(ArtifactKeys.REGEX_OVERALL_SUCCESS),
            regex_extraction_results=context.get_result_field(ArtifactKeys.REGEX_EXTRACTION_RESULTS),
            recursion_limit_reached=context.get_result_field(ArtifactKeys.RECURSION_LIMIT_REACHED, False),
            # Agentic parsing
            investigation_trace=context.get_result_field(ArtifactKeys.INVESTIGATION_TRACE),
            agentic_parsing_performed=context.get_result_field(ArtifactKeys.AGENTIC_PARSING_PERFORMED, False),
            abstention_check_performed=context.get_result_field(ArtifactKeys.ABSTENTION_CHECK_PERFORMED, False),
            abstention_detected=context.get_result_field(ArtifactKeys.ABSTENTION_DETECTED),
            abstention_override_applied=context.get_result_field(ArtifactKeys.ABSTENTION_OVERRIDE_APPLIED, False),
            abstention_reasoning=context.get_result_field(ArtifactKeys.ABSTENTION_REASONING),
            sufficiency_check_performed=context.get_result_field(ArtifactKeys.SUFFICIENCY_CHECK_PERFORMED, False),
            sufficiency_detected=context.get_result_field(ArtifactKeys.SUFFICIENCY_DETECTED),
            sufficiency_override_applied=context.get_result_field(ArtifactKeys.SUFFICIENCY_OVERRIDE_APPLIED, False),
            sufficiency_reasoning=context.get_result_field(ArtifactKeys.SUFFICIENCY_REASONING),
            answering_mcp_servers=answering_mcp_servers,
            usage_metadata=usage_metadata,
            agent_metrics=agent_metrics,
        )

        # Create rubric subclass (if rubric evaluation was performed)
        rubric_result = None
        agentic_evaluation_performed = context.get_result_field(ArtifactKeys.AGENTIC_RUBRIC_EVALUATION_PERFORMED, False)

        if rubric_evaluation_performed or agentic_evaluation_performed:
            # Split verify_rubric into separate trait score dicts
            verify_rubric = context.get_result_field(ArtifactKeys.VERIFY_RUBRIC)
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
            metric_trait_scores_dict = context.get_result_field(ArtifactKeys.METRIC_TRAIT_METRICS)

            # Note: trace filtering fields and evaluation_rubric are no longer stored per-result
            # - trace filtering fields are at root level of VerificationResult
            # - evaluation_rubric goes in shared_data at export time
            # Get llm_trait_labels for literal-kind LLM traits (maps trait name to class name)
            llm_trait_labels = context.get_result_field(ArtifactKeys.LLM_TRAIT_LABELS)

            # Get agentic trait evaluation results (populated by Stage 11b)
            agentic_trait_scores = context.get_result_field(ArtifactKeys.AGENTIC_TRAIT_SCORES)
            agentic_trait_traces_raw = context.get_result_field(ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_TRACES)
            # Filter out None traces from failed investigations to satisfy dict[str, str] schema
            agentic_trait_traces = (
                {k: v for k, v in agentic_trait_traces_raw.items() if v is not None}
                if agentic_trait_traces_raw
                else None
            )

            # Get dynamic rubric metadata (populated by presence check pre-processing)
            dynamic_skipped = context.get_result_field(ArtifactKeys.DYNAMIC_RUBRIC_SKIPPED_TRAITS)
            dynamic_promoted = context.get_result_field(ArtifactKeys.DYNAMIC_RUBRIC_PROMOTED_TRAITS)

            rubric_result = VerificationResultRubric(
                rubric_evaluation_performed=rubric_evaluation_performed,
                rubric_evaluation_strategy=context.get_result_field(ArtifactKeys.RUBRIC_EVALUATION_STRATEGY),
                llm_trait_scores=llm_trait_scores,
                llm_trait_labels=llm_trait_labels,
                regex_trait_scores=regex_trait_scores,
                callable_trait_scores=callable_trait_scores,
                metric_trait_scores=metric_trait_scores_dict,
                metric_trait_confusion_lists=context.get_result_field(ArtifactKeys.METRIC_TRAIT_CONFUSION_LISTS),
                agentic_trait_scores=agentic_trait_scores,
                agentic_trait_investigation_traces=agentic_trait_traces,
                dynamic_rubric_skipped_traits=dynamic_skipped,
                dynamic_rubric_promoted_traits=dynamic_promoted,
                trait_provenance=context.trait_provenance,
            )

        # Create deep-judgment subclass (if enabled)
        deep_judgment = None
        if context.get_result_field(ArtifactKeys.DEEP_JUDGMENT_ENABLED, False):
            deep_judgment = VerificationResultDeepJudgment(
                deep_judgment_enabled=context.get_result_field(ArtifactKeys.DEEP_JUDGMENT_ENABLED, False),
                deep_judgment_performed=context.get_result_field(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, False),
                extracted_excerpts=context.get_result_field(ArtifactKeys.EXTRACTED_EXCERPTS),
                attribute_reasoning=context.get_result_field(ArtifactKeys.ATTRIBUTE_REASONING),
                deep_judgment_stages_completed=context.get_result_field(ArtifactKeys.DEEP_JUDGMENT_STAGES_COMPLETED),
                deep_judgment_model_calls=context.get_result_field(ArtifactKeys.DEEP_JUDGMENT_MODEL_CALLS, 0),
                deep_judgment_excerpt_retry_count=context.get_result_field(
                    ArtifactKeys.DEEP_JUDGMENT_EXCERPT_RETRY_COUNT, 0
                ),
                attributes_without_excerpts=context.get_result_field(ArtifactKeys.ATTRIBUTES_WITHOUT_EXCERPTS),
                deep_judgment_search_enabled=context.get_result_field(ArtifactKeys.DEEP_JUDGMENT_SEARCH_ENABLED, False),
                hallucination_risk_assessment=context.get_result_field(ArtifactKeys.HALLUCINATION_RISK_ASSESSMENT),
            )

        # Create deep-judgment rubric subclass (if performed)
        deep_judgment_rubric = None
        if context.get_result_field(ArtifactKeys.DEEP_JUDGMENT_RUBRIC_PERFORMED, False):
            from karenina.schemas.verification import VerificationResultDeepJudgmentRubric

            deep_judgment_rubric = VerificationResultDeepJudgmentRubric(
                deep_judgment_rubric_performed=context.get_result_field(
                    ArtifactKeys.DEEP_JUDGMENT_RUBRIC_PERFORMED, False
                ),
                extracted_rubric_excerpts=context.get_result_field(ArtifactKeys.EXTRACTED_RUBRIC_EXCERPTS),
                rubric_trait_reasoning=context.get_result_field(ArtifactKeys.RUBRIC_TRAIT_REASONING),
                deep_judgment_rubric_scores=context.get_result_field(ArtifactKeys.DEEP_JUDGMENT_RUBRIC_SCORES),
                standard_rubric_scores=context.get_result_field(ArtifactKeys.STANDARD_RUBRIC_SCORES),
                trait_metadata=context.get_result_field(ArtifactKeys.TRAIT_METADATA),
                traits_without_valid_excerpts=context.get_result_field(ArtifactKeys.TRAITS_WITHOUT_VALID_EXCERPTS),
                rubric_hallucination_risk_assessment=context.get_result_field(
                    ArtifactKeys.RUBRIC_HALLUCINATION_RISK_ASSESSMENT
                ),
                total_deep_judgment_model_calls=context.get_result_field(
                    ArtifactKeys.TOTAL_DEEP_JUDGMENT_MODEL_CALLS, 0
                ),
                total_traits_evaluated=context.get_result_field(ArtifactKeys.TOTAL_TRAITS_EVALUATED, 0),
                total_excerpt_retries=context.get_result_field(ArtifactKeys.TOTAL_EXCERPT_RETRIES, 0),
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
            evaluation_input=context.get_result_field(ArtifactKeys.EVALUATION_INPUT),
            used_full_trace=context.get_result_field(ArtifactKeys.USED_FULL_TRACE, True),
            trace_extraction_error=context.get_result_field(ArtifactKeys.TRACE_EXTRACTION_ERROR),
        )

        # Store final result
        context.set_artifact(ArtifactKeys.FINAL_RESULT, result)

        # Clean up workspace working copies (never originals)
        if context.workspace_path and context.workspace_cleanup and context.workspace_is_copy:
            try:
                shutil.rmtree(context.workspace_path)
                logger.debug("Cleaned up workspace: %s", context.workspace_path)
            except Exception:
                logger.warning(
                    "Failed to clean up workspace: %s",
                    context.workspace_path,
                    exc_info=True,
                )

    @staticmethod
    def _format_strategy(strategy: Any) -> str:
        """Format a composition strategy node as a human-readable string.

        Args:
            strategy: A composition strategy node (AllOf, AnyOf, AtLeastN).

        Returns:
            String like "all_of", "any_of", or "at_least_n(2)".
        """
        from karenina.schemas.entities.composition import AllOf, AnyOf, AtLeastN

        if isinstance(strategy, AnyOf):
            return "any_of"
        elif isinstance(strategy, AtLeastN):
            return f"at_least_n({strategy.n})"
        elif isinstance(strategy, AllOf):
            return "all_of"
        return str(strategy.__class__.__name__).lower()
