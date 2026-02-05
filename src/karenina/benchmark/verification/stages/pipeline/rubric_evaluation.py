"""Rubric evaluation stage.

Evaluates LLM responses against qualitative rubric criteria.
"""

import logging

from karenina.schemas.verification.model_identity import ModelIdentity

from ...evaluators import RubricEvaluator
from ...utils import prepare_evaluation_input
from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext
from ..helpers.deep_judgment_helpers import apply_deep_judgment_config_to_traits

# Set up logger
logger = logging.getLogger(__name__)


class RubricEvaluationStage(BaseVerificationStage):
    """
    Evaluates answer against rubric criteria.

    This stage:
    1. Only runs if rubric is configured
    2. Creates RubricEvaluator with parsing model
    3. Evaluates standard rubric traits (score/binary)
    4. Evaluates metric traits separately (confusion matrix analysis)
    5. Handles evaluation errors gracefully (doesn't fail pipeline)

    Requires:
        - "raw_llm_response": Raw LLM response text (for evaluation)

    Produces:
        - "rubric_result": Dict of trait scores (dict or None)
        - "metric_confusion_lists": Confusion lists per metric trait (dict or None)
        - "metric_results": Computed metrics per metric trait (dict or None)

    Error Handling:
        Rubric evaluation errors are non-fatal. If evaluation fails,
        rubric_result is set to None and pipeline continues.

    Note:
        Rubric evaluation uses raw LLM response, not parsed response,
        so it can work independently of template parsing success.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "RubricEvaluation"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        return [ArtifactKeys.RAW_LLM_RESPONSE]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            ArtifactKeys.RUBRIC_RESULT,
            ArtifactKeys.LLM_TRAIT_LABELS,
            ArtifactKeys.METRIC_CONFUSION_LISTS,
            ArtifactKeys.METRIC_RESULTS,
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run only if rubric is configured and has traits.

        Rubric evaluation is independent of other verification stages,
        so it runs even if template verification failed.

        However, if trace validation failed (trace doesn't end with AI message)
        and we need to extract the final AI message (use_full_trace_for_rubric=False),
        then skip rubric evaluation since extraction would fail.

        Inherits error-checking from BaseVerificationStage.
        """
        if not super().should_run(context):
            return False

        rubric = context.rubric
        if rubric is None:
            return False

        # Check if any trait lists are non-empty
        has_traits = any([rubric.llm_traits, rubric.regex_traits, rubric.callable_traits, rubric.metric_traits])
        if not has_traits:
            return False

        # If trace validation failed and we need extraction, skip rubric evaluation
        trace_validation_failed = context.get_artifact(ArtifactKeys.TRACE_VALIDATION_FAILED, False)
        if trace_validation_failed and not context.use_full_trace_for_rubric:
            logger.info(
                f"Skipping rubric evaluation for question {context.question_id}: "
                f"trace validation failed and use_full_trace_for_rubric=False"
            )
            return False

        return True

    def execute(self, context: VerificationContext) -> None:
        """
        Evaluate answer against rubric.

        Args:
            context: Verification context

        Side Effects:
            - Sets rubric evaluation artifacts
            - Sets result fields for rubric metadata
            - Does NOT set context.error on rubric evaluation failure
        """
        raw_llm_response = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
        rubric = context.rubric

        # Retrieve usage tracker from previous stage or create new one
        usage_tracker = self.get_or_create_usage_tracker(context)

        # Determine what input to pass to rubric evaluation based on config
        # Prefer structured trace_messages when available
        use_full_trace = context.use_full_trace_for_rubric
        trace_messages = context.get_artifact(ArtifactKeys.TRACE_MESSAGES)
        trace_input = trace_messages if trace_messages else raw_llm_response
        rubric_evaluation_input, extraction_error = prepare_evaluation_input(trace_input, use_full_trace)

        if extraction_error is not None:
            # Extraction failed - mark as error and stop
            error_msg = f"Rubric evaluation: {extraction_error}"
            logger.error(error_msg)
            context.mark_error(error_msg)
            # Note: trace filtering fields are stored at root level by parse_template stage
            return

        if not use_full_trace:
            logger.info("Using final AI message only for rubric evaluation")

        rubric_result = None
        llm_trait_labels = None
        metric_confusion_lists = None
        metric_results = None

        # Build model string for tracking via ModelIdentity
        parsing_model = context.parsing_model
        parsing_model_str = ModelIdentity.from_model_config(parsing_model, role="parsing").display_string

        try:
            # Create rubric evaluator with parsing model and strategy from config
            evaluator = RubricEvaluator(
                context.parsing_model,
                evaluation_strategy=context.rubric_evaluation_strategy,
                prompt_config=context.prompt_config,
            )

            # Apply deep judgment configuration resolution to LLM traits
            configured_rubric = rubric
            if rubric is not None and rubric.llm_traits:
                # Apply configuration resolution to get final trait settings
                configured_llm_traits = apply_deep_judgment_config_to_traits(
                    rubric.llm_traits,
                    context.question_id,
                    context,  # Pass context which has config fields
                )

                # Create modified rubric with configured traits using Pydantic's model_copy
                # This is more efficient than deepcopy since we only need to replace llm_traits
                configured_rubric = rubric.model_copy(update={"llm_traits": configured_llm_traits})

            # Check if any LLM traits have deep judgment enabled (after configuration)
            has_deep_judgment_traits = False
            if configured_rubric is not None and configured_rubric.llm_traits:
                has_deep_judgment_traits = any(t.deep_judgment_enabled for t in configured_rubric.llm_traits)

            # Evaluate rubric traits
            if configured_rubric is not None:
                if has_deep_judgment_traits:
                    # Route to deep judgment evaluation
                    logger.info(f"Deep judgment enabled for rubric traits in question {context.question_id}")

                    # Create a minimal VerificationConfig for deep judgment settings
                    from karenina.schemas.verification import VerificationConfig

                    dj_config = VerificationConfig(
                        answering_models=[context.answering_model],
                        parsing_models=[context.parsing_model],
                        deep_judgment_rubric_enabled=True,
                        deep_judgment_rubric_max_excerpts_default=getattr(
                            context, "deep_judgment_max_excerpts_per_attribute", 3
                        ),
                        deep_judgment_rubric_fuzzy_match_threshold_default=getattr(
                            context, "deep_judgment_fuzzy_match_threshold", 0.80
                        ),
                        deep_judgment_rubric_excerpt_retry_attempts_default=getattr(
                            context, "deep_judgment_excerpt_retry_attempts", 2
                        ),
                        deep_judgment_rubric_search_enabled=getattr(context, "deep_judgment_search_enabled", False),
                        deep_judgment_rubric_search_tool=getattr(context, "deep_judgment_search_tool", "tavily"),
                    )

                    dj_result = evaluator.evaluate_rubric_with_deep_judgment(
                        question=context.question_text,
                        answer=rubric_evaluation_input,  # Use filtered or full trace based on config
                        rubric=configured_rubric,  # Use configured rubric with resolved settings
                        config=dj_config,  # Pass config for deep judgment settings
                    )

                    # Combine scores for backward compatibility
                    rubric_result = {}
                    rubric_result.update(dj_result["deep_judgment_scores"])
                    rubric_result.update(dj_result["standard_scores"])

                    # Get literal trait labels from standard evaluation (if any)
                    llm_trait_labels = dj_result.get("standard_labels")

                    # Store deep judgment metadata in result fields
                    context.set_result_field(ArtifactKeys.DEEP_JUDGMENT_RUBRIC_PERFORMED, True)
                    context.set_result_field(ArtifactKeys.EXTRACTED_RUBRIC_EXCERPTS, dj_result["excerpts"])
                    context.set_result_field(ArtifactKeys.RUBRIC_TRAIT_REASONING, dj_result["reasoning"])
                    context.set_result_field(
                        ArtifactKeys.DEEP_JUDGMENT_RUBRIC_SCORES, dj_result["deep_judgment_scores"]
                    )
                    context.set_result_field(ArtifactKeys.STANDARD_RUBRIC_SCORES, dj_result["standard_scores"])
                    context.set_result_field(ArtifactKeys.TRAIT_METADATA, dj_result["metadata"])
                    context.set_result_field(
                        ArtifactKeys.TRAITS_WITHOUT_VALID_EXCERPTS, dj_result["traits_without_valid_excerpts"]
                    )
                    if dj_result["hallucination_risks"]:
                        context.set_result_field(
                            ArtifactKeys.RUBRIC_HALLUCINATION_RISK_ASSESSMENT, dj_result["hallucination_risks"]
                        )

                    # Calculate aggregated statistics
                    total_model_calls = sum(m.get("model_calls", 0) for m in dj_result["metadata"].values())
                    total_retries = sum(m.get("excerpt_retry_count", 0) for m in dj_result["metadata"].values())
                    context.set_result_field(ArtifactKeys.TOTAL_DEEP_JUDGMENT_MODEL_CALLS, total_model_calls)
                    context.set_result_field(
                        ArtifactKeys.TOTAL_TRAITS_EVALUATED, len(dj_result["deep_judgment_scores"])
                    )
                    context.set_result_field(ArtifactKeys.TOTAL_EXCERPT_RETRIES, total_retries)

                    # Track deep judgment usage metadata
                    usage_metadata_list = dj_result.get("usage_metadata_list", [])
                    for usage_metadata in usage_metadata_list:
                        if usage_metadata:
                            usage_tracker.track_call("rubric_evaluation", parsing_model_str, usage_metadata)
                    logger.debug(
                        f"Deep judgment used {total_model_calls} model calls with {total_retries} retries, "
                        f"tracked {len(usage_metadata_list)} usage metadata entries"
                    )

                else:
                    # Standard rubric evaluation (no deep judgment)
                    rubric_result, llm_trait_labels, usage_metadata_list = evaluator.evaluate_rubric(
                        question=context.question_text,
                        answer=rubric_evaluation_input,  # Use filtered or full trace based on config
                        rubric=configured_rubric,  # Use configured rubric
                    )

                    # Track rubric evaluation calls
                    for usage_metadata in usage_metadata_list:
                        if usage_metadata:
                            usage_tracker.track_call("rubric_evaluation", parsing_model_str, usage_metadata)

            # Evaluate metric traits separately
            if rubric is not None and rubric.metric_traits:
                logger.info(
                    f"Evaluating {len(rubric.metric_traits)} metric trait(s) for question {context.question_id}"
                )
                for trait in rubric.metric_traits:
                    logger.debug(f"  - Trait: {trait.name} (mode: {trait.evaluation_mode}, metrics: {trait.metrics})")

                metric_confusion_lists, metric_results, metric_usage_metadata_list = evaluator.evaluate_metric_traits(
                    question=context.question_text,
                    answer=rubric_evaluation_input,  # Use filtered or full trace based on config
                    metric_traits=rubric.metric_traits,
                )

                # Track metric trait evaluation calls
                for usage_metadata in metric_usage_metadata_list:
                    if usage_metadata:
                        usage_tracker.track_call("rubric_evaluation", parsing_model_str, usage_metadata)

                logger.info(
                    f"Metric evaluation complete. Results: {list(metric_results.keys()) if metric_results else 'None'}"
                )
                if metric_results:
                    for trait_name, metrics in metric_results.items():
                        logger.debug(f"     {trait_name}: {metrics}")

        except (ValueError, RuntimeError) as e:
            # Handle specific rubric evaluator errors (non-fatal)
            logger.warning(
                f"Rubric evaluator initialization/configuration failed for question {context.question_id}: {e}"
            )
            rubric_result = None  # type: ignore[assignment]
        except Exception as e:
            # Don't fail the entire verification if rubric evaluation fails
            logger.warning(f"Rubric evaluation failed for question {context.question_id}: {e}")
            rubric_result = None  # type: ignore[assignment]

        # Store results (even if None)
        context.set_artifact(ArtifactKeys.RUBRIC_RESULT, rubric_result)
        context.set_artifact(ArtifactKeys.LLM_TRAIT_LABELS, llm_trait_labels)
        context.set_artifact(ArtifactKeys.METRIC_CONFUSION_LISTS, metric_confusion_lists)
        context.set_artifact(ArtifactKeys.METRIC_RESULTS, metric_results)

        # Store updated usage tracker for next stages
        context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)

        # Store in result builder
        context.set_result_field(ArtifactKeys.VERIFY_RUBRIC, rubric_result)
        context.set_result_field(ArtifactKeys.LLM_TRAIT_LABELS, llm_trait_labels)
        context.set_result_field(ArtifactKeys.METRIC_TRAIT_CONFUSION_LISTS, metric_confusion_lists)
        context.set_result_field(ArtifactKeys.METRIC_TRAIT_METRICS, metric_results)
        # Note: evaluation_rubric is now stored in shared_data at export time (not per-result)
        context.set_result_field(ArtifactKeys.RUBRIC_EVALUATION_STRATEGY, context.rubric_evaluation_strategy)
        # Note: trace filtering fields (evaluation_input, used_full_trace, trace_extraction_error)
        # are stored at root level by parse_template stage, not duplicated here for rubric
