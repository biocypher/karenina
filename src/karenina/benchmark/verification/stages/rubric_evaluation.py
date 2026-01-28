"""Rubric evaluation stage.

Evaluates LLM responses against qualitative rubric criteria.
"""

import logging
from typing import Any

from ....schemas.domain import LLMRubricTrait
from ....schemas.workflow.verification.config import DeepJudgmentTraitConfig
from ..evaluators import RubricEvaluator
from ..utils import extract_final_ai_message
from .base import BaseVerificationStage, VerificationContext

# Set up logger
logger = logging.getLogger(__name__)


def resolve_deep_judgment_config_for_trait(
    trait: LLMRubricTrait,
    question_id: str | None,
    config: Any,  # VerificationConfig
) -> DeepJudgmentTraitConfig:
    """
    Resolve deep judgment configuration for a single trait based on mode hierarchy.

    Resolution priority (first match wins):
    1. disabled mode: Deep judgment OFF
    2. enable_all mode: Deep judgment ON for all traits (respects global excerpt toggle)
    3. use_checkpoint mode: Use settings from trait object (loaded from checkpoint)
    4. custom mode: Look up trait in config dict (question-specific → global → disabled)

    Args:
        trait: The rubric trait to resolve configuration for
        question_id: Optional question ID for question-specific config lookup
        config: VerificationConfig with deep judgment mode and settings

    Returns:
        DeepJudgmentTraitConfig with resolved settings
    """
    mode = getattr(config, "deep_judgment_rubric_mode", "disabled")
    logger.debug(f"Resolving deep judgment config for trait '{trait.name}', mode='{mode}'")

    if mode == "disabled":
        # Explicit: Deep judgment OFF
        return DeepJudgmentTraitConfig(enabled=False)

    elif mode == "enable_all":
        # Apply to all traits with global excerpt toggle
        return DeepJudgmentTraitConfig(
            enabled=True,
            excerpt_enabled=getattr(config, "deep_judgment_rubric_global_excerpts", True),
            max_excerpts=getattr(config, "deep_judgment_rubric_max_excerpts_default", 7),
            fuzzy_match_threshold=getattr(config, "deep_judgment_rubric_fuzzy_match_threshold_default", 0.80),
            excerpt_retry_attempts=getattr(config, "deep_judgment_rubric_excerpt_retry_attempts_default", 2),
            search_enabled=getattr(config, "deep_judgment_rubric_search_enabled", False),
        )

    elif mode == "use_checkpoint":
        # Use settings from trait (loaded from checkpoint)
        return DeepJudgmentTraitConfig(
            enabled=trait.deep_judgment_enabled,
            excerpt_enabled=trait.deep_judgment_excerpt_enabled,
            max_excerpts=trait.deep_judgment_max_excerpts,
            fuzzy_match_threshold=trait.deep_judgment_fuzzy_match_threshold,
            excerpt_retry_attempts=trait.deep_judgment_excerpt_retry_attempts,
            search_enabled=trait.deep_judgment_search_enabled,
        )

    elif mode == "custom":
        # Navigate nested config structure
        config_dict = getattr(config, "deep_judgment_rubric_config", None) or {}

        # Try question-specific first
        if (
            question_id
            and "question_specific" in config_dict
            and question_id in config_dict["question_specific"]
            and trait.name in config_dict["question_specific"][question_id]
        ):
            trait_config = config_dict["question_specific"][question_id][trait.name]
            # Validate dict against model
            return DeepJudgmentTraitConfig(**trait_config)

        # Fall back to global
        if "global" in config_dict and trait.name in config_dict["global"]:
            trait_config = config_dict["global"][trait.name]
            # Validate dict against model
            return DeepJudgmentTraitConfig(**trait_config)

        # No config found, disabled
        return DeepJudgmentTraitConfig(enabled=False)

    else:
        # Unknown mode, default to disabled
        logger.warning(f"Unknown deep_judgment_rubric_mode: {mode}, defaulting to disabled")
        return DeepJudgmentTraitConfig(enabled=False)


def apply_deep_judgment_config_to_traits(
    traits: list[LLMRubricTrait],
    question_id: str | None,
    config: Any,  # VerificationConfig
) -> list[LLMRubricTrait]:
    """
    Apply resolved deep judgment configuration to a list of traits.

    Creates shallow copies of traits with resolved deep judgment settings.
    Uses Pydantic's model_copy() for efficient copying since we only modify
    scalar config fields (bool, int, float).

    Args:
        traits: List of traits to configure
        question_id: Optional question ID for question-specific config
        config: VerificationConfig with deep judgment settings

    Returns:
        List of traits with resolved deep judgment configuration applied
    """
    configured_traits = []

    for trait in traits:
        # Resolve configuration for this trait (before copy to avoid unnecessary work)
        dj_config = resolve_deep_judgment_config_for_trait(trait, question_id, config)

        # Use Pydantic's model_copy with update dict for efficient shallow copy
        # This avoids expensive deepcopy for scalar field updates
        trait_copy = trait.model_copy(
            update={
                "deep_judgment_enabled": dj_config.enabled,
                "deep_judgment_excerpt_enabled": dj_config.excerpt_enabled,
                "deep_judgment_max_excerpts": dj_config.max_excerpts,
                "deep_judgment_fuzzy_match_threshold": dj_config.fuzzy_match_threshold,
                "deep_judgment_excerpt_retry_attempts": dj_config.excerpt_retry_attempts,
                "deep_judgment_search_enabled": dj_config.search_enabled,
            }
        )

        configured_traits.append(trait_copy)

    return configured_traits


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
        return ["raw_llm_response"]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            "rubric_result",
            "llm_trait_labels",
            "metric_confusion_lists",
            "metric_results",
            "rubric_evaluation_input",
            "used_full_trace_for_rubric",
            "rubric_trace_extraction_error",
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
        trace_validation_failed = context.get_artifact("trace_validation_failed", False)
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
        raw_llm_response = context.get_artifact("raw_llm_response")
        rubric = context.rubric

        # Retrieve usage tracker from previous stage or create new one
        usage_tracker = self.get_or_create_usage_tracker(context)

        # Determine what input to pass to rubric evaluation based on config
        use_full_trace = context.use_full_trace_for_rubric
        rubric_trace_extraction_error = None
        rubric_evaluation_input = raw_llm_response  # Default to full trace

        if not use_full_trace:
            # Extract only the final AI message
            extracted_message, error = extract_final_ai_message(raw_llm_response)

            if error is not None:
                # Extraction failed - mark as error and stop
                error_msg = f"Failed to extract final AI message for rubric evaluation: {error}"
                logger.error(error_msg)
                rubric_trace_extraction_error = error
                context.mark_error(error_msg)

                # Store metadata before returning (artifacts only - result fields are at root level)
                context.set_artifact("used_full_trace_for_rubric", use_full_trace)
                context.set_artifact("rubric_trace_extraction_error", rubric_trace_extraction_error)
                context.set_artifact("rubric_evaluation_input", None)
                # Note: trace filtering fields (evaluation_input, used_full_trace, trace_extraction_error)
                # are now stored at the root level by parse_template stage
                return
            else:
                # Extraction successful - use extracted message
                rubric_evaluation_input = extracted_message
                logger.info("Using final AI message only for rubric evaluation")

        # Store trace filtering metadata
        context.set_artifact("used_full_trace_for_rubric", use_full_trace)
        context.set_artifact("rubric_evaluation_input", rubric_evaluation_input)
        context.set_artifact("rubric_trace_extraction_error", rubric_trace_extraction_error)

        rubric_result = None
        llm_trait_labels = None
        metric_confusion_lists = None
        metric_results = None

        # Build model string for tracking (centralized via adapter registry)
        parsing_model = context.parsing_model
        parsing_model_str = self.get_model_string(parsing_model)

        try:
            # Create rubric evaluator with parsing model and strategy from config
            evaluator = RubricEvaluator(context.parsing_model, evaluation_strategy=context.rubric_evaluation_strategy)

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
                    from ....schemas.workflow import VerificationConfig

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
                    context.set_result_field("deep_judgment_rubric_performed", True)
                    context.set_result_field("extracted_rubric_excerpts", dj_result["excerpts"])
                    context.set_result_field("rubric_trait_reasoning", dj_result["reasoning"])
                    context.set_result_field("deep_judgment_rubric_scores", dj_result["deep_judgment_scores"])
                    context.set_result_field("standard_rubric_scores", dj_result["standard_scores"])
                    context.set_result_field("trait_metadata", dj_result["metadata"])
                    context.set_result_field(
                        "traits_without_valid_excerpts", dj_result["traits_without_valid_excerpts"]
                    )
                    if dj_result["hallucination_risks"]:
                        context.set_result_field(
                            "rubric_hallucination_risk_assessment", dj_result["hallucination_risks"]
                        )

                    # Calculate aggregated statistics
                    total_model_calls = sum(m.get("model_calls", 0) for m in dj_result["metadata"].values())
                    total_retries = sum(m.get("excerpt_retry_count", 0) for m in dj_result["metadata"].values())
                    context.set_result_field("total_deep_judgment_model_calls", total_model_calls)
                    context.set_result_field("total_traits_evaluated", len(dj_result["deep_judgment_scores"]))
                    context.set_result_field("total_excerpt_retries", total_retries)

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
        context.set_artifact("rubric_result", rubric_result)
        context.set_artifact("llm_trait_labels", llm_trait_labels)
        context.set_artifact("metric_confusion_lists", metric_confusion_lists)
        context.set_artifact("metric_results", metric_results)

        # Store updated usage tracker for next stages
        context.set_artifact("usage_tracker", usage_tracker)

        # Store in result builder
        context.set_result_field("verify_rubric", rubric_result)
        context.set_result_field("llm_trait_labels", llm_trait_labels)
        context.set_result_field("metric_trait_confusion_lists", metric_confusion_lists)
        context.set_result_field("metric_trait_metrics", metric_results)
        # Note: evaluation_rubric is now stored in shared_data at export time (not per-result)
        context.set_result_field("rubric_evaluation_strategy", context.rubric_evaluation_strategy)

        # Store rubric-specific trace filtering fields
        context.set_result_field("used_full_trace_for_rubric", use_full_trace)
        context.set_result_field("rubric_evaluation_input", rubric_evaluation_input)
        context.set_result_field("rubric_trace_extraction_error", rubric_trace_extraction_error)
