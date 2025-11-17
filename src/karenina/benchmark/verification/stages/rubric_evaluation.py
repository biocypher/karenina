"""Rubric evaluation stage.

Evaluates LLM responses against qualitative rubric criteria.
"""

import logging

from ..evaluators.rubric_evaluator import RubricEvaluator
from ..stage import BaseVerificationStage, VerificationContext
from ..utils import UsageTracker

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
        return ["raw_llm_response"]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            "rubric_result",
            "metric_confusion_lists",
            "metric_results",
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run only if rubric is configured and has traits.

        Rubric evaluation is independent of other verification stages,
        so it runs even if template verification failed.
        """
        if context.error:
            return False

        rubric = context.rubric
        if rubric is None:
            return False

        # Check if any trait lists are non-empty
        has_traits = any([rubric.traits, rubric.regex_traits, rubric.callable_traits, rubric.metric_traits])
        return has_traits

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

        # Retrieve usage tracker from previous stage or initialize new one
        usage_tracker = context.get_artifact("usage_tracker")
        if usage_tracker is None:
            usage_tracker = UsageTracker()
            logger.warning("No usage tracker found in context, initializing new one")

        rubric_result = None
        metric_confusion_lists = None
        metric_results = None

        # Build model string for tracking
        parsing_model = context.parsing_model
        if parsing_model.interface == "openrouter":
            parsing_model_str = parsing_model.model_name
        else:
            parsing_model_str = f"{parsing_model.model_provider}/{parsing_model.model_name}"

        try:
            # Create rubric evaluator with parsing model
            evaluator = RubricEvaluator(context.parsing_model)

            # Evaluate standard rubric traits
            if rubric is not None:
                rubric_result, usage_metadata_list = evaluator.evaluate_rubric(
                    question=context.question_text,
                    answer=raw_llm_response,
                    rubric=rubric,
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
                    answer=raw_llm_response,
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
            rubric_result = None
        except Exception as e:
            # Don't fail the entire verification if rubric evaluation fails
            logger.warning(f"Rubric evaluation failed for question {context.question_id}: {e}")
            rubric_result = None

        # Store results (even if None)
        context.set_artifact("rubric_result", rubric_result)
        context.set_artifact("metric_confusion_lists", metric_confusion_lists)
        context.set_artifact("metric_results", metric_results)

        # Store updated usage tracker for next stages
        context.set_artifact("usage_tracker", usage_tracker)

        # Store in result builder
        context.set_result_field("verify_rubric", rubric_result)
        context.set_result_field("metric_trait_confusion_lists", metric_confusion_lists)
        context.set_result_field("metric_trait_metrics", metric_results)
        context.set_result_field("evaluation_rubric", rubric.model_dump() if rubric else None)
