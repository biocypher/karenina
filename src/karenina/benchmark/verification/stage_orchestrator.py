"""Stage orchestrator for verification pipeline.

Manages stage execution, dependencies, and error handling.
"""

import logging
import time

from ...schemas.rubric_class import Rubric
from ..models import ModelConfig, VerificationResult
from .stage import StageList, StageRegistry, VerificationContext
from .stages import (
    AbstentionCheckStage,
    DeepJudgmentAutoFailStage,
    EmbeddingCheckStage,
    FinalizeResultStage,
    GenerateAnswerStage,
    ParseTemplateStage,
    RubricEvaluationStage,
    ValidateTemplateStage,
    VerifyTemplateStage,
)

# Set up logger
logger = logging.getLogger(__name__)


class StageOrchestrator:
    """
    Orchestrates verification pipeline stages.

    The orchestrator:
    1. Builds a list of stages based on configuration
    2. Validates stage dependencies
    3. Executes stages in order
    4. Handles errors and stage skipping
    5. Returns the final VerificationResult

    Architecture:
        - Stages are executed sequentially in a defined order
        - Each stage can decide whether to run via should_run()
        - If a stage fails and sets context.error, remaining stages are skipped
        - FinalizeResultStage always runs to build the result object

    Stage Order:
        1. ValidateTemplateStage (always first)
        2. GenerateAnswerStage (always after validate)
        3. ParseTemplateStage (requires raw_answer)
        4. VerifyTemplateStage (requires parsed_answer)
        5. EmbeddingCheckStage (optional, after verify)
        6. AbstentionCheckStage (optional, after verify)
        7. DeepJudgmentAutoFailStage (optional, after verify)
        8. RubricEvaluationStage (optional, after generate)
        9. FinalizeResultStage (always last)
    """

    def __init__(self, stages: StageList) -> None:
        """
        Initialize orchestrator with stage list.

        Args:
            stages: Ordered list of stages to execute
        """
        self.stages = stages
        self.registry = StageRegistry()

        # Register all stages
        for stage in stages:
            self.registry.register(stage)

    @classmethod
    def from_config(
        cls,
        answering_model: ModelConfig,  # noqa: ARG003
        parsing_model: ModelConfig,  # noqa: ARG003
        rubric: Rubric | None = None,
        abstention_enabled: bool = False,
        deep_judgment_enabled: bool = False,
        evaluation_mode: str = "template_only",
    ) -> "StageOrchestrator":
        """
        Build orchestrator from configuration.

        This method determines which stages to include based on the
        configuration flags and evaluation mode.

        Args:
            answering_model: Answering model configuration (reserved for future use)
            parsing_model: Parsing model configuration (reserved for future use)
            rubric: Optional rubric for evaluation
            abstention_enabled: Whether abstention detection is enabled
            deep_judgment_enabled: Whether deep-judgment parsing is enabled
            evaluation_mode: Evaluation mode determining which stages run:
                - "template_only": Template verification only (default)
                - "template_and_rubric": Template verification + rubric evaluation
                - "rubric_only": Skip template, only evaluate rubrics

        Returns:
            Configured StageOrchestrator instance

        Note:
            answering_model and parsing_model parameters are currently unused
            but reserved for future stage configuration needs.
        """
        stages: StageList = []

        if evaluation_mode == "rubric_only":
            # Rubric-only mode: Skip template verification stages
            # Only generate answer, optionally check abstention, evaluate rubric, finalize
            stages.append(GenerateAnswerStage())

            # Optional abstention check (can run on raw response)
            if abstention_enabled:
                stages.append(AbstentionCheckStage())

            # Rubric evaluation (required for rubric_only mode)
            if rubric and (rubric.traits or rubric.manual_traits or rubric.metric_traits):
                stages.append(RubricEvaluationStage())

            # Finalize result (always last)
            stages.append(FinalizeResultStage())

        else:
            # template_only or template_and_rubric modes
            # Include all template verification stages
            stages.extend(
                [
                    ValidateTemplateStage(),
                    GenerateAnswerStage(),
                    ParseTemplateStage(),
                    VerifyTemplateStage(),
                ]
            )

            # Optional verification enhancement stages
            # Note: Embedding check stage has its own should_run() logic
            # It only runs if field verification failed
            stages.append(EmbeddingCheckStage())

            if abstention_enabled:
                stages.append(AbstentionCheckStage())

            if deep_judgment_enabled:
                stages.append(DeepJudgmentAutoFailStage())

            # Rubric evaluation (for template_and_rubric mode)
            if (
                evaluation_mode == "template_and_rubric"
                and rubric
                and (rubric.traits or rubric.manual_traits or rubric.metric_traits)
            ):
                stages.append(RubricEvaluationStage())

            # Finalize result (always last)
            stages.append(FinalizeResultStage())

        return cls(stages=stages)

    def validate_dependencies(self) -> list[str]:
        """
        Validate that all stage dependencies can be satisfied.

        Returns:
            List of error messages (empty if valid)
        """
        return self.registry.validate_dependencies(self.stages)

    def execute(self, context: VerificationContext) -> VerificationResult:
        """
        Execute the verification pipeline.

        Runs each stage in sequence, respecting should_run() conditions
        and handling errors gracefully.

        Args:
            context: Verification context (modified in-place)

        Returns:
            Final VerificationResult object

        Raises:
            ValueError: If stage dependencies are invalid
            RuntimeError: If FinalizeResultStage doesn't produce a result
        """
        # Initialize timing
        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        context.set_result_field("timestamp", timestamp)

        # Validate dependencies before execution
        errors = self.validate_dependencies()
        if errors:
            error_msg = "Stage dependency validation failed:\n" + "\n".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Execute stages in order
        for stage in self.stages:
            # Check if stage should run
            if not stage.should_run(context):
                logger.debug(f"Skipping stage {stage.name} (should_run returned False)")
                continue

            # Execute stage
            try:
                logger.debug(f"Executing stage: {stage.name}")
                stage.execute(context)

                # If error was set during execution, log it
                if context.error:
                    logger.warning(f"Stage {stage.name} set error: {context.error}")
                    # Don't break - FinalizeResultStage needs to run

            except Exception as e:
                # Stage execution failed - mark error and continue to finalize
                error_msg = f"Stage {stage.name} raised exception: {type(e).__name__}: {e}"
                logger.error(error_msg, exc_info=True)
                context.mark_error(error_msg)
                # Continue to FinalizeResultStage even on error

        # Calculate execution time
        execution_time = time.time() - start_time
        context.set_result_field("execution_time", execution_time)

        # Extract final result
        final_result = context.get_artifact("final_result")
        if final_result is None:
            error_msg = "FinalizeResultStage did not produce a final_result"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not isinstance(final_result, VerificationResult):
            error_msg = (
                f"FinalizeResultStage produced invalid result type: "
                f"{type(final_result).__name__} (expected VerificationResult)"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(
            f"Verification pipeline complete for question {context.question_id} "
            f"(execution_time: {execution_time:.2f}s, success: {context.completed_without_errors})"
        )

        return final_result

    def __repr__(self) -> str:
        """String representation for debugging."""
        stage_names = [stage.name for stage in self.stages]
        return f"StageOrchestrator(stages={stage_names})"
