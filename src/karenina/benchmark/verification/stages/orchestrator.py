"""Stage orchestrator for verification pipeline.

Manages stage execution, dependencies, and error handling.
"""

import logging
import time

from ....schemas.domain import Rubric
from ....schemas.workflow import VerificationResult
from .abstention_check import AbstentionCheckStage
from .base import ArtifactKeys, StageList, StageRegistry, VerificationContext
from .deep_judgment_autofail import DeepJudgmentAutoFailStage
from .deep_judgment_rubric_auto_fail import DeepJudgmentRubricAutoFailStage
from .embedding_check import EmbeddingCheckStage
from .finalize_result import FinalizeResultStage
from .generate_answer import GenerateAnswerStage
from .parse_template import ParseTemplateStage
from .recursion_limit_autofail import RecursionLimitAutoFailStage
from .rubric_evaluation import RubricEvaluationStage
from .sufficiency_check import SufficiencyCheckStage
from .trace_validation_autofail import TraceValidationAutoFailStage
from .validate_template import ValidateTemplateStage
from .verify_template import VerifyTemplateStage

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
        3. RecursionLimitAutoFailStage (auto-fail if recursion limit hit)
        4. TraceValidationAutoFailStage (auto-fail if trace doesn't end with AI message)
        5. AbstentionCheckStage (optional, before parse - skips parsing if detected)
        6. SufficiencyCheckStage (optional, before parse - skips parsing if insufficient)
        7. ParseTemplateStage (requires raw_answer)
        8. VerifyTemplateStage (requires parsed_answer)
        9. EmbeddingCheckStage (optional, after verify)
        10. DeepJudgmentAutoFailStage (optional, after verify)
        11. RubricEvaluationStage (optional, after generate)
        12. DeepJudgmentRubricAutoFailStage (optional, after rubric)
        13. FinalizeResultStage (always last)
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
        rubric: Rubric | None = None,
        abstention_enabled: bool = False,
        sufficiency_enabled: bool = False,
        deep_judgment_enabled: bool = False,
        evaluation_mode: str = "template_only",
    ) -> "StageOrchestrator":
        """
        Build orchestrator from configuration.

        This method determines which stages to include based on the
        configuration flags and evaluation mode.

        Args:
            rubric: Optional rubric for evaluation
            abstention_enabled: Whether abstention detection is enabled
            sufficiency_enabled: Whether trace sufficiency detection is enabled
            deep_judgment_enabled: Whether deep-judgment parsing is enabled
            evaluation_mode: Evaluation mode determining which stages run:
                - "template_only": Template verification only (default)
                - "template_and_rubric": Template verification + rubric evaluation
                - "rubric_only": Skip template, only evaluate rubrics

        Returns:
            Configured StageOrchestrator instance
        """
        stages: StageList = []

        if evaluation_mode == "rubric_only":
            # Rubric-only mode: Skip template verification stages
            # Only generate answer, optionally check abstention, evaluate rubric, finalize
            stages.append(GenerateAnswerStage())

            # Auto-fail if recursion limit hit (always runs after GenerateAnswer)
            stages.append(RecursionLimitAutoFailStage())

            # Auto-fail if trace doesn't end with AI message
            stages.append(TraceValidationAutoFailStage())

            # Optional abstention check (can run on raw response)
            if abstention_enabled:
                stages.append(AbstentionCheckStage())

            # Rubric evaluation (required for rubric_only mode)
            if rubric and (rubric.llm_traits or rubric.regex_traits or rubric.callable_traits or rubric.metric_traits):
                stages.append(RubricEvaluationStage())
                # Deep judgment rubric auto-fail (if deep judgment traits were evaluated)
                stages.append(DeepJudgmentRubricAutoFailStage())

            # Finalize result (always last)
            stages.append(FinalizeResultStage())

        else:
            # template_only or template_and_rubric modes
            # Include all template verification stages
            stages.extend(
                [
                    ValidateTemplateStage(),
                    GenerateAnswerStage(),
                    RecursionLimitAutoFailStage(),  # Auto-fail if recursion limit hit
                    TraceValidationAutoFailStage(),  # Auto-fail if trace doesn't end with AI message
                ]
            )

            # Abstention check runs before parsing to skip expensive LLM calls
            # when model refused to answer
            if abstention_enabled:
                stages.append(AbstentionCheckStage())

            # Sufficiency check runs before parsing to skip expensive LLM calls
            # when response lacks information to populate the template
            if sufficiency_enabled:
                stages.append(SufficiencyCheckStage())

            # Template parsing and verification
            stages.extend(
                [
                    ParseTemplateStage(),
                    VerifyTemplateStage(),
                ]
            )

            # Optional verification enhancement stages
            # Note: Embedding check stage has its own should_run() logic
            # It only runs if field verification failed
            stages.append(EmbeddingCheckStage())

            if deep_judgment_enabled:
                stages.append(DeepJudgmentAutoFailStage())

            # Rubric evaluation (for template_and_rubric mode)
            if (
                evaluation_mode == "template_and_rubric"
                and rubric
                and (rubric.llm_traits or rubric.regex_traits or rubric.callable_traits or rubric.metric_traits)
            ):
                stages.append(RubricEvaluationStage())
                # Deep judgment rubric auto-fail (if deep judgment traits were evaluated)
                stages.append(DeepJudgmentRubricAutoFailStage())

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
        context.set_result_field(ArtifactKeys.TIMESTAMP, timestamp)

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

            # Update execution time after each stage so FinalizeResultStage has access to it
            execution_time = time.time() - start_time
            context.set_result_field(ArtifactKeys.EXECUTION_TIME, execution_time)

        # Extract final result
        final_result = context.get_artifact(ArtifactKeys.FINAL_RESULT)
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
