"""Stage orchestrator for verification pipeline.

Manages stage execution, dependencies, and error handling.
"""

import copy
import logging
import time

from karenina.schemas.entities import Rubric
from karenina.schemas.entities.rubric import DynamicRubric
from karenina.schemas.verification import VerificationResult
from karenina.utils.retry_policy import RetryPolicy, track_retries

from ..pipeline.abstention_check import AbstentionCheckStage
from ..pipeline.agentic_parse_template import AgenticParseTemplateStage
from ..pipeline.agentic_rubric_evaluation import AgenticRubricEvaluationStage
from ..pipeline.deep_judgment_autofail import DeepJudgmentAutoFailStage
from ..pipeline.deep_judgment_rubric_auto_fail import DeepJudgmentRubricAutoFailStage
from ..pipeline.embedding_check import EmbeddingCheckStage
from ..pipeline.finalize_result import FinalizeResultStage
from ..pipeline.generate_answer import GenerateAnswerStage
from ..pipeline.parse_template import ParseTemplateStage
from ..pipeline.placeholder_retry_autofail import PlaceholderRetryAutoFailStage
from ..pipeline.recursion_limit_autofail import RecursionLimitAutoFailStage
from ..pipeline.rubric_evaluation import RubricEvaluationStage
from ..pipeline.sufficiency_check import SufficiencyCheckStage
from ..pipeline.trace_validation_autofail import TraceValidationAutoFailStage
from ..pipeline.validate_template import ValidateTemplateStage
from ..pipeline.verify_template import VerifyTemplateStage
from .base import ArtifactKeys, StageList, StageRegistry, VerificationContext

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
        11b. AgenticRubricEvaluationStage (optional, after rubric evaluation)
        12. DeepJudgmentRubricAutoFailStage (optional, after rubric)
        13. FinalizeResultStage (always last)
    """

    def __init__(self, stages: StageList) -> None:
        """
        Initialize orchestrator with stage list.

        The ``stages`` list is kept intact on ``self.stages`` so external
        callers and tests can inspect the full configured pipeline (including
        the trailing ``FinalizeResultStage``). Internally, ``execute`` runs
        the non-finalize stages in its main try/except loop and then invokes
        the finalize stage exactly once, unconditionally, so a raise in any
        earlier stage (including stage 1) still produces a populated
        ``VerificationResult``.

        Args:
            stages: Ordered list of stages to execute
        """
        # Validate finalize placement before registering stages so layout
        # errors surface as ValueError (not as a downstream "already
        # registered" duplicate when two FinalizeResultStage instances share
        # the same stage name). Reject any layout that would cause the
        # synthesized trailing finalize to run alongside a duplicate or
        # mid-list one.
        finalize_positions = [
            index for index, candidate in enumerate(stages) if isinstance(candidate, FinalizeResultStage)
        ]
        if len(finalize_positions) > 1:
            raise ValueError(
                "FinalizeResultStage must appear at most once in the pipeline; "
                f"found {len(finalize_positions)} occurrences at indices "
                f"{finalize_positions}"
            )
        if finalize_positions and finalize_positions[0] != len(stages) - 1:
            raise ValueError(
                "FinalizeResultStage must be the last stage in the pipeline; "
                f"found at index {finalize_positions[0]} of {len(stages)}"
            )

        # Reject stages that claim the "FinalizeResult" name without inheriting
        # from FinalizeResultStage. Such stages would be partitioned into
        # _main_stages, silently coexisting with a synthesized trailing
        # finalize and bypassing the final_result contract (issue 203).
        impostors = [
            index
            for index, candidate in enumerate(stages)
            if candidate.name == "FinalizeResult" and not isinstance(candidate, FinalizeResultStage)
        ]
        if impostors:
            raise ValueError(
                "Stages named 'FinalizeResult' must inherit from FinalizeResultStage; "
                f"found non-FinalizeResultStage impostor(s) at index/indices {impostors}"
            )

        self.stages = stages
        self.registry = StageRegistry()

        # Register all stages
        for stage in stages:
            self.registry.register(stage)

        # Partition the stage list so finalize runs exactly once after the
        # main loop, regardless of exceptions in earlier stages. When no
        # FinalizeResultStage is present at all, synthesise a fresh one so
        # minimal stage lists (e.g. unit tests) still produce a finalized
        # VerificationResult.
        main_stages: StageList = list(stages)
        if finalize_positions:
            popped = main_stages.pop()
            assert isinstance(popped, FinalizeResultStage)  # noqa: S101 - narrow type for mypy
            self._finalize_stage: FinalizeResultStage = popped
        else:
            self._finalize_stage = FinalizeResultStage()
        self._main_stages: StageList = main_stages

    @classmethod
    def from_config(
        cls,
        rubric: Rubric | None = None,
        abstention_enabled: bool = False,
        sufficiency_enabled: bool = False,
        deep_judgment_enabled: bool = False,  # Whether any template deep-judgment mode is active
        evaluation_mode: str = "template_only",
        agentic_parsing: bool = False,
        dynamic_rubric: DynamicRubric | None = None,
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
            agentic_parsing: Whether to use agentic parsing (Stage 7b) instead of
                classical parsing (Stage 7a). Requires AgentPort support.
            dynamic_rubric: Optional dynamic rubric whose traits are conditionally
                evaluated based on concept presence in the response.

        Returns:
            Configured StageOrchestrator instance
        """
        stages: StageList = []

        # Check whether the dynamic rubric contributes non-agentic or agentic traits
        _dynamic_has_non_agentic = dynamic_rubric is not None and (
            dynamic_rubric.llm_traits
            or dynamic_rubric.regex_traits
            or dynamic_rubric.callable_traits
            or dynamic_rubric.metric_traits
        )
        _dynamic_has_agentic = dynamic_rubric is not None and bool(dynamic_rubric.agentic_traits)

        if evaluation_mode == "rubric_only":
            # Rubric-only mode: Skip template verification stages
            # Only generate answer, optionally check abstention, evaluate rubric, finalize
            stages.append(GenerateAnswerStage())

            # Auto-fail if recursion limit hit (always runs after GenerateAnswer)
            stages.append(RecursionLimitAutoFailStage())

            # Auto-fail if trace doesn't end with AI message
            stages.append(TraceValidationAutoFailStage())

            # Auto-fail if trace is solely a ModelRetryMiddleware exhaustion placeholder
            stages.append(PlaceholderRetryAutoFailStage())

            # Optional abstention check (can run on raw response)
            if abstention_enabled:
                stages.append(AbstentionCheckStage())

            # Sufficiency check is not applicable in rubric_only mode (no template parsing)
            if sufficiency_enabled:
                logger.info(
                    "sufficiency_enabled=True ignored in rubric_only mode: "
                    "sufficiency check requires template parsing which is skipped"
                )

            # Rubric evaluation (required for rubric_only mode)
            _rubric_has_non_agentic = rubric and (
                rubric.llm_traits or rubric.regex_traits or rubric.callable_traits or rubric.metric_traits
            )
            if _rubric_has_non_agentic or _dynamic_has_non_agentic:
                stages.append(RubricEvaluationStage())
                # Deep judgment rubric auto-fail (only when deep judgment is enabled)
                if deep_judgment_enabled:
                    stages.append(DeepJudgmentRubricAutoFailStage())

            # Stage 11b: Agentic rubric evaluation
            if (rubric and rubric.agentic_traits) or _dynamic_has_agentic:
                stages.append(AgenticRubricEvaluationStage())

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
                    PlaceholderRetryAutoFailStage(),  # Auto-fail on ModelRetryMiddleware placeholder
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

            # Template parsing: classical or agentic
            if agentic_parsing:
                stages.append(AgenticParseTemplateStage())
            else:
                stages.append(ParseTemplateStage())
            stages.append(VerifyTemplateStage())

            # Optional verification enhancement stages
            # Note: Embedding check stage has its own should_run() logic
            # It only runs if field verification failed
            stages.append(EmbeddingCheckStage())

            if deep_judgment_enabled:
                stages.append(DeepJudgmentAutoFailStage())

            # Rubric evaluation (for template_and_rubric mode)
            _rubric_has_non_agentic = rubric and (
                rubric.llm_traits or rubric.regex_traits or rubric.callable_traits or rubric.metric_traits
            )
            if evaluation_mode == "template_and_rubric" and (_rubric_has_non_agentic or _dynamic_has_non_agentic):
                stages.append(RubricEvaluationStage())
                # Deep judgment rubric auto-fail (only when deep judgment is enabled)
                if deep_judgment_enabled:
                    stages.append(DeepJudgmentRubricAutoFailStage())

            # Stage 11b: Agentic rubric evaluation (after Stage 11)
            if evaluation_mode == "template_and_rubric" and (
                (rubric and rubric.agentic_traits) or _dynamic_has_agentic
            ):
                stages.append(AgenticRubricEvaluationStage())

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

        Runs each non-finalize stage in sequence, respecting should_run()
        conditions and catching any exceptions so that the final
        ``FinalizeResultStage`` is invoked exactly once at the end of the
        run. Exceptions raised by stage ``execute()`` implementations are
        logged at ERROR level (with ``exc_info=True``), attributed to the
        raising stage via ``context.mark_error(..., stage=stage.name)``, and
        then swallowed so finalization can proceed. When the raising stage is
        ``ValidateTemplateStage``, the raised exception message is also
        mirrored into the ``TEMPLATE_VALIDATION_ERROR`` artifact so the
        failure classifier can attribute the failure to template validation.

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

        # Bind a fresh retry tracker for the duration of this pipeline run.
        # The tracker is pre-populated with per-category budgets from the
        # answering model's RetryPolicy (which is the same policy stamped onto
        # both answering and parsing models in batch_runner.py). Every retry
        # observed by an adapter's RetryExecutor increments the matching
        # category's "used" counter. FinalizeResultStage reads the snapshot
        # below and stores it on VerificationResultMetadata.retry_counts so
        # callers can see, per result, how many transient failures were
        # recovered from and what budget was available for each category.
        active_policy = (
            context.answering_model.retry_policy
            if context.answering_model is not None and context.answering_model.retry_policy is not None
            else RetryPolicy()
        )
        execution_time = 0.0
        with track_retries(active_policy) as retry_counts:
            # Seed the artifact with the initial budgets (used=0) so even
            # stages running first observe the snapshot (e.g. when
            # FinalizeResultStage is the only stage in a minimal pipeline).
            context.set_artifact(ArtifactKeys.RETRY_COUNTS, copy.deepcopy(retry_counts))

            # Execute main (non-finalize) stages in order. Exceptions are
            # caught so the guaranteed finalize call below still runs.
            for stage in self._main_stages:
                # Record the originating stage so mark_error() can attribute
                # failures even when callers omit the stage kwarg.
                context.begin_stage(stage.name)

                # Check if stage should run
                if not stage.should_run(context):
                    logger.debug("Skipping stage %s (should_run returned False)", stage.name)
                    continue

                # Execute stage
                error_before = context.error
                try:
                    logger.debug("Executing stage: %s", stage.name)
                    stage.execute(context)

                    # Log only if this stage introduced or changed the error
                    if context.error and context.error != error_before:
                        logger.warning("Stage %s set error: %s", stage.name, context.error)
                        # Don't break - FinalizeResultStage needs to run

                except Exception as e:
                    # Stage execution failed: log at ERROR, mark the error on
                    # the context, and let the guaranteed finalize call below
                    # produce a populated result.
                    error_msg = f"Stage {stage.name} raised exception: {type(e).__name__}: {e}"
                    logger.error("Stage %s raised exception", stage.name, exc_info=True)
                    context.mark_error(
                        error_msg,
                        category=context.error_registry.classify(e),
                        stage=stage.name,
                    )
                    # Mirror the raised message into TEMPLATE_VALIDATION_ERROR
                    # so classify_failure attributes the failure to template
                    # validation (rule 5). ValidateTemplateStage catches its
                    # own known-bad templates and sets this artifact itself;
                    # this path handles the raise-from-inside-execute case.
                    if isinstance(stage, ValidateTemplateStage):
                        context.set_artifact(
                            ArtifactKeys.TEMPLATE_VALIDATION_ERROR,
                            str(e),
                        )

                # Update execution time after each stage so FinalizeResultStage
                # has access to it
                execution_time = time.time() - start_time
                context.set_result_field(ArtifactKeys.EXECUTION_TIME, execution_time)

                # Snapshot the retry tracker so FinalizeResultStage (which
                # runs at the end of this loop) sees the budgets and counts
                # as a plain nested dict copy, not a live reference.
                context.set_artifact(ArtifactKeys.RETRY_COUNTS, copy.deepcopy(retry_counts))

            # Guaranteed finalize: runs exactly once regardless of whether any
            # earlier stage raised. Kept inside the track_retries context so
            # the snapshot visible to FinalizeResultStage reflects all retries
            # observed during the run.
            context.begin_stage(self._finalize_stage.name)
            execution_time = time.time() - start_time
            context.set_result_field(ArtifactKeys.EXECUTION_TIME, execution_time)
            context.set_artifact(ArtifactKeys.RETRY_COUNTS, copy.deepcopy(retry_counts))
            try:
                self._finalize_stage.execute(context)
            except Exception:
                logger.error("FinalizeResultStage raised exception", exc_info=True)
                raise

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
            "Verification pipeline complete for question %s (execution_time: %.2fs, success: %s)",
            context.question_id,
            execution_time,
            context.completed_without_errors,
        )

        return final_result

    def __repr__(self) -> str:
        """String representation for debugging."""
        stage_names = [stage.name for stage in self.stages]
        return f"StageOrchestrator(stages={stage_names})"
