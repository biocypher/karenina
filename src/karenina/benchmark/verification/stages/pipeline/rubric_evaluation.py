"""Rubric evaluation stage.

Evaluates LLM responses against qualitative rubric criteria. When a
dynamic rubric is present, a concept presence check is run first: only
traits whose concepts appear in the response are promoted into the
static rubric for evaluation; absent traits are skipped.
"""

import logging
from dataclasses import asdict
from typing import Any

from karenina.adapters import get_llm
from karenina.benchmark.verification.evaluators import RubricEvaluator
from karenina.benchmark.verification.prompts import PromptAssembler, PromptTask
from karenina.benchmark.verification.prompts.rubric.presence_check import PresenceCheckPromptBuilder
from karenina.benchmark.verification.utils import prepare_evaluation_input
from karenina.benchmark.verification.utils.llm_judge_helpers import extract_judge_result
from karenina.ports import LLMResponse
from karenina.ports.capabilities import PortCapabilities
from karenina.schemas.entities import Rubric
from karenina.schemas.entities.rubric import (
    CallableRubricTrait,
    LLMRubricTrait,
    MetricRubricTrait,
    RegexRubricTrait,
)
from karenina.schemas.outputs.rubric import ConceptPresenceResult
from karenina.schemas.verification.model_identity import ModelIdentity

from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext
from ..helpers.deep_judgment_helpers import apply_deep_judgment_config_to_traits

logger = logging.getLogger(__name__)


class RubricEvaluationStage(BaseVerificationStage):
    """Evaluates answer against rubric criteria.

    This stage:
    1. Resolves dynamic rubric traits via concept presence check (if configured)
    2. Only runs if rubric is configured (static or promoted from dynamic)
    3. Creates RubricEvaluator with parsing model
    4. Evaluates standard rubric traits (score/binary)
    5. Evaluates metric traits separately (confusion matrix analysis)
    6. Handles evaluation errors gracefully (does not fail pipeline)

    Requires:
        - "raw_llm_response": Raw LLM response text (for evaluation)

    Produces:
        - "rubric_result": Dict of trait scores (dict or None)
        - "metric_confusion_lists": Confusion lists per metric trait (dict or None)
        - "metric_results": Computed metrics per metric trait (dict or None)
        - "dynamic_rubric_promoted_traits": List of promoted trait names (list or None)
        - "dynamic_rubric_skipped_traits": Dict of skipped trait name to reason (dict or None)

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
        """Run if rubric has traits or dynamic rubric has non-agentic traits.

        Rubric evaluation is independent of other verification stages,
        so it runs even if template verification failed.

        However, if trace validation failed (trace does not end with AI message)
        and we need to extract the final AI message (use_full_trace_for_rubric=False),
        then skip rubric evaluation since extraction would fail.

        Inherits error-checking from BaseVerificationStage.
        """
        if not super().should_run(context):
            return False

        # Check dynamic rubric first: if it has non-agentic traits, the stage
        # must run so _resolve_dynamic_rubric can promote them into context.rubric.
        if context.dynamic_rubric and not context.dynamic_rubric.is_empty():
            has_non_agentic = (
                context.dynamic_rubric.llm_traits
                or context.dynamic_rubric.regex_traits
                or context.dynamic_rubric.callable_traits
                or context.dynamic_rubric.metric_traits
            )
            if has_non_agentic:
                return True

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
                "Skipping rubric evaluation for question %s: "
                "trace validation failed and use_full_trace_for_rubric=False",
                context.question_id,
            )
            return False

        return True

    def execute(self, context: VerificationContext) -> None:
        """Evaluate answer against rubric.

        Args:
            context: Verification context

        Side Effects:
            - Sets rubric evaluation artifacts
            - Sets result fields for rubric metadata
            - Does NOT set context.error on rubric evaluation failure
        """
        self._resolve_dynamic_rubric(context)

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
            # Extraction failed; mark as error and stop
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

        # If rubric is None after dynamic resolution (all traits absent), skip evaluation
        if rubric is None or not any(
            [rubric.llm_traits, rubric.regex_traits, rubric.callable_traits, rubric.metric_traits]
        ):
            logger.info(
                "No evaluable rubric traits for question %s after dynamic resolution; skipping evaluation",
                context.question_id,
            )
            self._store_results(context, rubric_result, llm_trait_labels, metric_confusion_lists, metric_results)
            context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)
            return

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
                    logger.info("Deep judgment enabled for rubric traits in question %s", context.question_id)

                    # Create a minimal VerificationConfig for deep judgment settings
                    from karenina.schemas.verification import VerificationConfig

                    dj_config = VerificationConfig(
                        answering_models=[context.answering_model],
                        parsing_models=[context.parsing_model],
                        deep_judgment_rubric_max_excerpts_default=getattr(
                            context, "deep_judgment_max_excerpts_per_attribute", 3
                        ),
                        deep_judgment_rubric_fuzzy_match_threshold_default=getattr(
                            context, "deep_judgment_fuzzy_match_threshold", 0.80
                        ),
                        deep_judgment_rubric_excerpt_retry_attempts_default=getattr(
                            context, "deep_judgment_excerpt_retry_attempts", 2
                        ),
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
                        "Deep judgment used %d model calls with %d retries, tracked %d usage metadata entries",
                        total_model_calls,
                        total_retries,
                        len(usage_metadata_list),
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
                    "Evaluating %d metric trait(s) for question %s",
                    len(rubric.metric_traits),
                    context.question_id,
                )
                for trait in rubric.metric_traits:
                    logger.debug(
                        "  - Trait: %s (mode: %s, metrics: %s)", trait.name, trait.evaluation_mode, trait.metrics
                    )

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
                    "Metric evaluation complete. Results: %s",
                    list(metric_results.keys()) if metric_results else "None",
                )
                if metric_results:
                    for trait_name, metrics in metric_results.items():
                        logger.debug("     %s: %s", trait_name, metrics)

        except (ValueError, RuntimeError) as e:
            # Handle specific rubric evaluator errors (non-fatal)
            logger.warning(
                "Rubric evaluator initialization/configuration failed for question %s: %s",
                context.question_id,
                e,
            )
            rubric_result = None  # type: ignore[assignment]
        except Exception as e:
            # Don't fail the entire verification if rubric evaluation fails
            logger.warning("Rubric evaluation failed for question %s: %s", context.question_id, e)
            rubric_result = None  # type: ignore[assignment]

        self._store_results(context, rubric_result, llm_trait_labels, metric_confusion_lists, metric_results)

        # Store updated usage tracker for next stages
        context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)

    # ------------------------------------------------------------------
    # Dynamic rubric resolution
    # ------------------------------------------------------------------

    def _resolve_dynamic_rubric(self, context: VerificationContext) -> None:
        """Resolve dynamic rubric: call LLM for concept presence, promote or skip traits.

        For each non-agentic trait in the dynamic rubric, this method checks whether
        its concept is present in the response and either promotes it into
        ``context.rubric`` or records it as skipped. The ``rubric_trait_names``
        filter is applied after presence checking: a trait that is present but
        excluded by the filter is skipped with an appropriate reason.

        Name conflicts between dynamic and static rubric traits raise ValueError
        so that duplicate evaluation does not silently produce wrong results.

        Args:
            context: Verification context (modified in-place).

        Raises:
            ValueError: If a dynamic trait name collides with an existing
                static rubric trait name.
        """
        dynamic = context.dynamic_rubric
        if dynamic is None or dynamic.is_empty():
            return

        # Collect non-agentic traits only (agentic traits are handled by a separate stage)
        non_agentic_traits = (
            list(dynamic.llm_traits)
            + list(dynamic.regex_traits)
            + list(dynamic.callable_traits)
            + list(dynamic.metric_traits)
        )

        if not non_agentic_traits:
            return

        # Check for same-type name conflicts with existing static rubric.
        # Cross-type overlaps are allowed (results use type-segregated dicts).
        if context.rubric is not None:
            static_by_type: dict[type, set[str]] = {
                LLMRubricTrait: {t.name for t in context.rubric.llm_traits},
                RegexRubricTrait: {t.name for t in context.rubric.regex_traits},
                CallableRubricTrait: {t.name for t in context.rubric.callable_traits},
                MetricRubricTrait: {t.name for t in context.rubric.metric_traits},
            }
            for trait in non_agentic_traits:
                names = static_by_type.get(type(trait), set())
                if trait.name in names:
                    raise ValueError(
                        f"Dynamic {type(trait).__name__} trait '{trait.name}' conflicts with static rubric"
                    )

        # Call LLM for presence check
        presence_map = self._call_presence_check(context, non_agentic_traits)

        # Partition traits into promoted vs skipped
        promoted_names: list[str] = []
        skipped: dict[str, str] = {}
        trait_filter = set(context.rubric_trait_names) if context.rubric_trait_names else None

        promote_llm: list[Any] = []
        promote_regex: list[Any] = []
        promote_callable: list[Any] = []
        promote_metric: list[Any] = []

        for trait in non_agentic_traits:
            present = presence_map.get(trait.name, False)
            if not present:
                skipped[trait.name] = "concept not present in response"
                continue
            if trait_filter is not None and trait.name not in trait_filter:
                skipped[trait.name] = "excluded by rubric_trait_names filter"
                continue

            # Promote: route to the correct list by type
            promoted_names.append(trait.name)
            if isinstance(trait, LLMRubricTrait):
                promote_llm.append(trait)
            elif isinstance(trait, RegexRubricTrait):
                promote_regex.append(trait)
            elif isinstance(trait, CallableRubricTrait):
                promote_callable.append(trait)
            elif isinstance(trait, MetricRubricTrait):
                promote_metric.append(trait)

        # Merge promoted traits into context.rubric
        if promoted_names:
            if context.rubric is None:
                context.rubric = Rubric()
            context.rubric = context.rubric.model_copy(
                update={
                    "llm_traits": list(context.rubric.llm_traits) + promote_llm,
                    "regex_traits": list(context.rubric.regex_traits) + promote_regex,
                    "callable_traits": list(context.rubric.callable_traits) + promote_callable,
                    "metric_traits": list(context.rubric.metric_traits) + promote_metric,
                }
            )

        # Annotate promoted traits with "dynamic" provenance
        if promoted_names and context.trait_provenance is not None:
            for name in promoted_names:
                context.trait_provenance[name] = "dynamic"

        # Record artifacts for downstream consumers and result serialization
        context.set_artifact(ArtifactKeys.DYNAMIC_RUBRIC_PROMOTED_TRAITS, promoted_names or None)
        context.set_artifact(ArtifactKeys.DYNAMIC_RUBRIC_SKIPPED_TRAITS, skipped or None)
        context.set_result_field(ArtifactKeys.DYNAMIC_RUBRIC_PROMOTED_TRAITS, promoted_names or None)
        context.set_result_field(ArtifactKeys.DYNAMIC_RUBRIC_SKIPPED_TRAITS, skipped or None)

        logger.info(
            "Dynamic rubric resolved for question %s: promoted=%s, skipped=%s",
            context.question_id,
            promoted_names,
            list(skipped.keys()),
        )

    def _call_presence_check(
        self,
        context: VerificationContext,
        traits: list[Any],
    ) -> dict[str, bool]:
        """Call the LLM to check concept presence for dynamic rubric traits.

        Uses the parsing model, PresenceCheckPromptBuilder for prompts,
        PromptAssembler with PromptTask.RUBRIC_DYNAMIC_PRESENCE_CHECK,
        and ConceptPresenceResult as the structured output schema.

        The response text is extracted the same way as for rubric evaluation
        (respecting ``use_full_trace_for_rubric``).

        Args:
            context: Verification context.
            traits: Non-agentic traits requiring presence checking.

        Returns:
            Mapping of trait name to presence boolean.
        """
        # Determine response text (same logic as rubric evaluation input)
        raw_llm_response = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
        use_full_trace = context.use_full_trace_for_rubric
        trace_messages = context.get_artifact(ArtifactKeys.TRACE_MESSAGES)
        trace_input = trace_messages if trace_messages else raw_llm_response
        response_text, extraction_error = prepare_evaluation_input(trace_input, use_full_trace)

        if extraction_error is not None:
            logger.warning(
                "Cannot extract response text for presence check (question %s): %s. "
                "Marking all dynamic traits as absent.",
                context.question_id,
                extraction_error,
            )
            return {t.name: False for t in traits}

        # Build prompts
        builder = PresenceCheckPromptBuilder()
        system_text = builder.build_system_prompt()
        user_text = builder.build_user_prompt(traits, response_text)
        example_json = builder.build_example_json(traits)

        # Assemble with tri-section pattern
        parsing_model = context.parsing_model
        assembler = PromptAssembler(
            task=PromptTask.RUBRIC_DYNAMIC_PRESENCE_CHECK,
            interface=parsing_model.interface,
            capabilities=PortCapabilities(),
        )

        user_instructions = None
        if context.prompt_config is not None:
            user_instructions = context.prompt_config.get_for_task(PromptTask.RUBRIC_DYNAMIC_PRESENCE_CHECK.value)

        instruction_context: dict[str, object] = {
            "json_schema": ConceptPresenceResult.model_json_schema(),
            "example_json": example_json,
            "output_format_hint": 'Return a JSON object with a "results" key containing presence booleans.',
        }

        messages = assembler.assemble(
            system_text=system_text,
            user_text=user_text,
            user_instructions=user_instructions,
            instruction_context=instruction_context,
        )

        # Invoke structured LLM
        detection_config = parsing_model.model_copy(update={"temperature": 0.0})
        llm = get_llm(detection_config)
        structured_llm = llm.with_structured_output(ConceptPresenceResult)
        response: LLMResponse = structured_llm.invoke(messages)

        # Track usage
        usage_tracker = self.get_or_create_usage_tracker(context)
        usage_metadata = asdict(response.usage) if response.usage else {}
        if usage_metadata:
            model_str = ModelIdentity.from_model_config(parsing_model, role="parsing").display_string
            usage_tracker.track_call("dynamic_rubric_presence_check", model_str, usage_metadata)
            context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)

        # Extract result
        result = extract_judge_result(response, ConceptPresenceResult, "results")
        if result is not None:
            return result.to_dict()

        # Fallback: if structured output failed, try raw
        if isinstance(response.raw, ConceptPresenceResult):
            return response.raw.to_dict()

        logger.warning(
            "Presence check returned unexpected output for question %s. Marking all dynamic traits as absent.",
            context.question_id,
        )
        return {t.name: False for t in traits}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _store_results(
        context: VerificationContext,
        rubric_result: dict[str, Any] | None,
        llm_trait_labels: dict[str, str] | None,
        metric_confusion_lists: dict[str, Any] | None,
        metric_results: dict[str, Any] | None,
    ) -> None:
        """Store rubric evaluation results into context artifacts and result builder."""
        context.set_artifact(ArtifactKeys.RUBRIC_RESULT, rubric_result)
        context.set_artifact(ArtifactKeys.LLM_TRAIT_LABELS, llm_trait_labels)
        context.set_artifact(ArtifactKeys.METRIC_CONFUSION_LISTS, metric_confusion_lists)
        context.set_artifact(ArtifactKeys.METRIC_RESULTS, metric_results)

        context.set_result_field(ArtifactKeys.VERIFY_RUBRIC, rubric_result)
        context.set_result_field(ArtifactKeys.LLM_TRAIT_LABELS, llm_trait_labels)
        context.set_result_field(ArtifactKeys.METRIC_TRAIT_CONFUSION_LISTS, metric_confusion_lists)
        context.set_result_field(ArtifactKeys.METRIC_TRAIT_METRICS, metric_results)
        # Note: evaluation_rubric is now stored in shared_data at export time (not per-result)
        context.set_result_field(ArtifactKeys.RUBRIC_EVALUATION_STRATEGY, context.rubric_evaluation_strategy)
        # Note: trace filtering fields (evaluation_input, used_full_trace, trace_extraction_error)
        # are stored at root level by parse_template stage, not duplicated here for rubric
