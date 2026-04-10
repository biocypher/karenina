"""Agentic rubric evaluation stage (Stage 11b).

Evaluates AgenticRubricTrait instances by launching an agent per trait
(individual strategy) or a single shared agent for all traits (shared
strategy). Each trait produces a score and an investigation trace.
"""

import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from karenina.adapters.registry import AdapterRegistry
from karenina.benchmark.verification.evaluators import AgenticTraitEvaluator
from karenina.benchmark.verification.prompts import PromptAssembler, PromptTask
from karenina.ports import PortCapabilities
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import AgenticRubricTrait

from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext

logger = logging.getLogger(__name__)


class AgenticRubricEvaluationStage(BaseVerificationStage):
    """Evaluate agentic rubric traits via agent investigation and extraction.

    This stage sits between Stage 11 (RubricEvaluation) and Stage 12
    (DeepJudgmentRubricAutoFail) in the pipeline. It handles only
    AgenticRubricTrait instances; standard rubric traits are evaluated
    by RubricEvaluationStage.

    Strategies:
        individual: One agent per trait (default). Each trait may use a
            different model via model_override.
        shared: One shared agent investigates all traits, then per-trait
            extraction pulls out individual scores. All traits must resolve
            to the same model; falls back to individual otherwise.

    Requires:
        raw_llm_response: The answering model's raw response text.

    Produces:
        agentic_rubric_evaluation_performed: True when this stage ran.
        agentic_trait_scores: Dict mapping trait name to score (or None on failure).
        agentic_trait_investigation_traces: Dict mapping trait name to investigation trace.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "AgenticRubricEvaluation"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        return [ArtifactKeys.RAW_LLM_RESPONSE]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            ArtifactKeys.AGENTIC_RUBRIC_EVALUATION_PERFORMED,
            ArtifactKeys.AGENTIC_TRAIT_SCORES,
            ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_TRACES,
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """Run only when agentic rubric traits are configured and no prior errors.

        The orchestrator gates stage inclusion by evaluation_mode, so
        this method only checks runtime conditions: error state and
        whether the rubric actually contains agentic_traits.
        """
        if not super().should_run(context):
            return False

        rubric = context.rubric
        if rubric is None:
            return False

        return bool(rubric.agentic_traits)

    def execute(self, context: VerificationContext) -> None:
        """Evaluate all agentic rubric traits.

        Dispatches to the individual or shared strategy based on
        context.agentic_rubric_strategy. When any trait has
        ``materialize_trace=True``, the answering agent trace is written
        to a file once (shared across all traits) before evaluation begins.
        The file is cleaned up afterwards unless any trait sets
        ``persist_trace=True``.

        Args:
            context: Verification context with rubric and artifacts.

        Raises:
            ValueError: If no agentic trait can be evaluated because
                the resolved model interface lacks ``agent_tier='deep_agent'``.
        """
        traits = context.rubric.agentic_traits  # type: ignore[union-attr]
        self._validate_agent_support(traits, context)
        raw_response = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
        workspace_path = context.workspace_path

        # Stage-level trace materialization: write once, share across traits
        needs_materialization = any(t.materialize_trace for t in traits)
        should_persist = any(t.persist_trace for t in traits)
        trace_file_path: Path | None = None

        if needs_materialization and raw_response:
            scenario_turn = getattr(context, "scenario_turn", None)
            trace_file_path = self._write_trace_file(
                workspace_path=workspace_path,
                trace=raw_response,
                question_id=context.question_id,
                scenario_turn=scenario_turn,
            )
            logger.info(
                "Materialized trace file for question '%s': %s",
                context.question_id,
                trace_file_path,
            )

        try:
            strategy = context.agentic_rubric_strategy
            if strategy == "shared":
                scores, traces = self._execute_shared(
                    traits,
                    context,
                    raw_response,
                    workspace_path,
                )
            else:
                scores, traces = self._execute_individual(
                    traits,
                    context,
                    raw_response,
                    workspace_path,
                )

            # Store in artifacts
            self.set_artifact_and_result(
                context,
                ArtifactKeys.AGENTIC_RUBRIC_EVALUATION_PERFORMED,
                True,
            )
            self.set_artifact_and_result(
                context,
                ArtifactKeys.AGENTIC_TRAIT_SCORES,
                scores,
            )
            self.set_artifact_and_result(
                context,
                ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_TRACES,
                traces,
            )

            evaluated = sum(1 for v in scores.values() if v is not None)
            logger.info(
                "Agentic rubric evaluation complete: %d/%d traits scored",
                evaluated,
                len(traits),
            )
        finally:
            # Cleanup trace file unless persistence requested
            if trace_file_path is not None and not should_persist:
                try:
                    trace_file_path.unlink(missing_ok=True)
                    logger.debug("Cleaned up trace file: %s", trace_file_path)
                except Exception:
                    logger.warning(
                        "Failed to clean up trace file: %s",
                        trace_file_path,
                        exc_info=True,
                    )

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _execute_individual(
        self,
        traits: list[AgenticRubricTrait],
        context: VerificationContext,
        raw_response: str,
        workspace_path: Path | None,
    ) -> tuple[dict[str, int | bool | dict[str, Any] | None], dict[str, str | None]]:
        """Evaluate each trait with its own agent.

        Returns:
            Tuple of (scores_dict, traces_dict).
        """
        # TODO: when context.agentic_rubric_parallel is True, evaluate traits concurrently
        scores: dict[str, int | bool | dict[str, Any] | None] = {}
        traces: dict[str, str | None] = {}

        for trait in traits:
            model = self._resolve_model(trait, context)
            if model is None:
                logger.warning(
                    "Skipping trait '%s': resolved model lacks agent support",
                    trait.name,
                )
                scores[trait.name] = None
                traces[trait.name] = None
                continue

            evaluator = AgenticTraitEvaluator(
                model_config=model,
                prompt_config=context.prompt_config,
            )
            score, trace = evaluator.evaluate_trait(
                trait=trait,
                question_text=context.question_text,
                raw_llm_response=raw_response,
                workspace_path=workspace_path,
            )
            if trait.is_template_kind and isinstance(score, dict):
                for field_name, value in score.items():
                    scores[f"{trait.name}.{field_name}"] = value
            else:
                scores[trait.name] = score
            traces[trait.name] = trace

        return scores, traces

    def _execute_shared(
        self,
        traits: list[AgenticRubricTrait],
        context: VerificationContext,
        raw_response: str,
        workspace_path: Path | None,
    ) -> tuple[dict[str, int | bool | dict[str, Any] | None], dict[str, str | None]]:
        """Evaluate all traits with a single shared agent.

        All traits must resolve to the same model (by interface, provider,
        and model_name). If they differ, falls back to individual strategy.

        Returns:
            Tuple of (scores_dict, traces_dict).
        """
        # Resolve all models and check they are identical
        resolved: list[tuple[AgenticRubricTrait, ModelConfig | None]] = [
            (trait, self._resolve_model(trait, context)) for trait in traits
        ]

        valid = [(trait, m) for trait, m in resolved if m is not None]
        if not valid:
            logger.warning("No traits have agent support; all skipped")
            empty_scores: dict[str, int | bool | dict[str, Any] | None] = {t.name: None for t in traits}
            empty_traces: dict[str, str | None] = {t.name: None for t in traits}
            return empty_scores, empty_traces

        # Check that all valid models match on interface + provider + model_name
        first_model = valid[0][1]
        first_key = (first_model.interface, first_model.model_provider, first_model.model_name)
        if not all((m.interface, m.model_provider, m.model_name) == first_key for _, m in valid):
            logger.info("Shared strategy: traits resolve to different models, falling back to individual strategy")
            return self._execute_individual(
                traits,
                context,
                raw_response,
                workspace_path,
            )

        # Build a combined investigation prompt
        evaluator = AgenticTraitEvaluator(
            model_config=first_model,
            prompt_config=context.prompt_config,
        )
        valid_traits = [trait for trait, _ in valid]

        combined_desc_parts = [f"- {trait.name}: {trait.description}" for trait in valid_traits]
        combined_description = "\n".join(combined_desc_parts)

        # Compute union of context modes and max of turns/timeout across traits
        include_trace = any(t.context_mode in ("trace_and_workspace", "trace_only") for t in valid_traits)
        include_workspace = any(t.context_mode in ("trace_and_workspace", "workspace_only") for t in valid_traits)
        shared_max_turns = max(t.max_turns for t in valid_traits)
        shared_timeout = max(t.timeout_seconds for t in valid_traits)

        # Run a single shared investigation
        try:
            from karenina.adapters import get_agent
            from karenina.ports import AgentConfig

            agent = get_agent(first_model)
            system_text = (
                "You are an evaluation agent investigating the quality of an LLM "
                "response. You have access to tools and can examine files, run "
                "code, and navigate the workspace.\n\n"
                "Evaluate ALL of the following criteria:\n"
                f"{combined_description}\n\n"
                "After investigating, report your findings for each criterion "
                "clearly so scores can be extracted per trait."
            )

            user_parts: list[str] = [f"Question: {context.question_text}"]

            if include_trace and raw_response:
                user_parts.append(f"\n--- ANSWERING AGENT TRACE ---\n{raw_response}\n--- END TRACE ---")

            if workspace_path and include_workspace:
                user_parts.append(f"\nWorkspace directory: {workspace_path}")

            user_text = "\n".join(user_parts)

            # Assemble with adapter + user instructions
            assembler = PromptAssembler(
                task=PromptTask.RUBRIC_AGENTIC_TRAIT_INVESTIGATION,
                interface=first_model.interface,
                capabilities=PortCapabilities(),
            )
            user_instructions = (
                context.prompt_config.get_for_task(PromptTask.RUBRIC_AGENTIC_TRAIT_INVESTIGATION.value)
                if context.prompt_config
                else None
            )
            messages = assembler.assemble(
                system_text=system_text,
                user_text=user_text,
                user_instructions=user_instructions,
            )

            agent_config = AgentConfig(
                max_turns=shared_max_turns,
                timeout=float(shared_timeout),
                workspace_path=workspace_path if workspace_path else None,
            )

            result = agent.run(messages=messages, config=agent_config)
            shared_trace = result.raw_trace
        except Exception:
            logger.warning(
                "Shared agentic investigation failed; falling back to individual",
                exc_info=True,
            )
            return self._execute_individual(
                traits,
                context,
                raw_response,
                workspace_path,
            )

        # Per-trait extraction from the shared trace
        result_scores: dict[str, int | bool | dict[str, Any] | None] = {}
        result_traces: dict[str, str | None] = {}

        for trait, model in resolved:
            if model is None:
                result_scores[trait.name] = None
                result_traces[trait.name] = None
                continue

            result_traces[trait.name] = shared_trace
            try:
                extracted = evaluator.run_extraction(trait, shared_trace)
                if trait.is_template_kind and isinstance(extracted, dict):
                    for field_name, value in extracted.items():
                        result_scores[f"{trait.name}.{field_name}"] = value
                else:
                    result_scores[trait.name] = extracted
            except Exception:
                logger.warning(
                    "Extraction failed for trait '%s' in shared strategy",
                    trait.name,
                    exc_info=True,
                )
                result_scores[trait.name] = None

        return result_scores, result_traces

    # ------------------------------------------------------------------
    # Trace materialization
    # ------------------------------------------------------------------

    @staticmethod
    def _write_trace_file(
        workspace_path: Path | None,
        trace: str,
        question_id: str,
        scenario_turn: int | None,
    ) -> Path:
        """Write trace to a file for agent grep/search access.

        When ``workspace_path`` is provided, the trace file is placed under
        ``<workspace>/traces/``. Otherwise a temporary directory
        is created as a fallback.

        Args:
            workspace_path: Resolved workspace directory, or None.
            trace: The full answering agent trace text.
            question_id: Question identifier (sanitized for filesystem safety).
            scenario_turn: Turn index for multi-turn scenarios, or None for
                single-turn evaluations.

        Returns:
            Path to the written trace file.
        """
        if workspace_path is None:
            trace_dir = Path(tempfile.mkdtemp(prefix="karenina_traces_"))
        else:
            trace_dir = Path(workspace_path) / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)

        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", question_id)
        filename = f"{safe_id}_turn{scenario_turn}_trace.txt" if scenario_turn is not None else f"{safe_id}_trace.txt"

        trace_path = trace_dir / filename
        trace_path.write_text(trace, encoding="utf-8")
        return trace_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_agent_support(
        traits: list[AgenticRubricTrait],
        context: VerificationContext,
    ) -> None:
        """Validate that every agentic trait can be evaluated.

        Checks each trait's resolved model (model_override or parsing_model)
        for ``agent_tier='deep_agent'``. Raises if any trait cannot be
        evaluated, since silent skipping hides configuration errors.

        Raises:
            ValueError: If any trait resolves to an interface that lacks
                ``agent_tier='deep_agent'``.
        """
        default_model = context.parsing_model
        default_spec = AdapterRegistry.get_spec(default_model.interface)
        default_ok = default_spec is not None and default_spec.agent_tier == "deep_agent"

        unsupported: list[str] = []
        for trait in traits:
            if trait.model_override is not None:
                spec = AdapterRegistry.get_spec(trait.model_override.interface)
                if spec is None or spec.agent_tier != "deep_agent":
                    unsupported.append(trait.name)
            elif not default_ok:
                unsupported.append(trait.name)

        if unsupported:
            trait_names = ", ".join(f"'{n}'" for n in unsupported)
            raise ValueError(
                f"Agentic rubric traits ({trait_names}) require an interface with "
                f"agent_tier='deep_agent' (e.g., 'claude_agent_sdk' or "
                f"'langchain_deep_agents'), but the resolved model uses "
                f"interface='{default_model.interface}' "
                f"(agent_tier='{default_spec.agent_tier if default_spec else 'unknown'}'). "
                f"Either change the parsing model interface or set model_override "
                f"on each agentic trait."
            )

    @staticmethod
    def _resolve_model(
        trait: AgenticRubricTrait,
        context: VerificationContext,
    ) -> ModelConfig | None:
        """Resolve the model to use for a given trait.

        Returns trait.model_override if set; otherwise context.parsing_model.
        Validates that the resolved model's interface has
        agent_tier='deep_agent'. Returns None if the interface lacks deep
        agent support (the trait will be skipped).
        """
        model = trait.model_override or context.parsing_model
        spec = AdapterRegistry.get_spec(model.interface)
        if spec is None or spec.agent_tier != "deep_agent":
            logger.warning(
                "Interface '%s' has agent_tier='%s' (not 'deep_agent'); trait '%s' will be skipped",
                model.interface,
                spec.agent_tier if spec else "unknown",
                trait.name,
            )
            return None
        return model
