"""Agentic rubric evaluation stage (Stage 11b).

Evaluates AgenticRubricTrait instances by launching an agent per trait
(individual strategy) or a single shared agent for all traits (shared
strategy). Each trait produces a score and an investigation trace.
"""

import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from karenina.adapters.agent_runtime import workspace_path_for_prompt
from karenina.adapters.registry import AdapterRegistry, close_adapter
from karenina.benchmark.verification.evaluators import AgenticTraitEvaluator
from karenina.benchmark.verification.evaluators.rubric.agentic_trait import build_mcp_servers
from karenina.benchmark.verification.prompts import PromptAssembler, PromptTask
from karenina.benchmark.verification.utils.trace_usage_tracker import UsageTracker, usage_to_callback_metadata
from karenina.ports import UsageMetadata
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import AgenticRubricTrait
from karenina.schemas.verification.model_identity import ModelIdentity

from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext

logger = logging.getLogger(__name__)

# UsageTracker stage key for the judge's investigation and extraction calls.
AGENTIC_RUBRIC_USAGE_STAGE = "agentic_rubric_evaluation"


@dataclass
class _AgenticRubricOutputs:
    """Per-trait outputs collected by one strategy run."""

    scores: dict[str, int | bool | dict[str, Any] | None] = field(default_factory=dict)
    traces: dict[str, str | None] = field(default_factory=dict)
    extraction_metadata: dict[str, dict[str, str | None]] = field(default_factory=dict)

    def record_score(self, trait: AgenticRubricTrait, score: int | bool | dict[str, Any] | None) -> None:
        """Store a trait score, expanding template kind dicts into dotted keys."""
        if trait.is_template_kind and isinstance(score, dict):
            for field_name, value in score.items():
                self.scores[f"{trait.name}.{field_name}"] = value
        else:
            self.scores[trait.name] = score

    def record_skipped(self, trait: AgenticRubricTrait) -> None:
        """Store an empty entry for a trait whose model lacks agent support."""
        self.scores[trait.name] = None
        self.traces[trait.name] = None


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
        agentic_trait_extraction_metadata: Dict mapping trait name to extraction provenance.

    The judge's investigation and extraction usage is recorded in the usage
    tracker under the ``agentic_rubric_evaluation`` stage.
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
            ArtifactKeys.AGENTIC_TRAIT_EXTRACTION_METADATA,
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

        usage_tracker = self.get_or_create_usage_tracker(context)

        try:
            strategy = context.agentic_rubric_strategy
            if strategy == "shared":
                outputs = self._execute_shared(
                    traits,
                    context,
                    raw_response,
                    workspace_path,
                    usage_tracker,
                )
            else:
                outputs = self._execute_individual(
                    traits,
                    context,
                    raw_response,
                    workspace_path,
                    usage_tracker,
                )
            context.set_artifact(ArtifactKeys.USAGE_TRACKER, usage_tracker)

            # Store in artifacts
            self.set_artifact_and_result(
                context,
                ArtifactKeys.AGENTIC_RUBRIC_EVALUATION_PERFORMED,
                True,
            )
            self.set_artifact_and_result(
                context,
                ArtifactKeys.AGENTIC_TRAIT_SCORES,
                outputs.scores,
            )
            self.set_artifact_and_result(
                context,
                ArtifactKeys.AGENTIC_TRAIT_INVESTIGATION_TRACES,
                outputs.traces,
            )
            self.set_artifact_and_result(
                context,
                ArtifactKeys.AGENTIC_TRAIT_EXTRACTION_METADATA,
                outputs.extraction_metadata,
            )

            evaluated = sum(1 for v in outputs.scores.values() if v is not None)
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
        usage_tracker: UsageTracker,
    ) -> _AgenticRubricOutputs:
        """Evaluate each trait with its own agent.

        Returns:
            The per-trait scores, traces and extraction metadata.
        """
        # TODO: when context.agentic_rubric_parallel is True, evaluate traits concurrently
        outputs = _AgenticRubricOutputs()

        for trait in traits:
            model = self._resolve_model(trait, context)
            if model is None:
                logger.warning(
                    "Skipping trait '%s': resolved model lacks agent support",
                    trait.name,
                )
                outputs.record_skipped(trait)
                continue

            evaluator = AgenticTraitEvaluator(
                model_config=model,
                prompt_config=context.prompt_config,
            )
            evaluation = evaluator.evaluate_trait(
                trait=trait,
                question_text=context.question_text,
                raw_llm_response=raw_response,
                workspace_path=workspace_path,
            )
            outputs.record_score(trait, evaluation.score)
            outputs.traces[trait.name] = evaluation.trace
            self._track_usage(usage_tracker, model, evaluation.investigation_usage)
            self._track_usage(usage_tracker, model, evaluation.extraction_usage)
            metadata = getattr(evaluator, "last_extraction_metadata", None)
            if isinstance(metadata, dict):
                outputs.extraction_metadata[trait.name] = metadata

        return outputs

    def _execute_shared(
        self,
        traits: list[AgenticRubricTrait],
        context: VerificationContext,
        raw_response: str,
        workspace_path: Path | None,
        usage_tracker: UsageTracker,
    ) -> _AgenticRubricOutputs:
        """Evaluate all traits with a single shared agent.

        All traits must resolve to the same model (by interface, provider,
        and model_name). If they differ, falls back to individual strategy.

        Returns:
            The per-trait scores, traces and extraction metadata.
        """
        # Resolve all models and check they are identical
        resolved: list[tuple[AgenticRubricTrait, ModelConfig | None]] = [
            (trait, self._resolve_model(trait, context)) for trait in traits
        ]

        valid = [(trait, m) for trait, m in resolved if m is not None]
        if not valid:
            logger.warning("No traits have agent support; all skipped")
            skipped = _AgenticRubricOutputs()
            for trait in traits:
                skipped.record_skipped(trait)
            return skipped

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
                usage_tracker,
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
        agent: Any | None = None
        try:
            from karenina.adapters import get_agent
            from karenina.ports import AgentConfig

            agent = get_agent(first_model)
            capabilities = agent.capabilities
            if capabilities.supports_code_execution:
                access_text = (
                    "You have access to tools and can examine files, execute code in the sandbox, "
                    "and navigate the workspace."
                    if capabilities.uses_sandboxed_execution
                    else "You have access to tools and can examine files, run code, and navigate the workspace."
                )
            else:
                access_text = (
                    "You have access to tools and can examine files and navigate the workspace, "
                    "but command execution is not available."
                )
            system_text = (
                "You are an evaluation agent investigating the quality of an LLM "
                f"response. {access_text}\n\n"
                "Evaluate ALL of the following criteria:\n"
                f"{combined_description}\n\n"
                "After investigating, report your findings for each criterion "
                "clearly so scores can be extracted per trait."
            )

            user_parts: list[str] = [f"Question: {context.question_text}"]

            if include_trace and raw_response:
                user_parts.append(f"\n--- ANSWERING AGENT TRACE ---\n{raw_response}\n--- END TRACE ---")

            if workspace_path and include_workspace:
                prompt_workspace = workspace_path_for_prompt(first_model, workspace_path)
                user_parts.append(f"\nWorkspace directory: {prompt_workspace}")

            user_text = "\n".join(user_parts)

            # Assemble with adapter + user instructions
            assembler = PromptAssembler(
                task=PromptTask.RUBRIC_AGENTIC_TRAIT_INVESTIGATION,
                interface=first_model.interface,
                capabilities=capabilities,
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

            result = agent.run(
                messages=messages,
                mcp_servers=build_mcp_servers(first_model),
                config=agent_config,
            )
            shared_trace = result.raw_trace
            self._track_usage(usage_tracker, first_model, result.usage)
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
                usage_tracker,
            )
        finally:
            if agent is not None:
                close_adapter(agent)

        # Per-trait extraction from the shared trace
        outputs = _AgenticRubricOutputs()

        for trait, model in resolved:
            if model is None:
                outputs.record_skipped(trait)
                continue

            outputs.traces[trait.name] = shared_trace
            try:
                outputs.record_score(trait, evaluator.run_extraction(trait, shared_trace))
            except Exception:
                logger.warning(
                    "Extraction failed for trait '%s' in shared strategy",
                    trait.name,
                    exc_info=True,
                )
                outputs.scores[trait.name] = None
            self._track_usage(usage_tracker, first_model, evaluator.last_extraction_usage)
            metadata = getattr(evaluator, "last_extraction_metadata", None)
            if isinstance(metadata, dict):
                outputs.extraction_metadata[trait.name] = metadata

        return outputs

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    @staticmethod
    def _track_usage(
        usage_tracker: UsageTracker,
        model: ModelConfig,
        usage: UsageMetadata | None,
    ) -> None:
        """Record one judge agent run or call under the agentic rubric usage stage.

        Usage that is missing or reports no tokens and no web searches is skipped.
        """
        if not isinstance(usage, UsageMetadata):
            return
        if not (usage.input_tokens or usage.output_tokens or usage.web_search_queries):
            return
        model_str = ModelIdentity.from_model_config(model, role="parsing").display_string
        usage_tracker.track_call(
            AGENTIC_RUBRIC_USAGE_STAGE,
            model_str,
            usage_to_callback_metadata(model_str, usage),
        )

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
