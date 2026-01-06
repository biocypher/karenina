"""GEPA adapter for Karenina benchmark verification.

This module implements the GEPAAdapter interface using karenina's
verification pipeline as the evaluation metric.
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Any

try:
    from gepa import EvaluationBatch, GEPAAdapter

    GEPA_AVAILABLE = True
except ImportError:
    # Create stub classes for type checking when gepa not installed
    class GEPAAdapter:  # type: ignore[no-redef]
        pass

    class EvaluationBatch:  # type: ignore[no-redef]
        pass

    GEPA_AVAILABLE = False

from karenina.integrations.gepa.config import OptimizationTarget
from karenina.integrations.gepa.data_types import KareninaDataInst, KareninaTrajectory
from karenina.integrations.gepa.feedback import LLMFeedbackGenerator
from karenina.integrations.gepa.scoring import (
    compute_single_score,
    extract_failed_fields,
)

if TYPE_CHECKING:
    from karenina.benchmark.benchmark import Benchmark
    from karenina.schemas.workflow.models import ModelConfig
    from karenina.schemas.workflow.verification.config import VerificationConfig
    from karenina.schemas.workflow.verification.result import VerificationResult
    from karenina.schemas.workflow.verification.result_set import VerificationResultSet


class KareninaAdapter(GEPAAdapter):  # type: ignore[misc]
    """GEPA adapter using karenina verification as evaluation metric.

    This adapter enables GEPA to optimize text components (prompts, instructions,
    tool descriptions) by evaluating candidates against karenina's verification
    pipeline.

    Features:
    - Multi-model combinatorial testing (all models × all questions)
    - Per-model objective scores for Pareto optimization
    - Knowledge distillation from successful to failed models in reflection
    - Configurable template/rubric weighting

    Example:
        >>> adapter = KareninaAdapter(
        ...     benchmark=benchmark,
        ...     base_config=verification_config,
        ...     targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        ... )
        >>> result = gepa.optimize(
        ...     seed_candidate={"answering_system_prompt": "You are helpful."},
        ...     trainset=train_data,
        ...     valset=val_data,
        ...     adapter=adapter,
        ... )
    """

    def __init__(
        self,
        benchmark: "Benchmark",
        base_config: "VerificationConfig",
        targets: list[OptimizationTarget],
        template_weight: float = 0.7,
        rubric_weight: float = 0.3,
        feedback_model_config: "ModelConfig | None" = None,
        enable_differential_analysis: bool = True,
    ):
        """Initialize the adapter.

        Args:
            benchmark: Karenina Benchmark object with questions to verify
            base_config: Base VerificationConfig to use (will be modified with
                         optimized components for each candidate)
            targets: List of OptimizationTarget specifying what to optimize
            template_weight: Weight for template pass/fail in scoring (0.0-1.0)
            rubric_weight: Weight for rubric scores in scoring (0.0-1.0)
            feedback_model_config: Optional ModelConfig for LLM-generated feedback.
                If provided, uses an LLM to generate rich diagnostic feedback.
                If None, falls back to programmatic feedback.
            enable_differential_analysis: When True and feedback_model_config is set,
                performs differential analysis comparing successful vs failed traces.
        """
        if not GEPA_AVAILABLE:
            raise ImportError("gepa package is required for KareninaAdapter. Install with: pip install gepa")

        self.benchmark = benchmark
        self.base_config = base_config
        self.targets = targets
        self.template_weight = template_weight
        self.rubric_weight = rubric_weight
        self.enable_differential_analysis = enable_differential_analysis

        # Initialize feedback generator if model config provided
        self.feedback_generator: LLMFeedbackGenerator | None = None
        if feedback_model_config is not None:
            self.feedback_generator = LLMFeedbackGenerator(feedback_model_config)

        # Cache model configs for quick lookup
        self._model_configs: dict[str, ModelConfig] = {}
        for model in base_config.answering_models:
            if model.model_name:
                self._model_configs[model.model_name] = model

    def evaluate(
        self,
        batch: list[KareninaDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate a candidate using karenina verification.

        This is the main GEPA interface method. For each candidate, it:
        1. Injects optimized text into the verification config
        2. Runs verification on all questions × all models
        3. Computes scores and collects trajectories

        Args:
            batch: List of KareninaDataInst to evaluate
            candidate: Dict mapping component names to optimized text
            capture_traces: Whether to collect execution traces for reflection

        Returns:
            EvaluationBatch with outputs, scores, trajectories, and objective_scores
        """
        # Inject candidate text into verification config
        config = self._inject_candidate(candidate)

        # Run verification on all questions
        question_ids = [inst.question_id for inst in batch]
        results = self.benchmark.run_verification(
            config,
            question_ids=question_ids,
        )

        # Build outputs, scores, trajectories
        outputs: list[dict[str, VerificationResult]] = []
        scores: list[float] = []
        trajectories: list[KareninaTrajectory] | None = [] if capture_traces else None
        # objective_scores is per-example: list of dicts mapping model_name -> score for Pareto optimization
        objective_scores: list[dict[str, float]] = []

        for inst in batch:
            # Collect results for this question across all models
            question_results = self._get_results_for_question(results, inst.question_id)

            # Aggregate score across models (average)
            model_scores_for_question: list[float] = []
            # Per-question objective scores for multi-model Pareto optimization
            per_question_objectives: dict[str, float] = {}
            for model_name, result in question_results.items():
                score = compute_single_score(result, self.template_weight, self.rubric_weight)
                model_scores_for_question.append(score)

                # Track per-model objective scores for this question
                per_question_objectives[model_name] = score

                # Capture trajectory for reflection
                if capture_traces and trajectories is not None:
                    trajectories.append(
                        KareninaTrajectory(
                            data_inst=inst,
                            model_name=model_name,
                            model_config=self._get_model_config(model_name),
                            optimized_components=candidate,
                            verification_result=result,
                            score=score,
                            raw_llm_response=(result.template.raw_llm_response if result.template else None),
                            parsing_error=result.metadata.error,
                            failed_fields=extract_failed_fields(result),
                            rubric_scores=self._extract_rubric_scores(result),
                        )
                    )

            # Average score across models for this question
            avg_score = (
                sum(model_scores_for_question) / len(model_scores_for_question) if model_scores_for_question else 0.0
            )
            scores.append(avg_score)
            outputs.append(question_results)
            # Append per-question objective scores
            objective_scores.append(per_question_objectives)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores if objective_scores else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build JSON-serializable feedback for GEPA's reflection LLM.

        This method extracts actionable feedback from verification failures
        to guide GEPA's prompt mutation. When an LLM feedback generator is
        configured, it produces rich diagnostic feedback including:
        - Differential analysis (comparing successful vs failed traces)
        - Rubric-specific feedback (when rubrics are attached)

        Falls back to programmatic feedback when no feedback generator is set.

        Args:
            candidate: The evaluated candidate text components
            eval_batch: EvaluationBatch from evaluate()
            components_to_update: List of component names being optimized

        Returns:
            Dict mapping component names to lists of feedback examples.
            Each example has "Inputs", "Generated Outputs", "Feedback" keys.
        """
        result: dict[str, list[dict[str, Any]]] = {comp: [] for comp in components_to_update}
        trajectories = eval_batch.trajectories or []

        # Group trajectories by question for cross-model analysis
        by_question = self._group_by_question(trajectories)

        for _question_id, question_trajs in by_question.items():
            successes = [t for t in question_trajs if t.passed()]
            failures = [t for t in question_trajs if not t.passed()]

            # Create feedback for each failed trajectory
            for traj in failures:
                if self.feedback_generator:
                    # Use LLM-generated feedback with optional differential analysis
                    feedback_str = self.feedback_generator.generate_complete_feedback(
                        failed_trajectory=traj,
                        successful_trajectories=successes if self.enable_differential_analysis else None,
                        rubric_scores=traj.rubric_scores,
                    )
                else:
                    # Fallback to programmatic feedback
                    feedback_str = self._build_programmatic_feedback(traj, successes)

                # Add to each component being optimized
                for comp in components_to_update:
                    result[comp].append(
                        {
                            "Inputs": {
                                "question": traj.data_inst.question_text,
                                "model": traj.model_name,
                                "current_prompt": candidate.get(comp, ""),
                            },
                            "Generated Outputs": traj.raw_llm_response or "(no response)",
                            "Feedback": feedback_str,
                        }
                    )

        return result

    def _build_programmatic_feedback(
        self,
        traj: KareninaTrajectory,
        successes: list[KareninaTrajectory],
    ) -> str:
        """Build programmatic feedback without LLM (fallback).

        Args:
            traj: The failed trajectory
            successes: List of successful trajectories for the same question

        Returns:
            Feedback string with parsing errors, failed fields, and knowledge distillation.
        """
        feedback_parts: list[str] = []

        if traj.parsing_error:
            feedback_parts.append(f"Parsing error: {traj.parsing_error}")

        if traj.failed_fields:
            feedback_parts.append(f"Failed fields: {', '.join(traj.failed_fields)}")

        feedback_parts.append(f"Expected answer: {traj.data_inst.raw_answer}")

        # Knowledge distillation insight
        if successes:
            successful_patterns = self._extract_successful_patterns(successes)
            successful_models = successful_patterns.get("successful_models", [])
            response_sample = successful_patterns.get("response_structures", [""])[0]
            feedback_parts.append(
                f"Note: Models {successful_models} succeeded on this question. "
                f"Their responses included: {response_sample[:100]}..."
            )

        return "\n".join(feedback_parts)

    def _inject_candidate(self, candidate: dict[str, str]) -> "VerificationConfig":
        """Create config with candidate's optimized text injected.

        Args:
            candidate: Dict mapping component names to optimized text

        Returns:
            Modified VerificationConfig with optimized components
        """
        config = self.base_config.model_copy(deep=True)

        if OptimizationTarget.ANSWERING_SYSTEM_PROMPT in self.targets:
            prompt = candidate.get("answering_system_prompt")
            if prompt:
                for model in config.answering_models:
                    model.system_prompt = prompt

        if OptimizationTarget.PARSING_INSTRUCTIONS in self.targets:
            instructions = candidate.get("parsing_instructions")
            if instructions:
                # Store as override in config (will be used by template evaluator)
                if not hasattr(config, "parsing_instructions_override"):
                    # Add dynamically if not present
                    object.__setattr__(config, "parsing_instructions_override", instructions)
                else:
                    config.parsing_instructions_override = instructions

        if OptimizationTarget.MCP_TOOL_DESCRIPTIONS in self.targets:
            # Collect MCP tool description overrides from candidate
            tool_overrides: dict[str, str] = {}
            for key, value in candidate.items():
                if key.startswith("mcp_tool_"):
                    tool_name = key[9:]  # Remove "mcp_tool_" prefix
                    tool_overrides[tool_name] = value

            if tool_overrides:
                for model in config.answering_models:
                    if model.mcp_urls_dict:
                        # Store overrides (will be applied when creating MCP client)
                        if not hasattr(model, "mcp_tool_description_overrides"):
                            object.__setattr__(model, "mcp_tool_description_overrides", tool_overrides)
                        else:
                            model.mcp_tool_description_overrides = tool_overrides

        return config

    def _get_results_for_question(
        self,
        results: "VerificationResultSet",
        question_id: str,
    ) -> dict[str, "VerificationResult"]:
        """Get all results for a specific question, keyed by model name.

        Args:
            results: VerificationResultSet from run_verification
            question_id: Question ID to filter for

        Returns:
            Dict mapping model names to their results for this question
        """
        question_results: dict[str, VerificationResult] = {}

        # VerificationResultSet is iterable
        for result in results:
            if result.metadata.question_id == question_id:
                model_name = result.metadata.answering_model or "unknown"
                question_results[model_name] = result

        return question_results

    def _get_model_config(self, model_name: str) -> "ModelConfig":
        """Get ModelConfig for a model by name."""
        if model_name in self._model_configs:
            return self._model_configs[model_name]
        # Return first model as fallback
        return self.base_config.answering_models[0]

    def _group_by_question(
        self,
        trajectories: list[KareninaTrajectory],
    ) -> dict[str, list[KareninaTrajectory]]:
        """Group trajectories by question ID."""
        groups: dict[str, list[KareninaTrajectory]] = defaultdict(list)
        for traj in trajectories:
            groups[traj.data_inst.question_id].append(traj)
        return groups

    def _extract_successful_patterns(
        self,
        successes: list[KareninaTrajectory],
    ) -> dict[str, Any]:
        """Extract common patterns from successful model responses.

        Used for knowledge distillation in reflection feedback.
        """
        return {
            "response_structures": [t.raw_llm_response[:200] if t.raw_llm_response else "" for t in successes],
            "successful_models": [t.model_name for t in successes],
        }

    def _extract_rubric_scores(
        self,
        result: "VerificationResult",
    ) -> dict[str, float] | None:
        """Extract per-trait rubric scores from a result."""
        if not result.rubric or not result.rubric.rubric_evaluation_performed:
            return None

        scores: dict[str, float] = {}
        # Use get_all_trait_scores() which aggregates all trait types
        all_scores = result.rubric.get_all_trait_scores()
        for trait_name, trait_result in all_scores.items():
            if isinstance(trait_result, bool):
                scores[trait_name] = 1.0 if trait_result else 0.0
            elif isinstance(trait_result, int | float):
                scores[trait_name] = float(trait_result)
            elif isinstance(trait_result, dict) and "f1" in trait_result:
                # Metric trait - use F1 score if available
                scores[trait_name] = float(trait_result["f1"])

        return scores if scores else None
