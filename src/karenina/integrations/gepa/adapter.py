"""GEPA adapter for Karenina benchmark verification.

This module implements the GEPAAdapter interface using karenina's
verification pipeline as the evaluation metric.

Supports parallel feedback generation controlled by environment variables:
- KARENINA_ASYNC_ENABLED: Enable/disable parallel execution (default: true)
- KARENINA_ASYNC_MAX_WORKERS: Maximum parallel workers (default: 8)
"""

import logging
import os
import queue
import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from anyio.from_thread import start_blocking_portal

logger = logging.getLogger(__name__)

# Default settings for parallel execution
DEFAULT_ASYNC_ENABLED = True
DEFAULT_MAX_WORKERS = 8

try:
    from gepa import EvaluationBatch, GEPAAdapter  # type: ignore[attr-defined]

    GEPA_AVAILABLE = True
except ImportError:
    # Create stub classes for type checking when gepa not installed
    class GEPAAdapter:  # type: ignore[no-redef]
        pass

    class EvaluationBatch:  # type: ignore[no-redef]
        pass

    GEPA_AVAILABLE = False

# Imports after optional dependency handling
from karenina.integrations.gepa.config import ObjectiveConfig, OptimizationTarget  # noqa: E402
from karenina.integrations.gepa.data_types import KareninaDataInst, KareninaTrajectory  # noqa: E402
from karenina.integrations.gepa.feedback import LLMFeedbackGenerator  # noqa: E402
from karenina.integrations.gepa.scoring import (  # noqa: E402
    compute_objective_scores,
    extract_failed_fields,
)

if TYPE_CHECKING:
    from karenina.benchmark.benchmark import Benchmark
    from karenina.schemas.workflow.models import ModelConfig
    from karenina.schemas.workflow.verification.config import VerificationConfig
    from karenina.schemas.workflow.verification.result import VerificationResult
    from karenina.schemas.workflow.verification.result_set import VerificationResultSet


class KareninaAdapter(GEPAAdapter):  # type: ignore[type-arg]
    """GEPA adapter using karenina verification as evaluation metric.

    This adapter enables GEPA to optimize text components (prompts, instructions,
    tool descriptions) by evaluating candidates against karenina's verification
    pipeline.

    Features:
    - Multi-model combinatorial testing (all models × all questions)
    - Multi-objective Pareto optimization (template + rubric traits as separate objectives)
    - Knowledge distillation from successful to failed models in reflection
    - Configurable objective dimensions via ObjectiveConfig

    Example:
        >>> adapter = KareninaAdapter(
        ...     benchmark=benchmark,
        ...     base_config=verification_config,
        ...     targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        ...     objective_config=ObjectiveConfig(
        ...         include_template=True,
        ...         trait_mode=TraitSelectionMode.ALL,
        ...     ),
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
        objective_config: ObjectiveConfig,
        feedback_model_config: "ModelConfig | None" = None,
        enable_differential_analysis: bool = True,
        seed_mcp_tool_descriptions: dict[str, str] | None = None,
        auto_fetch_tool_descriptions: bool = True,
    ):
        """Initialize the adapter.

        Args:
            benchmark: Karenina Benchmark object with questions to verify
            base_config: Base VerificationConfig to use (will be modified with
                         optimized components for each candidate)
            targets: List of OptimizationTarget specifying what to optimize
            objective_config: Configuration for multi-objective optimization.
                Controls which dimensions (template, rubric traits) become
                separate Pareto objectives.
            feedback_model_config: Optional ModelConfig for LLM-generated feedback.
                If provided, uses an LLM to generate rich diagnostic feedback.
                If None, falls back to programmatic feedback.
            enable_differential_analysis: When True and feedback_model_config is set,
                performs differential analysis comparing successful vs failed traces.
            seed_mcp_tool_descriptions: Optional dict of seed tool descriptions.
                If provided, uses these as initial values for MCP tool optimization.
                If None and auto_fetch_tool_descriptions is True, fetches from MCP servers.
            auto_fetch_tool_descriptions: If True and MCP_TOOL_DESCRIPTIONS is in targets
                and seed_mcp_tool_descriptions is None, automatically fetch descriptions
                from configured MCP servers.
        """
        if not GEPA_AVAILABLE:
            raise ImportError("gepa package is required for KareninaAdapter. Install with: pip install gepa")

        self.benchmark = benchmark
        self.base_config = base_config
        self.targets = targets
        self.objective_config = objective_config
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

        # Store seed tool descriptions
        self._seed_tool_descriptions = seed_mcp_tool_descriptions

        # Cache trait max_scores and directionalities from global rubric for proper normalization
        self._trait_max_scores: dict[str, int] = {}
        self._trait_directionalities: dict[str, bool] = {}
        global_rubric = benchmark.get_global_rubric()
        if global_rubric:
            self._trait_max_scores = global_rubric.get_trait_max_scores()
            self._trait_directionalities = global_rubric.get_trait_directionalities()

        # Auto-fetch tool descriptions if targeting MCP tools and no seed provided
        if (
            OptimizationTarget.MCP_TOOL_DESCRIPTIONS in targets
            and self._seed_tool_descriptions is None
            and auto_fetch_tool_descriptions
        ):
            try:
                self._seed_tool_descriptions = self.fetch_seed_tool_descriptions()
                logger.info(
                    f"Auto-fetched seed tool descriptions for {len(self._seed_tool_descriptions)} tools: "
                    f"{list(self._seed_tool_descriptions.keys())}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to auto-fetch tool descriptions: {e}. "
                    f"Provide seed_mcp_tool_descriptions manually or ensure MCP servers are available."
                )

    @property
    def seed_tool_descriptions(self) -> dict[str, str] | None:
        """Get the seed tool descriptions (auto-fetched or manually provided)."""
        return self._seed_tool_descriptions

    def fetch_seed_tool_descriptions(self) -> dict[str, str]:
        """Fetch tool descriptions from configured MCP servers.

        Connects to MCP servers configured in answering_models and retrieves
        the current tool descriptions. Useful for initializing GEPA optimization
        with actual tool docstrings as seeds.

        Returns:
            Dict mapping tool names to descriptions.

        Raises:
            ValueError: If no MCP servers are configured.
            RuntimeError: If fetching fails.
        """
        from karenina.utils.mcp import fetch_tool_descriptions

        # Find first model with MCP configuration
        mcp_urls_dict = None
        mcp_tool_filter = None

        for model in self.base_config.answering_models:
            if model.mcp_urls_dict:
                mcp_urls_dict = model.mcp_urls_dict
                mcp_tool_filter = model.mcp_tool_filter
                break

        if not mcp_urls_dict:
            raise ValueError("No MCP servers configured in answering_models. Cannot fetch tool descriptions.")

        logger.info(f"Fetching tool descriptions from MCP servers: {list(mcp_urls_dict.keys())}")

        try:
            descriptions = fetch_tool_descriptions(mcp_urls_dict, mcp_tool_filter)
            logger.info(f"Fetched descriptions for {len(descriptions)} tools")
            return descriptions
        except Exception as e:
            raise RuntimeError(f"Failed to fetch tool descriptions: {e}") from e

    def evaluate(
        self,
        batch: list[KareninaDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:  # type: ignore[type-arg]
        """Evaluate a candidate using karenina verification.

        This is the main GEPA interface method. For each candidate, it:
        1. Injects optimized text into the verification config
        2. Runs verification on all questions × all models
        3. Computes multi-objective scores and collects trajectories

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
        # objective_scores: list of dicts with compound 'model:dimension' keys for Pareto optimization
        objective_scores: list[dict[str, float]] = []

        for inst in batch:
            # Collect results for this question across all models
            question_results = self._get_results_for_question(results, inst.question_id)

            # Per-question objective scores with compound keys
            per_question_objectives: dict[str, float] = {}

            for model_name, result in question_results.items():
                # Compute multi-objective scores for this model
                model_objectives = compute_objective_scores(
                    result,
                    model_name,
                    self.objective_config,
                    self._trait_max_scores,
                    self._trait_directionalities,
                )
                per_question_objectives.update(model_objectives)

                # Capture trajectory for reflection
                if capture_traces and trajectories is not None:
                    trajectories.append(
                        KareninaTrajectory(
                            data_inst=inst,
                            model_name=model_name,
                            model_config=self._get_model_config(model_name),
                            optimized_components=candidate,
                            verification_result=result,
                            raw_llm_response=(result.template.raw_llm_response if result.template else None),
                            parsing_error=result.metadata.error,
                            failed_fields=extract_failed_fields(result),
                            rubric_scores=self._extract_rubric_scores(result),
                        )
                    )

            # scores field: average of all objectives for simple ranking
            avg_score = (
                sum(per_question_objectives.values()) / len(per_question_objectives) if per_question_objectives else 0.0
            )
            scores.append(avg_score)
            outputs.append(question_results)
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
        eval_batch: EvaluationBatch,  # type: ignore[type-arg]
        components_to_update: list[str],
        async_enabled: bool | None = None,
        max_workers: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Build JSON-serializable feedback for GEPA's reflection LLM.

        This method extracts actionable feedback from verification failures
        to guide GEPA's prompt mutation. When an LLM feedback generator is
        configured, it produces rich diagnostic feedback including:
        - Differential analysis (comparing successful vs failed traces)
        - Rubric-specific feedback (when rubrics are attached)

        Falls back to programmatic feedback when no feedback generator is set.

        Supports parallel LLM feedback generation when async_enabled is True
        (controlled by KARENINA_ASYNC_ENABLED env var, default: true).

        Args:
            candidate: The evaluated candidate text components
            eval_batch: EvaluationBatch from evaluate()
            components_to_update: List of component names being optimized
            async_enabled: Whether to run feedback generation in parallel.
                Defaults to KARENINA_ASYNC_ENABLED env var (default: true).
            max_workers: Maximum parallel workers for feedback generation.
                Defaults to KARENINA_ASYNC_MAX_WORKERS env var (default: 8).

        Returns:
            Dict mapping component names to lists of feedback examples.
            Each example has "Inputs", "Generated Outputs", "Feedback" keys.
        """
        # Determine async mode from env var if not specified
        if async_enabled is None:
            async_enabled = os.getenv("KARENINA_ASYNC_ENABLED", "true").lower() == "true"

        if max_workers is None:
            max_workers = int(os.getenv("KARENINA_ASYNC_MAX_WORKERS", str(DEFAULT_MAX_WORKERS)))

        trajectories = eval_batch.trajectories or []

        # Group trajectories by question for cross-model analysis
        by_question = self._group_by_question(trajectories)

        # Build list of feedback tasks: (traj, successes) pairs
        feedback_tasks: list[tuple[KareninaTrajectory, list[KareninaTrajectory]]] = []
        for _question_id, question_trajs in by_question.items():
            successes = [t for t in question_trajs if t.passed()]
            failures = [t for t in question_trajs if not t.passed()]
            for traj in failures:
                feedback_tasks.append((traj, successes))

        # Generate feedback (parallel or sequential)
        if async_enabled and self.feedback_generator and len(feedback_tasks) > 1:
            logger.info(
                f"Generating feedback in parallel for {len(feedback_tasks)} failures (max_workers={max_workers})"
            )
            feedback_results = self._generate_feedback_parallel(feedback_tasks, max_workers)
        else:
            logger.debug(f"Generating feedback sequentially for {len(feedback_tasks)} failures")
            feedback_results = self._generate_feedback_sequential(feedback_tasks)

        # Build result structure
        result: dict[str, list[dict[str, Any]]] = {comp: [] for comp in components_to_update}
        for (traj, _successes), feedback_str in zip(feedback_tasks, feedback_results, strict=True):
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

    def _generate_feedback_sequential(
        self,
        tasks: list[tuple[KareninaTrajectory, list[KareninaTrajectory]]],
    ) -> list[str]:
        """Generate feedback for all tasks sequentially.

        Args:
            tasks: List of (failed_trajectory, successful_trajectories) tuples

        Returns:
            List of feedback strings in same order as tasks
        """
        results: list[str] = []
        for traj, successes in tasks:
            if self.feedback_generator:
                feedback_str = self.feedback_generator.generate_complete_feedback(
                    failed_trajectory=traj,
                    successful_trajectories=successes if self.enable_differential_analysis else None,
                    rubric_scores=traj.rubric_scores,
                )
            else:
                feedback_str = self._build_programmatic_feedback(traj, successes)
            results.append(feedback_str)
        return results

    def _generate_feedback_parallel(
        self,
        tasks: list[tuple[KareninaTrajectory, list[KareninaTrajectory]]],
        max_workers: int,
    ) -> list[str]:
        """Generate feedback for all tasks in parallel using threading.

        Uses the same pattern as batch_runner.py with BlockingPortal for
        async LLM invocations from worker threads.

        Args:
            tasks: List of (failed_trajectory, successful_trajectories) tuples
            max_workers: Maximum number of parallel workers

        Returns:
            List of feedback strings in same order as tasks
        """
        if not self.feedback_generator:
            # Fallback to sequential for programmatic feedback (fast anyway)
            return self._generate_feedback_sequential(tasks)

        # Create indexed task list
        indexed_tasks = list(enumerate(tasks))
        total = len(indexed_tasks)

        # Create work queue
        work_queue: queue.Queue[tuple[int, KareninaTrajectory, list[KareninaTrajectory]] | None] = queue.Queue()
        for idx, (traj, successes) in indexed_tasks:
            work_queue.put((idx, traj, successes))

        # Thread-safe storage for results
        results_lock = threading.Lock()
        results_by_index: dict[int, str] = {}

        # Completion tracking
        tasks_completed_event = threading.Event()

        def worker(portal: Any) -> None:
            """Worker function that processes feedback tasks from queue."""
            while True:
                try:
                    item = work_queue.get(timeout=1.0)
                except queue.Empty:
                    # Check if all tasks are completed
                    with results_lock:
                        if len(results_by_index) == total:
                            tasks_completed_event.set()
                    continue

                # Check for shutdown signal
                if item is None:
                    work_queue.task_done()
                    break

                idx, traj, successes = item

                try:
                    # Use async feedback generation via portal
                    feedback_str = portal.call(
                        self.feedback_generator.generate_complete_feedback_async,  # type: ignore[union-attr]
                        traj,
                        successes if self.enable_differential_analysis else None,
                        traj.rubric_scores,
                    )

                    # Store result
                    with results_lock:
                        results_by_index[idx] = feedback_str
                        if len(results_by_index) == total:
                            tasks_completed_event.set()

                except Exception as e:
                    logger.error(f"Feedback generation failed for task {idx}: {e}")
                    # Store error message as feedback
                    with results_lock:
                        results_by_index[idx] = f"Error generating feedback: {e}"
                        if len(results_by_index) == total:
                            tasks_completed_event.set()

                finally:
                    work_queue.task_done()

        # Use BlockingPortal for async operations from worker threads
        with start_blocking_portal(backend="asyncio") as portal:
            # Start worker threads
            workers = []
            effective_workers = min(max_workers, total)
            for _ in range(effective_workers):
                t = threading.Thread(target=worker, args=(portal,), daemon=True)
                t.start()
                workers.append(t)

            # Wait for all tasks to complete
            tasks_completed_event.wait()

            # Send shutdown signal to workers
            for _ in range(effective_workers):
                work_queue.put(None)

            # Wait for workers to finish
            for t in workers:
                t.join(timeout=5.0)

        # Return results in original order
        return [results_by_index[i] for i in range(total)]

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
