"""LLM-based feedback generation for GEPA optimization.

This module provides rich diagnostic feedback by using an LLM to analyze
verification failures. It supports:
- Single trajectory analysis (when only one model fails)
- Differential analysis (comparing successful vs failed traces)
- Rubric-specific feedback (when rubrics are attached to questions)
- Async/parallel feedback generation for improved performance
"""

import asyncio
from typing import TYPE_CHECKING, Any

from karenina.adapters.factory import get_llm
from karenina.ports import LLMPort, Message
from karenina.schemas.config import INTERFACES_NO_PROVIDER_REQUIRED

if TYPE_CHECKING:
    from karenina.integrations.gepa.data_types import KareninaTrajectory
    from karenina.schemas.config import ModelConfig


from karenina.integrations.gepa.prompts.feedback import (
    DIFFERENTIAL_FEEDBACK_SYSTEM_PROMPT,
    RUBRIC_FEEDBACK_SYSTEM_PROMPT,
    SINGLE_FEEDBACK_SYSTEM_PROMPT,
)


class LLMFeedbackGenerator:
    """Generates rich diagnostic feedback using an LLM.

    This class provides LLM-powered analysis of verification failures to produce
    more actionable feedback than simple programmatic string concatenation.

    Example:
        >>> from karenina.schemas.config import ModelConfig
        >>> config = ModelConfig(
        ...     id="feedback-llm",
        ...     model_provider="openai",
        ...     model_name="gpt-4o-mini",
        ...     temperature=0.7,
        ... )
        >>> generator = LLMFeedbackGenerator(config)
        >>> feedback = generator.generate_complete_feedback(
        ...     failed_trajectory=traj,
        ...     successful_trajectories=successes,
        ...     rubric_scores=traj.rubric_scores,
        ... )
    """

    def __init__(self, model_config: "ModelConfig"):
        """Initialize the feedback generator.

        Args:
            model_config: Configuration for the feedback LLM.

        Raises:
            ValueError: If model_config is missing required fields.
            RuntimeError: If LLM initialization fails.
        """
        if not model_config.model_name:
            raise ValueError("model_name is required in model configuration")

        if model_config.interface not in INTERFACES_NO_PROVIDER_REQUIRED and not model_config.model_provider:
            raise ValueError(f"model_provider is required for {model_config.interface} interface")

        self.model_config = model_config

        try:
            self.llm: LLMPort = get_llm(model_config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize feedback LLM: {e}") from e

    def generate_single_feedback(
        self,
        trajectory: "KareninaTrajectory",
    ) -> str:
        """Generate feedback for a single failed trajectory.

        Args:
            trajectory: The failed trajectory to analyze.

        Returns:
            LLM-generated feedback explaining the failure and suggesting improvements.
        """
        prompt = self._build_single_feedback_prompt(trajectory)

        messages = [
            Message.system(SINGLE_FEEDBACK_SYSTEM_PROMPT),
            Message.user(prompt),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def generate_differential_feedback(
        self,
        failed_trajectory: "KareninaTrajectory",
        successful_trajectories: list["KareninaTrajectory"],
    ) -> str:
        """Generate feedback by comparing failed vs successful traces.

        This method performs differential analysis to identify what successful
        models did differently that allowed them to pass verification.

        Args:
            failed_trajectory: The trajectory that failed verification.
            successful_trajectories: List of trajectories that passed verification.

        Returns:
            LLM-generated feedback with differential analysis.
        """
        prompt = self._build_differential_feedback_prompt(failed_trajectory, successful_trajectories)

        messages = [
            Message.system(DIFFERENTIAL_FEEDBACK_SYSTEM_PROMPT),
            Message.user(prompt),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def generate_rubric_feedback(
        self,
        trajectory: "KareninaTrajectory",
        rubric_scores: dict[str, Any],
    ) -> str:
        """Generate feedback for rubric evaluation results.

        Args:
            trajectory: The trajectory with rubric evaluation.
            rubric_scores: Per-trait rubric scores.

        Returns:
            LLM-generated feedback explaining rubric failures.
        """
        prompt = self._build_rubric_feedback_prompt(trajectory, rubric_scores)

        messages = [
            Message.system(RUBRIC_FEEDBACK_SYSTEM_PROMPT),
            Message.user(prompt),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def generate_complete_feedback(
        self,
        failed_trajectory: "KareninaTrajectory",
        successful_trajectories: list["KareninaTrajectory"] | None,
        rubric_scores: dict[str, Any] | None,
    ) -> str:
        """Generate combined template verification + rubric feedback.

        This is the main entry point for feedback generation. It generates:
        1. Template verification feedback (differential if successes exist, single otherwise)
        2. Rubric evaluation feedback (if rubrics are present)

        Args:
            failed_trajectory: The trajectory that failed verification.
            successful_trajectories: Optional list of trajectories that passed.
            rubric_scores: Optional per-trait rubric scores.

        Returns:
            Combined feedback string with both template and rubric analysis.
        """
        parts: list[str] = []

        # 1. Template verification feedback
        parts.append("--- TEMPLATE VERIFICATION FEEDBACK ---")
        if successful_trajectories:
            parts.append(self.generate_differential_feedback(failed_trajectory, successful_trajectories))
        else:
            parts.append(self.generate_single_feedback(failed_trajectory))

        # 2. Rubric feedback (if rubrics present)
        if rubric_scores:
            parts.append("\n--- RUBRIC EVALUATION FEEDBACK ---")
            parts.append(self.generate_rubric_feedback(failed_trajectory, rubric_scores))

        return "\n".join(parts)

    # =========================================================================
    # Async Methods for Parallel Feedback Generation
    # =========================================================================

    async def generate_single_feedback_async(
        self,
        trajectory: "KareninaTrajectory",
    ) -> str:
        """Async version of generate_single_feedback.

        Args:
            trajectory: The failed trajectory to analyze.

        Returns:
            LLM-generated feedback explaining the failure and suggesting improvements.
        """
        prompt = self._build_single_feedback_prompt(trajectory)

        messages = [
            Message.system(SINGLE_FEEDBACK_SYSTEM_PROMPT),
            Message.user(prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    async def generate_differential_feedback_async(
        self,
        failed_trajectory: "KareninaTrajectory",
        successful_trajectories: list["KareninaTrajectory"],
    ) -> str:
        """Async version of generate_differential_feedback.

        Args:
            failed_trajectory: The trajectory that failed verification.
            successful_trajectories: List of trajectories that passed verification.

        Returns:
            LLM-generated feedback with differential analysis.
        """
        prompt = self._build_differential_feedback_prompt(failed_trajectory, successful_trajectories)

        messages = [
            Message.system(DIFFERENTIAL_FEEDBACK_SYSTEM_PROMPT),
            Message.user(prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    async def generate_rubric_feedback_async(
        self,
        trajectory: "KareninaTrajectory",
        rubric_scores: dict[str, Any],
    ) -> str:
        """Async version of generate_rubric_feedback.

        Args:
            trajectory: The trajectory with rubric evaluation.
            rubric_scores: Per-trait rubric scores.

        Returns:
            LLM-generated feedback explaining rubric failures.
        """
        prompt = self._build_rubric_feedback_prompt(trajectory, rubric_scores)

        messages = [
            Message.system(RUBRIC_FEEDBACK_SYSTEM_PROMPT),
            Message.user(prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    async def generate_complete_feedback_async(
        self,
        failed_trajectory: "KareninaTrajectory",
        successful_trajectories: list["KareninaTrajectory"] | None,
        rubric_scores: dict[str, Any] | None,
    ) -> str:
        """Async version of generate_complete_feedback with parallel LLM calls.

        This method runs template and rubric feedback generation in parallel
        when both are needed, providing significant speedup.

        Args:
            failed_trajectory: The trajectory that failed verification.
            successful_trajectories: Optional list of trajectories that passed.
            rubric_scores: Optional per-trait rubric scores.

        Returns:
            Combined feedback string with both template and rubric analysis.
        """
        # Build tasks for parallel execution
        tasks: list[asyncio.Task[str]] = []

        # Template verification feedback task
        if successful_trajectories:
            template_task = asyncio.create_task(
                self.generate_differential_feedback_async(failed_trajectory, successful_trajectories)
            )
        else:
            template_task = asyncio.create_task(self.generate_single_feedback_async(failed_trajectory))
        tasks.append(template_task)

        # Rubric feedback task (if rubrics present)
        rubric_task: asyncio.Task[str] | None = None
        if rubric_scores:
            rubric_task = asyncio.create_task(self.generate_rubric_feedback_async(failed_trajectory, rubric_scores))
            tasks.append(rubric_task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Build result
        parts: list[str] = []
        parts.append("--- TEMPLATE VERIFICATION FEEDBACK ---")
        parts.append(template_task.result())

        if rubric_task:
            parts.append("\n--- RUBRIC EVALUATION FEEDBACK ---")
            parts.append(rubric_task.result())

        return "\n".join(parts)

    def _build_single_feedback_prompt(self, trajectory: "KareninaTrajectory") -> str:
        """Build the prompt for single trajectory analysis."""
        parts = [
            "## Question",
            trajectory.data_inst.question_text,
            "",
            "## Expected Answer",
            trajectory.data_inst.raw_answer,
            "",
            "## Model Response",
            f"Model: {trajectory.model_name}",
            "",
            trajectory.raw_llm_response or "(no response)",
            "",
            "## Verification Result",
        ]

        if trajectory.parsing_error:
            parts.append(f"Parsing Error: {trajectory.parsing_error}")

        if trajectory.failed_fields:
            parts.append(f"Failed Fields: {', '.join(trajectory.failed_fields)}")

        if not trajectory.parsing_error and not trajectory.failed_fields:
            parts.append("Verification failed (no specific error details available)")

        parts.extend(
            [
                "",
                "## Task",
                "Analyze why this response failed verification and suggest prompt improvements.",
            ]
        )

        return "\n".join(parts)

    def _build_differential_feedback_prompt(
        self,
        failed_trajectory: "KareninaTrajectory",
        successful_trajectories: list["KareninaTrajectory"],
    ) -> str:
        """Build the prompt for differential analysis."""
        parts = [
            "## Question",
            failed_trajectory.data_inst.question_text,
            "",
            "## Expected Answer",
            failed_trajectory.data_inst.raw_answer,
            "",
            "## SUCCESSFUL TRACES (models that passed)",
        ]

        # Include full traces from successful models
        for i, success in enumerate(successful_trajectories, 1):
            parts.extend(
                [
                    f"### Successful Trace {i}: {success.model_name}",
                    "",
                    success.raw_llm_response or "(no response)",
                    "",
                ]
            )

            # Include parsed fields if available
            if success.verification_result.template and success.verification_result.template.parsed_llm_response:
                parts.extend(
                    [
                        "Parsed Fields:",
                        str(success.verification_result.template.parsed_llm_response),
                        "",
                    ]
                )

        parts.extend(
            [
                "## FAILED TRACE (model that failed)",
                f"Model: {failed_trajectory.model_name}",
                "",
                failed_trajectory.raw_llm_response or "(no response)",
                "",
            ]
        )

        if failed_trajectory.parsing_error:
            parts.append(f"Parsing Error: {failed_trajectory.parsing_error}")

        if failed_trajectory.failed_fields:
            parts.append(f"Failed Fields: {', '.join(failed_trajectory.failed_fields)}")

        parts.extend(
            [
                "",
                "## Task",
                "Compare the successful and failed traces. Identify what the successful models",
                "did differently and suggest specific prompt improvements for the failing model.",
            ]
        )

        return "\n".join(parts)

    def _build_rubric_feedback_prompt(
        self,
        trajectory: "KareninaTrajectory",
        rubric_scores: dict[str, Any],
    ) -> str:
        """Build the prompt for rubric evaluation analysis."""
        parts = [
            "## Question",
            trajectory.data_inst.question_text,
            "",
            "## Model Response",
            trajectory.raw_llm_response or "(no response)",
            "",
            "## Rubric Evaluation Results",
        ]

        # Format each trait result
        for trait_name, score in rubric_scores.items():
            if isinstance(score, bool):
                result_str = "PASSED" if score else "FAILED"
            elif isinstance(score, int | float):
                result_str = f"{score:.2f}" if isinstance(score, float) else str(score)
            elif isinstance(score, dict):
                # Metric trait with precision/recall/f1
                result_str = ", ".join(f"{k}: {v:.2f}" for k, v in score.items() if isinstance(v, int | float))
            else:
                result_str = str(score)

            parts.append(f"- {trait_name}: {result_str}")

        # Identify failed traits for focus
        failed_traits = [
            name
            for name, score in rubric_scores.items()
            if (isinstance(score, bool) and not score)
            or (isinstance(score, int | float) and score < 0.5)
            or (isinstance(score, dict) and score.get("f1", 1.0) < 0.5)
        ]

        if failed_traits:
            parts.extend(
                [
                    "",
                    f"## Failed/Low-Scoring Traits: {', '.join(failed_traits)}",
                ]
            )

        parts.extend(
            [
                "",
                "## Task",
                "Analyze why the failed rubric traits didn't meet criteria.",
                "Suggest specific improvements to the response.",
            ]
        )

        return "\n".join(parts)
