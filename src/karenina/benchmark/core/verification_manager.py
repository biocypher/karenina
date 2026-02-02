"""Verification management functionality for benchmarks."""

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BenchmarkBase
    from .rubrics import RubricManager

from ...schemas.workflow import (
    FinishedTemplate,
    VerificationConfig,
    VerificationResult,
    VerificationResultSet,
)
from ..verification import run_verification_batch
from ..verification.utils.template_validation import validate_answer_template


class VerificationManager:
    """Manager for verification workflows and orchestration."""

    def __init__(self, base: "BenchmarkBase", rubric_manager: "RubricManager") -> None:
        """Initialize with reference to benchmark base and rubric manager."""
        self.base = base
        self.rubric_manager = rubric_manager

    def run_verification(
        self,
        config: VerificationConfig,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        async_enabled: bool | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> VerificationResultSet:
        """
        Run verification on the benchmark using existing execution system.

        Args:
            config: Verification configuration
            question_ids: Optional list of question IDs to verify (default: all finished)
            run_name: Optional run name for tracking
            async_enabled: Optional async control (overrides KARENINA_ASYNC_ENABLED env var if provided)
            progress_callback: Optional callback for progress updates

        Returns:
            VerificationResultSet containing all verification results
        """
        # If no question IDs provided, verify all finished questions
        if question_ids is None:
            question_ids = [q_id for q_id, q in self.base._questions_cache.items() if q.get("finished", False)]

        # Validate that all requested questions exist and are ready
        for q_id in question_ids:
            if q_id not in self.base._questions_cache:
                raise ValueError(f"Question not found: {q_id}")

            q_data = self.base._questions_cache[q_id]
            template_code = q_data.get("answer_template")

            if not template_code:
                raise ValueError(f"Question {q_id} has no answer template")

            # Validate template
            is_valid, error_msg, _ = validate_answer_template(template_code)
            if not is_valid:
                raise ValueError(f"Invalid template for question {q_id}: {error_msg}")

        # Build FinishedTemplate objects from benchmark questions
        templates = []
        for q_id in question_ids:
            q_data = self.base._questions_cache[q_id]

            # Convert question_rubric from list of traits to dict if needed
            question_rubric_dict = None
            if q_data.get("question_rubric"):
                question_rubric_raw = q_data["question_rubric"]
                # If it's already a dict, use it as-is
                if isinstance(question_rubric_raw, dict):
                    question_rubric_dict = question_rubric_raw
                # If it's a list of trait objects, convert to Rubric and dump
                elif isinstance(question_rubric_raw, list):
                    from ...schemas.domain.rubric import (
                        CallableTrait,
                        LLMRubricTrait,
                        MetricRubricTrait,
                        RegexTrait,
                        Rubric,
                    )

                    # Separate traits by type
                    llm_traits = [t for t in question_rubric_raw if isinstance(t, LLMRubricTrait)]
                    regex_traits = [t for t in question_rubric_raw if isinstance(t, RegexTrait)]
                    callable_traits = [t for t in question_rubric_raw if isinstance(t, CallableTrait)]
                    metric_traits = [t for t in question_rubric_raw if isinstance(t, MetricRubricTrait)]

                    # Create Rubric and convert to dict
                    rubric = Rubric(
                        llm_traits=llm_traits,
                        regex_traits=regex_traits,
                        callable_traits=callable_traits,
                        metric_traits=metric_traits,
                    )
                    question_rubric_dict = rubric.model_dump()

            template = FinishedTemplate(
                question_id=q_id,
                question_text=q_data["question"],
                question_preview=q_data["question"][:100],
                template_code=q_data["answer_template"],
                last_modified=q_data.get("dateModified", datetime.now().isoformat()),
                few_shot_examples=q_data.get("few_shot_examples"),
                question_rubric=question_rubric_dict,
                keywords=q_data.get("keywords"),
            )
            templates.append(template)

        # Get global rubric
        global_rubric = self.rubric_manager.get_global_rubric()

        # Determine storage_url from config if db_config is present
        storage_url = None
        if config.db_config is not None:
            storage_url = config.db_config.storage_url

        # Create progress callback adapter if needed
        # Benchmark expects: Callable[[float, str], None] (percentage, message)
        # batch_runner expects: Callable[[int, int, VerificationResult | None], None] (current, total, result)
        batch_progress_callback = None
        if progress_callback:

            def adapter(current: int, total: int, result: VerificationResult | None) -> None:
                # Convert to percentage and create message
                # This is called BEFORE starting each task to show current item being processed
                percentage = ((current - 1) / total) * 100 if total > 0 else 0
                if result:
                    message = f"Verifying {result.question_id} ({current}/{total})"
                    progress_callback(percentage, message)

            batch_progress_callback = adapter

        # Call batch runner with all templates
        results = run_verification_batch(
            templates=templates,
            config=config,
            run_name=run_name,
            global_rubric=global_rubric,
            async_enabled=async_enabled,  # Can override env var
            storage_url=storage_url,
            benchmark_name=self.base.name,
            progress_callback=batch_progress_callback,
        )

        return results
