"""Verification management functionality for benchmarks."""

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BenchmarkBase
    from .results import ResultsManager
    from .rubrics import RubricManager

from ..models import VerificationConfig, VerificationResult
from ..verification.orchestrator import run_question_verification
from ..verification.validation import validate_answer_template


class VerificationManager:
    """Manager for verification workflows and orchestration."""

    def __init__(self, base: "BenchmarkBase", rubric_manager: "RubricManager") -> None:
        """Initialize with reference to benchmark base and rubric manager."""
        self.base = base
        self.rubric_manager = rubric_manager
        # We'll need the results manager for auto-storage
        self._results_manager: ResultsManager | None = None

    def verify_question(
        self,
        question_id: str,
        config: VerificationConfig,
        run_name: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Verify a single question.

        Args:
            question_id: The question ID to verify
            config: Verification configuration
            run_name: Optional run name for tracking
            job_id: Optional job ID for tracking

        Returns:
            Dictionary mapping result keys to VerificationResult objects

        Raises:
            ValueError: If question not found or not ready for verification
        """
        return self.run_verification(
            config=config,
            question_ids=[question_id],
            run_name=run_name,
            job_id=job_id,
        )

    def verify_questions(
        self,
        question_ids: list[str],
        config: VerificationConfig,
        run_name: str | None = None,
        job_id: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Verify multiple specific questions.

        Args:
            question_ids: List of question IDs to verify
            config: Verification configuration
            run_name: Optional run name for tracking
            job_id: Optional job ID for tracking
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping result keys to VerificationResult objects
        """
        return self.run_verification(
            config=config,
            question_ids=question_ids,
            run_name=run_name,
            job_id=job_id,
            progress_callback=progress_callback,
        )

    def verify_filtered(
        self,
        config: VerificationConfig,
        finished: bool | None = True,
        has_template: bool | None = True,
        has_rubric: bool | None = None,
        author: str | None = None,
        run_name: str | None = None,
        job_id: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Verify questions matching specific criteria.

        Args:
            config: Verification configuration
            finished: Filter by finished status
            has_template: Filter by template existence
            has_rubric: Filter by rubric existence
            author: Filter by author name
            run_name: Optional run name for tracking
            job_id: Optional job ID for tracking
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping result keys to VerificationResult objects
        """
        # Use question manager to get filtered questions (assume it's available on base)
        from .questions import QuestionManager

        question_manager = QuestionManager(self.base)

        filtered_questions = question_manager.filter_questions(
            finished=finished,
            has_template=has_template,
            has_rubric=has_rubric,
            author=author,
        )

        question_ids = [q["id"] for q in filtered_questions]

        return self.run_verification(
            config=config,
            question_ids=question_ids,
            run_name=run_name,
            job_id=job_id,
            progress_callback=progress_callback,
        )

    def verify_all_finished(
        self,
        config: VerificationConfig,
        run_name: str | None = None,
        job_id: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Verify all finished questions in the benchmark.

        Args:
            config: Verification configuration
            run_name: Optional run name for tracking
            job_id: Optional job ID for tracking
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping result keys to VerificationResult objects
        """
        return self.run_verification(
            config=config,
            question_ids=None,  # This defaults to all finished questions
            run_name=run_name,
            job_id=job_id,
            progress_callback=progress_callback,
        )

    def verify_custom(
        self,
        question_selector: Callable[[dict[str, Any]], bool],
        config: VerificationConfig,
        run_name: str | None = None,
        job_id: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Verify questions selected by a custom function.

        Args:
            question_selector: Function that takes question data and returns bool
            config: Verification configuration
            run_name: Optional run name for tracking
            job_id: Optional job ID for tracking
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping result keys to VerificationResult objects
        """
        # Select questions using the custom selector
        selected_questions = []
        for q_data in self.base._questions_cache.values():
            if question_selector(q_data):
                selected_questions.append(q_data["id"])

        return self.run_verification(
            config=config,
            question_ids=selected_questions,
            run_name=run_name,
            job_id=job_id,
            progress_callback=progress_callback,
        )

    def verify_dry_run(
        self,
        config: VerificationConfig,  # noqa: ARG002
        question_ids: list[str] | None = None,
    ) -> dict[str, bool]:
        """
        Perform a dry run verification (validate without executing).

        Args:
            config: Verification configuration to validate
            question_ids: Optional list of question IDs (default: all finished)

        Returns:
            Dictionary mapping question IDs to readiness status (True/False)
        """
        # If no question IDs provided, use all finished questions
        if question_ids is None:
            question_ids = [q_id for q_id, q in self.base._questions_cache.items() if q.get("finished", False)]

        results = {}

        for q_id in question_ids:
            try:
                # Check if question exists
                if q_id not in self.base._questions_cache:
                    results[q_id] = False
                    continue

                q_data = self.base._questions_cache[q_id]
                template_code = q_data.get("answer_template")

                # Check if template exists
                if not template_code:
                    results[q_id] = False
                    continue

                # Validate template
                is_valid, _, _ = validate_answer_template(template_code)
                results[q_id] = is_valid

            except Exception:
                results[q_id] = False

        return results

    def run_verification(
        self,
        config: VerificationConfig,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        job_id: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Run verification on the benchmark using existing execution system.

        Args:
            config: Verification configuration
            question_ids: Optional list of question IDs to verify (default: all finished)
            run_name: Optional run name for tracking
            job_id: Optional job ID for tracking
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping question IDs to VerificationResult objects
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

        results: dict[str, VerificationResult] = {}
        total_questions = len(question_ids)

        for i, q_id in enumerate(question_ids):
            q_data = self.base._questions_cache[q_id]
            template_code = q_data["answer_template"]

            # Update progress
            if progress_callback:
                progress = (i / total_questions) * 100
                progress_callback(progress, f"Verifying question {q_id}")

            # Merge global and question-specific rubrics
            rubric = self.rubric_manager.get_merged_rubric_for_question(q_id)

            try:
                # Run verification for this question using the orchestrator
                question_results = run_question_verification(
                    question_id=q_id,
                    question_text=q_data["question"],
                    template_code=template_code,
                    config=config,
                    rubric=rubric,
                )

                # Store all results from this question (may be multiple due to model combinations)
                results.update(question_results)

            except Exception as e:
                # Create error result for this question
                error_result = VerificationResult(
                    question_id=q_id,
                    success=False,
                    error=f"Verification failed: {str(e)}",
                    question_text=q_data["question"],
                    raw_llm_response="",
                    answering_model="unknown",
                    parsing_model="unknown",
                    execution_time=0.0,
                    timestamp=datetime.now().isoformat(),
                    run_name=run_name,
                    job_id=job_id,
                )
                results[q_id] = error_result

        # Final progress update
        if progress_callback:
            progress_callback(100.0, "Verification complete")

        # Note: Results are not auto-stored in checkpoint anymore
        # They remain in memory only and must be explicitly exported if needed

        return results

    def verify_with_mixed_configs(
        self,
        question_configs: dict[str, VerificationConfig],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, dict[str, VerificationResult]]:
        """
        Verify different questions with different configurations.

        Args:
            question_configs: Dictionary mapping question IDs to their configurations
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping question IDs to their verification results
        """
        all_results = {}
        total_questions = len(question_configs)

        for i, (question_id, config) in enumerate(question_configs.items()):
            # Update progress
            if progress_callback:
                progress = (i / total_questions) * 100
                progress_callback(progress, f"Verifying question {question_id} with custom config")

            try:
                question_results = self.verify_question(
                    question_id=question_id,
                    config=config,
                    run_name=f"mixed_config_{question_id}",
                )
                all_results[question_id] = question_results
            except Exception as e:
                # Create error result
                error_result = VerificationResult(
                    question_id=question_id,
                    success=False,
                    error=f"Mixed config verification failed: {str(e)}",
                    question_text=self.base._questions_cache.get(question_id, {}).get("question", ""),
                    raw_llm_response="",
                    answering_model="unknown",
                    parsing_model="unknown",
                    execution_time=0.0,
                    timestamp=datetime.now().isoformat(),
                )
                all_results[question_id] = {f"{question_id}_error": error_result}

        # Final progress update
        if progress_callback:
            progress_callback(100.0, "Mixed config verification complete")

        return all_results

    def verify_comparative(
        self,
        question_ids: list[str],
        configs: list[VerificationConfig],
        run_names: list[str],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, dict[str, VerificationResult]]:
        """
        Run same questions with multiple configurations for comparison.

        Args:
            question_ids: List of question IDs to verify
            configs: List of configurations to test
            run_names: List of run names for each configuration
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping run names to their results
        """
        if len(configs) != len(run_names):
            raise ValueError("Number of configs must match number of run names")

        all_results = {}
        total_runs = len(configs)

        for i, (config, run_name) in enumerate(zip(configs, run_names, strict=False)):
            # Update progress
            if progress_callback:
                progress = (i / total_runs) * 100
                progress_callback(progress, f"Running comparative verification: {run_name}")

            try:
                run_results = self.verify_questions(
                    question_ids=question_ids,
                    config=config,
                    run_name=run_name,
                )
                all_results[run_name] = run_results
            except Exception as e:
                # Create error results for this run
                error_results = {}
                for q_id in question_ids:
                    error_result = VerificationResult(
                        question_id=q_id,
                        success=False,
                        error=f"Comparative verification failed: {str(e)}",
                        question_text=self.base._questions_cache.get(q_id, {}).get("question", ""),
                        raw_llm_response="",
                        answering_model="unknown",
                        parsing_model="unknown",
                        execution_time=0.0,
                        timestamp=datetime.now().isoformat(),
                        run_name=run_name,
                    )
                    error_results[f"{q_id}_error"] = error_result
                all_results[run_name] = error_results

        # Final progress update
        if progress_callback:
            progress_callback(100.0, "Comparative verification complete")

        return all_results

    def verify_progressive(
        self,
        config: VerificationConfig,
        batch_size: int = 5,
        run_name: str | None = None,
        resume_from: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Verify questions in batches with ability to resume from interruptions.

        Args:
            config: Verification configuration
            batch_size: Number of questions to verify per batch
            run_name: Optional run name for tracking
            resume_from: Optional question ID to resume from
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping result keys to VerificationResult objects
        """
        # Get all finished questions
        finished_questions = [q_id for q_id, q in self.base._questions_cache.items() if q.get("finished", False)]

        # If resuming, find the starting point
        start_index = 0
        if resume_from and resume_from in finished_questions:
            start_index = finished_questions.index(resume_from)

        # Process in batches
        all_results = {}
        remaining_questions = finished_questions[start_index:]
        total_batches = (len(remaining_questions) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(remaining_questions))
            batch_questions = remaining_questions[start_idx:end_idx]

            # Update progress
            if progress_callback:
                batch_progress = (batch_num / total_batches) * 100
                progress_callback(batch_progress, f"Processing batch {batch_num + 1}/{total_batches}")

            try:
                batch_results = self.verify_questions(
                    question_ids=batch_questions,
                    config=config,
                    run_name=f"{run_name}_batch_{batch_num + 1}" if run_name else None,
                )
                all_results.update(batch_results)

            except Exception as e:
                # Log the error but continue with remaining batches
                if progress_callback:
                    progress_callback(
                        (batch_num / total_batches) * 100, f"Batch {batch_num + 1} failed: {str(e)[:50]}..."
                    )
                continue

        # Final progress update
        if progress_callback:
            progress_callback(100.0, "Progressive verification complete")

        return all_results
