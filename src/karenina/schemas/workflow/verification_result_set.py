"""
VerificationResultSet class - main container for verification results.

This module provides the top-level container returned by run_verification,
with accessor methods for specialized result views (rubrics, templates, judgments).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .judgment_results import JudgmentResults
from .rubric_judgment_results import RubricJudgmentResults
from .rubric_results import RubricResults
from .template_results import TemplateResults

if TYPE_CHECKING:
    from .verification import VerificationResult


class VerificationResultSet(BaseModel):
    """
    Container for all verification results from a run_verification call.

    This is the main class returned by Benchmark.run_verification(). It stores
    all individual VerificationResult objects and provides three specialized
    accessor methods for working with different evaluation types:
    - get_rubrics_results(): Access rubric evaluation data
    - get_template_results(): Access template verification data
    - get_judgment_results(): Access deep judgment data

    The class also provides filtering, grouping, and iteration capabilities.

    Attributes:
        results: List of all VerificationResult objects from the verification run
    """

    results: list[VerificationResult] = Field(description="List of all verification results from the run")

    model_config = {"arbitrary_types_allowed": True}

    # ========================================================================
    # Accessor Methods (Main Interface)
    # ========================================================================

    def get_rubrics_results(self, include_deep_judgment: bool = False) -> RubricResults:
        """
        Get rubric evaluation results.

        Returns a RubricResults object that provides access to:
        - LLM trait scores (1-5 scale)
        - Manual trait scores (boolean)
        - Metric trait scores (precision, recall, f1, etc.)
        - Confusion matrices for metric traits
        - Aggregation and grouping capabilities
        - Deep judgment columns (optional, when include_deep_judgment=True)

        Args:
            include_deep_judgment: Whether to include deep judgment columns in DataFrame exports.
                When True, LLM traits will have three additional columns:
                - trait_reasoning: Reasoning text for the trait score
                - trait_excerpts: JSON-serialized list of excerpts
                - trait_hallucination_risk: Hallucination risk assessment
                Default: False (backward compatible)

        Returns:
            RubricResults instance wrapping all verification results

        Example:
            ```python
            result_set = benchmark.run_verification(config)

            # Standard rubric results (backward compatible)
            rubric_results = result_set.get_rubrics_results()
            df = rubric_results.to_dataframe()

            # With deep judgment columns
            rubric_results = result_set.get_rubrics_results(include_deep_judgment=True)
            df = rubric_results.to_dataframe()
            # df now has trait_reasoning, trait_excerpts, trait_hallucination_risk columns

            # Aggregate by question
            aggregated = rubric_results.aggregate_llm_traits(
                strategy="mean",
                by="question"
            )

            # Filter and group
            filtered = rubric_results.filter(question_ids=["q1", "q2"])
            by_model = filtered.group_by_model()
            ```
        """
        return RubricResults(results=self.results, include_deep_judgment=include_deep_judgment)

    def get_template_results(self) -> TemplateResults:
        """
        Get template verification results.

        Returns a TemplateResults object that provides access to:
        - Template verification pass/fail status
        - Embedding similarity scores
        - Regex validation results
        - Abstention detection
        - MCP tool usage metrics
        - Parsed responses (LLM vs ground truth)

        Returns:
            TemplateResults instance wrapping all verification results

        Example:
            ```python
            result_set = benchmark.run_verification(config)
            template_results = result_set.get_template_results()

            # Get pass rate by question
            pass_rates = template_results.aggregate_pass_rate(by="question")

            # Get embedding scores
            embedding_scores = template_results.get_embedding_scores()

            # Aggregate embedding similarity
            avg_similarity = template_results.aggregate_embedding_scores(
                strategy="mean",
                by="model"
            )

            # Filter to only passed results
            passed = template_results.filter(passed_only=True)
            ```
        """
        return TemplateResults(results=self.results)

    def get_judgment_results(self) -> JudgmentResults:
        """
        Get deep judgment evaluation results.

        Returns a JudgmentResults object that provides access to:
        - Extracted excerpts per attribute
        - Reasoning traces for each attribute
        - Hallucination risk assessments
        - Search validation results
        - Processing metrics (model calls, retries, etc.)

        Returns:
            JudgmentResults instance wrapping all verification results

        Example:
            ```python
            result_set = benchmark.run_verification(config)
            judgment_results = result_set.get_judgment_results()

            # Get extracted excerpts
            excerpts = judgment_results.get_extracted_excerpts(
                attribute_name="location"
            )

            # Get hallucination risk distribution
            risk_dist = judgment_results.aggregate_hallucination_risk_distribution(
                by="question"
            )

            # Get processing metrics
            metrics = judgment_results.get_processing_metrics()

            # Filter to results with search enabled
            with_search = judgment_results.filter(with_search_only=True)
            ```
        """
        return JudgmentResults(results=self.results)

    def get_rubric_judgments_results(self) -> RubricJudgmentResults:
        """
        Get detailed deep judgment rubric results with excerpt-level explosion.

        Returns a RubricJudgmentResults object that provides fine-grained access to:
        - Per-excerpt data (text, confidence, similarity scores)
        - Per-trait metadata (model calls, retry counts, stages completed)
        - Reasoning for each trait evaluation
        - Hallucination risk assessments per excerpt
        - Search validation results (if search enabled)

        Unlike get_rubrics_results() which returns one row per trait, this method
        explodes each trait into multiple rows (one per excerpt) for detailed analysis.

        Key Differences:
            - get_rubrics_results(): Standard export, one row per trait
            - get_rubric_judgments_results(): Detailed export, one row per (trait × excerpt)

        Returns:
            RubricJudgmentResults instance with excerpt-exploded data

        Example:
            ```python
            result_set = benchmark.run_verification(config)
            rubric_judgments = result_set.get_rubric_judgments_results()

            # Get detailed DataFrame with excerpt explosion
            df = rubric_judgments.to_dataframe()

            # Analyze per-excerpt confidence
            avg_confidence = df.groupby("trait_name")["excerpt_confidence"].mean()

            # Examine traits with validation failures
            failed = df[df["trait_validation_failed"] == True]
            print(failed[["trait_name", "trait_excerpt_retries", "excerpt_text"]])

            # Study hallucination risks
            risky = df[df["excerpt_hallucination_risk"].notna()]
            print(risky[["excerpt_text", "excerpt_hallucination_risk", "excerpt_hallucination_justification"]])

            # Get summary statistics
            excerpt_summary = rubric_judgments.get_excerpt_count_summary()
            retry_summary = rubric_judgments.get_retry_summary()
            ```
        """
        return RubricJudgmentResults(results=self.results)

    # ========================================================================
    # Filtering
    # ========================================================================

    def filter(
        self,
        question_ids: list[str] | None = None,
        answering_models: list[str] | None = None,
        parsing_models: list[str] | None = None,
        replicates: list[int] | None = None,
        completed_only: bool = False,
        has_template: bool = False,
        has_rubric: bool = False,
        has_judgment: bool = False,
    ) -> VerificationResultSet:
        """
        Filter results by various criteria.

        Args:
            question_ids: Filter by question IDs
            answering_models: Filter by answering model names
            parsing_models: Filter by parsing model names
            replicates: Filter by replicate numbers
            completed_only: Only include results that completed without errors
            has_template: Only include results with template verification data
            has_rubric: Only include results with rubric evaluation data
            has_judgment: Only include results with deep judgment data

        Returns:
            New VerificationResultSet with filtered results

        Example:
            ```python
            # Filter to specific questions
            filtered = result_set.filter(question_ids=["q1", "q2"])

            # Filter to successful results with rubric data
            filtered = result_set.filter(
                completed_only=True,
                has_rubric=True
            )

            # Filter to specific model and replicate
            filtered = result_set.filter(
                answering_models=["gpt-4"],
                replicates=[1]
            )
            ```
        """
        filtered = self.results

        if question_ids:
            filtered = [r for r in filtered if r.metadata.question_id in question_ids]

        if answering_models:
            filtered = [r for r in filtered if r.metadata.answering_model in answering_models]

        if parsing_models:
            filtered = [r for r in filtered if r.metadata.parsing_model in parsing_models]

        if replicates:
            filtered = [
                r
                for r in filtered
                if r.metadata.answering_replicate in replicates or r.metadata.parsing_replicate in replicates
            ]

        if completed_only:
            filtered = [r for r in filtered if r.metadata.completed_without_errors]

        if has_template:
            filtered = [r for r in filtered if r.template is not None and r.template.template_verification_performed]

        if has_rubric:
            filtered = [r for r in filtered if r.rubric is not None and r.rubric.rubric_evaluation_performed]

        if has_judgment:
            filtered = [r for r in filtered if r.deep_judgment is not None and r.deep_judgment.deep_judgment_performed]

        return VerificationResultSet(results=filtered)

    # ========================================================================
    # Grouping
    # ========================================================================

    def group_by_question(self) -> dict[str, VerificationResultSet]:
        """
        Group results by question ID.

        Returns:
            Dictionary mapping question IDs to VerificationResultSet instances

        Example:
            ```python
            by_question = result_set.group_by_question()
            for question_id, question_results in by_question.items():
                print(f"Question {question_id}: {len(question_results)} results")

                # Get rubric results for this question
                rubric_results = question_results.get_rubrics_results()
            ```
        """
        grouped: dict[str, list[VerificationResult]] = {}
        for result in self.results:
            qid = result.metadata.question_id
            if qid not in grouped:
                grouped[qid] = []
            grouped[qid].append(result)

        return {qid: VerificationResultSet(results=results) for qid, results in grouped.items()}

    def group_by_model(self) -> dict[str, VerificationResultSet]:
        """
        Group results by answering model.

        Returns:
            Dictionary mapping model names to VerificationResultSet instances

        Example:
            ```python
            by_model = result_set.group_by_model()
            for model_name, model_results in by_model.items():
                print(f"Model {model_name}: {len(model_results)} results")

                # Get template results for this model
                template_results = model_results.get_template_results()
                pass_rate = template_results.aggregate_pass_rate(by="question")
            ```
        """
        grouped: dict[str, list[VerificationResult]] = {}
        for result in self.results:
            model = result.metadata.answering_model
            if model not in grouped:
                grouped[model] = []
            grouped[model].append(result)

        return {model: VerificationResultSet(results=results) for model, results in grouped.items()}

    def group_by_replicate(self) -> dict[int, VerificationResultSet]:
        """
        Group results by replicate number.

        Returns:
            Dictionary mapping replicate numbers to VerificationResultSet instances

        Example:
            ```python
            by_replicate = result_set.group_by_replicate()
            for rep_num, rep_results in by_replicate.items():
                print(f"Replicate {rep_num}: {len(rep_results)} results")
            ```
        """
        grouped: dict[int, list[VerificationResult]] = {}
        for result in self.results:
            rep = result.metadata.answering_replicate or 0
            if rep not in grouped:
                grouped[rep] = []
            grouped[rep].append(result)

        return {rep: VerificationResultSet(results=results) for rep, results in grouped.items()}

    # ========================================================================
    # Summary and Statistics
    # ========================================================================

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics for the entire result set.

        Returns:
            Dictionary with summary statistics:
            - num_results: Total number of results
            - num_completed: Number of results that completed without errors
            - num_with_template: Number with template verification
            - num_with_rubric: Number with rubric evaluation
            - num_with_judgment: Number with deep judgment
            - num_questions: Number of unique questions
            - num_models: Number of unique answering models
            - num_replicates: Number of unique replicates

        Example:
            ```python
            summary = result_set.get_summary()
            print(f"Total results: {summary['num_results']}")
            print(f"Questions: {summary['num_questions']}")
            print(f"Models: {summary['num_models']}")
            ```
        """
        num_completed = 0
        num_with_template = 0
        num_with_rubric = 0
        num_with_judgment = 0
        questions = set()
        models = set()
        replicates = set()

        for result in self.results:
            if result.metadata.completed_without_errors:
                num_completed += 1

            if result.template and result.template.template_verification_performed:
                num_with_template += 1

            if result.rubric and result.rubric.rubric_evaluation_performed:
                num_with_rubric += 1

            if result.deep_judgment and result.deep_judgment.deep_judgment_performed:
                num_with_judgment += 1

            questions.add(result.metadata.question_id)
            models.add(result.metadata.answering_model)
            if result.metadata.answering_replicate is not None:
                replicates.add(result.metadata.answering_replicate)

        return {
            "num_results": len(self.results),
            "num_completed": num_completed,
            "num_with_template": num_with_template,
            "num_with_rubric": num_with_rubric,
            "num_with_judgment": num_with_judgment,
            "num_questions": len(questions),
            "num_models": len(models),
            "num_replicates": len(replicates),
        }

    def get_question_ids(self) -> list[str]:
        """
        Get list of all unique question IDs in the result set.

        Returns:
            Sorted list of question IDs

        Example:
            ```python
            question_ids = result_set.get_question_ids()
            print(f"Questions: {', '.join(question_ids)}")
            ```
        """
        return sorted({r.metadata.question_id for r in self.results})

    def get_model_names(self) -> list[str]:
        """
        Get list of all unique answering model names in the result set.

        Returns:
            Sorted list of model names

        Example:
            ```python
            models = result_set.get_model_names()
            print(f"Models: {', '.join(models)}")
            ```
        """
        return sorted({r.metadata.answering_model for r in self.results})

    # ========================================================================
    # Special Methods
    # ========================================================================

    def __len__(self) -> int:
        """
        Return number of results in the set.

        Example:
            ```python
            print(f"Result set contains {len(result_set)} results")
            ```
        """
        return len(self.results)

    def __iter__(self) -> Any:
        """
        Iterate over individual VerificationResult objects.

        Example:
            ```python
            for result in result_set:
                print(f"Question: {result.metadata.question_id}")
                if result.template:
                    print(f"  Passed: {result.template.verify_result}")
            ```
        """
        return iter(self.results)

    def __getitem__(self, index: int) -> VerificationResult:
        """
        Access individual result by index.

        Example:
            ```python
            first_result = result_set[0]
            last_result = result_set[-1]
            ```
        """
        return self.results[index]

    def __repr__(self) -> str:
        """String representation of the result set."""
        summary = self.get_summary()
        return (
            f"VerificationResultSet("
            f"results={summary['num_results']}, "
            f"questions={summary['num_questions']}, "
            f"models={summary['num_models']})"
        )

    # ========================================================================
    # Legacy Compatibility
    # ========================================================================

    def to_legacy_dict(self) -> dict[str, VerificationResult]:
        """
        Convert to legacy dict format for backward compatibility.

        This method recreates the dictionary format used by the old run_verification
        API where keys were in the format:
        {question_id}_{answering_model}_{parsing_model}_rep{N}_{timestamp}

        Returns:
            Dictionary mapping result keys to VerificationResult objects

        Example:
            ```python
            result_set = benchmark.run_verification(config)

            # Convert to old dict format for legacy code
            legacy_dict = result_set.to_legacy_dict()

            # Keys like: "q1_gpt-4_gpt-4_rep1_1699123456789"
            for result_key, result in legacy_dict.items():
                print(f"{result_key}: {result.metadata.question_id}")
            ```
        """
        import time

        result_dict = {}
        for result in self.results:
            # Generate key matching the old format:
            # {question_id}_{answering}_{parsing}_rep{N}_{timestamp}
            key_parts = [
                result.metadata.question_id,
                result.metadata.answering_model,
                result.metadata.parsing_model,
            ]

            # Add replicate if present
            if result.metadata.answering_replicate is not None:
                key_parts.append(f"rep{result.metadata.answering_replicate}")

            # Use the result's timestamp if available, otherwise current time
            if result.metadata.timestamp:
                # Try to parse ISO timestamp to milliseconds
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(result.metadata.timestamp.replace("Z", "+00:00"))
                    timestamp_ms = str(int(dt.timestamp() * 1000))
                except (ValueError, AttributeError):
                    # Fallback to current time
                    timestamp_ms = str(int(time.time() * 1000))
            else:
                timestamp_ms = str(int(time.time() * 1000))

            key_parts.append(timestamp_ms)

            result_key = "_".join(key_parts)
            result_dict[result_key] = result

        return result_dict
