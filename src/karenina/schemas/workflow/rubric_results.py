"""
RubricResults class for collating and analyzing rubric evaluation results.

This module provides functionality to work with rubric evaluation results across
multiple verification runs, supporting aggregation and analysis of trait scores.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .aggregation import AggregatorRegistry, create_default_registry

if TYPE_CHECKING:
    from .verification import VerificationResult


TraitType = Literal["llm", "regex", "callable", "metric"]


class RubricResults(BaseModel):
    """
    Collated rubric evaluation results from multiple verification runs.

    This class provides a unified interface for accessing, filtering, and
    aggregating rubric trait scores across multiple verification results.
    It handles three trait types (LLM, manual, metric) and supports
    extensible aggregation strategies.

    Attributes:
        results: List of VerificationResult objects containing rubric data
        aggregator_registry: Registry of aggregation strategies
    """

    results: list[VerificationResult] = Field(description="List of verification results containing rubric data")

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, results: list[VerificationResult], include_deep_judgment: bool = False, **data: Any) -> None:
        """
        Initialize RubricResults with verification results.

        Args:
            results: List of VerificationResult objects
            include_deep_judgment: Whether to include deep judgment columns (default: False)
            **data: Additional pydantic model data
        """
        super().__init__(results=results, **data)
        # Store configuration
        self.include_deep_judgment = include_deep_judgment
        # Create default aggregator registry
        self._aggregator_registry: AggregatorRegistry = create_default_registry()

    # ========================================================================
    # DataFrame Conversion
    # ========================================================================

    def to_dataframe(
        self, trait_type: Literal["llm_score", "llm_binary", "llm", "regex", "callable", "metric", "all"] = "all"
    ) -> Any:
        """
        Convert rubric evaluation results to pandas DataFrame.

        Creates one row per trait (for llm/regex/callable) or per metric (for metric traits).
        Supports filtering by trait type or returning all traits combined.

        Args:
            trait_type: Type of traits to include
                - "llm_score": LLM traits with 1-5 scale scores
                - "llm_binary": LLM traits with boolean scores
                - "llm": All LLM traits (both score and binary)
                - "regex": Regex traits (boolean)
                - "callable": Callable traits (boolean or score)
                - "metric": Metric traits (precision, recall, f1) - EXPLODED by metric
                - "all": All trait types combined (default)

        Column ordering:
            1. Status: completed_without_errors, error
            2. Identification: question_id, template_id, question_text, keywords, replicate
            3. Model Config: answering_model, parsing_model, system_prompts
            4. Rubric Data: trait_name, trait_score, trait_type (+ metric_name for metrics)
            5. Rubric Metadata: evaluation_rubric
            6. Execution Metadata: execution_time, timestamp, run_name, job_id
            7. Deep Judgment (if include_deep_judgment=True):
               - trait_reasoning: Reasoning text for the trait score
               - trait_excerpts: JSON-serialized list of excerpts
               - trait_hallucination_risk: Hallucination risk assessment

        Deep Judgment Columns:
            When `include_deep_judgment=True` is set during initialization,
            three additional columns are added for LLM traits:
            - trait_reasoning: String containing the reasoning for the trait score
            - trait_excerpts: JSON string of excerpt objects with text, confidence, similarity
            - trait_hallucination_risk: Optional hallucination risk assessment object

        Returns:
            pandas.DataFrame: Exploded DataFrame with one row per trait/metric

        Example:
            >>> # Standard rubric results (backward compatible)
            >>> rubric_results = result_set.get_rubrics()
            >>> df = rubric_results.to_dataframe(trait_type="llm_score")
            >>> avg_scores = df[df['trait_name'] == 'clarity'].groupby('question_id')['trait_score'].mean()

            >>> # With deep judgment columns
            >>> rubric_results = result_set.get_rubrics(include_deep_judgment=True)
            >>> df = rubric_results.to_dataframe(trait_type="llm")
            >>> # Analyze reasoning and excerpts
            >>> df[['trait_name', 'trait_score', 'trait_reasoning']].head()
        """
        import pandas as pd

        rows = []

        for result in self.results:
            if result.rubric is None or not result.rubric.rubric_evaluation_performed:
                # No rubric data - create single row with minimal info
                rows.append(self._create_empty_rubric_row(result))
                continue

            # Process LLM traits
            if trait_type in ("llm_score", "llm_binary", "llm", "all") and result.rubric.llm_trait_scores:
                for trait_name, trait_score in result.rubric.llm_trait_scores.items():
                    # Determine if score or binary based on value type
                    is_binary = isinstance(trait_score, bool)
                    score_type = "llm_binary" if is_binary else "llm_score"

                    # Filter by requested type
                    if trait_type == "llm_score" and is_binary:
                        continue
                    if trait_type == "llm_binary" and not is_binary:
                        continue

                    rows.append(self._create_llm_trait_row(result, trait_name, trait_score, score_type))

            # Process regex traits
            if trait_type in ("regex", "all") and result.rubric.regex_trait_scores:
                for trait_name, trait_score in result.rubric.regex_trait_scores.items():
                    rows.append(self._create_regex_trait_row(result, trait_name, trait_score))

            # Process callable traits
            if trait_type in ("callable", "all") and result.rubric.callable_trait_scores:
                for trait_name, trait_score in result.rubric.callable_trait_scores.items():
                    rows.append(self._create_callable_trait_row(result, trait_name, trait_score))

            # Process metric traits (EXPLODED by metric)
            if trait_type in ("metric", "all") and result.rubric.metric_trait_scores:
                for trait_name, metrics in result.rubric.metric_trait_scores.items():
                    # Each metric gets its own row
                    for metric_name, metric_score in metrics.items():
                        # Get confusion matrix data for this trait
                        confusion_data = None
                        if result.rubric.metric_trait_confusion_lists:
                            confusion_data = result.rubric.metric_trait_confusion_lists.get(trait_name)

                        rows.append(
                            self._create_metric_trait_row(result, trait_name, metric_name, metric_score, confusion_data)
                        )

        return pd.DataFrame(rows)

    def _create_llm_trait_row(
        self,
        result: VerificationResult,
        trait_name: str,
        trait_score: int | bool,
        score_type: str,
    ) -> dict[str, Any]:
        """Create DataFrame row for LLM trait."""
        metadata = result.metadata
        rubric = result.rubric

        # Unified replicate
        replicate = metadata.answering_replicate
        if replicate is None:
            replicate = metadata.parsing_replicate

        row = {
            # === Status ===
            "completed_without_errors": metadata.completed_without_errors,
            "error": metadata.error,
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": replicate,
            # === Model Configuration ===
            "answering_model": metadata.answering_model,
            "parsing_model": metadata.parsing_model,
            "answering_system_prompt": metadata.answering_system_prompt,
            "parsing_system_prompt": metadata.parsing_system_prompt,
            # === Rubric Data ===
            "trait_name": trait_name,
            "trait_score": trait_score,
            "trait_type": score_type,
            # === Rubric Metadata ===
            "evaluation_rubric": rubric.evaluation_rubric if rubric else None,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
            "job_id": metadata.job_id,
        }

        # Add deep judgment columns if requested
        if self.include_deep_judgment:
            # Get deep judgment rubric data
            rubric_dj = result.deep_judgment_rubric

            # Add trait_reasoning
            row["trait_reasoning"] = (
                rubric_dj.rubric_trait_reasoning.get(trait_name)
                if rubric_dj and rubric_dj.rubric_trait_reasoning
                else None
            )

            # Add trait_excerpts (JSON serialized)
            row["trait_excerpts"] = json.dumps(
                rubric_dj.extracted_rubric_excerpts.get(trait_name, [])
                if rubric_dj and rubric_dj.extracted_rubric_excerpts
                else []
            )

            # Add trait_hallucination_risk
            row["trait_hallucination_risk"] = (
                rubric_dj.rubric_hallucination_risk_assessment.get(trait_name)
                if rubric_dj and rubric_dj.rubric_hallucination_risk_assessment
                else None
            )

        return row

    def _create_regex_trait_row(
        self,
        result: VerificationResult,
        trait_name: str,
        trait_score: bool,
    ) -> dict[str, Any]:
        """Create DataFrame row for regex trait."""
        metadata = result.metadata
        rubric = result.rubric

        # Unified replicate
        replicate = metadata.answering_replicate
        if replicate is None:
            replicate = metadata.parsing_replicate

        return {
            # === Status ===
            "completed_without_errors": metadata.completed_without_errors,
            "error": metadata.error,
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": replicate,
            # === Model Configuration ===
            "answering_model": metadata.answering_model,
            "parsing_model": metadata.parsing_model,
            "answering_system_prompt": metadata.answering_system_prompt,
            "parsing_system_prompt": metadata.parsing_system_prompt,
            # === Rubric Data ===
            "trait_name": trait_name,
            "trait_score": trait_score,
            "trait_type": "regex",
            # === Rubric Metadata ===
            "evaluation_rubric": rubric.evaluation_rubric if rubric else None,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
            "job_id": metadata.job_id,
        }

    def _create_callable_trait_row(
        self,
        result: VerificationResult,
        trait_name: str,
        trait_score: bool | int,
    ) -> dict[str, Any]:
        """Create DataFrame row for callable trait."""
        metadata = result.metadata
        rubric = result.rubric

        # Unified replicate
        replicate = metadata.answering_replicate
        if replicate is None:
            replicate = metadata.parsing_replicate

        return {
            # === Status ===
            "completed_without_errors": metadata.completed_without_errors,
            "error": metadata.error,
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": replicate,
            # === Model Configuration ===
            "answering_model": metadata.answering_model,
            "parsing_model": metadata.parsing_model,
            "answering_system_prompt": metadata.answering_system_prompt,
            "parsing_system_prompt": metadata.parsing_system_prompt,
            # === Rubric Data ===
            "trait_name": trait_name,
            "trait_score": trait_score,
            "trait_type": "callable",
            # === Rubric Metadata ===
            "evaluation_rubric": rubric.evaluation_rubric if rubric else None,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
            "job_id": metadata.job_id,
        }

    def _create_metric_trait_row(
        self,
        result: VerificationResult,
        trait_name: str,
        metric_name: str,
        metric_score: float,
        confusion_data: dict[str, list[str]] | None,
    ) -> dict[str, Any]:
        """Create DataFrame row for metric trait (EXPLODED by metric)."""
        metadata = result.metadata
        rubric = result.rubric

        # Unified replicate
        replicate = metadata.answering_replicate
        if replicate is None:
            replicate = metadata.parsing_replicate

        return {
            # === Status ===
            "completed_without_errors": metadata.completed_without_errors,
            "error": metadata.error,
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": replicate,
            # === Model Configuration ===
            "answering_model": metadata.answering_model,
            "parsing_model": metadata.parsing_model,
            "answering_system_prompt": metadata.answering_system_prompt,
            "parsing_system_prompt": metadata.parsing_system_prompt,
            # === Metric Trait Data (EXPLODED) ===
            "trait_name": trait_name,
            "metric_name": metric_name,
            "metric_score": metric_score,
            "trait_type": "metric",
            # === Confusion Matrix Metadata ===
            "confusion_tp": confusion_data.get("tp") if confusion_data else None,
            "confusion_fp": confusion_data.get("fp") if confusion_data else None,
            "confusion_fn": confusion_data.get("fn") if confusion_data else None,
            "confusion_tn": confusion_data.get("tn") if confusion_data else None,
            # === Rubric Metadata ===
            "evaluation_rubric": rubric.evaluation_rubric if rubric else None,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
            "job_id": metadata.job_id,
        }

    def _create_empty_rubric_row(self, result: VerificationResult) -> dict[str, Any]:
        """Create empty DataFrame row for results without rubric data."""
        metadata = result.metadata

        # Unified replicate
        replicate = metadata.answering_replicate
        if replicate is None:
            replicate = metadata.parsing_replicate

        return {
            # === Status ===
            "completed_without_errors": metadata.completed_without_errors,
            "error": metadata.error,
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": replicate,
            # === Model Configuration ===
            "answering_model": metadata.answering_model,
            "parsing_model": metadata.parsing_model,
            "answering_system_prompt": metadata.answering_system_prompt,
            "parsing_system_prompt": metadata.parsing_system_prompt,
            # === Rubric Data (None) ===
            "trait_name": None,
            "trait_score": None,
            "trait_type": None,
            # === Rubric Metadata ===
            "evaluation_rubric": None,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
            "job_id": metadata.job_id,
        }

    # ========================================================================
    # Core Data Access
    # ========================================================================

    def get_results_with_rubric(self) -> list[VerificationResult]:
        """
        Get only results that have rubric evaluation data.

        Returns:
            List of results where rubric evaluation was performed
        """
        return [r for r in self.results if r.rubric is not None and r.rubric.rubric_evaluation_performed]

    def get_llm_trait_scores(
        self, question_id: str | None = None, trait_name: str | None = None
    ) -> dict[str, dict[str, int]]:
        """
        Get LLM trait scores from all results.

        Args:
            question_id: Optional question ID to filter by
            trait_name: Optional trait name to filter by

        Returns:
            Dictionary mapping result identifiers to LLM trait scores
            Format: {result_id: {trait_name: score}}
        """
        scores = {}
        for result in self.get_results_with_rubric():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.rubric and result.rubric.llm_trait_scores:
                result_id = self._get_result_id(result)
                trait_scores = result.rubric.llm_trait_scores

                # Filter by trait name if specified
                if trait_name:
                    if trait_name in trait_scores:
                        scores[result_id] = {trait_name: trait_scores[trait_name]}
                else:
                    scores[result_id] = trait_scores

        return scores

    def get_regex_trait_scores(
        self, question_id: str | None = None, trait_name: str | None = None
    ) -> dict[str, dict[str, bool]]:
        """
        Get regex trait scores from all results.

        Args:
            question_id: Optional question ID to filter by
            trait_name: Optional trait name to filter by

        Returns:
            Dictionary mapping result identifiers to regex trait scores
            Format: {result_id: {trait_name: bool}}
        """
        scores = {}
        for result in self.get_results_with_rubric():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.rubric and result.rubric.regex_trait_scores:
                result_id = self._get_result_id(result)
                trait_scores = result.rubric.regex_trait_scores

                # Filter by trait name if specified
                if trait_name:
                    if trait_name in trait_scores:
                        scores[result_id] = {trait_name: trait_scores[trait_name]}
                else:
                    scores[result_id] = trait_scores

        return scores

    def get_callable_trait_scores(
        self, question_id: str | None = None, trait_name: str | None = None
    ) -> dict[str, dict[str, bool | int]]:
        """
        Get callable trait scores from all results.

        Args:
            question_id: Optional question ID to filter by
            trait_name: Optional trait name to filter by

        Returns:
            Dictionary mapping result identifiers to callable trait scores
            Format: {result_id: {trait_name: bool | int}}
        """
        scores = {}
        for result in self.get_results_with_rubric():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.rubric and result.rubric.callable_trait_scores:
                result_id = self._get_result_id(result)
                trait_scores = result.rubric.callable_trait_scores

                # Filter by trait name if specified
                if trait_name:
                    if trait_name in trait_scores:
                        scores[result_id] = {trait_name: trait_scores[trait_name]}
                else:
                    scores[result_id] = trait_scores

        return scores

    def get_metric_trait_scores(
        self, question_id: str | None = None, trait_name: str | None = None
    ) -> dict[str, dict[str, dict[str, float]]]:
        """
        Get metric trait scores from all results.

        Args:
            question_id: Optional question ID to filter by
            trait_name: Optional trait name to filter by

        Returns:
            Dictionary mapping result identifiers to metric trait scores
            Format: {result_id: {trait_name: {metric_name: value}}}
        """
        scores = {}
        for result in self.get_results_with_rubric():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.rubric and result.rubric.metric_trait_scores:
                result_id = self._get_result_id(result)
                trait_scores = result.rubric.metric_trait_scores

                # Filter by trait name if specified
                if trait_name:
                    if trait_name in trait_scores:
                        scores[result_id] = {trait_name: trait_scores[trait_name]}
                else:
                    scores[result_id] = trait_scores

        return scores

    def get_all_trait_scores(self, question_id: str | None = None) -> dict[str, dict[str, Any]]:
        """
        Get all trait scores (LLM, manual, and metric) from all results.

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary mapping result identifiers to all trait scores
            Format: {result_id: {trait_name: score_value}}
            Note: Metric traits will have dict values, others will have int/bool
        """
        all_scores = {}
        for result in self.get_results_with_rubric():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.rubric:
                result_id = self._get_result_id(result)
                # Use the helper method from VerificationResultRubric
                all_scores[result_id] = result.rubric.get_all_trait_scores()

        return all_scores

    def get_confusion_matrices(
        self, question_id: str | None = None, trait_name: str | None = None
    ) -> dict[str, dict[str, dict[str, list[str]]]]:
        """
        Get confusion matrix data for metric traits.

        Args:
            question_id: Optional question ID to filter by
            trait_name: Optional trait name to filter by

        Returns:
            Dictionary mapping result identifiers to confusion matrices
            Format: {result_id: {trait_name: {category: [excerpts]}}}
            Categories: "tp", "tn", "fp", "fn"
        """
        matrices = {}
        for result in self.get_results_with_rubric():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.rubric and result.rubric.metric_trait_confusion_lists:
                result_id = self._get_result_id(result)
                confusion_data = result.rubric.metric_trait_confusion_lists

                # Filter by trait name if specified
                if trait_name:
                    if trait_name in confusion_data:
                        matrices[result_id] = {trait_name: confusion_data[trait_name]}
                else:
                    matrices[result_id] = confusion_data

        return matrices

    # ========================================================================
    # Aggregation
    # ========================================================================

    def aggregate_llm_traits(
        self,
        strategy: str = "mean",
        by: str = "question_id",
        **kwargs: Any,
    ) -> dict[str, dict[str, float]]:
        """
        Aggregate LLM trait scores using specified strategy.

        Uses pandas DataFrame groupby for efficient aggregation.

        Args:
            strategy: Aggregation strategy name (e.g., "mean", "median", "mode")
            by: Column name to group by (e.g., "question_id", "answering_model", "replicate")
            **kwargs: Additional parameters for the aggregator

        Returns:
            Dictionary mapping group identifiers to aggregated trait scores
            Format: {group_id: {trait_name: aggregated_score}}

        Example:
            >>> results.aggregate_llm_traits(strategy="mean", by="question_id")
            {'q1': {'clarity': 4.5, 'accuracy': 4.0}, 'q2': {...}}
        """

        aggregator = self._aggregator_registry.get(strategy)

        # Get DataFrame with only LLM traits (both score and binary)
        df = self.to_dataframe(trait_type="llm")

        if len(df) == 0:
            return {}

        # Group by specified column and trait_name, then aggregate scores
        grouped = df.groupby([by, "trait_name"])["trait_score"].agg(lambda s: aggregator.aggregate(s, **kwargs))

        # Convert to nested dictionary format
        result: dict[str, dict[str, float]] = {}
        for (group_id, trait_name), score in grouped.items():
            if group_id not in result:
                result[group_id] = {}
            result[group_id][trait_name] = float(score)

        return result

    def aggregate_regex_traits(
        self,
        strategy: str = "majority_vote",
        by: str = "question_id",
        **kwargs: Any,
    ) -> dict[str, dict[str, bool]]:
        """
        Aggregate regex trait scores using specified strategy.

        Uses pandas DataFrame groupby for efficient aggregation.

        Args:
            strategy: Aggregation strategy name (default: "majority_vote")
            by: Column name to group by (e.g., "question_id", "answering_model", "replicate")
            **kwargs: Additional parameters for the aggregator

        Returns:
            Dictionary mapping group identifiers to aggregated trait scores
            Format: {group_id: {trait_name: aggregated_bool}}

        Example:
            >>> results.aggregate_regex_traits(strategy="majority_vote", by="question_id")
            {'q1': {'contains_url': True, 'has_code': False}, 'q2': {...}}
        """

        aggregator = self._aggregator_registry.get(strategy)

        # Get DataFrame with only regex traits
        df = self.to_dataframe(trait_type="regex")

        if len(df) == 0:
            return {}

        # Group by specified column and trait_name, then aggregate scores
        grouped = df.groupby([by, "trait_name"])["trait_score"].agg(lambda s: aggregator.aggregate(s, **kwargs))

        # Convert to nested dictionary format
        result: dict[str, dict[str, bool]] = {}
        for (group_id, trait_name), score in grouped.items():
            if group_id not in result:
                result[group_id] = {}
            result[group_id][trait_name] = bool(score)

        return result

    def aggregate_callable_traits(
        self,
        strategy: str = "majority_vote",
        by: str = "question_id",
        **kwargs: Any,
    ) -> dict[str, dict[str, bool | float]]:
        """
        Aggregate callable trait scores using specified strategy.

        For boolean traits, uses majority vote. For score traits, uses mean.

        Uses pandas DataFrame groupby for efficient aggregation.

        Args:
            strategy: Aggregation strategy name (default: "majority_vote" for bool, "mean" for scores)
            by: Column name to group by (e.g., "question_id", "answering_model", "replicate")
            **kwargs: Additional parameters for the aggregator

        Returns:
            Dictionary mapping group identifiers to aggregated trait scores
            Format: {group_id: {trait_name: aggregated_value}}

        Example:
            >>> results.aggregate_callable_traits(strategy="majority_vote", by="question_id")
            {'q1': {'passes_check': True, 'quality_score': 4.2}, 'q2': {...}}
        """

        aggregator = self._aggregator_registry.get(strategy)

        # Get DataFrame with only callable traits
        df = self.to_dataframe(trait_type="callable")

        if len(df) == 0:
            return {}

        # Group by specified column and trait_name, then aggregate scores
        grouped = df.groupby([by, "trait_name"])["trait_score"].agg(lambda s: aggregator.aggregate(s, **kwargs))

        # Convert to nested dictionary format
        result: dict[str, dict[str, bool | float]] = {}
        for (group_id, trait_name), score in grouped.items():
            if group_id not in result:
                result[group_id] = {}
            # Keep original type (bool or numeric)
            result[group_id][trait_name] = score

        return result

    def aggregate_metric_traits(
        self,
        metric_name: str,
        strategy: str = "mean",
        by: str = "question_id",
        **kwargs: Any,
    ) -> dict[str, dict[str, float]]:
        """
        Aggregate specific metric from metric trait scores.

        Uses pandas DataFrame groupby for efficient aggregation.

        Args:
            metric_name: Name of metric to aggregate (e.g., "precision", "recall", "f1")
            strategy: Aggregation strategy name (default: "mean")
            by: Column name to group by (e.g., "question_id", "answering_model", "replicate")
            **kwargs: Additional parameters for the aggregator

        Returns:
            Dictionary mapping group identifiers to aggregated metric values
            Format: {group_id: {trait_name: aggregated_metric_value}}

        Example:
            >>> results.aggregate_metric_traits(metric_name="f1", strategy="mean", by="question_id")
            {'q1': {'accuracy': 0.95, 'completeness': 0.88}, 'q2': {...}}
        """

        aggregator = self._aggregator_registry.get(strategy)

        # Get DataFrame with only metric traits (already exploded by metric)
        df = self.to_dataframe(trait_type="metric")

        if len(df) == 0:
            return {}

        # Filter to only the requested metric
        df_filtered = df[df["metric_name"] == metric_name]

        if len(df_filtered) == 0:
            return {}

        # Group by specified column and trait_name, then aggregate scores
        grouped = df_filtered.groupby([by, "trait_name"])["metric_score"].agg(
            lambda s: aggregator.aggregate(s, **kwargs)
        )

        # Convert to nested dictionary format
        result: dict[str, dict[str, float]] = {}
        for (group_id, trait_name), score in grouped.items():
            if group_id not in result:
                result[group_id] = {}
            result[group_id][trait_name] = float(score)

        return result

    # ========================================================================
    # Aggregator Registry Management
    # ========================================================================

    def register_aggregator(self, name: str, aggregator: Any) -> None:
        """
        Register a custom aggregation strategy.

        Args:
            name: Unique name for the aggregator
            aggregator: Aggregator instance implementing ResultAggregator protocol

        Example:
            ```python
            class CustomAggregator:
                def aggregate(self, values, **kwargs):
                    # Custom aggregation logic
                    return sum(values) / len(values)

            rubric_results.register_aggregator("custom", CustomAggregator())
            rubric_results.aggregate_llm_traits(strategy="custom")
            ```
        """
        self._aggregator_registry.register(name, aggregator)

    def list_aggregators(self) -> list[str]:
        """
        List all available aggregation strategies.

        Returns:
            List of registered aggregator names
        """
        return self._aggregator_registry.list_aggregators()

    # ========================================================================
    # Filtering and Grouping
    # ========================================================================

    def filter(
        self,
        question_ids: list[str] | None = None,
        answering_models: list[str] | None = None,
        parsing_models: list[str] | None = None,
        replicates: list[int] | None = None,
    ) -> RubricResults:
        """
        Filter results by various criteria.

        Args:
            question_ids: Filter by question IDs
            answering_models: Filter by answering model names
            parsing_models: Filter by parsing model names
            replicates: Filter by replicate numbers

        Returns:
            New RubricResults instance with filtered results
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

        return RubricResults(results=filtered)

    def group_by_question(self) -> dict[str, RubricResults]:
        """
        Group results by question ID.

        Returns:
            Dictionary mapping question IDs to RubricResults instances
        """
        grouped: dict[str, list[VerificationResult]] = {}
        for result in self.results:
            qid = result.metadata.question_id
            if qid not in grouped:
                grouped[qid] = []
            grouped[qid].append(result)

        return {qid: RubricResults(results=results) for qid, results in grouped.items()}

    def group_by_model(self) -> dict[str, RubricResults]:
        """
        Group results by answering model.

        Returns:
            Dictionary mapping model names to RubricResults instances
        """
        grouped: dict[str, list[VerificationResult]] = {}
        for result in self.results:
            model = result.metadata.answering_model
            if model not in grouped:
                grouped[model] = []
            grouped[model].append(result)

        return {model: RubricResults(results=results) for model, results in grouped.items()}

    # ========================================================================
    # Summary Statistics
    # ========================================================================

    def get_trait_summary(self) -> dict[str, Any]:
        """
        Get summary statistics for all traits.

        Returns:
            Dictionary with summary statistics:
            - num_results: Number of results with rubric data
            - llm_traits: List of LLM trait names
            - regex_traits: List of regex trait names
            - callable_traits: List of callable trait names
            - metric_traits: List of metric trait names
            - num_questions: Number of unique questions
        """
        results_with_rubric = self.get_results_with_rubric()

        llm_traits: set[str] = set()
        regex_traits: set[str] = set()
        callable_traits: set[str] = set()
        metric_traits: set[str] = set()
        questions: set[str] = set()

        for result in results_with_rubric:
            questions.add(result.metadata.question_id)

            if result.rubric:
                if result.rubric.llm_trait_scores:
                    llm_traits.update(result.rubric.llm_trait_scores.keys())
                if result.rubric.regex_trait_scores:
                    regex_traits.update(result.rubric.regex_trait_scores.keys())
                if result.rubric.callable_trait_scores:
                    callable_traits.update(result.rubric.callable_trait_scores.keys())
                if result.rubric.metric_trait_scores:
                    metric_traits.update(result.rubric.metric_trait_scores.keys())

        return {
            "num_results": len(results_with_rubric),
            "llm_traits": sorted(llm_traits),
            "regex_traits": sorted(regex_traits),
            "callable_traits": sorted(callable_traits),
            "metric_traits": sorted(metric_traits),
            "num_questions": len(questions),
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_result_id(self, result: VerificationResult) -> str:
        """
        Generate a unique identifier for a result.

        Args:
            result: VerificationResult to identify

        Returns:
            Unique string identifier
        """
        parts = [
            result.metadata.question_id,
            result.metadata.answering_model,
            result.metadata.parsing_model,
        ]

        if result.metadata.answering_replicate is not None:
            parts.append(f"rep{result.metadata.answering_replicate}")

        if result.metadata.timestamp:
            parts.append(str(result.metadata.timestamp))

        return "_".join(parts)

    # ========================================================================
    # Special Methods
    # ========================================================================

    def __len__(self) -> int:
        """Return number of results."""
        return len(self.results)

    def __iter__(self) -> Any:
        """Iterate over results."""
        return iter(self.results)

    def __getitem__(self, index: int) -> VerificationResult:
        """Access result by index."""
        return self.results[index]
