"""
RubricResults class for collating and analyzing rubric evaluation results.

This module provides functionality to work with rubric evaluation results across
multiple verification runs, supporting aggregation and analysis of trait scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .aggregation import (
    AggregatorRegistry,
    GroupByRegistry,
    create_default_groupby_registry,
    create_default_registry,
)

if TYPE_CHECKING:
    from .verification import VerificationResult


TraitType = Literal["llm", "manual", "metric"]


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

    def __init__(self, results: list[VerificationResult], **data: Any) -> None:
        """
        Initialize RubricResults with verification results.

        Args:
            results: List of VerificationResult objects
            **data: Additional pydantic model data
        """
        super().__init__(results=results, **data)
        # Create default aggregator and groupby registries
        self._aggregator_registry: AggregatorRegistry = create_default_registry()
        self._groupby_registry: GroupByRegistry = create_default_groupby_registry()

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

    def get_manual_trait_scores(
        self, question_id: str | None = None, trait_name: str | None = None
    ) -> dict[str, dict[str, bool]]:
        """
        Get manual trait scores from all results.

        Args:
            question_id: Optional question ID to filter by
            trait_name: Optional trait name to filter by

        Returns:
            Dictionary mapping result identifiers to manual trait scores
            Format: {result_id: {trait_name: bool}}
        """
        scores = {}
        for result in self.get_results_with_rubric():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.rubric and result.rubric.manual_trait_scores:
                result_id = self._get_result_id(result)
                trait_scores = result.rubric.manual_trait_scores

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
        by: str = "question",
        **kwargs: Any,
    ) -> dict[str, dict[str, float]]:
        """
        Aggregate LLM trait scores using specified strategy.

        Args:
            strategy: Aggregation strategy name (e.g., "mean", "median", "mode")
            by: Grouping strategy name (e.g., "question", "model", "replicate", "model_pair")
            **kwargs: Additional parameters for the aggregator

        Returns:
            Dictionary mapping group identifiers to aggregated trait scores
            Format: {group_id: {trait_name: aggregated_score}}
        """
        aggregator = self._aggregator_registry.get(strategy)

        # Group results by specified axis
        grouped = self._group_results(by)

        # Aggregate trait scores for each group
        aggregated: dict[str, dict[str, float]] = {}
        for group_id, group_results in grouped.items():
            trait_scores: dict[str, list[int]] = {}

            # Collect all trait scores for this group
            for result in group_results:
                if result.rubric and result.rubric.llm_trait_scores:
                    for trait_name, score in result.rubric.llm_trait_scores.items():
                        if trait_name not in trait_scores:
                            trait_scores[trait_name] = []
                        trait_scores[trait_name].append(score)

            # Aggregate each trait
            aggregated[group_id] = {}
            for trait_name, scores in trait_scores.items():
                aggregated[group_id][trait_name] = aggregator.aggregate(scores, **kwargs)

        return aggregated

    def aggregate_manual_traits(
        self,
        strategy: str = "majority_vote",
        by: str = "question",
        **kwargs: Any,
    ) -> dict[str, dict[str, bool]]:
        """
        Aggregate manual trait scores using specified strategy.

        Args:
            strategy: Aggregation strategy name (default: "majority_vote")
            by: Grouping axis ("question", "model", "replicate")
            **kwargs: Additional parameters for the aggregator

        Returns:
            Dictionary mapping group identifiers to aggregated trait scores
            Format: {group_id: {trait_name: aggregated_bool}}
        """
        aggregator = self._aggregator_registry.get(strategy)

        # Group results by specified axis
        grouped = self._group_results(by)

        # Aggregate trait scores for each group
        aggregated: dict[str, dict[str, bool]] = {}
        for group_id, group_results in grouped.items():
            trait_scores: dict[str, list[bool]] = {}

            # Collect all trait scores for this group
            for result in group_results:
                if result.rubric and result.rubric.manual_trait_scores:
                    for trait_name, score in result.rubric.manual_trait_scores.items():
                        if trait_name not in trait_scores:
                            trait_scores[trait_name] = []
                        trait_scores[trait_name].append(score)

            # Aggregate each trait
            aggregated[group_id] = {}
            for trait_name, scores in trait_scores.items():
                aggregated[group_id][trait_name] = aggregator.aggregate(scores, **kwargs)

        return aggregated

    def aggregate_metric_traits(
        self,
        metric_name: str,
        strategy: str = "mean",
        by: str = "question",
        **kwargs: Any,
    ) -> dict[str, dict[str, float]]:
        """
        Aggregate specific metric from metric trait scores.

        Args:
            metric_name: Name of metric to aggregate (e.g., "precision", "recall", "f1")
            strategy: Aggregation strategy name (default: "mean")
            by: Grouping axis ("question", "model", "replicate")
            **kwargs: Additional parameters for the aggregator

        Returns:
            Dictionary mapping group identifiers to aggregated metric values
            Format: {group_id: {trait_name: aggregated_metric_value}}
        """
        aggregator = self._aggregator_registry.get(strategy)

        # Group results by specified axis
        grouped = self._group_results(by)

        # Aggregate metric values for each group
        aggregated: dict[str, dict[str, float]] = {}
        for group_id, group_results in grouped.items():
            trait_metrics: dict[str, list[float]] = {}

            # Collect metric values for this group
            for result in group_results:
                if result.rubric and result.rubric.metric_trait_scores:
                    for trait_name, metrics in result.rubric.metric_trait_scores.items():
                        if metric_name in metrics:
                            if trait_name not in trait_metrics:
                                trait_metrics[trait_name] = []
                            trait_metrics[trait_name].append(metrics[metric_name])

            # Aggregate each trait's metric
            aggregated[group_id] = {}
            for trait_name, metric_values in trait_metrics.items():
                aggregated[group_id][trait_name] = aggregator.aggregate(metric_values, **kwargs)

        return aggregated

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

    def register_groupby_strategy(self, name: str, strategy: Any) -> None:
        """
        Register a custom grouping strategy.

        Args:
            name: Unique name for the grouping strategy
            strategy: Strategy instance implementing GroupByStrategy protocol

        Example:
            ```python
            class ByTemplateStrategy:
                def get_group_key(self, result):
                    return result.metadata.template_id

            rubric_results.register_groupby_strategy("template", ByTemplateStrategy())
            rubric_results.aggregate_llm_traits(by="template")
            ```
        """
        self._groupby_registry.register(name, strategy)

    def list_groupby_strategies(self) -> list[str]:
        """
        List all available grouping strategies.

        Returns:
            List of registered groupby strategy names
        """
        return self._groupby_registry.list_strategies()

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
            - manual_traits: List of manual trait names
            - metric_traits: List of metric trait names
            - num_questions: Number of unique questions
        """
        results_with_rubric = self.get_results_with_rubric()

        llm_traits: set[str] = set()
        manual_traits: set[str] = set()
        metric_traits: set[str] = set()
        questions: set[str] = set()

        for result in results_with_rubric:
            questions.add(result.metadata.question_id)

            if result.rubric:
                if result.rubric.llm_trait_scores:
                    llm_traits.update(result.rubric.llm_trait_scores.keys())
                if result.rubric.manual_trait_scores:
                    manual_traits.update(result.rubric.manual_trait_scores.keys())
                if result.rubric.metric_trait_scores:
                    metric_traits.update(result.rubric.metric_trait_scores.keys())

        return {
            "num_results": len(results_with_rubric),
            "llm_traits": sorted(llm_traits),
            "manual_traits": sorted(manual_traits),
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

    def _group_results(self, by: str) -> dict[str, list[VerificationResult]]:
        """
        Group results by specified grouping strategy.

        Args:
            by: Grouping strategy name (must be registered in groupby registry)

        Returns:
            Dictionary mapping group keys to lists of results

        Raises:
            KeyError: If grouping strategy not found in registry
        """
        strategy = self._groupby_registry.get(by)
        grouped: dict[str, list[VerificationResult]] = {}

        for result in self.get_results_with_rubric():
            # Get group key using the strategy
            key = strategy.get_group_key(result)

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)

        return grouped

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
