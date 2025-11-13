"""
TemplateResults class for collating and analyzing template verification results.

This module provides functionality to work with template verification results,
including embedding checks, regex validation, abstention detection, and MCP metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .aggregation import (
    AggregatorRegistry,
    GroupByRegistry,
    create_default_groupby_registry,
    create_default_registry,
)

if TYPE_CHECKING:
    from .verification import VerificationResult


class TemplateResults(BaseModel):
    """
    Collated template verification results from multiple verification runs.

    This class provides a unified interface for accessing, filtering, and
    aggregating template verification data including:
    - Template verification pass/fail rates
    - Embedding similarity scores
    - Regex validation results
    - Abstention detection
    - MCP tool usage metrics

    Attributes:
        results: List of VerificationResult objects containing template data
    """

    results: list[VerificationResult] = Field(description="List of verification results containing template data")

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, results: list[VerificationResult], **data: Any) -> None:
        """
        Initialize TemplateResults with verification results.

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

    def get_results_with_template(self) -> list[VerificationResult]:
        """
        Get only results that have template verification data.

        Returns:
            List of results where template verification was performed
        """
        return [r for r in self.results if r.template is not None and r.template.template_verification_performed]

    def get_verification_results(self, question_id: str | None = None) -> dict[str, bool | None]:
        """
        Get template verification pass/fail results.

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary mapping result identifiers to verification results
            Format: {result_id: bool | None}
        """
        results_dict = {}
        for result in self.get_results_with_template():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.template:
                result_id = self._get_result_id(result)
                results_dict[result_id] = result.template.verify_result

        return results_dict

    def get_embedding_scores(self, question_id: str | None = None) -> dict[str, float | None]:
        """
        Get embedding similarity scores from all results.

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary mapping result identifiers to similarity scores
            Format: {result_id: float | None}
        """
        scores = {}
        for result in self.get_results_with_template():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.template and result.template.embedding_check_performed:
                result_id = self._get_result_id(result)
                scores[result_id] = result.template.embedding_similarity_score

        return scores

    def get_embedding_overrides(self, question_id: str | None = None) -> dict[str, bool]:
        """
        Get embedding override status from all results.

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary mapping result identifiers to override status
            Format: {result_id: bool}
        """
        overrides = {}
        for result in self.get_results_with_template():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.template and result.template.embedding_check_performed:
                result_id = self._get_result_id(result)
                overrides[result_id] = result.template.embedding_override_applied

        return overrides

    def get_regex_results(
        self, question_id: str | None = None, pattern_name: str | None = None
    ) -> dict[str, dict[str, bool] | None]:
        """
        Get regex validation results from all results.

        Args:
            question_id: Optional question ID to filter by
            pattern_name: Optional pattern name to filter by

        Returns:
            Dictionary mapping result identifiers to regex validation results
            Format: {result_id: {pattern_name: bool} | None}
        """
        regex_results: dict[str, dict[str, bool] | None] = {}
        for result in self.get_results_with_template():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.template and result.template.regex_validations_performed:
                result_id = self._get_result_id(result)
                validation_results = result.template.regex_validation_results

                # Filter by pattern name if specified
                if pattern_name and validation_results:
                    if pattern_name in validation_results:
                        regex_results[result_id] = {pattern_name: validation_results[pattern_name]}
                elif validation_results is not None:
                    regex_results[result_id] = validation_results

        return regex_results

    def get_abstention_detections(self, question_id: str | None = None) -> dict[str, bool | None]:
        """
        Get abstention detection results from all results.

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary mapping result identifiers to abstention detected status
            Format: {result_id: bool | None}
        """
        abstentions: dict[str, bool | None] = {}
        for result in self.get_results_with_template():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.template and result.template.abstention_check_performed:
                result_id = self._get_result_id(result)
                abstentions[result_id] = result.template.abstention_detected

        return abstentions

    def get_mcp_usage(self, question_id: str | None = None) -> dict[str, dict[str, Any]]:
        """
        Get MCP tool usage data from all results.

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary mapping result identifiers to MCP usage data
            Format: {result_id: {
                "servers": list[str],
                "metrics": dict
            }}
        """
        mcp_usage = {}
        for result in self.get_results_with_template():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.template:
                result_id = self._get_result_id(result)
                mcp_usage[result_id] = {
                    "servers": result.template.answering_mcp_servers or [],
                    "metrics": result.template.agent_metrics or {},
                }

        return mcp_usage

    def get_parsed_responses(self, question_id: str | None = None) -> dict[str, dict[str, Any]]:
        """
        Get parsed LLM and ground truth responses.

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary mapping result identifiers to parsed responses
            Format: {result_id: {
                "llm": dict,
                "ground_truth": dict
            }}
        """
        responses = {}
        for result in self.get_results_with_template():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.template:
                result_id = self._get_result_id(result)
                responses[result_id] = {
                    "llm": result.template.parsed_llm_response,
                    "ground_truth": result.template.parsed_gt_response,
                }

        return responses

    # ========================================================================
    # Aggregation
    # ========================================================================

    def aggregate_pass_rate(
        self,
        by: str = "question",
        **kwargs: Any,  # noqa: ARG002
    ) -> dict[str, float]:
        """
        Calculate template verification pass rate by group.

        Args:
            by: Grouping axis ("question", "model", "replicate")
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping group identifiers to pass rates (0.0 to 1.0)
            Format: {group_id: pass_rate}
        """
        # Group results by specified axis
        grouped = self._group_results(by)

        # Calculate pass rate for each group
        pass_rates = {}
        for group_id, group_results in grouped.items():
            verify_results = []

            for result in group_results:
                if result.template and result.template.verify_result is not None:
                    verify_results.append(result.template.verify_result)

            if verify_results:
                pass_count = sum(1 for v in verify_results if v)
                pass_rates[group_id] = pass_count / len(verify_results)
            else:
                pass_rates[group_id] = 0.0

        return pass_rates

    def aggregate_embedding_scores(
        self,
        strategy: str = "mean",
        by: str = "question",
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Aggregate embedding similarity scores using specified strategy.

        Args:
            strategy: Aggregation strategy name (e.g., "mean", "median")
            by: Grouping axis ("question", "model", "replicate")
            **kwargs: Additional parameters for the aggregator

        Returns:
            Dictionary mapping group identifiers to aggregated scores
            Format: {group_id: aggregated_score}
        """
        aggregator = self._aggregator_registry.get(strategy)

        # Group results by specified axis
        grouped = self._group_results(by)

        # Aggregate scores for each group
        aggregated = {}
        for group_id, group_results in grouped.items():
            scores = []

            for result in group_results:
                if (
                    result.template
                    and result.template.embedding_check_performed
                    and result.template.embedding_similarity_score is not None
                ):
                    scores.append(result.template.embedding_similarity_score)

            aggregated[group_id] = aggregator.aggregate(scores, **kwargs)

        return aggregated

    def aggregate_regex_success_rate(
        self,
        pattern_name: str | None = None,
        by: str = "question",
        **kwargs: Any,  # noqa: ARG002
    ) -> dict[str, dict[str, float]] | dict[str, float]:
        """
        Calculate regex validation success rate.

        Args:
            pattern_name: Optional specific pattern to analyze
            by: Grouping axis ("question", "model", "replicate")
            **kwargs: Additional parameters

        Returns:
            If pattern_name specified: {group_id: success_rate}
            If pattern_name None: {group_id: {pattern_name: success_rate}}
        """
        # Group results by specified axis
        grouped = self._group_results(by)

        if pattern_name:
            # Single pattern analysis
            success_rates: dict[str, float] = {}
            for group_id, group_results in grouped.items():
                pattern_results = []

                for result in group_results:
                    if (
                        result.template
                        and result.template.regex_validations_performed
                        and result.template.regex_validation_results
                        and pattern_name in result.template.regex_validation_results
                    ):
                        pattern_results.append(result.template.regex_validation_results[pattern_name])

                if pattern_results:
                    success_count = sum(1 for v in pattern_results if v)
                    success_rates[group_id] = success_count / len(pattern_results)
                else:
                    success_rates[group_id] = 0.0

            return success_rates
        else:
            # All patterns analysis
            all_success_rates: dict[str, dict[str, float]] = {}
            for group_id, group_results in grouped.items():
                pattern_results_by_name: dict[str, list[bool]] = {}

                for result in group_results:
                    if (
                        result.template
                        and result.template.regex_validations_performed
                        and result.template.regex_validation_results
                    ):
                        for (
                            pname,
                            presult,
                        ) in result.template.regex_validation_results.items():
                            if pname not in pattern_results_by_name:
                                pattern_results_by_name[pname] = []
                            pattern_results_by_name[pname].append(presult)

                # Calculate success rate for each pattern
                all_success_rates[group_id] = {}
                for pname, presults in pattern_results_by_name.items():
                    if presults:
                        success_count = sum(1 for v in presults if v)
                        all_success_rates[group_id][pname] = success_count / len(presults)
                    else:
                        all_success_rates[group_id][pname] = 0.0

            return all_success_rates

    def aggregate_abstention_rate(
        self,
        by: str = "question",
        **kwargs: Any,  # noqa: ARG002
    ) -> dict[str, float]:
        """
        Calculate abstention detection rate by group.

        Args:
            by: Grouping axis ("question", "model", "replicate")
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping group identifiers to abstention rates (0.0 to 1.0)
            Format: {group_id: abstention_rate}
        """
        # Group results by specified axis
        grouped = self._group_results(by)

        # Calculate abstention rate for each group
        abstention_rates = {}
        for group_id, group_results in grouped.items():
            abstention_results = []

            for result in group_results:
                if result.template and result.template.abstention_check_performed:
                    abstention_results.append(result.template.abstention_detected)

            if abstention_results:
                abstention_count = sum(1 for v in abstention_results if v)
                abstention_rates[group_id] = abstention_count / len(abstention_results)
            else:
                abstention_rates[group_id] = 0.0

        return abstention_rates

    # ========================================================================
    # Aggregator Registry Management
    # ========================================================================

    def register_aggregator(self, name: str, aggregator: Any) -> None:
        """
        Register a custom aggregation strategy.

        Args:
            name: Unique name for the aggregator
            aggregator: Aggregator instance implementing ResultAggregator protocol
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
        passed_only: bool = False,
        failed_only: bool = False,
    ) -> TemplateResults:
        """
        Filter results by various criteria.

        Args:
            question_ids: Filter by question IDs
            answering_models: Filter by answering model names
            parsing_models: Filter by parsing model names
            replicates: Filter by replicate numbers
            passed_only: Only include results that passed template verification
            failed_only: Only include results that failed template verification

        Returns:
            New TemplateResults instance with filtered results
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

        if passed_only:
            filtered = [r for r in filtered if r.template and r.template.verify_result is True]

        if failed_only:
            filtered = [r for r in filtered if r.template and r.template.verify_result is False]

        return TemplateResults(results=filtered)

    def group_by_question(self) -> dict[str, TemplateResults]:
        """
        Group results by question ID.

        Returns:
            Dictionary mapping question IDs to TemplateResults instances
        """
        grouped: dict[str, list[VerificationResult]] = {}
        for result in self.results:
            qid = result.metadata.question_id
            if qid not in grouped:
                grouped[qid] = []
            grouped[qid].append(result)

        return {qid: TemplateResults(results=results) for qid, results in grouped.items()}

    def group_by_model(self) -> dict[str, TemplateResults]:
        """
        Group results by answering model.

        Returns:
            Dictionary mapping model names to TemplateResults instances
        """
        grouped: dict[str, list[VerificationResult]] = {}
        for result in self.results:
            model = result.metadata.answering_model
            if model not in grouped:
                grouped[model] = []
            grouped[model].append(result)

        return {model: TemplateResults(results=results) for model, results in grouped.items()}

    # ========================================================================
    # Summary Statistics
    # ========================================================================

    def get_template_summary(self) -> dict[str, Any]:
        """
        Get summary statistics for template verification.

        Returns:
            Dictionary with summary statistics:
            - num_results: Number of results with template data
            - num_passed: Number of results that passed
            - num_failed: Number of results that failed
            - pass_rate: Overall pass rate
            - num_with_embedding: Number with embedding check
            - num_with_regex: Number with regex validation
            - num_with_abstention: Number with abstention detection
            - num_questions: Number of unique questions
        """
        results_with_template = self.get_results_with_template()

        num_passed = 0
        num_failed = 0
        num_with_embedding = 0
        num_with_regex = 0
        num_with_abstention = 0
        questions = set()

        for result in results_with_template:
            questions.add(result.metadata.question_id)

            if result.template:
                if result.template.verify_result is True:
                    num_passed += 1
                elif result.template.verify_result is False:
                    num_failed += 1

                if result.template.embedding_check_performed:
                    num_with_embedding += 1

                if result.template.regex_validations_performed:
                    num_with_regex += 1

                if result.template.abstention_check_performed:
                    num_with_abstention += 1

        num_results = len(results_with_template)
        pass_rate = num_passed / num_results if num_results > 0 else 0.0

        return {
            "num_results": num_results,
            "num_passed": num_passed,
            "num_failed": num_failed,
            "pass_rate": pass_rate,
            "num_with_embedding": num_with_embedding,
            "num_with_regex": num_with_regex,
            "num_with_abstention": num_with_abstention,
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

        for result in self.get_results_with_template():
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
