"""
JudgmentResults class for collating and analyzing deep judgment results.

This module provides functionality to work with deep judgment evaluation results,
including excerpt extraction, reasoning traces, and hallucination risk assessments.
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


class JudgmentResults(BaseModel):
    """
    Collated deep judgment results from multiple verification runs.

    This class provides a unified interface for accessing, filtering, and
    aggregating deep judgment data including:
    - Extracted excerpts per attribute
    - Reasoning traces
    - Hallucination risk assessments
    - Search validation results
    - Multi-stage processing metrics

    Attributes:
        results: List of VerificationResult objects containing deep judgment data
    """

    results: list[VerificationResult] = Field(description="List of verification results containing deep judgment data")

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, results: list[VerificationResult], **data: Any) -> None:
        """
        Initialize JudgmentResults with verification results.

        Args:
            results: List of VerificationResult objects
            **data: Additional pydantic model data
        """
        super().__init__(results=results, **data)
        # Create default aggregator and groupby registries
        self._aggregator_registry: AggregatorRegistry = create_default_registry()
        self._groupby_registry: GroupByRegistry = create_default_groupby_registry()

    # ========================================================================
    # DataFrame Conversion
    # ========================================================================

    def to_dataframe(self) -> Any:
        """
        Convert deep judgment results to pandas DataFrame.

        Creates one row per (attribute × excerpt) combination.
        Attributes with no excerpts get one row with excerpt data as None.

        Column ordering:
            1. Status: completed_without_errors, error, recursion_limit_reached
            2. Identification: question_id, template_id, question_text, keywords, replicate, answering_mcp_servers
            3. Model Config: answering_model, parsing_model, system_prompts
            4. Response Data: raw_llm_response, parsed_gt_response, parsed_llm_response
            5. Deep Judgment Config: deep_judgment_enabled, deep_judgment_performed, deep_judgment_search_enabled
            6. Attribute Data: attribute_name, gt_attribute_value, llm_attribute_value, attribute_match
            7. Excerpt Data: excerpt_index, excerpt_text, excerpt_confidence, excerpt_similarity_score
            8. Search Enhancement: excerpt_search_results, excerpt_hallucination_risk, excerpt_hallucination_justification
            9. Attribute Metadata: attribute_reasoning, attribute_overall_risk, attribute_has_excerpts
            10. Processing Metrics: deep_judgment_model_calls, deep_judgment_excerpt_retries, stages_completed
            11. Execution Metadata: execution_time, timestamp, run_name, job_id

        Returns:
            pandas.DataFrame: Exploded DataFrame with one row per (attribute × excerpt)

        Example:
            >>> judgment_results = result_set.get_judgment_results()
            >>> df = judgment_results.to_dataframe()
            >>> # Filter to specific attribute
            >>> location_df = df[df['attribute_name'] == 'location']
            >>> # Count excerpts per question
            >>> excerpt_counts = df.groupby('question_id')['excerpt_index'].count()
        """
        import pandas as pd

        rows = []

        for result in self.results:
            if result.deep_judgment is None or not result.deep_judgment.deep_judgment_performed:
                # No deep judgment data - create single row with minimal info
                rows.append(self._create_empty_judgment_row(result))
                continue

            # Get template data for parsed responses
            parsed_gt = result.template.parsed_gt_response if result.template else None
            parsed_llm = result.template.parsed_llm_response if result.template else None

            # Get all attributes from extracted_excerpts
            if result.deep_judgment.extracted_excerpts:
                for attribute_name, excerpt_list in result.deep_judgment.extracted_excerpts.items():
                    # Get ground truth and LLM values for this attribute
                    gt_value = parsed_gt.get(attribute_name) if parsed_gt else None
                    llm_value = parsed_llm.get(attribute_name) if parsed_llm else None

                    # Get attribute-level metadata
                    attribute_reasoning = None
                    if result.deep_judgment.attribute_reasoning:
                        attribute_reasoning = result.deep_judgment.attribute_reasoning.get(attribute_name)

                    attribute_risk = None
                    if result.deep_judgment.hallucination_risk_assessment:
                        attribute_risk = result.deep_judgment.hallucination_risk_assessment.get(attribute_name)

                    if excerpt_list:
                        # Create one row per excerpt
                        for idx, excerpt in enumerate(excerpt_list):
                            rows.append(
                                self._create_judgment_row(
                                    result,
                                    attribute_name,
                                    gt_value,
                                    llm_value,
                                    idx,
                                    excerpt,
                                    attribute_reasoning,
                                    attribute_risk,
                                )
                            )
                    else:
                        # No excerpts found for this attribute - create one row with None excerpt data
                        rows.append(
                            self._create_judgment_row(
                                result,
                                attribute_name,
                                gt_value,
                                llm_value,
                                None,
                                None,
                                attribute_reasoning,
                                attribute_risk,
                            )
                        )
            else:
                # No extracted_excerpts at all
                rows.append(self._create_empty_judgment_row(result))

        return pd.DataFrame(rows)

    def _create_judgment_row(
        self,
        result: VerificationResult,
        attribute_name: str,
        gt_value: Any,
        llm_value: Any,
        excerpt_index: int | None,
        excerpt: dict[str, Any] | None,
        attribute_reasoning: str | None,
        attribute_risk: str | None,
    ) -> dict[str, Any]:
        """Create DataFrame row for judgment (attribute × excerpt)."""
        metadata = result.metadata
        template = result.template
        deep_judgment = result.deep_judgment

        # Unified replicate
        replicate = metadata.answering_replicate
        if replicate is None:
            replicate = metadata.parsing_replicate

        # Determine attribute match
        attribute_match = None
        if gt_value is not None or llm_value is not None:
            attribute_match = gt_value == llm_value

        # Extract excerpt data
        excerpt_text = None
        excerpt_confidence = None
        excerpt_similarity_score = None
        excerpt_search_results = None
        excerpt_hallucination_risk = None
        excerpt_hallucination_justification = None

        if excerpt:
            excerpt_text = excerpt.get("text")
            excerpt_confidence = excerpt.get("confidence")
            excerpt_similarity_score = excerpt.get("similarity_score")
            excerpt_search_results = excerpt.get("search_results")
            excerpt_hallucination_risk = excerpt.get("hallucination_risk")
            excerpt_hallucination_justification = excerpt.get("hallucination_justification")

        return {
            # === Status ===
            "completed_without_errors": metadata.completed_without_errors,
            "error": metadata.error,
            "recursion_limit_reached": template.recursion_limit_reached if template else None,
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": replicate,
            "answering_mcp_servers": template.answering_mcp_servers if template else None,
            # === Model Configuration ===
            "answering_model": metadata.answering_model,
            "parsing_model": metadata.parsing_model,
            "answering_system_prompt": metadata.answering_system_prompt,
            "parsing_system_prompt": metadata.parsing_system_prompt,
            # === Response Data ===
            "raw_llm_response": template.raw_llm_response if template else None,
            "parsed_gt_response": template.parsed_gt_response if template else None,
            "parsed_llm_response": template.parsed_llm_response if template else None,
            # === Deep Judgment Configuration ===
            "deep_judgment_enabled": deep_judgment.deep_judgment_enabled if deep_judgment else False,
            "deep_judgment_performed": deep_judgment.deep_judgment_performed if deep_judgment else False,
            "deep_judgment_search_enabled": deep_judgment.deep_judgment_search_enabled if deep_judgment else False,
            # === Attribute Information ===
            "attribute_name": attribute_name,
            "gt_attribute_value": gt_value,
            "llm_attribute_value": llm_value,
            "attribute_match": attribute_match,
            # === Excerpt Information ===
            "excerpt_index": excerpt_index,
            "excerpt_text": excerpt_text,
            "excerpt_confidence": excerpt_confidence,
            "excerpt_similarity_score": excerpt_similarity_score,
            # === Search Enhancement ===
            "excerpt_search_results": excerpt_search_results,
            "excerpt_hallucination_risk": excerpt_hallucination_risk,
            "excerpt_hallucination_justification": excerpt_hallucination_justification,
            # === Attribute Metadata ===
            "attribute_reasoning": attribute_reasoning,
            "attribute_overall_risk": attribute_risk,
            "attribute_has_excerpts": excerpt is not None,
            # === Processing Metrics ===
            "deep_judgment_model_calls": deep_judgment.deep_judgment_model_calls if deep_judgment else 0,
            "deep_judgment_excerpt_retries": deep_judgment.deep_judgment_excerpt_retry_count if deep_judgment else 0,
            "stages_completed": deep_judgment.deep_judgment_stages_completed if deep_judgment else None,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
            "job_id": metadata.job_id,
        }

    def _create_empty_judgment_row(self, result: VerificationResult) -> dict[str, Any]:
        """Create minimal DataFrame row when no judgment data exists."""
        metadata = result.metadata
        template = result.template

        # Unified replicate
        replicate = metadata.answering_replicate
        if replicate is None:
            replicate = metadata.parsing_replicate

        return {
            # === Status ===
            "completed_without_errors": metadata.completed_without_errors,
            "error": metadata.error,
            "recursion_limit_reached": template.recursion_limit_reached if template else None,
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": replicate,
            "answering_mcp_servers": template.answering_mcp_servers if template else None,
            # === Model Configuration ===
            "answering_model": metadata.answering_model,
            "parsing_model": metadata.parsing_model,
            "answering_system_prompt": metadata.answering_system_prompt,
            "parsing_system_prompt": metadata.parsing_system_prompt,
            # === Response Data ===
            "raw_llm_response": template.raw_llm_response if template else None,
            "parsed_gt_response": template.parsed_gt_response if template else None,
            "parsed_llm_response": template.parsed_llm_response if template else None,
            # === Deep Judgment Configuration ===
            "deep_judgment_enabled": False,
            "deep_judgment_performed": False,
            "deep_judgment_search_enabled": False,
            # === Attribute Information ===
            "attribute_name": None,
            "gt_attribute_value": None,
            "llm_attribute_value": None,
            "attribute_match": None,
            # === Excerpt Information ===
            "excerpt_index": None,
            "excerpt_text": None,
            "excerpt_confidence": None,
            "excerpt_similarity_score": None,
            # === Search Enhancement ===
            "excerpt_search_results": None,
            "excerpt_hallucination_risk": None,
            "excerpt_hallucination_justification": None,
            # === Attribute Metadata ===
            "attribute_reasoning": None,
            "attribute_overall_risk": None,
            "attribute_has_excerpts": False,
            # === Processing Metrics ===
            "deep_judgment_model_calls": 0,
            "deep_judgment_excerpt_retries": 0,
            "stages_completed": None,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
            "job_id": metadata.job_id,
        }

    # ========================================================================
    # Core Data Access
    # ========================================================================

    def get_results_with_judgment(self) -> list[VerificationResult]:
        """
        Get only results that have deep judgment data.

        Returns:
            List of results where deep judgment was performed
        """
        return [r for r in self.results if r.deep_judgment is not None and r.deep_judgment.deep_judgment_performed]

    def get_extracted_excerpts(
        self,
        question_id: str | None = None,
        attribute_name: str | None = None,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """
        Get extracted excerpts from all results.

        Args:
            question_id: Optional question ID to filter by
            attribute_name: Optional attribute name to filter by

        Returns:
            Dictionary mapping result identifiers to excerpts
            Format: {result_id: {attribute_name: [excerpt_dicts]}}
            Each excerpt dict contains: text, confidence, similarity_score, etc.
        """
        excerpts = {}
        for result in self.get_results_with_judgment():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.deep_judgment and result.deep_judgment.extracted_excerpts:
                result_id = self._get_result_id(result)
                excerpt_data = result.deep_judgment.extracted_excerpts

                # Filter by attribute name if specified
                if attribute_name:
                    if attribute_name in excerpt_data:
                        excerpts[result_id] = {attribute_name: excerpt_data[attribute_name]}
                else:
                    excerpts[result_id] = excerpt_data

        return excerpts

    def get_attribute_reasoning(
        self,
        question_id: str | None = None,
        attribute_name: str | None = None,
    ) -> dict[str, dict[str, str]]:
        """
        Get reasoning traces for attributes.

        Args:
            question_id: Optional question ID to filter by
            attribute_name: Optional attribute name to filter by

        Returns:
            Dictionary mapping result identifiers to reasoning traces
            Format: {result_id: {attribute_name: reasoning_text}}
        """
        reasoning = {}
        for result in self.get_results_with_judgment():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.deep_judgment and result.deep_judgment.attribute_reasoning:
                result_id = self._get_result_id(result)
                reasoning_data = result.deep_judgment.attribute_reasoning

                # Filter by attribute name if specified
                if attribute_name:
                    if attribute_name in reasoning_data:
                        reasoning[result_id] = {attribute_name: reasoning_data[attribute_name]}
                else:
                    reasoning[result_id] = reasoning_data

        return reasoning

    def get_hallucination_risks(
        self,
        question_id: str | None = None,
        attribute_name: str | None = None,
    ) -> dict[str, dict[str, str]]:
        """
        Get hallucination risk assessments.

        Args:
            question_id: Optional question ID to filter by
            attribute_name: Optional attribute name to filter by

        Returns:
            Dictionary mapping result identifiers to risk assessments
            Format: {result_id: {attribute_name: risk_level}}
            Risk levels: "none", "low", "medium", "high"
        """
        risks = {}
        for result in self.get_results_with_judgment():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.deep_judgment and result.deep_judgment.hallucination_risk_assessment:
                result_id = self._get_result_id(result)
                risk_data = result.deep_judgment.hallucination_risk_assessment

                # Filter by attribute name if specified
                if attribute_name:
                    if attribute_name in risk_data:
                        risks[result_id] = {attribute_name: risk_data[attribute_name]}
                else:
                    risks[result_id] = risk_data

        return risks

    def get_attributes_without_excerpts(self, question_id: str | None = None) -> dict[str, list[str]]:
        """
        Get attributes that had no excerpts found.

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary mapping result identifiers to lists of attribute names
            Format: {result_id: [attribute_name, ...]}
        """
        no_excerpts = {}
        for result in self.get_results_with_judgment():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.deep_judgment and result.deep_judgment.attributes_without_excerpts:
                result_id = self._get_result_id(result)
                no_excerpts[result_id] = result.deep_judgment.attributes_without_excerpts

        return no_excerpts

    def get_search_results(self, question_id: str | None = None) -> dict[str, str | None]:
        """
        Get search validation results (external evidence).

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary mapping result identifiers to search results text
            Format: {result_id: search_text | None}
        """
        search_data: dict[str, Any] = {}
        for result in self.get_results_with_judgment():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.deep_judgment and result.deep_judgment.deep_judgment_search_enabled:
                # result_id = self._get_result_id(result)
                # TODO: search_results attribute needs to be added to VerificationResultDeepJudgment schema
                # search_data[result_id] = result.deep_judgment.search_results
                pass

        return search_data

    def get_processing_metrics(self, question_id: str | None = None) -> dict[str, dict[str, Any]]:
        """
        Get deep judgment processing metrics.

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary mapping result identifiers to processing metrics
            Format: {result_id: {
                "stages_completed": list[str],
                "model_calls": int,
                "retry_count": int,
                "search_enabled": bool
            }}
        """
        metrics = {}
        for result in self.get_results_with_judgment():
            # Filter by question if specified
            if question_id and result.metadata.question_id != question_id:
                continue

            if result.deep_judgment:
                result_id = self._get_result_id(result)
                metrics[result_id] = {
                    "stages_completed": result.deep_judgment.deep_judgment_stages_completed or [],
                    "model_calls": result.deep_judgment.deep_judgment_model_calls or 0,
                    "retry_count": result.deep_judgment.deep_judgment_excerpt_retry_count or 0,
                    "search_enabled": result.deep_judgment.deep_judgment_search_enabled or False,
                }

        return metrics

    # ========================================================================
    # Aggregation
    # ========================================================================

    def aggregate_excerpt_counts(
        self,
        attribute_name: str | None = None,
        by: str = "question",
        **kwargs: Any,  # noqa: ARG002
    ) -> dict[str, dict[str, float]] | dict[str, float]:
        """
        Aggregate excerpt counts per attribute.

        Args:
            attribute_name: Optional specific attribute to analyze
            by: Grouping axis ("question", "model", "replicate")
            **kwargs: Additional parameters

        Returns:
            If attribute_name specified: {group_id: mean_excerpt_count}
            If attribute_name None: {group_id: {attribute_name: mean_count}}
        """
        # Group results by specified axis
        grouped = self._group_results(by)

        if attribute_name:
            # Single attribute analysis
            counts: dict[str, float] = {}
            for group_id, group_results in grouped.items():
                excerpt_counts = []

                for result in group_results:
                    if (
                        result.deep_judgment
                        and result.deep_judgment.extracted_excerpts
                        and attribute_name in result.deep_judgment.extracted_excerpts
                    ):
                        excerpt_list = result.deep_judgment.extracted_excerpts[attribute_name]
                        excerpt_counts.append(len(excerpt_list))

                if excerpt_counts:
                    counts[group_id] = sum(excerpt_counts) / len(excerpt_counts)
                else:
                    counts[group_id] = 0.0

            return counts
        else:
            # All attributes analysis
            all_counts: dict[str, dict[str, float]] = {}
            for group_id, group_results in grouped.items():
                attribute_counts: dict[str, list[int]] = {}

                for result in group_results:
                    if result.deep_judgment and result.deep_judgment.extracted_excerpts:
                        for (
                            attr,
                            excerpts,
                        ) in result.deep_judgment.extracted_excerpts.items():
                            if attr not in attribute_counts:
                                attribute_counts[attr] = []
                            attribute_counts[attr].append(len(excerpts))

                # Calculate mean count for each attribute
                all_counts[group_id] = {}
                for attr, counts_list in attribute_counts.items():
                    if counts_list:
                        all_counts[group_id][attr] = sum(counts_list) / len(counts_list)
                    else:
                        all_counts[group_id][attr] = 0.0

            return all_counts

    def aggregate_hallucination_risk_distribution(
        self,
        attribute_name: str | None = None,
        by: str = "question",
        **kwargs: Any,  # noqa: ARG002
    ) -> dict[str, dict[str, dict[str, float]]] | dict[str, dict[str, float]]:
        """
        Calculate distribution of hallucination risk levels.

        Args:
            attribute_name: Optional specific attribute to analyze
            by: Grouping axis ("question", "model", "replicate")
            **kwargs: Additional parameters

        Returns:
            If attribute_name specified: {group_id: {risk_level: proportion}}
            If attribute_name None: {group_id: {attribute: {risk_level: proportion}}}
        """
        # Group results by specified axis
        grouped = self._group_results(by)

        if attribute_name:
            # Single attribute analysis
            distributions: dict[str, dict[str, float]] = {}
            for group_id, group_results in grouped.items():
                risk_levels = []

                for result in group_results:
                    if (
                        result.deep_judgment
                        and result.deep_judgment.hallucination_risk_assessment
                        and attribute_name in result.deep_judgment.hallucination_risk_assessment
                    ):
                        risk_level = result.deep_judgment.hallucination_risk_assessment[attribute_name]
                        risk_levels.append(risk_level)

                # Calculate proportions
                if risk_levels:
                    total = len(risk_levels)
                    distributions[group_id] = {
                        "none": risk_levels.count("none") / total,
                        "low": risk_levels.count("low") / total,
                        "medium": risk_levels.count("medium") / total,
                        "high": risk_levels.count("high") / total,
                    }
                else:
                    distributions[group_id] = {
                        "none": 0.0,
                        "low": 0.0,
                        "medium": 0.0,
                        "high": 0.0,
                    }

            return distributions
        else:
            # All attributes analysis
            all_distributions: dict[str, dict[str, dict[str, float]]] = {}
            for group_id, group_results in grouped.items():
                attribute_risks: dict[str, list[str]] = {}

                for result in group_results:
                    if result.deep_judgment and result.deep_judgment.hallucination_risk_assessment:
                        for (
                            attr,
                            risk,
                        ) in result.deep_judgment.hallucination_risk_assessment.items():
                            if attr not in attribute_risks:
                                attribute_risks[attr] = []
                            attribute_risks[attr].append(risk)

                # Calculate proportions for each attribute
                all_distributions[group_id] = {}
                for attr, risk_levels in attribute_risks.items():
                    if risk_levels:
                        total = len(risk_levels)
                        all_distributions[group_id][attr] = {
                            "none": risk_levels.count("none") / total,
                            "low": risk_levels.count("low") / total,
                            "medium": risk_levels.count("medium") / total,
                            "high": risk_levels.count("high") / total,
                        }
                    else:
                        all_distributions[group_id][attr] = {
                            "none": 0.0,
                            "low": 0.0,
                            "medium": 0.0,
                            "high": 0.0,
                        }

            return all_distributions

    def aggregate_model_calls(
        self,
        strategy: str = "mean",
        by: str = "question",
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Aggregate number of model calls using specified strategy.

        Args:
            strategy: Aggregation strategy name (e.g., "mean", "median", "mode")
            by: Grouping axis ("question", "model", "replicate")
            **kwargs: Additional parameters for the aggregator

        Returns:
            Dictionary mapping group identifiers to aggregated model call counts
            Format: {group_id: aggregated_count}
        """
        aggregator = self._aggregator_registry.get(strategy)

        # Group results by specified axis
        grouped = self._group_results(by)

        # Aggregate model calls for each group
        aggregated = {}
        for group_id, group_results in grouped.items():
            model_calls = []

            for result in group_results:
                if result.deep_judgment and result.deep_judgment.deep_judgment_model_calls:
                    model_calls.append(result.deep_judgment.deep_judgment_model_calls)

            aggregated[group_id] = aggregator.aggregate(model_calls, **kwargs)

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
        with_search_only: bool = False,
    ) -> JudgmentResults:
        """
        Filter results by various criteria.

        Args:
            question_ids: Filter by question IDs
            answering_models: Filter by answering model names
            parsing_models: Filter by parsing model names
            replicates: Filter by replicate numbers
            with_search_only: Only include results with search validation enabled

        Returns:
            New JudgmentResults instance with filtered results
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

        if with_search_only:
            filtered = [r for r in filtered if r.deep_judgment and r.deep_judgment.deep_judgment_search_enabled]

        return JudgmentResults(results=filtered)

    def group_by_question(self) -> dict[str, JudgmentResults]:
        """
        Group results by question ID.

        Returns:
            Dictionary mapping question IDs to JudgmentResults instances
        """
        grouped: dict[str, list[VerificationResult]] = {}
        for result in self.results:
            qid = result.metadata.question_id
            if qid not in grouped:
                grouped[qid] = []
            grouped[qid].append(result)

        return {qid: JudgmentResults(results=results) for qid, results in grouped.items()}

    def group_by_model(self) -> dict[str, JudgmentResults]:
        """
        Group results by answering model.

        Returns:
            Dictionary mapping model names to JudgmentResults instances
        """
        grouped: dict[str, list[VerificationResult]] = {}
        for result in self.results:
            model = result.metadata.answering_model
            if model not in grouped:
                grouped[model] = []
            grouped[model].append(result)

        return {model: JudgmentResults(results=results) for model, results in grouped.items()}

    # ========================================================================
    # Summary Statistics
    # ========================================================================

    def get_judgment_summary(self) -> dict[str, Any]:
        """
        Get summary statistics for deep judgment.

        Returns:
            Dictionary with summary statistics:
            - num_results: Number of results with deep judgment data
            - num_with_excerpts: Number with at least one excerpt
            - num_with_search: Number with search validation
            - mean_model_calls: Average number of model calls
            - mean_retry_count: Average number of retries
            - num_questions: Number of unique questions
            - attributes: List of all attribute names processed
        """
        results_with_judgment = self.get_results_with_judgment()

        num_with_excerpts = 0
        num_with_search = 0
        total_model_calls = 0
        total_retry_count = 0
        questions: set[str] = set()
        all_attributes: set[str] = set()

        for result in results_with_judgment:
            questions.add(result.metadata.question_id)

            if result.deep_judgment:
                # Check for excerpts
                if result.deep_judgment.extracted_excerpts:
                    has_excerpts = any(
                        len(excerpts) > 0 for excerpts in result.deep_judgment.extracted_excerpts.values()
                    )
                    if has_excerpts:
                        num_with_excerpts += 1

                    # Collect attribute names
                    all_attributes.update(result.deep_judgment.extracted_excerpts.keys())

                # Check for search
                if result.deep_judgment.deep_judgment_search_enabled:
                    num_with_search += 1

                # Aggregate metrics
                if result.deep_judgment.deep_judgment_model_calls:
                    total_model_calls += result.deep_judgment.deep_judgment_model_calls

                if result.deep_judgment.deep_judgment_excerpt_retry_count:
                    total_retry_count += result.deep_judgment.deep_judgment_excerpt_retry_count

        num_results = len(results_with_judgment)
        mean_model_calls = total_model_calls / num_results if num_results > 0 else 0.0
        mean_retry_count = total_retry_count / num_results if num_results > 0 else 0.0

        return {
            "num_results": num_results,
            "num_with_excerpts": num_with_excerpts,
            "num_with_search": num_with_search,
            "mean_model_calls": mean_model_calls,
            "mean_retry_count": mean_retry_count,
            "num_questions": len(questions),
            "attributes": sorted(all_attributes),
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

        for result in self.get_results_with_judgment():
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
