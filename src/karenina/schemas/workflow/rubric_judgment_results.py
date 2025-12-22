"""
RubricJudgmentResults class for detailed deep judgment rubric analysis.

This module provides excerpt-level explosion of deep judgment rubric evaluation
results, enabling fine-grained analysis of extracted evidence and reasoning.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .verification import VerificationResult


class RubricJudgmentResults(BaseModel):
    """
    Detailed deep judgment rubric evaluation results with excerpt-level explosion.

    This class provides a specialized view of deep judgment rubric evaluation
    results, exploding each trait's excerpts into separate rows for fine-grained
    analysis. Each row represents one (trait × excerpt) combination, with full
    metadata about the excerpt, reasoning, and evaluation process.

    Explosion Strategy:
        - Traits WITH excerpts: One row per excerpt (N rows for N excerpts)
        - Traits WITHOUT excerpts: Single row with excerpt fields set to None

    Use Cases:
        - Analyzing individual excerpts and their confidence scores
        - Examining excerpt-level hallucination risks
        - Studying retry patterns and validation failures
        - Deep-diving into reasoning process for specific traits
        - Understanding model behavior during excerpt extraction

    Comparison with RubricResults:
        - RubricResults: One row per trait (standard export, backward compatible)
        - RubricJudgmentResults: Multiple rows per trait (excerpt explosion, detailed)

    Attributes:
        results: List of VerificationResult objects containing deep judgment data
    """

    results: list[VerificationResult] = Field(
        description="List of verification results containing deep judgment rubric data"
    )

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, results: list[VerificationResult], **data: Any) -> None:
        """
        Initialize RubricJudgmentResults with verification results.

        Args:
            results: List of VerificationResult objects
            **data: Additional pydantic model data
        """
        super().__init__(results=results, **data)

    # ========================================================================
    # DataFrame Conversion
    # ========================================================================

    def to_dataframe(self) -> Any:
        """
        Convert deep judgment rubric results to pandas DataFrame with excerpt explosion.

        Creates multiple rows per trait (one per excerpt) for comprehensive analysis.
        Traits without excerpts get a single row with excerpt fields set to None.

        Column Categories:
            1. Status: completed_without_errors, error
            2. Identification: question_id, template_id, question_text, keywords, replicate
            3. Model Config: answering_model, parsing_model, system_prompts
            4. Trait Identification: trait_name, trait_score, trait_type
            5. Trait Reasoning: trait_reasoning
            6. Trait Metadata:
               - trait_model_calls: Number of LLM calls for this trait
               - trait_excerpt_retries: Number of excerpt extraction retries
               - trait_stages_completed: JSON list of completed stages
               - trait_validation_failed: Whether validation failed
               - trait_had_excerpts: Whether trait used excerpt extraction
            7. Excerpt Data (exploded, per-excerpt):
               - excerpt_index: Index of this excerpt (0-based)
               - excerpt_text: Verbatim text excerpt
               - excerpt_confidence: Confidence score (if available)
               - excerpt_similarity_score: Fuzzy match similarity score
            8. Excerpt Hallucination (if search enabled):
               - excerpt_hallucination_risk: Risk level assessment
               - excerpt_hallucination_justification: Reasoning for risk
               - excerpt_search_results: JSON of search validation results
            9. Execution Metadata: execution_time, timestamp, run_name

        Returns:
            pandas.DataFrame: Exploded DataFrame with one row per (trait × excerpt)

        Example:
            >>> result_set = benchmark.run_verification_new(config)["question_id"]
            >>> rubric_judgments = result_set.get_rubric_judgments_results()
            >>> df = rubric_judgments.to_dataframe()
            >>> # Analyze per-excerpt confidence
            >>> df.groupby("trait_name")["excerpt_confidence"].mean()
            >>> # Examine traits with validation failures
            >>> df[df["trait_validation_failed"] == True][["trait_name", "trait_excerpt_retries"]]
            >>> # Study hallucination risks
            >>> df[df["excerpt_hallucination_risk"].notna()][["excerpt_text", "excerpt_hallucination_risk"]]
        """
        import pandas as pd

        rows = []

        for result in self.results:
            # Only process results with deep judgment rubric evaluation
            if (
                not hasattr(result, "deep_judgment_rubric")
                or result.deep_judgment_rubric is None
                or not result.deep_judgment_rubric.deep_judgment_rubric_performed
            ):
                continue

            rubric_dj = result.deep_judgment_rubric
            metadata = result.metadata

            # Process each deep judgment trait
            if rubric_dj.deep_judgment_rubric_scores:
                for trait_name, trait_score in rubric_dj.deep_judgment_rubric_scores.items():
                    # Get trait-level data
                    trait_reasoning = (
                        rubric_dj.rubric_trait_reasoning.get(trait_name) if rubric_dj.rubric_trait_reasoning else None
                    )
                    trait_excerpts = (
                        rubric_dj.extracted_rubric_excerpts.get(trait_name, [])
                        if rubric_dj.extracted_rubric_excerpts
                        else []
                    )
                    trait_metadata = rubric_dj.trait_metadata.get(trait_name, {}) if rubric_dj.trait_metadata else {}
                    trait_hallucination_risk = (
                        rubric_dj.rubric_hallucination_risk_assessment.get(trait_name)
                        if rubric_dj.rubric_hallucination_risk_assessment
                        else None
                    )

                    # Determine trait type (score vs binary)
                    is_binary = isinstance(trait_score, bool)
                    score_type = "llm_binary" if is_binary else "llm_score"

                    # Base row data (shared across all excerpts for this trait)
                    base_row = {
                        # === Status ===
                        "completed_without_errors": metadata.completed_without_errors,
                        "error": metadata.error,
                        # === Identification Metadata ===
                        "question_id": metadata.question_id,
                        "template_id": metadata.template_id,
                        "question_text": metadata.question_text,
                        "keywords": metadata.keywords,
                        "replicate": metadata.replicate,
                        # === Model Configuration ===
                        "answering_model": metadata.answering_model,
                        "parsing_model": metadata.parsing_model,
                        "answering_system_prompt": metadata.answering_system_prompt,
                        "parsing_system_prompt": metadata.parsing_system_prompt,
                        # === Trait Identification ===
                        "trait_name": trait_name,
                        "trait_score": trait_score,
                        "trait_type": score_type,
                        # === Trait Reasoning ===
                        "trait_reasoning": trait_reasoning,
                        # === Trait Metadata ===
                        "trait_model_calls": trait_metadata.get("model_calls", 0),
                        "trait_excerpt_retries": trait_metadata.get("excerpt_retry_count", 0),
                        "trait_stages_completed": json.dumps(trait_metadata.get("stages_completed", [])),
                        "trait_validation_failed": trait_metadata.get("excerpt_validation_failed", False),
                        # === Execution Metadata ===
                        "execution_time": metadata.execution_time,
                        "timestamp": metadata.timestamp,
                        "run_name": metadata.run_name,
                    }

                    if trait_excerpts:
                        # Explode: One row per excerpt
                        for excerpt_index, excerpt in enumerate(trait_excerpts):
                            row = base_row.copy()
                            row["trait_had_excerpts"] = True

                            # === Excerpt Data ===
                            row["excerpt_index"] = excerpt_index
                            row["excerpt_text"] = excerpt.get("text")  # Changed from "excerpt_text" to "text"
                            row["excerpt_confidence"] = excerpt.get("confidence")
                            row["excerpt_similarity_score"] = excerpt.get("similarity_score")

                            # === Excerpt Hallucination ===
                            if trait_hallucination_risk and excerpt.get("hallucination_assessment"):
                                hallucination_data = excerpt["hallucination_assessment"]
                                row["excerpt_hallucination_risk"] = hallucination_data.get("risk_level")
                                row["excerpt_hallucination_justification"] = hallucination_data.get("justification")
                                row["excerpt_search_results"] = json.dumps(hallucination_data.get("search_results", []))
                            else:
                                row["excerpt_hallucination_risk"] = None
                                row["excerpt_hallucination_justification"] = None
                                row["excerpt_search_results"] = None

                            rows.append(row)
                    else:
                        # No excerpts: Single row with None values
                        row = base_row.copy()
                        row["trait_had_excerpts"] = False

                        # === Excerpt Data (None) ===
                        row["excerpt_index"] = None
                        row["excerpt_text"] = None
                        row["excerpt_confidence"] = None
                        row["excerpt_similarity_score"] = None

                        # === Excerpt Hallucination (None) ===
                        row["excerpt_hallucination_risk"] = None
                        row["excerpt_hallucination_justification"] = None
                        row["excerpt_search_results"] = None

                        rows.append(row)

        return pd.DataFrame(rows)

    # ========================================================================
    # Data Access
    # ========================================================================

    def get_results_with_deep_judgment(self) -> list[VerificationResult]:
        """
        Get only results that have deep judgment rubric data.

        Returns:
            List of results where deep judgment rubric evaluation was performed
        """
        return [
            r
            for r in self.results
            if hasattr(r, "deep_judgment_rubric")
            and r.deep_judgment_rubric is not None
            and r.deep_judgment_rubric.deep_judgment_rubric_performed
        ]

    def get_excerpt_count_summary(self) -> dict[str, dict[str, int]]:
        """
        Get summary of excerpt counts per trait across all results.

        Returns:
            Dictionary mapping trait names to excerpt statistics
            Format: {trait_name: {"total_excerpts": N, "avg_excerpts": M, "results_with_trait": K}}

        Example:
            >>> summary = rubric_judgments.get_excerpt_count_summary()
            >>> print(summary["clarity"])
            {'total_excerpts': 35, 'avg_excerpts': 7.0, 'results_with_trait': 5}
        """
        trait_stats: dict[str, dict[str, Any]] = {}

        for result in self.get_results_with_deep_judgment():
            rubric_dj = result.deep_judgment_rubric
            if not rubric_dj or not rubric_dj.deep_judgment_rubric_scores:
                continue

            for trait_name in rubric_dj.deep_judgment_rubric_scores:
                if trait_name not in trait_stats:
                    trait_stats[trait_name] = {"total_excerpts": 0, "result_count": 0}

                trait_stats[trait_name]["result_count"] += 1

                excerpts = (
                    rubric_dj.extracted_rubric_excerpts.get(trait_name, [])
                    if rubric_dj.extracted_rubric_excerpts
                    else []
                )
                trait_stats[trait_name]["total_excerpts"] += len(excerpts)

        # Calculate averages
        summary = {}
        for trait_name, stats in trait_stats.items():
            avg_excerpts = stats["total_excerpts"] / stats["result_count"] if stats["result_count"] > 0 else 0
            summary[trait_name] = {
                "total_excerpts": stats["total_excerpts"],
                "avg_excerpts": round(avg_excerpts, 2),
                "results_with_trait": stats["result_count"],
            }

        return summary

    def get_retry_summary(self) -> dict[str, dict[str, Any]]:
        """
        Get summary of excerpt extraction retries per trait.

        Returns:
            Dictionary mapping trait names to retry statistics
            Format: {trait_name: {"total_retries": N, "max_retries": M, "traits_with_retries": K}}

        Example:
            >>> summary = rubric_judgments.get_retry_summary()
            >>> print(summary["specificity"])
            {'total_retries': 8, 'max_retries': 3, 'traits_with_retries': 4}
        """
        trait_stats: dict[str, dict[str, Any]] = {}

        for result in self.get_results_with_deep_judgment():
            rubric_dj = result.deep_judgment_rubric
            if not rubric_dj or not rubric_dj.trait_metadata:
                continue

            for trait_name, metadata in rubric_dj.trait_metadata.items():
                retry_count = metadata.get("excerpt_retry_count", 0)

                if trait_name not in trait_stats:
                    trait_stats[trait_name] = {
                        "total_retries": 0,
                        "max_retries": 0,
                        "traits_with_retries": 0,
                    }

                trait_stats[trait_name]["total_retries"] += retry_count
                trait_stats[trait_name]["max_retries"] = max(trait_stats[trait_name]["max_retries"], retry_count)
                if retry_count > 0:
                    trait_stats[trait_name]["traits_with_retries"] += 1

        return trait_stats

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
