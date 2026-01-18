"""
DataFrame builders for rubric evaluation results.

This module provides functionality to convert rubric evaluation results
to pandas DataFrames for analysis.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .verification import VerificationResult, VerificationResultDeepJudgmentRubric


class RubricDataFrameBuilder:
    """
    Builder class for converting rubric evaluation results to DataFrames.

    This class handles the conversion of VerificationResult objects to pandas
    DataFrames for rubric analysis. It supports all trait types (LLM, regex,
    callable, metric) and optional deep judgment columns.

    Attributes:
        results: List of VerificationResult objects containing rubric data
        include_deep_judgment: Whether to include deep judgment columns
    """

    def __init__(self, results: list[VerificationResult], include_deep_judgment: bool = False) -> None:
        """
        Initialize RubricDataFrameBuilder with verification results.

        Args:
            results: List of VerificationResult objects
            include_deep_judgment: Whether to include deep judgment columns (default: False)
        """
        self._results = results
        self._include_deep_judgment = include_deep_judgment

    def build_dataframe(
        self,
        trait_type: Literal[
            "llm_score", "llm_binary", "llm_literal", "llm", "regex", "callable", "metric", "all"
        ] = "all",
    ) -> Any:
        """
        Convert rubric evaluation results to pandas DataFrame.

        Creates one row per trait (for llm/regex/callable) or per metric (for metric traits).
        Supports filtering by trait type or returning all traits combined.

        Args:
            trait_type: Type of traits to include
                - "llm_score": LLM traits with 1-5 scale scores
                - "llm_binary": LLM traits with boolean scores
                - "llm_literal": LLM traits with literal kind (categorical classification)
                - "llm": All LLM traits (score, binary, and literal)
                - "regex": Regex traits (boolean)
                - "callable": Callable traits (boolean or score)
                - "metric": Metric traits (precision, recall, f1) - EXPLODED by metric
                - "all": All trait types combined (default)

        Column ordering:
            1. Status: completed_without_errors, error
            2. Identification: question_id, template_id, question_text, keywords, replicate
            3. Model Config: answering_model, parsing_model, system_prompts
            4. Rubric Data: trait_name, trait_score, trait_label, trait_type, metric_name
            5. Confusion Matrix: confusion_tp, confusion_fp, confusion_fn, confusion_tn (for metrics only)
            6. Execution Metadata: execution_time, timestamp, run_name
            7. Deep Judgment (if include_deep_judgment=True):
               - trait_reasoning: Reasoning text for the trait score
               - trait_excerpts: JSON-serialized list of excerpts
               - trait_hallucination_risk: Hallucination risk assessment

        Note on literal kind traits:
            For literal kind LLM traits, `trait_score` contains the integer index (0 to N-1)
            and `trait_label` contains the human-readable class name. This allows numeric
            aggregation on scores while preserving the categorical label for display.
            Error state is indicated by score=-1 with label containing the invalid value.

        Returns:
            pandas.DataFrame: Exploded DataFrame with one row per trait/metric

        Example:
            >>> builder = RubricDataFrameBuilder(results)
            >>> df = builder.build_dataframe(trait_type="llm_score")
            >>> avg_scores = df[df['trait_name'] == 'clarity'].groupby('question_id')['trait_score'].mean()

            >>> # For literal traits, access labels for display
            >>> df_literal = builder.build_dataframe(trait_type="llm_literal")
            >>> df_literal[['trait_name', 'trait_score', 'trait_label']].head()
        """
        import pandas as pd

        rows = []

        for result in self._results:
            if result.rubric is None or not result.rubric.rubric_evaluation_performed:
                # No rubric data - create single row with minimal info
                rows.append(self._create_empty_rubric_row(result))
                continue

            # Process LLM traits
            if (
                trait_type in ("llm_score", "llm_binary", "llm_literal", "llm", "all")
                and result.rubric.llm_trait_scores
            ):
                # Get labels for literal kind traits (if any)
                llm_trait_labels = result.rubric.get_llm_trait_labels()

                for trait_name, trait_score in result.rubric.llm_trait_scores.items():
                    # Determine trait type based on value and presence in labels
                    is_binary = isinstance(trait_score, bool)
                    is_literal = trait_name in llm_trait_labels

                    if is_binary:
                        score_type = "llm_binary"
                        trait_label = None
                    elif is_literal:
                        score_type = "llm_literal"
                        trait_label = llm_trait_labels[trait_name]
                    else:
                        score_type = "llm_score"
                        trait_label = None

                    # Filter by requested type
                    if trait_type == "llm_score" and (is_binary or is_literal):
                        continue
                    if trait_type == "llm_binary" and not is_binary:
                        continue
                    if trait_type == "llm_literal" and not is_literal:
                        continue

                    rows.append(self._create_llm_trait_row(result, trait_name, trait_score, score_type, trait_label))

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

        df = pd.DataFrame(rows)

        # Reorder columns for consistent structure
        column_order = self._get_column_order(df)

        return df[column_order]

    def _get_column_order(self, df: Any) -> list[str]:
        """
        Get the desired column order for the DataFrame.

        Args:
            df: The DataFrame to reorder columns for

        Returns:
            List of column names in desired order
        """
        # Define desired column order
        desired_order = [
            # Status
            "completed_without_errors",
            "error",
            # Identification
            "question_id",
            "template_id",
            "question_text",
            "keywords",
            "replicate",
            # Model Config
            "answering_model",
            "parsing_model",
            "answering_system_prompt",
            "parsing_system_prompt",
            # Rubric Data
            "trait_name",
            "trait_score",
            "trait_label",  # For literal kind LLM traits: class name (score is index)
            "trait_type",
            "metric_name",
            # Confusion Matrix (for metric traits)
            "confusion_tp",
            "confusion_fp",
            "confusion_fn",
            "confusion_tn",
            # Execution Metadata
            "execution_time",
            "timestamp",
            "run_name",
            # Deep Judgment (if included)
            "trait_reasoning",
            "trait_excerpts",
            "trait_hallucination_risk",
        ]

        # Only include columns that exist in the DataFrame
        column_order = [col for col in desired_order if col in df.columns]

        # Columns to explicitly exclude from output
        excluded_columns = {"job_id"}

        # Add any columns that weren't in our desired order (shouldn't happen, but defensive)
        for col in df.columns:
            if col not in column_order and col not in excluded_columns:
                column_order.append(col)

        return column_order

    def _create_llm_trait_row(
        self,
        result: VerificationResult,
        trait_name: str,
        trait_score: int | bool,
        score_type: str,
        trait_label: str | None = None,
    ) -> dict[str, Any]:
        """Create DataFrame row for LLM trait.

        Args:
            result: The verification result
            trait_name: Name of the trait
            trait_score: Score value (bool for binary, int for score/literal)
            score_type: Type of LLM trait ("llm_binary", "llm_score", or "llm_literal")
            trait_label: For literal kind traits, the class name (human-readable label).
                         Score is the index, label is the class name.
        """
        metadata = result.metadata

        row: dict[str, Any] = {
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
            # === Rubric Data ===
            "trait_name": trait_name,
            "trait_type": score_type,
            "trait_score": trait_score,
            "trait_label": trait_label,  # Class name for literal kind traits, None otherwise
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
        }

        # Add deep judgment columns if requested
        if self._include_deep_judgment:
            self._add_deep_judgment_columns(row, result.deep_judgment_rubric, trait_name)

        return row

    def _create_regex_trait_row(
        self,
        result: VerificationResult,
        trait_name: str,
        trait_score: bool,
    ) -> dict[str, Any]:
        """Create DataFrame row for regex trait."""
        metadata = result.metadata

        return {
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
            # === Rubric Data ===
            "trait_name": trait_name,
            "trait_type": "regex",
            "trait_score": trait_score,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
        }

    def _create_callable_trait_row(
        self,
        result: VerificationResult,
        trait_name: str,
        trait_score: bool | int,
    ) -> dict[str, Any]:
        """Create DataFrame row for callable trait."""
        metadata = result.metadata

        return {
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
            # === Rubric Data ===
            "trait_name": trait_name,
            "trait_type": "callable",
            "trait_score": trait_score,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
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

        return {
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
            # === Metric Trait Data (EXPLODED) ===
            "trait_name": trait_name,
            "trait_type": "metric",
            "metric_name": metric_name,
            "trait_score": metric_score,
            # === Confusion Matrix Metadata ===
            "confusion_tp": confusion_data.get("tp") if confusion_data else None,
            "confusion_fp": confusion_data.get("fp") if confusion_data else None,
            "confusion_fn": confusion_data.get("fn") if confusion_data else None,
            "confusion_tn": confusion_data.get("tn") if confusion_data else None,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
        }

    def _create_empty_rubric_row(self, result: VerificationResult) -> dict[str, Any]:
        """Create empty DataFrame row for results without rubric data."""
        metadata = result.metadata

        return {
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
            # === Rubric Data (None) ===
            "trait_name": None,
            "trait_score": None,
            "trait_type": None,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
        }

    def _add_deep_judgment_columns(
        self,
        row: dict[str, Any],
        rubric_dj: VerificationResultDeepJudgmentRubric | None,
        trait_name: str,
    ) -> None:
        """
        Add deep judgment columns to a row.

        Args:
            row: The row dictionary to add columns to
            rubric_dj: Deep judgment rubric data (may be None)
            trait_name: Name of the trait
        """
        # Add trait_reasoning
        row["trait_reasoning"] = (
            rubric_dj.rubric_trait_reasoning.get(trait_name) if rubric_dj and rubric_dj.rubric_trait_reasoning else None
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
