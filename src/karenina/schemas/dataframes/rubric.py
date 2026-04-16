"""
DataFrame builders for rubric evaluation results.

This module provides functionality to convert rubric evaluation results
to pandas DataFrames for analysis.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..verification import VerificationResult, VerificationResultDeepJudgmentRubric
    from ..verification.result_components import VerificationResultMetadata


def _failure_columns(metadata: VerificationResultMetadata) -> dict[str, Any]:
    """Return the unified ``success``/``failure_*``/``caveats`` columns.

    Args:
        metadata: The verification result metadata to translate.

    Returns:
        Mapping with six keys: ``success`` (bool), ``failure_category``,
        ``failure_group``, ``failure_stage``, ``failure_reason`` (each a
        ``str | None``), and ``caveats`` (comma-joined, possibly empty).
    """
    failure = metadata.failure
    return {
        "success": failure is None,
        "failure_category": failure.category.value if failure else None,
        "failure_group": failure.group.value if failure else None,
        "failure_stage": failure.stage if failure else None,
        "failure_reason": failure.reason if failure else None,
        "caveats": ",".join(c.value for c in metadata.caveats),
    }


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
            "llm_score", "llm_binary", "llm_literal", "llm", "regex", "callable", "metric", "agentic", "all"
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
                - "agentic": Agentic traits (boolean or score)
                - "all": All trait types combined (default)

        Column ordering:
            1. Status: success, failure_category, failure_group, failure_stage,
               failure_reason, caveats
            2. Identification: question_id, template_id, question_text, keywords, replicate,
               scenario_id, scenario_node, scenario_turn, scenario_path
            3. Model Config: answering_model, parsing_model, system_prompts
            4. Rubric Evaluation Metadata: rubric_evaluation_performed, rubric_evaluation_strategy
            5. Rubric Data: trait_name, trait_score, trait_label, trait_type,
               evaluation_method, metric_name
            6. Confusion Matrix: confusion_tp, confusion_fp, confusion_fn, confusion_tn (for metrics only)
            7. Execution Metadata: execution_time, timestamp, run_name
            8. Deep Judgment (if include_deep_judgment=True):
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
        row_result_ids: list[str] = []  # Parallel list tracking which result produced each row

        for result in self._results:
            if result.rubric is None or not result.rubric.rubric_evaluation_performed:
                # No rubric data: skip entirely to avoid ghost rows with
                # trait_name=None that pollute len(df) and value_counts().
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
                    row_result_ids.append(result.metadata.result_id)

            # Process regex traits
            if trait_type in ("regex", "all") and result.rubric.regex_trait_scores:
                for trait_name, trait_score in result.rubric.regex_trait_scores.items():
                    rows.append(self._create_regex_trait_row(result, trait_name, trait_score))
                    row_result_ids.append(result.metadata.result_id)

            # Process callable traits
            if trait_type in ("callable", "all") and result.rubric.callable_trait_scores:
                for trait_name, trait_score in result.rubric.callable_trait_scores.items():
                    rows.append(self._create_callable_trait_row(result, trait_name, trait_score))
                    row_result_ids.append(result.metadata.result_id)

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
                        row_result_ids.append(result.metadata.result_id)

            # Process agentic traits
            if trait_type in ("agentic", "all") and result.rubric.agentic_trait_scores:
                for trait_name, agentic_score in result.rubric.agentic_trait_scores.items():
                    rows.append(self._create_agentic_trait_row(result, trait_name, agentic_score))
                    row_result_ids.append(result.metadata.result_id)

        df = pd.DataFrame(rows)
        if "replicate" in df.columns:
            df["replicate"] = df["replicate"].astype(pd.Int64Dtype())

        # Add dynamic rubric _skipped columns if any result has dynamic rubric data
        df = self._add_dynamic_rubric_skipped_columns(df, row_result_ids)

        # Reorder columns for consistent structure
        column_order = self._get_column_order(df)

        return df[column_order]

    def _add_dynamic_rubric_skipped_columns(self, df: Any, row_result_ids: list[str]) -> Any:
        """Add {trait_name}_skipped columns for dynamic rubric metadata.

        When at least one result in the dataset has dynamic rubric data
        (promoted or skipped traits), this method adds boolean companion
        columns indicating whether each dynamic trait was skipped (True),
        promoted (False), or not applicable (NaN).

        Args:
            df: The DataFrame to augment with _skipped columns.
            row_result_ids: Parallel list mapping each DataFrame row to
                its source result_id, built during row construction.

        Returns:
            The DataFrame, potentially with new _skipped columns appended.
        """
        import numpy as np

        if df.empty:
            return df

        # Collect all dynamic trait names and per-result status
        has_dynamic_data = False
        all_dynamic_traits: set[str] = set()

        for result in self._results:
            if result.rubric is None:
                continue
            if result.rubric.dynamic_rubric_skipped_traits:
                has_dynamic_data = True
                all_dynamic_traits.update(result.rubric.dynamic_rubric_skipped_traits.keys())
            if result.rubric.dynamic_rubric_promoted_traits:
                has_dynamic_data = True
                all_dynamic_traits.update(result.rubric.dynamic_rubric_promoted_traits)

        if not has_dynamic_data or not all_dynamic_traits:
            return df

        # Build a mapping from result_id to dynamic rubric status.
        # Each result maps to: trait_name -> True (skipped) / False (promoted)
        result_dynamic_status: dict[str, dict[str, bool]] = {}
        for result in self._results:
            if result.rubric is None:
                continue
            key = result.metadata.result_id
            status: dict[str, bool] = {}
            if result.rubric.dynamic_rubric_skipped_traits:
                for trait_name in result.rubric.dynamic_rubric_skipped_traits:
                    status[trait_name] = True
            if result.rubric.dynamic_rubric_promoted_traits:
                for trait_name in result.rubric.dynamic_rubric_promoted_traits:
                    status[trait_name] = False
            if status:
                result_dynamic_status[key] = status

        # Add _skipped columns for each dynamic trait (sorted for determinism)
        for trait_name in sorted(all_dynamic_traits):
            col_name = f"{trait_name}_skipped"
            values = []
            for rid in row_result_ids:
                rid_status = result_dynamic_status.get(rid)
                if rid_status is None or trait_name not in rid_status:
                    values.append(np.nan)
                else:
                    values.append(rid_status[trait_name])
            df[col_name] = values

        return df

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
            "success",
            "failure_category",
            "failure_group",
            "failure_stage",
            "failure_reason",
            "caveats",
            # Identification
            "question_id",
            "template_id",
            "question_text",
            "keywords",
            "replicate",
            # Scenario Metadata
            "scenario_id",
            "scenario_node",
            "scenario_turn",
            "scenario_path",
            # Model Config
            "answering_model",
            "parsing_model",
            "answering_system_prompt",
            "parsing_system_prompt",
            # Rubric Evaluation Metadata
            "rubric_evaluation_performed",
            "rubric_evaluation_strategy",
            # Rubric Data
            "trait_name",
            "trait_score",
            "trait_label",  # For literal kind LLM traits: class name (score is index)
            "trait_type",
            "evaluation_method",
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
            "trait_provenance",
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
            **_failure_columns(metadata),
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": metadata.replicate,
            # === Scenario Metadata ===
            "scenario_id": metadata.scenario_id,
            "scenario_node": metadata.scenario_node,
            "scenario_turn": metadata.scenario_turn,
            "scenario_path": metadata.scenario_path,
            # === Model Configuration ===
            "answering_model": metadata.answering_model,
            "parsing_model": metadata.parsing_model,
            "answering_system_prompt": metadata.answering_system_prompt,
            "parsing_system_prompt": metadata.parsing_system_prompt,
            # === Rubric Evaluation Metadata ===
            "rubric_evaluation_performed": result.rubric.rubric_evaluation_performed if result.rubric else False,
            "rubric_evaluation_strategy": result.rubric.rubric_evaluation_strategy if result.rubric else None,
            # === Rubric Data ===
            "trait_name": trait_name,
            "trait_type": score_type,
            "evaluation_method": "llm",
            "trait_score": trait_score,
            "trait_label": trait_label,  # Class name for literal kind traits, None otherwise
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
            "trait_provenance": (
                result.rubric.trait_provenance.get(trait_name)
                if result.rubric and result.rubric.trait_provenance
                else None
            ),
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
            **_failure_columns(metadata),
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": metadata.replicate,
            # === Scenario Metadata ===
            "scenario_id": metadata.scenario_id,
            "scenario_node": metadata.scenario_node,
            "scenario_turn": metadata.scenario_turn,
            "scenario_path": metadata.scenario_path,
            # === Model Configuration ===
            "answering_model": metadata.answering_model,
            "parsing_model": metadata.parsing_model,
            "answering_system_prompt": metadata.answering_system_prompt,
            "parsing_system_prompt": metadata.parsing_system_prompt,
            # === Rubric Evaluation Metadata ===
            "rubric_evaluation_performed": result.rubric.rubric_evaluation_performed if result.rubric else False,
            "rubric_evaluation_strategy": result.rubric.rubric_evaluation_strategy if result.rubric else None,
            # === Rubric Data ===
            "trait_name": trait_name,
            "trait_type": "regex",
            "evaluation_method": "regex",
            "trait_score": trait_score,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
            "trait_provenance": (
                result.rubric.trait_provenance.get(trait_name)
                if result.rubric and result.rubric.trait_provenance
                else None
            ),
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
            **_failure_columns(metadata),
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": metadata.replicate,
            # === Scenario Metadata ===
            "scenario_id": metadata.scenario_id,
            "scenario_node": metadata.scenario_node,
            "scenario_turn": metadata.scenario_turn,
            "scenario_path": metadata.scenario_path,
            # === Model Configuration ===
            "answering_model": metadata.answering_model,
            "parsing_model": metadata.parsing_model,
            "answering_system_prompt": metadata.answering_system_prompt,
            "parsing_system_prompt": metadata.parsing_system_prompt,
            # === Rubric Evaluation Metadata ===
            "rubric_evaluation_performed": result.rubric.rubric_evaluation_performed if result.rubric else False,
            "rubric_evaluation_strategy": result.rubric.rubric_evaluation_strategy if result.rubric else None,
            # === Rubric Data ===
            "trait_name": trait_name,
            "trait_type": "callable",
            "evaluation_method": "callable",
            "trait_score": trait_score,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
            "trait_provenance": (
                result.rubric.trait_provenance.get(trait_name)
                if result.rubric and result.rubric.trait_provenance
                else None
            ),
        }

    def _create_metric_trait_row(
        self,
        result: VerificationResult,
        trait_name: str,
        metric_name: str,
        metric_score: float,
        confusion_data: dict[str, list[str]] | None,
    ) -> dict[str, Any]:
        """Create DataFrame row for metric trait (EXPLODED by metric).

        Note on confusion data duplication:
            The per-trait confusion lists (tp, fp, fn, tn) are intentionally
            repeated across all metric rows for the same trait. For example,
            when a trait produces precision, recall, and f1 rows, all three
            rows carry the same confusion_tp/fp/fn/tn values. This is
            intentional denormalization for DataFrame usability: it allows
            filtering to any single metric row and still having the full
            confusion context without a join.

        Args:
            result: The verification result
            trait_name: Name of the trait
            metric_name: Name of the specific metric (e.g. "precision", "recall", "f1")
            metric_score: Computed score for this metric
            confusion_data: Optional per-trait confusion lists keyed by tp/fp/fn/tn
        """
        metadata = result.metadata

        return {
            # === Status ===
            **_failure_columns(metadata),
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": metadata.replicate,
            # === Scenario Metadata ===
            "scenario_id": metadata.scenario_id,
            "scenario_node": metadata.scenario_node,
            "scenario_turn": metadata.scenario_turn,
            "scenario_path": metadata.scenario_path,
            # === Model Configuration ===
            "answering_model": metadata.answering_model,
            "parsing_model": metadata.parsing_model,
            "answering_system_prompt": metadata.answering_system_prompt,
            "parsing_system_prompt": metadata.parsing_system_prompt,
            # === Rubric Evaluation Metadata ===
            "rubric_evaluation_performed": result.rubric.rubric_evaluation_performed if result.rubric else False,
            "rubric_evaluation_strategy": result.rubric.rubric_evaluation_strategy if result.rubric else None,
            # === Metric Trait Data (EXPLODED) ===
            "trait_name": trait_name,
            "trait_type": "metric",
            "evaluation_method": "metric",
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
            "trait_provenance": (
                result.rubric.trait_provenance.get(trait_name)
                if result.rubric and result.rubric.trait_provenance
                else None
            ),
        }

    def _create_agentic_trait_row(
        self,
        result: VerificationResult,
        trait_name: str,
        trait_score: int | bool | float | str | list[Any] | None,
    ) -> dict[str, Any]:
        """Create DataFrame row for agentic trait.

        Args:
            result: The verification result
            trait_name: Name of the trait
            trait_score: Score value (bool or int)
        """
        metadata = result.metadata

        return {
            # === Status ===
            **_failure_columns(metadata),
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": metadata.replicate,
            # === Scenario Metadata ===
            "scenario_id": metadata.scenario_id,
            "scenario_node": metadata.scenario_node,
            "scenario_turn": metadata.scenario_turn,
            "scenario_path": metadata.scenario_path,
            # === Model Configuration ===
            "answering_model": metadata.answering_model,
            "parsing_model": metadata.parsing_model,
            "answering_system_prompt": metadata.answering_system_prompt,
            "parsing_system_prompt": metadata.parsing_system_prompt,
            # === Rubric Evaluation Metadata ===
            "rubric_evaluation_performed": result.rubric.rubric_evaluation_performed if result.rubric else False,
            "rubric_evaluation_strategy": result.rubric.rubric_evaluation_strategy if result.rubric else None,
            # === Rubric Data ===
            "trait_name": trait_name,
            "trait_type": "agentic",
            "evaluation_method": "agentic",
            "trait_score": trait_score,
            "investigation_trace": (
                result.rubric.agentic_trait_investigation_traces.get(trait_name)
                if result.rubric and result.rubric.agentic_trait_investigation_traces
                else None
            ),
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
            "trait_provenance": (
                result.rubric.trait_provenance.get(trait_name)
                if result.rubric and result.rubric.trait_provenance
                else None
            ),
        }

    def _add_deep_judgment_columns(
        self,
        row: dict[str, Any],
        rubric_dj: VerificationResultDeepJudgmentRubric | None,
        trait_name: str,
    ) -> None:
        """Add deep judgment columns to a row.

        Deep judgment columns are intentionally added only to LLM trait rows.
        Stage 12 (deep_judgment_rubric) in the verification pipeline only
        evaluates LLM traits, so regex, callable, metric, and agentic trait
        rows never carry deep judgment data. If deep judgment is extended to
        other trait types in the future, the corresponding row creator methods
        would need the same _add_deep_judgment_columns call.

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

        # Add trait_hallucination_risk (extract overall_risk string from dict)
        risk_data = (
            rubric_dj.rubric_hallucination_risk_assessment.get(trait_name)
            if rubric_dj and rubric_dj.rubric_hallucination_risk_assessment
            else None
        )
        row["trait_hallucination_risk"] = risk_data.get("overall_risk") if isinstance(risk_data, dict) else risk_data
