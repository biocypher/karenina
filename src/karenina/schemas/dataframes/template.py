"""
DataFrame builders for template verification results.

This module provides functionality to convert template verification results
to pandas DataFrames for analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..verification import VerificationResult
    from ..verification.result_components import VerificationResultMetadata


def _failure_columns(metadata: VerificationResultMetadata) -> dict[str, Any]:
    """Return the unified ``success``/``failure_*``/``caveats`` columns.

    Args:
        metadata: The verification result metadata to translate.

    Returns:
        Mapping with six keys: ``success`` (bool), ``failure_category``,
        ``failure_group``, ``failure_stage``, ``failure_reason`` (each a
        ``str | None``), and ``caveats`` (a comma-joined string, possibly
        empty). Values are ``None`` on the failure-specific keys when the
        result is a pass.
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


class TemplateDataFrameBuilder:
    """
    Builder class for converting template verification results to DataFrames.

    This class handles the conversion of VerificationResult objects to various
    DataFrame formats for analysis:
    - Field-level comparison DataFrame
    - Regex validation DataFrame
    - Token usage DataFrame

    Attributes:
        results: List of VerificationResult objects containing template data
    """

    def __init__(self, results: list[VerificationResult]) -> None:
        """
        Initialize TemplateDataFrameBuilder with verification results.

        Args:
            results: List of VerificationResult objects
        """
        self._results = results

    def build_field_dataframe(self) -> Any:
        """
        Convert template verification results to pandas DataFrame.

        Creates one row per parsed field comparison (ground truth vs LLM response).
        Each field in the parsed responses gets its own row with field-level matching.

        Column ordering:
            1. Status: success, failure_category, failure_group, failure_stage,
               failure_reason, caveats, recursion_limit_reached
            2. Identification: question_id, template_id, question_text, keywords,
               replicate, answering_mcp_servers
            3. Scenario: scenario_id, scenario_node, scenario_turn, scenario_path
            4. Model Config: answering_model, parsing_model, system_prompts
            5. Template Response: raw_llm_response
            6. Field Comparison: field_name, gt_value, llm_value, field_match,
               field_type
            7. Verification Checks: verify_result, verify_granular_result,
               embedding, abstention, sufficiency, regex
            8. Execution Metadata: execution_time, timestamp, run_name

        Returns:
            pandas.DataFrame: Exploded DataFrame with one row per field comparison

        Example:
            >>> builder = TemplateDataFrameBuilder(results)
            >>> df = builder.build_field_dataframe()
            >>> # Filter and aggregate using pandas
            >>> match_rate = df.groupby('question_id')['field_match'].mean()
        """
        import pandas as pd

        rows = []

        for result_idx, result in enumerate(self._results):
            if result.template is None:
                # No template data - create single row with error info
                row = self._create_empty_row(result)
                row["result_index"] = result_idx
                rows.append(row)
                continue

            # Get parsed responses
            parsed_gt = result.template.parsed_gt_response or {}
            parsed_llm = result.template.parsed_llm_response or {}

            # Get all unique field names from both responses
            all_fields = set(parsed_gt.keys()) | set(parsed_llm.keys())

            if not all_fields:
                # No fields to compare - create single row
                row = self._create_field_row(result, None, None, None, None, None)
                row["result_index"] = result_idx
                rows.append(row)
            else:
                # Create one row per field
                for field_name in sorted(all_fields):
                    # Check if key exists (not just getting value)
                    gt_exists = field_name in parsed_gt
                    llm_exists = field_name in parsed_llm

                    gt_value = parsed_gt.get(field_name) if gt_exists else None
                    llm_value = parsed_llm.get(field_name) if llm_exists else None

                    row = self._create_field_row(result, field_name, gt_value, llm_value, gt_exists, llm_exists)
                    row["result_index"] = result_idx
                    rows.append(row)

        df = pd.DataFrame(rows)
        if "replicate" in df.columns:
            df["replicate"] = df["replicate"].astype(pd.Int64Dtype())
        return df

    def _create_field_row(
        self,
        result: VerificationResult,
        field_name: str | None,
        gt_value: Any | None,
        llm_value: Any | None,
        gt_exists: bool | None = None,
        llm_exists: bool | None = None,
    ) -> dict[str, Any]:
        """
        Create a single DataFrame row for a field comparison.

        Args:
            result: VerificationResult to extract data from
            field_name: Name of the field being compared
            gt_value: Ground truth value for this field
            llm_value: LLM response value for this field
            gt_exists: Whether the field exists in ground truth response
            llm_exists: Whether the field exists in LLM response

        Returns:
            Dictionary representing one DataFrame row
        """
        template = result.template
        metadata = result.metadata

        # Determine field match status
        field_match = None
        if field_name is not None:
            # Use stored primitive verification results if available (issue 152)
            if result.template and result.template.field_results and field_name in result.template.field_results:
                field_match = result.template.field_results[field_name]
            else:
                # Fallback to naive comparison for backward compatibility
                field_match = False if gt_exists != llm_exists else self._compare_values(gt_value, llm_value)

        # Per-field graded score (continuous companion to field_match). 1.0/0.0
        # for non-graded primitives; partial credit for NumericGraded. None when
        # not available (e.g. legacy results without field_scores).
        field_score = None
        if (
            field_name is not None
            and result.template
            and result.template.field_scores
            and field_name in result.template.field_scores
        ):
            field_score = result.template.field_scores[field_name]

        # Determine field type
        field_type = None
        if gt_value is not None:
            field_type = type(gt_value).__name__
        elif llm_value is not None:
            field_type = type(llm_value).__name__

        row = {
            # === Status (FIRST COLUMN) ===
            **_failure_columns(metadata),
            "recursion_limit_reached": template.recursion_limit_reached if template else False,
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": metadata.replicate,
            "answering_mcp_servers": template.answering_mcp_servers if template else None,
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
            # === Template Response ===
            "raw_llm_response": template.raw_llm_response if template else None,
            # === Field Comparison (EXPLODED DIMENSION) ===
            "field_name": field_name,
            "gt_value": gt_value,
            "llm_value": llm_value,
            "field_match": field_match,
            "field_score": field_score,
            "field_type": field_type,
            # === Verification Checks ===
            "verify_result": template.verify_result if template else None,
            "verify_granular_result": template.verify_granular_result if template else None,
            "embedding_check_performed": template.embedding_check_performed if template else False,
            "embedding_similarity_score": template.embedding_similarity_score if template else None,
            "embedding_model_used": template.embedding_model_used if template else None,
            "embedding_override_applied": template.embedding_override_applied if template else False,
            "abstention_check_performed": template.abstention_check_performed if template else False,
            "abstention_detected": template.abstention_detected if template else None,
            "abstention_reasoning": template.abstention_reasoning if template else None,
            "abstention_override_applied": template.abstention_override_applied if template else False,
            "sufficiency_check_performed": template.sufficiency_check_performed if template else False,
            "sufficiency_detected": template.sufficiency_detected if template else None,
            "sufficiency_reasoning": template.sufficiency_reasoning if template else None,
            "sufficiency_override_applied": template.sufficiency_override_applied if template else False,
            "regex_validations_performed": template.regex_validations_performed if template else False,
            "regex_overall_success": template.regex_overall_success if template else None,
            # === Execution Metadata (AT END) ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
        }

        return row

    def _create_empty_row(self, result: VerificationResult) -> dict[str, Any]:
        """
        Create an empty DataFrame row for results without template data.

        Args:
            result: VerificationResult with no template data

        Returns:
            Dictionary representing one DataFrame row with minimal data
        """
        metadata = result.metadata

        return {
            # === Status ===
            **_failure_columns(metadata),
            "recursion_limit_reached": False,
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": metadata.replicate,
            "answering_mcp_servers": None,
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
            # === Template Response ===
            "raw_llm_response": None,
            # === Field Comparison ===
            "field_name": None,
            "gt_value": None,
            "llm_value": None,
            "field_match": None,
            "field_type": None,
            # === Verification Checks ===
            "verify_result": None,
            "verify_granular_result": None,
            "embedding_check_performed": False,
            "embedding_similarity_score": None,
            "embedding_model_used": None,
            "embedding_override_applied": False,
            "abstention_check_performed": False,
            "abstention_detected": None,
            "abstention_reasoning": None,
            "abstention_override_applied": False,
            "sufficiency_check_performed": False,
            "sufficiency_detected": None,
            "sufficiency_reasoning": None,
            "sufficiency_override_applied": False,
            "regex_validations_performed": False,
            "regex_overall_success": None,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
        }

    def _compare_values(self, value1: Any, value2: Any) -> bool | None:
        """
        Compare two values for equality, handling various data types.

        When both values are None, the comparison is not meaningful (the field
        was absent or unpopulated on both sides), so None is returned to signal
        "not comparable". This aligns with the judgment builder, which also
        treats both-None as not comparable.

        Args:
            value1: First value to compare
            value2: Second value to compare

        Returns:
            True if values are equal, False if unequal, None if both values
            are None (not comparable).
        """
        # Handle None cases
        if value1 is None and value2 is None:
            return None
        if value1 is None or value2 is None:
            return False

        # Direct equality check (works for primitives, lists, dicts)
        try:
            return bool(value1 == value2)
        except Exception:  # noqa: BLE001
            # If comparison fails (e.g., unhashable types), return False
            return False

    def build_regex_dataframe(self) -> Any:
        """
        Convert regex validation results to pandas DataFrame.

        Creates one row per regex pattern tested.
        Provides detailed information about pattern matches, extraction, and positions.

        Column ordering:
            1. Status: success, failure_category, failure_group, failure_stage,
               failure_reason, caveats
            2. Identification: question_id, template_id, replicate
            3. Model Config: answering_model, parsing_model
            4. Regex Details: pattern_name, pattern_regex, matched, extracted_value, match positions
            5. Validation Context: raw_llm_response
            6. Execution Metadata: timestamp, run_name

        Returns:
            pandas.DataFrame: Exploded DataFrame with one row per regex pattern

        Example:
            >>> builder = TemplateDataFrameBuilder(results)
            >>> df = builder.build_regex_dataframe()
            >>> # Filter to failed patterns
            >>> failed = df[~df['matched']]
            >>> # Analyze pattern success rates
            >>> success_rates = df.groupby('pattern_name')['matched'].mean()
        """
        import pandas as pd

        rows = []

        for result in self._results:
            if result.template is None or not result.template.regex_validations_performed:
                # No regex validation data - skip this result
                continue

            template = result.template
            metadata = result.metadata

            # Get regex data
            validation_results = template.regex_validation_results or {}
            validation_details = template.regex_validation_details or {}
            extraction_results = template.regex_extraction_results or {}

            if not validation_results:
                # No patterns tested - skip
                continue

            # Create one row per pattern
            for pattern_name, matched in validation_results.items():
                # Get detailed match info
                details = validation_details.get(pattern_name, {})
                extracted = extraction_results.get(pattern_name)

                # Extract match details
                match_start = details.get("match_start")
                match_end = details.get("match_end")
                full_match = details.get("full_match")
                pattern_regex = details.get("pattern") or details.get("regex")

                rows.append(
                    {
                        # === Status ===
                        **_failure_columns(metadata),
                        # === Identification ===
                        "question_id": metadata.question_id,
                        "template_id": metadata.template_id,
                        "replicate": metadata.replicate,
                        # === Model Configuration ===
                        "answering_model": metadata.answering_model,
                        "parsing_model": metadata.parsing_model,
                        # === Regex Details ===
                        "pattern_name": pattern_name,
                        "pattern_regex": pattern_regex,
                        "matched": matched,
                        "extracted_value": extracted,
                        "match_start": match_start,
                        "match_end": match_end,
                        "full_match": full_match,
                        # === Validation Context ===
                        "raw_llm_response": template.raw_llm_response,
                        # === Execution Metadata ===
                        "timestamp": metadata.timestamp,
                        "run_name": metadata.run_name,
                    }
                )

        return pd.DataFrame(rows)

    def build_usage_dataframe(self, totals_only: bool = False) -> Any:
        """
        Convert token usage data to pandas DataFrame.

        By default, creates one row per usage stage (exploded).
        With totals_only=True, creates one row per verification with aggregated totals.

        Column ordering:
            1. Status: success, failure_category, failure_group, failure_stage,
               failure_reason, caveats
            2. Identification: question_id, template_id, replicate
            3. Model Config: answering_model, parsing_model
            4. Usage Stage: usage_stage (excluded if totals_only=True)
            5. Token Counts: input_tokens, output_tokens, total_tokens
            6. Model Used: model_used
            7. Detailed Breakdowns: input_audio_tokens, cache_read_tokens, etc.
            8. Agent Metrics: agent_iterations, agent_tool_calls, etc.
            9. Execution Metadata: timestamp, run_name

        Args:
            totals_only: If True, only include "total" stage (one row per verification)
                        If False (default), explode by stage (excluding "total")

        Returns:
            pandas.DataFrame: Usage data, exploded by stage or totals only

        Example:
            >>> builder = TemplateDataFrameBuilder(results)
            >>> # Get per-stage breakdown
            >>> df_stages = builder.build_usage_dataframe()
            >>> # Get totals only
            >>> df_totals = builder.build_usage_dataframe(totals_only=True)
            >>> # Analyze token usage by stage
            >>> by_stage = df_stages.groupby('usage_stage')['total_tokens'].sum()
        """
        import pandas as pd

        rows = []

        for result in self._results:
            if result.template is None or result.template.usage_metadata is None:
                # No usage data - skip this result
                continue

            template = result.template
            metadata = result.metadata

            # Get agent metrics (apply to all stages)
            agent_metrics = template.agent_metrics or {}

            # Process usage data
            usage_metadata = template.usage_metadata or {}
            for stage, usage_data in usage_metadata.items():
                # Apply filtering based on totals_only parameter
                if totals_only and stage != "total":
                    continue
                if not totals_only and stage == "total":
                    continue

                # Extract token counts
                input_tokens = usage_data.get("input_tokens")
                output_tokens = usage_data.get("output_tokens")
                total_tokens = usage_data.get("total_tokens")
                model_used = usage_data.get("model")

                # Extract detailed token breakdowns
                input_details = usage_data.get("input_token_details", {}) or {}
                output_details = usage_data.get("output_token_details", {}) or {}

                input_audio = input_details.get("audio")
                cache_read = input_details.get("cache_read")
                output_audio = output_details.get("audio")
                reasoning_tokens = output_details.get("reasoning")

                # Agent metrics (only meaningful for answer_generation stage)
                is_answer_stage = stage == "answer_generation"
                iterations = agent_metrics.get("iterations") if is_answer_stage else None
                tool_calls = agent_metrics.get("tool_calls") if is_answer_stage else None
                tools_used = agent_metrics.get("tools_used") if is_answer_stage else None
                suspected_failures = agent_metrics.get("suspect_failed_tool_calls") if is_answer_stage else None

                rows.append(
                    {
                        # === Status ===
                        **_failure_columns(metadata),
                        # === Identification ===
                        "question_id": metadata.question_id,
                        "template_id": metadata.template_id,
                        "replicate": metadata.replicate,
                        # === Model Configuration ===
                        "answering_model": metadata.answering_model,
                        "parsing_model": metadata.parsing_model,
                        # === Usage Stage ===
                        "usage_stage": stage if not totals_only else None,
                        # === Token Counts ===
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        # === Model Used ===
                        "model_used": model_used,
                        # === Detailed Breakdowns ===
                        "input_audio_tokens": input_audio,
                        "input_cache_read_tokens": cache_read,
                        "output_audio_tokens": output_audio,
                        "output_reasoning_tokens": reasoning_tokens,
                        # === Agent Metrics ===
                        "agent_iterations": iterations,
                        "agent_tool_calls": tool_calls,
                        "agent_tools_used": tools_used,
                        "agent_suspected_failures": suspected_failures,
                        # === Execution Metadata ===
                        "timestamp": metadata.timestamp,
                        "run_name": metadata.run_name,
                    }
                )

        return pd.DataFrame(rows)
