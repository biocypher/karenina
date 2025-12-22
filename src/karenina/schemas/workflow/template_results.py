"""
TemplateResults class for collating and analyzing template verification results.

This module provides functionality to work with template verification results,
including embedding checks, regex validation, abstention detection, and MCP metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .aggregation import AggregatorRegistry, create_default_registry

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
        # Create default aggregator registry
        self._aggregator_registry: AggregatorRegistry = create_default_registry()

    # ========================================================================
    # DataFrame Conversion
    # ========================================================================

    def to_dataframe(self) -> Any:
        """
        Convert template verification results to pandas DataFrame.

        Creates one row per parsed field comparison (ground truth vs LLM response).
        Each field in the parsed responses gets its own row with field-level matching.

        Column ordering:
            1. Status: completed_without_errors, error, recursion_limit_reached
            2. Identification: question_id, template_id, question_text, keywords,
               replicate, answering_mcp_servers
            3. Model Config: answering_model, parsing_model, system_prompts
            4. Template Response: raw_llm_response
            5. Field Comparison: field_name, gt_value, llm_value, field_match, field_type
            6. Verification Checks: embedding, abstention, regex
            7. Execution Metadata: execution_time, timestamp, run_name

        Returns:
            pandas.DataFrame: Exploded DataFrame with one row per field comparison

        Example:
            >>> template_results = result_set.get_template_results()
            >>> df = template_results.to_dataframe()
            >>> # Filter and aggregate using pandas
            >>> match_rate = df.groupby('question_id')['field_match'].mean()
        """
        import pandas as pd

        rows = []

        for result_idx, result in enumerate(self.results):
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

        return pd.DataFrame(rows)

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
            # If one exists but not the other, they don't match
            field_match = False if gt_exists != llm_exists else self._compare_values(gt_value, llm_value)

        # Determine field type
        field_type = None
        if gt_value is not None:
            field_type = type(gt_value).__name__
        elif llm_value is not None:
            field_type = type(llm_value).__name__

        row = {
            # === Status (FIRST COLUMN) ===
            "completed_without_errors": metadata.completed_without_errors,
            "error": metadata.error,
            "recursion_limit_reached": template.recursion_limit_reached if template else False,
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": metadata.replicate,
            "answering_mcp_servers": template.answering_mcp_servers if template else None,
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
            "field_type": field_type,
            # === Verification Checks ===
            "verify_result": template.verify_result if template else None,
            "embedding_check_performed": template.embedding_check_performed if template else False,
            "embedding_similarity_score": template.embedding_similarity_score if template else None,
            "embedding_model_used": template.embedding_model_used if template else None,
            "embedding_override_applied": template.embedding_override_applied if template else False,
            "abstention_check_performed": template.abstention_check_performed if template else False,
            "abstention_detected": template.abstention_detected if template else None,
            "abstention_reasoning": template.abstention_reasoning if template else None,
            "abstention_override_applied": template.abstention_override_applied if template else False,
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
            "completed_without_errors": metadata.completed_without_errors,
            "error": metadata.error,
            "recursion_limit_reached": False,
            # === Identification Metadata ===
            "question_id": metadata.question_id,
            "template_id": metadata.template_id,
            "question_text": metadata.question_text,
            "keywords": metadata.keywords,
            "replicate": metadata.replicate,
            "answering_mcp_servers": None,
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
            "embedding_check_performed": False,
            "embedding_similarity_score": None,
            "embedding_model_used": None,
            "embedding_override_applied": False,
            "abstention_check_performed": False,
            "abstention_detected": None,
            "abstention_reasoning": None,
            "abstention_override_applied": False,
            "regex_validations_performed": False,
            "regex_overall_success": None,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
        }

    def _compare_values(self, value1: Any, value2: Any) -> bool:
        """
        Compare two values for equality, handling various data types.

        Args:
            value1: First value to compare
            value2: Second value to compare

        Returns:
            True if values are equal, False otherwise
        """
        # Handle None cases
        if value1 is None and value2 is None:
            return True
        if value1 is None or value2 is None:
            return False

        # Direct equality check (works for primitives, lists, dicts)
        try:
            return bool(value1 == value2)
        except Exception:  # noqa: BLE001
            # If comparison fails (e.g., unhashable types), return False
            return False

    def to_regex_dataframe(self) -> Any:
        """
        Convert regex validation results to pandas DataFrame.

        Creates one row per regex pattern tested.
        Provides detailed information about pattern matches, extraction, and positions.

        Column ordering:
            1. Status: completed_without_errors, error
            2. Identification: question_id, template_id, replicate
            3. Model Config: answering_model, parsing_model
            4. Regex Details: pattern_name, pattern_regex, matched, extracted_value, match positions
            5. Validation Context: raw_llm_response
            6. Execution Metadata: timestamp, run_name

        Returns:
            pandas.DataFrame: Exploded DataFrame with one row per regex pattern

        Example:
            >>> template_results = result_set.get_templates()
            >>> df = template_results.to_regex_dataframe()
            >>> # Filter to failed patterns
            >>> failed = df[~df['matched']]
            >>> # Analyze pattern success rates
            >>> success_rates = df.groupby('pattern_name')['matched'].mean()
        """
        import pandas as pd

        rows = []

        for result in self.results:
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
                        "completed_without_errors": metadata.completed_without_errors,
                        "error": metadata.error,
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

    def to_usage_dataframe(self, totals_only: bool = False) -> Any:
        """
        Convert token usage data to pandas DataFrame.

        By default, creates one row per usage stage (exploded).
        With totals_only=True, creates one row per verification with aggregated totals.

        Column ordering:
            1. Status: completed_without_errors, error
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
            >>> template_results = result_set.get_templates()
            >>> # Get per-stage breakdown
            >>> df_stages = template_results.to_usage_dataframe()
            >>> # Get totals only
            >>> df_totals = template_results.to_usage_dataframe(totals_only=True)
            >>> # Analyze token usage by stage
            >>> by_stage = df_stages.groupby('usage_stage')['total_tokens'].sum()
        """
        import pandas as pd

        rows = []

        for result in self.results:
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

                # Agent metrics (only relevant for answer_generation stage typically)
                iterations = agent_metrics.get("iterations")
                tool_calls = agent_metrics.get("tool_calls")
                tools_used = agent_metrics.get("tools_used")
                suspected_failures = agent_metrics.get("suspect_failed_tool_calls")

                rows.append(
                    {
                        # === Status ===
                        "completed_without_errors": metadata.completed_without_errors,
                        "error": metadata.error,
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
        by: str = "question_id",
        **kwargs: Any,  # noqa: ARG002
    ) -> dict[str, float]:
        """
        Calculate template verification pass rate by group.

        Uses pandas DataFrame groupby for efficient aggregation.

        Args:
            by: Column name to group by (e.g., "question_id", "answering_model", "replicate")
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping group identifiers to pass rates (0.0 to 1.0)
            Format: {group_id: pass_rate}

        Example:
            >>> results.aggregate_pass_rate(by="question_id")
            {'q1': 0.8, 'q2': 1.0}
        """

        # Get DataFrame
        df = self.to_dataframe()

        if len(df) == 0:
            return {}

        # Deduplicate: Field-exploded DataFrame has multiple rows per result
        # Keep first occurrence of each unique result (by result_index)
        df_dedup = df.drop_duplicates(subset=["result_index"]).copy()

        # Filter to only rows where verify_result is not None
        df_filtered = df_dedup[df_dedup["verify_result"].notna()].copy()

        if len(df_filtered) == 0:
            return {}

        # Calculate pass rate per group
        pass_rates = df_filtered.groupby(by)["verify_result"].mean()

        return dict(pass_rates.to_dict())

    def aggregate_embedding_scores(
        self,
        strategy: str = "mean",
        by: str = "question_id",
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Aggregate embedding similarity scores using specified strategy.

        Uses pandas DataFrame groupby for efficient aggregation.

        Args:
            strategy: Aggregation strategy name (e.g., "mean", "median")
            by: Column name to group by (e.g., "question_id", "answering_model", "replicate")
            **kwargs: Additional parameters for the aggregator

        Returns:
            Dictionary mapping group identifiers to aggregated scores
            Format: {group_id: aggregated_score}

        Example:
            >>> results.aggregate_embedding_scores(strategy="mean", by="question_id")
            {'q1': 0.92, 'q2': 0.85}
        """

        aggregator = self._aggregator_registry.get(strategy)

        # Get DataFrame
        df = self.to_dataframe()

        if len(df) == 0:
            return {}

        # Deduplicate: Field-exploded DataFrame has multiple rows per result
        df_dedup = df.drop_duplicates(subset=["result_index"]).copy()

        # Filter to only rows where embedding check was performed and score exists
        df_filtered = df_dedup[
            (df_dedup["embedding_check_performed"] == True)  # noqa: E712
            & (df_dedup["embedding_similarity_score"].notna())
        ].copy()

        if len(df_filtered) == 0:
            return {}

        # Aggregate scores per group
        aggregated = df_filtered.groupby(by)["embedding_similarity_score"].agg(
            lambda s: aggregator.aggregate(s, **kwargs)
        )

        return dict(aggregated.to_dict())

    def aggregate_regex_success_rate(
        self,
        pattern_name: str | None = None,
        by: str = "question_id",
        **kwargs: Any,  # noqa: ARG002
    ) -> dict[str, dict[str, float]] | dict[str, float]:
        """
        Calculate regex validation success rate.

        Uses pandas DataFrame groupby for efficient aggregation.

        Args:
            pattern_name: Optional specific pattern to analyze
            by: Column name to group by (e.g., "question_id", "answering_model", "replicate")
            **kwargs: Additional parameters

        Returns:
            If pattern_name specified: {group_id: success_rate}
            If pattern_name None: {group_id: {pattern_name: success_rate}}

        Example:
            >>> # Single pattern
            >>> results.aggregate_regex_success_rate(pattern_name="email", by="question_id")
            {'q1': 1.0, 'q2': 0.5}
            >>> # All patterns
            >>> results.aggregate_regex_success_rate(by="question_id")
            {'q1': {'email': 1.0, 'url': 0.5}, 'q2': {'email': 0.8}}
        """

        # Get regex DataFrame
        df = self.to_regex_dataframe()

        if len(df) == 0:
            return {} if pattern_name else {}

        if pattern_name:
            # Single pattern analysis
            df_filtered = df[df["pattern_name"] == pattern_name].copy()

            if len(df_filtered) == 0:
                return {}

            # Calculate success rate per group
            success_rates = df_filtered.groupby(by)["matched"].mean()
            return dict(success_rates.to_dict())
        else:
            # All patterns analysis
            # Group by both by-column and pattern_name, then calculate mean
            success_rates = df.groupby([by, "pattern_name"])["matched"].mean()

            # Convert to nested dictionary format
            result: dict[str, dict[str, float]] = {}
            for (group_id, pname), rate in success_rates.items():
                if group_id not in result:
                    result[group_id] = {}
                result[group_id][pname] = float(rate)

            return result

    def aggregate_abstention_rate(
        self,
        by: str = "question_id",
        **kwargs: Any,  # noqa: ARG002
    ) -> dict[str, float]:
        """
        Calculate abstention detection rate by group.

        Uses pandas DataFrame groupby for efficient aggregation.

        Args:
            by: Column name to group by (e.g., "question_id", "answering_model", "replicate")
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping group identifiers to abstention rates (0.0 to 1.0)
            Format: {group_id: abstention_rate}

        Example:
            >>> results.aggregate_abstention_rate(by="question_id")
            {'q1': 0.0, 'q2': 0.25}
        """

        # Get DataFrame
        df = self.to_dataframe()

        if len(df) == 0:
            return {}

        # Deduplicate: Field-exploded DataFrame has multiple rows per result
        df_dedup = df.drop_duplicates(subset=["result_index"]).copy()

        # Filter to only rows where abstention check was performed
        df_filtered = df_dedup[df_dedup["abstention_check_performed"] == True].copy()  # noqa: E712

        if len(df_filtered) == 0:
            return {}

        # Calculate abstention rate per group
        abstention_rates = df_filtered.groupby(by)["abstention_detected"].mean()

        return dict(abstention_rates.to_dict())

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
            filtered = [r for r in filtered if r.metadata.replicate in replicates]

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
        Get the unique identifier for a result.

        Args:
            result: VerificationResult to identify

        Returns:
            The result's deterministic hash ID
        """
        return result.metadata.result_id

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
