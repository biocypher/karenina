"""
JudgmentDataFrameBuilder for converting deep judgment results to pandas DataFrames.

This module extracts DataFrame conversion logic from JudgmentResults,
following the same pattern as TemplateDataFrameBuilder and RubricDataFrameBuilder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..verification import VerificationResult


class JudgmentDataFrameBuilder:
    """
    Builder for converting deep judgment results to pandas DataFrames.

    Handles the conversion of VerificationResult objects with deep judgment data
    into structured DataFrames with one row per (attribute × excerpt) combination.

    This class is typically accessed via the `dataframe_builder` property of
    JudgmentResults rather than instantiated directly.

    Attributes:
        results: List of VerificationResult objects to convert
    """

    def __init__(self, results: list[VerificationResult]) -> None:
        """
        Initialize the builder with verification results.

        Args:
            results: List of VerificationResult objects containing deep judgment data
        """
        self.results = results

    def build_dataframe(self) -> Any:
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
            11. Execution Metadata: execution_time, timestamp, run_name

        Returns:
            pandas.DataFrame: Exploded DataFrame with one row per (attribute × excerpt)

        Example:
            >>> builder = JudgmentDataFrameBuilder(results)
            >>> df = builder.build_dataframe()
            >>> # Filter to specific attribute
            >>> location_df = df[df['attribute_name'] == 'location']
        """
        import pandas as pd

        rows = []

        for result_idx, result in enumerate(self.results):
            if result.deep_judgment is None or not result.deep_judgment.deep_judgment_performed:
                # No deep judgment data - create single row with minimal info
                row = self._create_empty_judgment_row(result)
                row["result_index"] = result_idx
                rows.append(row)
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
                            row = self._create_judgment_row(
                                result,
                                attribute_name,
                                gt_value,
                                llm_value,
                                idx,
                                excerpt,
                                attribute_reasoning,
                                attribute_risk,
                            )
                            row["result_index"] = result_idx
                            rows.append(row)
                    else:
                        # No excerpts found for this attribute - create one row with None excerpt data
                        row = self._create_judgment_row(
                            result,
                            attribute_name,
                            gt_value,
                            llm_value,
                            None,
                            None,
                            attribute_reasoning,
                            attribute_risk,
                        )
                        row["result_index"] = result_idx
                        rows.append(row)
            else:
                # No extracted_excerpts - but check if we have hallucination_risk_assessment
                if result.deep_judgment.hallucination_risk_assessment:
                    # Create rows for each attribute in hallucination_risk_assessment
                    for attribute_name, attribute_risk in result.deep_judgment.hallucination_risk_assessment.items():
                        gt_value = parsed_gt.get(attribute_name) if parsed_gt else None
                        llm_value = parsed_llm.get(attribute_name) if parsed_llm else None

                        attribute_reasoning = None
                        if result.deep_judgment.attribute_reasoning:
                            attribute_reasoning = result.deep_judgment.attribute_reasoning.get(attribute_name)

                        row = self._create_judgment_row(
                            result,
                            attribute_name,
                            gt_value,
                            llm_value,
                            None,  # No excerpt index
                            None,  # No excerpt data
                            attribute_reasoning,
                            attribute_risk,
                        )
                        row["result_index"] = result_idx
                        rows.append(row)
                else:
                    # No extracted_excerpts and no hallucination_risk_assessment - minimal row
                    row = self._create_empty_judgment_row(result)
                    row["result_index"] = result_idx
                    rows.append(row)

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
            "replicate": metadata.replicate,
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
        }

    def _create_empty_judgment_row(self, result: VerificationResult) -> dict[str, Any]:
        """Create minimal DataFrame row when no judgment data exists."""
        metadata = result.metadata
        template = result.template
        deep_judgment = result.deep_judgment

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
            "replicate": metadata.replicate,
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
            "deep_judgment_model_calls": deep_judgment.deep_judgment_model_calls if deep_judgment else 0,
            "deep_judgment_excerpt_retries": deep_judgment.deep_judgment_excerpt_retry_count if deep_judgment else 0,
            "stages_completed": deep_judgment.deep_judgment_stages_completed if deep_judgment else None,
            # === Execution Metadata ===
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name,
        }
