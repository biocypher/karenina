"""Metric trait evaluation for rubric assessment.

This module implements the MetricTraitEvaluator class, which handles
evaluation of MetricRubricTrait instances using confusion matrix analysis.

Metric traits compute precision, recall, F1, specificity, and accuracy
by categorizing answer content into true/false positives/negatives based
on TP and TN instructions defined in the trait.

Two evaluation modes are supported:
- tp_only: Only TP instructions provided; computes precision, recall, F1
- full_matrix: Both TP and TN instructions; computes all metrics including specificity

All LLM calls use LLMPort.with_structured_output() for consistent backend abstraction.
"""

import json
import logging
import re
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from .....ports import LLMPort, PortCapabilities
from .....schemas.domain import MetricRubricTrait
from ...prompts import PromptAssembler, PromptTask
from ...prompts.rubric.metric_trait import MetricTraitPromptBuilder

if TYPE_CHECKING:
    from .....schemas.config import ModelConfig
    from .....schemas.verification import PromptConfig

logger = logging.getLogger(__name__)


class MetricTraitEvaluator:
    """
    Evaluates metric rubric traits using LLM-based confusion matrix analysis.

    This evaluator handles MetricRubricTrait evaluation by:
    1. Prompting an LLM to categorize answer content into TP/FN/TN/FP buckets
    2. Optionally deduplicating extracted items
    3. Computing requested metrics (precision, recall, F1, etc.)

    All LLM calls use LLMPort.with_structured_output() for consistent
    backend abstraction.

    Example usage:
        evaluator = MetricTraitEvaluator(llm, model_config=config)
        confusion_lists, metrics, usage = evaluator.evaluate_metric_traits(
            question="What diseases affect the lungs?",
            answer="Asthma and bronchitis are common lung diseases.",
            metric_traits=[trait]
        )
    """

    def __init__(
        self,
        llm: LLMPort,
        *,
        model_config: "ModelConfig",
        prompt_config: "PromptConfig | None" = None,
    ):
        """
        Initialize the metric trait evaluator.

        Args:
            llm: LLMPort adapter for LLM operations.
            model_config: Model configuration for reference.
            prompt_config: Optional per-task-type user instructions for prompt assembly.
        """
        self.llm = llm
        self._model_config = model_config
        self._prompt_config = prompt_config
        self._prompt_builder = MetricTraitPromptBuilder()

    def evaluate_metric_traits(
        self, question: str, answer: str, metric_traits: list[MetricRubricTrait]
    ) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, float]], list[dict[str, Any]]]:
        """
        Evaluate metric traits and return confusion lists and computed metrics.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            metric_traits: List of metric traits to evaluate

        Returns:
            Tuple of (confusion_lists, metrics, usage_metadata_list) where:
            - confusion_lists: {trait_name: {tp: [...], tn: [...], fp: [...], fn: [...]}}
            - metrics: {trait_name: {precision: 0.85, recall: 0.92, ...}}
            - usage_metadata_list: List of usage metadata dicts from LLM calls

        Raises:
            Exception: If evaluation fails for all traits
        """
        confusion_lists: dict[str, dict[str, list[str]]] = {}
        metrics: dict[str, dict[str, float]] = {}
        usage_metadata_list: list[dict[str, Any]] = []

        for trait in metric_traits:
            try:
                trait_confusion, trait_metrics, usage_metadata = self._evaluate_single_metric_trait(
                    question, answer, trait
                )
                confusion_lists[trait.name] = trait_confusion
                metrics[trait.name] = trait_metrics
                if usage_metadata:
                    usage_metadata_list.append(usage_metadata)
            except Exception as e:
                logger.warning(f"Failed to evaluate metric trait '{trait.name}': {e}")
                # Store empty results for failed traits
                confusion_lists[trait.name] = {"tp": [], "tn": [], "fp": [], "fn": []}
                metrics[trait.name] = {}

        return confusion_lists, metrics, usage_metadata_list

    def _evaluate_single_metric_trait(
        self, question: str, answer: str, trait: MetricRubricTrait
    ) -> tuple[dict[str, list[str]], dict[str, float], dict[str, Any]]:
        """
        Evaluate a single metric trait using LLMPort.with_structured_output().

        Args:
            question: The original question
            answer: The answer to evaluate
            trait: The metric trait to evaluate

        Returns:
            Tuple of (confusion_lists, metrics, usage_metadata)
        """
        from .....schemas.workflow.rubric_outputs import ConfusionMatrixOutput

        # Build prompt
        system_prompt = self._prompt_builder.build_system_prompt()
        user_prompt = self._prompt_builder.build_user_prompt(question, answer, trait)

        # Build instruction_context for adapter instructions (format-specific content)
        instruction_context: dict[str, object] = {
            "json_schema": ConfusionMatrixOutput.model_json_schema(),
            "output_format_hint": '{"tp": [...], "fn": [...], "fp": [...]}',
            "example_json": '{"tp": ["asthma", "bronchitis"], "fn": ["pneumonia"], "fp": ["emphysema"]}',
        }

        # Assemble messages via PromptAssembler (applies adapter + user instructions)
        user_instructions = (
            self._prompt_config.get_for_task(PromptTask.RUBRIC_METRIC_TRAIT.value) if self._prompt_config else None
        )
        assembler = PromptAssembler(
            task=PromptTask.RUBRIC_METRIC_TRAIT,
            interface=self._model_config.interface,
            capabilities=PortCapabilities(),
        )
        messages = assembler.assemble(
            system_text=system_prompt,
            user_text=user_prompt,
            user_instructions=user_instructions,
            instruction_context=instruction_context,
        )

        # Use LLMPort.with_structured_output() for parsing
        structured_llm = self.llm.with_structured_output(ConfusionMatrixOutput)
        response = structured_llm.invoke(messages)

        # Extract usage metadata
        usage_metadata = asdict(response.usage) if response.usage else {}

        # Extract parsed result
        parsed_result = response.raw

        confusion_lists = {
            "tp": parsed_result.tp,
            "fn": parsed_result.fn,
            "fp": parsed_result.fp,
            "tn": parsed_result.tn,
        }

        # Apply deduplication if requested
        if trait.repeated_extraction:
            confusion_lists = self._deduplicate_confusion_lists(confusion_lists)

        # Compute metrics
        computed_metrics = self._compute_metrics(
            confusion_lists["tp"], confusion_lists["tn"], confusion_lists["fp"], confusion_lists["fn"], trait.metrics
        )

        return confusion_lists, computed_metrics, usage_metadata

    def _parse_metric_trait_response(self, response: str, trait: MetricRubricTrait) -> dict[str, list[str]]:
        """
        Parse the LLM response to extract confusion lists.

        Args:
            response: Raw LLM response
            trait: The metric trait being evaluated

        Returns:
            Dictionary with keys {tp, tn, fp, fn} and list values
        """
        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in metric trait response: {response[:200]}")

        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in metric trait response: {e}") from e

        # Initialize all buckets with empty lists
        confusion_lists: dict[str, list[str]] = {"tp": [], "tn": [], "fp": [], "fn": []}

        # Extract lists from response based on evaluation mode
        if trait.evaluation_mode == "tp_only":
            # TP-only mode: Extract TP, FN, FP (TN cannot be computed)
            if "tp" in result:
                confusion_lists["tp"] = result["tp"] if isinstance(result["tp"], list) else []
            if "fn" in result:
                confusion_lists["fn"] = result["fn"] if isinstance(result["fn"], list) else []
            if "fp" in result:
                confusion_lists["fp"] = result["fp"] if isinstance(result["fp"], list) else []
            # TN remains empty (cannot be computed in tp_only mode)

        else:  # full_matrix mode
            # Full matrix mode: Extract all four buckets
            if "tp" in result:
                confusion_lists["tp"] = result["tp"] if isinstance(result["tp"], list) else []
            if "fn" in result:
                confusion_lists["fn"] = result["fn"] if isinstance(result["fn"], list) else []
            if "tn" in result:
                confusion_lists["tn"] = result["tn"] if isinstance(result["tn"], list) else []
            if "fp" in result:
                confusion_lists["fp"] = result["fp"] if isinstance(result["fp"], list) else []

        return confusion_lists

    def _deduplicate_confusion_lists(self, confusion_lists: dict[str, list[str]]) -> dict[str, list[str]]:
        """
        Deduplicate excerpts within each confusion list bucket.

        Uses case-insensitive exact matching. Preserves the first occurrence of each unique excerpt.

        Args:
            confusion_lists: Dictionary of confusion matrix lists

        Returns:
            Dictionary with deduplicated lists
        """
        deduplicated = {}

        for bucket, excerpts in confusion_lists.items():
            deduplicated[bucket] = self._deduplicate_excerpts(excerpts)

        return deduplicated

    def _deduplicate_excerpts(self, excerpts: list[str]) -> list[str]:
        """
        Deduplicate a list of excerpts (case-insensitive exact matching).

        Args:
            excerpts: List of excerpt strings

        Returns:
            Deduplicated list preserving first occurrence order
        """
        seen_lower = set()
        deduplicated = []

        for excerpt in excerpts:
            excerpt_lower = excerpt.lower().strip()
            if excerpt_lower and excerpt_lower not in seen_lower:
                seen_lower.add(excerpt_lower)
                deduplicated.append(excerpt)

        return deduplicated

    def _compute_metrics(
        self, tp: list[str], tn: list[str], fp: list[str], fn: list[str], requested_metrics: list[str]
    ) -> dict[str, float]:
        """
        Compute classification metrics from confusion matrix lists.

        Args:
            tp: True positives list
            tn: True negatives list
            fp: False positives list
            fn: False negatives list
            requested_metrics: List of metric names to compute

        Returns:
            Dictionary mapping metric names to computed values
        """
        # Get counts
        tp_count = len(tp)
        tn_count = len(tn)
        fp_count = len(fp)
        fn_count = len(fn)

        metrics = {}

        for metric in requested_metrics:
            try:
                if metric == "precision":
                    # Precision = TP / (TP + FP)
                    denominator = tp_count + fp_count
                    metrics["precision"] = tp_count / denominator if denominator > 0 else 0.0

                elif metric == "recall":
                    # Recall = TP / (TP + FN)
                    denominator = tp_count + fn_count
                    metrics["recall"] = tp_count / denominator if denominator > 0 else 0.0

                elif metric == "specificity":
                    # Specificity = TN / (TN + FP)
                    denominator = tn_count + fp_count
                    metrics["specificity"] = tn_count / denominator if denominator > 0 else 0.0

                elif metric == "accuracy":
                    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
                    denominator = tp_count + tn_count + fp_count + fn_count
                    metrics["accuracy"] = (tp_count + tn_count) / denominator if denominator > 0 else 0.0

                elif metric == "f1":
                    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
                    precision_denom = tp_count + fp_count
                    recall_denom = tp_count + fn_count

                    if precision_denom > 0 and recall_denom > 0:
                        precision = tp_count / precision_denom
                        recall = tp_count / recall_denom

                        if precision + recall > 0:
                            metrics["f1"] = 2 * (precision * recall) / (precision + recall)
                        else:
                            metrics["f1"] = 0.0
                    else:
                        metrics["f1"] = 0.0

            except Exception as e:
                logger.warning(f"Failed to compute metric '{metric}': {e}")
                metrics[metric] = 0.0

        return metrics
