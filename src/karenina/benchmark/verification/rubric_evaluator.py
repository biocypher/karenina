"""
Rubric evaluation for qualitative assessment of LLM responses.
"""

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ...llm.interface import init_chat_model_unified
from ...schemas.rubric_class import ManualRubricTrait, MetricRubricTrait, Rubric, RubricTrait
from ..models import INTERFACES_NO_PROVIDER_REQUIRED, ModelConfig

logger = logging.getLogger(__name__)


class RubricEvaluator:
    """
    Evaluates LLM responses against a defined rubric using qualitative traits.
    """

    def __init__(self, model_config: ModelConfig, callable_registry: dict[str, Callable[[str], bool]] | None = None):
        """
        Initialize the rubric evaluator with an LLM model.

        Args:
            model_config: Configuration for the evaluation model
            callable_registry: Registry of callable functions for manual trait evaluation

        Raises:
            ValueError: If model configuration is invalid
            RuntimeError: If LLM initialization fails
        """
        if not model_config:
            raise ValueError("Model configuration is required")

        if not model_config.model_name:
            raise ValueError("Model name is required in model configuration")

        # Model provider is optional for OpenRouter and manual interfaces
        if model_config.interface not in INTERFACES_NO_PROVIDER_REQUIRED and not model_config.model_provider:
            raise ValueError(
                f"Model provider is required for model {model_config.id} "
                f"(interface: {model_config.interface}). Only {INTERFACES_NO_PROVIDER_REQUIRED} "
                f"interfaces allow empty providers."
            )

        self.model_config = model_config
        self.callable_registry = callable_registry or {}

        try:
            self.llm = init_chat_model_unified(
                model=model_config.model_name,
                provider=model_config.model_provider,
                temperature=model_config.temperature,
                interface=model_config.interface,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM for rubric evaluation: {e}") from e

    def evaluate_rubric(self, question: str, answer: str, rubric: Rubric) -> dict[str, int | bool]:
        """
        Evaluate an answer against a rubric's traits (both LLM and manual).

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            rubric: The rubric containing evaluation traits

        Returns:
            Dictionary mapping trait names to their evaluated scores

        Raises:
            Exception: If evaluation fails completely
        """
        results: dict[str, int | bool] = {}

        # Evaluate manual traits first (these are faster and deterministic)
        if rubric.manual_traits:
            manual_results = self._evaluate_manual_traits(answer, rubric.manual_traits)
            results.update(manual_results)

        # Evaluate LLM traits if present
        if rubric.traits:
            try:
                # Try batch evaluation first (more efficient)
                llm_results = self._evaluate_batch(question, answer, rubric)
                results.update(llm_results)
            except Exception as batch_error:
                # Fallback to sequential evaluation
                try:
                    llm_results = self._evaluate_sequential(question, answer, rubric)
                    results.update(llm_results)
                except Exception as seq_error:
                    # Log both errors and raise the sequential one
                    logger.error(f"Batch evaluation failed: {batch_error}")
                    logger.error(f"Sequential evaluation failed: {seq_error}")
                    raise seq_error

        return results

    def register_callable(self, name: str, func: Callable[[str], bool]) -> None:
        """
        Register a callable function for manual trait evaluation.

        Args:
            name: Name to register the function under
            func: Function that takes a string and returns a boolean

        Raises:
            ValueError: If function doesn't have correct signature
        """
        # Basic validation of function signature
        import inspect

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if len(params) != 1:
            raise ValueError(f"Callable '{name}' must have exactly one parameter, got {len(params)}")

        # Store the function
        self.callable_registry[name] = func

    def _evaluate_manual_traits(self, answer: str, manual_traits: list[ManualRubricTrait]) -> dict[str, int | bool]:
        """
        Evaluate manual traits using regex patterns or callable functions.

        Args:
            answer: The text to evaluate
            manual_traits: List of manual traits to evaluate

        Returns:
            Dictionary mapping trait names to boolean results

        Raises:
            RuntimeError: If evaluation of any trait fails
        """
        results: dict[str, int | bool] = {}

        for trait in manual_traits:
            try:
                result = trait.evaluate(answer, self.callable_registry)
                results[trait.name] = result
            except Exception as e:
                logger.warning(f"Failed to evaluate manual trait '{trait.name}': {e}")
                # Mark failed traits as None for consistency with LLM evaluation
                results[trait.name] = None  # type: ignore[assignment]

        return results

    def _evaluate_batch(self, question: str, answer: str, rubric: Rubric) -> dict[str, int | bool]:
        """
        Evaluate all traits in a single LLM call (more efficient).
        """
        system_prompt = self._build_batch_system_prompt()
        user_prompt = self._build_batch_user_prompt(question, answer, rubric.traits)

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        response = self.llm.invoke(messages)
        raw_response = response.content if hasattr(response, "content") else str(response)

        return self._parse_batch_response(raw_response, rubric.traits)

    def _evaluate_sequential(self, question: str, answer: str, rubric: Rubric) -> dict[str, int | bool]:
        """
        Evaluate traits one by one (fallback method).
        """
        results = {}

        for trait in rubric.traits:
            try:
                score = self._evaluate_single_trait(question, answer, trait)
                results[trait.name] = score
            except Exception as e:
                logger.warning(f"Failed to evaluate trait '{trait.name}': {e}")
                # Continue with other traits, mark this one as None
                results[trait.name] = None  # type: ignore[assignment]

        return results

    def _evaluate_single_trait(self, question: str, answer: str, trait: RubricTrait) -> int | bool:
        """
        Evaluate a single trait.
        """
        system_prompt = self._build_single_trait_system_prompt(trait)
        user_prompt = self._build_single_trait_user_prompt(question, answer, trait)

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        response = self.llm.invoke(messages)
        raw_response = response.content if hasattr(response, "content") else str(response)

        return self._parse_single_trait_response(raw_response, trait)

    def _build_batch_system_prompt(self) -> str:
        """Build system prompt for batch evaluation."""
        return """You are an expert evaluator assessing the quality of responses using a structured rubric.

Your task is to evaluate a given answer against multiple evaluation traits and return scores in a specific JSON format.

For each trait, you will be given:
- A trait name
- A description of what to evaluate
- The trait type (boolean or score)
- For score traits: the valid range (e.g., 1-5)

Your evaluation should be:
- Objective and consistent
- Based solely on the provided criteria
- Independent for each trait

Return your evaluation as a JSON object where keys are trait names and values are the scores."""

    def _build_batch_user_prompt(self, question: str, answer: str, traits: list[RubricTrait]) -> str:
        """Build user prompt for batch evaluation."""
        traits_description = []

        for trait in traits:
            if trait.kind == "boolean":
                trait_desc = f"- {trait.name}: {trait.description or 'Boolean evaluation'} (return true or false)"
            else:
                min_score = trait.min_score or 1
                max_score = trait.max_score or 5
                trait_desc = f"- {trait.name}: {trait.description or 'Score-based evaluation'} (return integer from {min_score} to {max_score})"
            traits_description.append(trait_desc)

        return f"""Please evaluate the following answer using these traits:

{chr(10).join(traits_description)}

QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

Return your evaluation as a JSON object with trait names as keys and scores as values. For example:
{{"trait1": true, "trait2": 4, "trait3": false}}

JSON Response:"""

    def _build_single_trait_system_prompt(self, trait: RubricTrait) -> str:
        """Build system prompt for single trait evaluation."""
        if trait.kind == "boolean":
            return f"""You are evaluating responses for the trait: {trait.name}

Description: {trait.description or "Boolean evaluation"}

Respond with only "true" or "false" based on whether the answer meets this criteria."""
        else:
            min_score = trait.min_score or 1
            max_score = trait.max_score or 5
            return f"""You are evaluating responses for the trait: {trait.name}

Description: {trait.description or "Score-based evaluation"}

Rate the answer on a scale from {min_score} to {max_score}, where:
- {min_score} = Poor/Does not meet criteria
- {max_score} = Excellent/Fully meets criteria

Respond with only the numeric score ({min_score}-{max_score})."""

    def _build_single_trait_user_prompt(self, question: str, answer: str, trait: RubricTrait) -> str:
        """Build user prompt for single trait evaluation."""
        return f"""QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

Please evaluate this answer for the trait "{trait.name}": {trait.description or "No description provided"}"""

    def _parse_batch_response(self, response: str, traits: list[RubricTrait]) -> dict[str, int | bool]:
        """Parse the batch evaluation response."""
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response}")

        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}") from e

        # Validate and convert the results
        validated_results = {}
        trait_map = {trait.name: trait for trait in traits}

        for trait_name, score in result.items():
            if trait_name in trait_map:
                trait = trait_map[trait_name]
                validated_score = self._validate_score(score, trait)
                validated_results[trait_name] = validated_score

        # Add None for missing traits
        for trait in traits:
            if trait.name not in validated_results:
                validated_results[trait.name] = None  # type: ignore[assignment]

        return validated_results

    def _parse_single_trait_response(self, response: str, trait: RubricTrait) -> int | bool:
        """Parse a single trait evaluation response."""
        response = response.strip().lower()

        if trait.kind == "boolean":
            if response in ["true", "yes", "1"]:
                return True
            elif response in ["false", "no", "0"]:
                return False
            else:
                # Try to extract boolean from longer response
                if "true" in response or "yes" in response:
                    return True
                elif "false" in response or "no" in response:
                    return False
                else:
                    raise ValueError(f"Could not parse boolean from: {response}")
        else:
            # Extract numeric score
            numbers = re.findall(r"\d+", response)
            if not numbers:
                raise ValueError(f"No numeric score found in: {response}")

            score = int(numbers[0])
            return self._validate_score(score, trait)

    def _validate_score(self, score: Any, trait: RubricTrait) -> int | bool:
        """Validate and convert a score for a trait."""
        if trait.kind == "boolean":
            if isinstance(score, bool):
                return score
            elif isinstance(score, int | str):
                return bool(score) and str(score).lower() not in ["false", "0", "no"]
            else:
                return bool(score)
        else:
            if not isinstance(score, int | float):
                try:
                    score = int(score)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid score type for trait {trait.name}: {type(score)}") from e

            min_score = trait.min_score or 1
            max_score = trait.max_score or 5

            # Clamp score to valid range
            clamped_score = max(min_score, min(max_score, int(score)))
            return clamped_score

    # ========== Metric Trait Evaluation Methods ==========

    def evaluate_metric_traits(
        self, question: str, answer: str, metric_traits: list[MetricRubricTrait]
    ) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, float]]]:
        """
        Evaluate metric traits and return confusion lists and computed metrics.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            metric_traits: List of metric traits to evaluate

        Returns:
            Tuple of (confusion_lists, metrics) where:
            - confusion_lists: {trait_name: {tp: [...], tn: [...], fp: [...], fn: [...]}}
            - metrics: {trait_name: {precision: 0.85, recall: 0.92, ...}}

        Raises:
            Exception: If evaluation fails for all traits
        """
        confusion_lists: dict[str, dict[str, list[str]]] = {}
        metrics: dict[str, dict[str, float]] = {}

        for trait in metric_traits:
            try:
                trait_confusion, trait_metrics = self._evaluate_single_metric_trait(question, answer, trait)
                confusion_lists[trait.name] = trait_confusion
                metrics[trait.name] = trait_metrics
            except Exception as e:
                logger.warning(f"Failed to evaluate metric trait '{trait.name}': {e}")
                # Store empty results for failed traits
                confusion_lists[trait.name] = {"tp": [], "tn": [], "fp": [], "fn": []}
                metrics[trait.name] = {}

        return confusion_lists, metrics

    def _evaluate_single_metric_trait(
        self, question: str, answer: str, trait: MetricRubricTrait
    ) -> tuple[dict[str, list[str]], dict[str, float]]:
        """
        Evaluate a single metric trait.

        Args:
            question: The original question
            answer: The answer to evaluate
            trait: The metric trait to evaluate

        Returns:
            Tuple of (confusion_lists, metrics)
        """
        # Build prompt
        system_prompt = self._build_metric_trait_system_prompt()
        user_prompt = self._build_metric_trait_user_prompt(question, answer, trait)

        # Invoke LLM
        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = self.llm.invoke(messages)
        raw_response = response.content if hasattr(response, "content") else str(response)

        # Parse response to get confusion lists
        confusion_lists = self._parse_metric_trait_response(raw_response, trait)

        # Apply deduplication if requested
        if trait.repeated_extraction:
            confusion_lists = self._deduplicate_confusion_lists(confusion_lists)

        # Compute metrics
        computed_metrics = self._compute_metrics(
            confusion_lists["tp"], confusion_lists["tn"], confusion_lists["fp"], confusion_lists["fn"], trait.metrics
        )

        return confusion_lists, computed_metrics

    def _build_metric_trait_system_prompt(self) -> str:
        """Build system prompt for metric trait evaluation."""
        return """You are an expert evaluator performing confusion-matrix analysis on text responses.

Your task is to analyze an answer and identify excerpts that match specific instruction categories.

You will be given instructions for identifying:
- True Positives (TP): Correct matches
- True Negatives (TN): Correct non-matches
- False Positives (FP): Incorrect matches
- False Negatives (FN): Missed matches

Your evaluation should be:
- Objective and thorough
- Based solely on the provided instructions
- Extract exact phrases or short excerpts from the answer

Return your analysis as a JSON object with arrays of excerpts for each category."""

    def _build_metric_trait_user_prompt(self, question: str, answer: str, trait: MetricRubricTrait) -> str:
        """Build user prompt for metric trait evaluation (mode-specific)."""
        prompt_parts = []

        prompt_parts.append(f"Analyze the following answer for: **{trait.name}**")
        if trait.description:
            prompt_parts.append(f"Description: {trait.description}")
        prompt_parts.append("")

        if trait.evaluation_mode == "tp_only":
            # TP-only mode: Check which TP instructions are present, identify extra content
            prompt_parts.append("TRUE POSITIVE INSTRUCTIONS (what SHOULD be present):")
            for i, instruction in enumerate(trait.tp_instructions, 1):
                prompt_parts.append(f"  {i}. {instruction}")
            prompt_parts.append("")

            prompt_parts.append("QUESTION:")
            prompt_parts.append(question)
            prompt_parts.append("")
            prompt_parts.append("ANSWER TO EVALUATE:")
            prompt_parts.append(answer)
            prompt_parts.append("")

            prompt_parts.append("Evaluate the answer and return a JSON object with:")
            prompt_parts.append(
                '- "tp": List of TP instructions that are present in the answer (by number or exact text)'
            )
            prompt_parts.append('- "fn": List of TP instructions that are missing from the answer')
            prompt_parts.append(
                '- "fp": List of any extra content/terms in the answer that are NOT covered by the TP instructions'
            )
            prompt_parts.append("")
            prompt_parts.append("Expected format:")
            prompt_parts.append(
                '{"tp": ["instruction 1", "instruction 2"], "fn": ["instruction 3"], "fp": ["extra term 1", "extra term 2"]}'
            )
            prompt_parts.append("")
            prompt_parts.append("JSON Response:")

        else:  # full_matrix mode
            # Full matrix: Check each TP and TN instruction against answer
            prompt_parts.append("TRUE POSITIVE INSTRUCTIONS (what SHOULD be present):")
            for i, instruction in enumerate(trait.tp_instructions, 1):
                prompt_parts.append(f"  {i}. {instruction}")
            prompt_parts.append("")

            prompt_parts.append("TRUE NEGATIVE INSTRUCTIONS (what SHOULD NOT be present):")
            for i, instruction in enumerate(trait.tn_instructions, 1):
                prompt_parts.append(f"  {i}. {instruction}")
            prompt_parts.append("")

            prompt_parts.append("QUESTION:")
            prompt_parts.append(question)
            prompt_parts.append("")
            prompt_parts.append("ANSWER TO EVALUATE:")
            prompt_parts.append(answer)
            prompt_parts.append("")

            prompt_parts.append("Evaluate the answer against each instruction and return a JSON object with:")
            prompt_parts.append('- "tp": TP instructions that are present in the answer (correct)')
            prompt_parts.append('- "fn": TP instructions that are missing from the answer (incorrect)')
            prompt_parts.append('- "tn": TN instructions that are correctly absent from the answer (correct)')
            prompt_parts.append('- "fp": TN instructions that are incorrectly present in the answer (incorrect)')
            prompt_parts.append("")
            prompt_parts.append("Expected format:")
            prompt_parts.append(
                '{"tp": ["instruction 1"], "fn": ["instruction 2"], "tn": ["instruction 3"], "fp": ["instruction 4"]}'
            )
            prompt_parts.append("")
            prompt_parts.append("JSON Response:")

        return "\n".join(prompt_parts)

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
