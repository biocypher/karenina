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

from .....ports import LLMPort, Message
from .....schemas.domain import MetricRubricTrait

if TYPE_CHECKING:
    from .....schemas.workflow.models import ModelConfig

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

    def __init__(self, llm: LLMPort, *, model_config: "ModelConfig"):
        """
        Initialize the metric trait evaluator.

        Args:
            llm: LLMPort adapter for LLM operations.
            model_config: Model configuration for reference.
        """
        self.llm = llm
        self._model_config = model_config

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
        system_prompt = self._build_metric_trait_system_prompt()
        user_prompt = self._build_metric_trait_user_prompt(question, answer, trait)

        messages: list[Message] = [Message.system(system_prompt), Message.user(user_prompt)]

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

    def _build_metric_trait_system_prompt(self) -> str:
        """Build system prompt for metric trait evaluation."""
        return """You are an expert evaluator performing confusion-matrix analysis on text responses.

Your task is to analyze an answer and categorize its content based on provided instructions.

**RESPONSE FORMAT:**
You will receive a JSON Schema specifying the exact output structure. Your response MUST conform to this schema.
Return ONLY valid JSON with the required structure. Empty arrays [] are valid.

Do NOT include any text before or after the JSON.

**CONFUSION MATRIX CATEGORIES:**

- **tp (True Positives)**: Content from the answer that SHOULD be present AND IS present
  → Extract actual excerpts/terms FROM THE ANSWER (not instruction text)

- **fn (False Negatives)**: Content that SHOULD be present BUT IS NOT in the answer
  → List what is missing (reference the instruction content)

- **tn (True Negatives)**: Content that SHOULD NOT be present AND IS correctly absent
  → List TN instructions correctly not mentioned in the answer

- **fp (False Positives)**: Content from the answer that SHOULD NOT be present BUT IS present
  → Extract actual excerpts/terms FROM THE ANSWER that should not be there

**MATCHING CRITERIA:**
- Accept exact matches and close variants
- Accept synonyms (e.g., "disease"/"illness", "tumor"/"neoplasm")
- Case insensitive matching unless instructed otherwise
- For partial matches: "lung cancer" satisfies "mention cancer"

**CRITICAL RULES:**
- For tp and fp: Extract ACTUAL text FROM THE ANSWER
- For fn and tn: Reference the instruction content
- Extract key terms or short phrases, not full sentences
- When uncertain, include the item (err on inclusivity)

**WHAT NOT TO DO:**
- Do NOT wrap JSON in markdown code blocks
- Do NOT add explanatory text
- Do NOT include duplicate items in the same list"""

    def _build_metric_trait_user_prompt(self, question: str, answer: str, trait: MetricRubricTrait) -> str:
        """Build user prompt for metric trait evaluation (mode-specific)."""
        from .....schemas.workflow.rubric_outputs import ConfusionMatrixOutput

        # Get JSON schema from Pydantic model
        json_schema = json.dumps(ConfusionMatrixOutput.model_json_schema(), indent=2)

        # Format TP instructions as numbered list
        tp_instructions_formatted = "\n".join(
            f"  {i}. {instruction}" for i, instruction in enumerate(trait.tp_instructions, 1)
        )

        # Build description line if present
        description_line = f"Description: {trait.description}\n" if trait.description else ""

        if trait.evaluation_mode == "tp_only":
            return f"""Analyze the following answer for: **{trait.name}**
{description_line}
**EVALUATION TASK:**
You are evaluating an answer against required content (TP instructions). Your job is to categorize content from the answer into:
1. **True Positives (TP)**: Content that correctly matches TP instructions
2. **False Negatives (FN)**: Required content from TP instructions that is missing
3. **False Positives (FP)**: Content that LOOKS like it should match TP instructions but is actually incorrect

**TRUE POSITIVE INSTRUCTIONS (required content):**
{tp_instructions_formatted}

**QUESTION:**
{question}

**ANSWER TO EVALUATE:**
{answer}

**EVALUATION GUIDELINES:**

**For True Positives (TP):**
- Extract actual terms/excerpts from the answer that match TP instructions
- Accept exact matches, synonyms, and semantically equivalent expressions
- Example: If TP instruction is "mention asthma" and answer says "asthma", extract "asthma"
- Example: If TP instruction is "mention tumor" and answer says "neoplasm", extract "neoplasm" (synonym)

**For False Negatives (FN):**
- List the content from TP instructions that is NOT found in the answer
- Reference the actual missing content
- Example: If TP instruction is "mention pneumonia" but it's not in the answer, add "pneumonia"

**For False Positives (FP):**
- Extract terms from the answer that appear to be attempting to satisfy TP instructions but are actually INCORRECT
- Focus on terms in the same domain/category as TP instructions that LOOK like valid answers but aren't
- Example: If TP instructions ask for inflammatory lung diseases (asthma, bronchitis, pneumonia) but answer includes restrictive lung diseases (pulmonary fibrosis, sarcoidosis), those are FP
- DO NOT include: generic filler text, explanations, or content clearly not attempting to match TP instructions
- If unsure whether something is FP, consider: "Is this term in the same category as TP instructions but not actually correct?"

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**OUTPUT FORMAT:**

Return ONLY a valid JSON object:
{{"tp": [<excerpts from answer matching TP instructions>], "fn": [<missing TP instruction content>], "fp": [<incorrect terms from answer that look like TPs but aren't>]}}

Example:
{{"tp": ["asthma", "bronchitis"], "fn": ["pneumonia", "pleurisy"], "fp": ["pulmonary fibrosis", "emphysema", "sarcoidosis"]}}

Your JSON response:"""

        else:  # full_matrix mode
            # Format TN instructions as numbered list
            tn_instructions_formatted = "\n".join(
                f"  {i}. {instruction}" for i, instruction in enumerate(trait.tn_instructions, 1)
            )

            return f"""Analyze the following answer for: **{trait.name}**
{description_line}
**EVALUATION TASK:**
You are evaluating an answer against two instruction sets:
- **TP instructions**: Content that SHOULD be present
- **TN instructions**: Content that SHOULD NOT be present

Categorize the answer content into four confusion matrix categories.

**TRUE POSITIVE INSTRUCTIONS (what SHOULD be present):**
{tp_instructions_formatted}

**TRUE NEGATIVE INSTRUCTIONS (what SHOULD NOT be present):**
{tn_instructions_formatted}

**QUESTION:**
{question}

**ANSWER TO EVALUATE:**
{answer}

**EVALUATION GUIDELINES:**

**For True Positives (TP):**
- Extract actual terms/excerpts from the answer that match TP instructions
- Accept exact matches, synonyms, and semantically equivalent expressions

**For False Negatives (FN):**
- List content from TP instructions that is NOT found in the answer
- These are required items that are missing

**For True Negatives (TN):**
- List TN instructions that are correctly NOT present in the answer
- Reference the instruction content
- These represent unwanted content that is correctly absent

**For False Positives (FP):**
- Extract terms/excerpts from the answer that match TN instructions
- These are items that SHOULD NOT be there but ARE present
- Use the same matching criteria as TP (accept synonyms, etc.)

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**OUTPUT FORMAT:**

Return ONLY a valid JSON object:
{{"tp": [<excerpts matching TP instructions>], "fn": [<missing TP content>], "tn": [<TN instructions correctly absent>], "fp": [<excerpts matching TN instructions>]}}

Example:
{{"tp": ["asthma"], "fn": ["bronchitis"], "tn": ["pulmonary fibrosis"], "fp": ["emphysema"]}}

Your JSON response:"""

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
