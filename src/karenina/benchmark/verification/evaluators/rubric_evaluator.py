"""
Rubric evaluation for qualitative assessment of LLM responses.
"""

import json
import logging
import re
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ....infrastructure.llm.interface import init_chat_model_unified
from ....schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait, Rubric
from ....schemas.workflow import INTERFACES_NO_PROVIDER_REQUIRED, ModelConfig

logger = logging.getLogger(__name__)


class RubricEvaluator:
    """
    Evaluates LLM responses against a defined rubric using qualitative traits.
    """

    def __init__(self, model_config: ModelConfig, evaluation_strategy: str = "batch"):
        """
        Initialize the rubric evaluator with an LLM model.

        Args:
            model_config: Configuration for the evaluation model
            evaluation_strategy: Strategy for evaluating LLM traits ("batch" or "sequential")
                - "batch": Evaluate all traits in single LLM call (efficient)
                - "sequential": Evaluate traits one-by-one (reliable)

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
        self.evaluation_strategy = evaluation_strategy

        try:
            # Build kwargs for model initialization
            model_kwargs: dict[str, Any] = {
                "model": model_config.model_name,
                "provider": model_config.model_provider,
                "temperature": model_config.temperature,
                "interface": model_config.interface,
            }

            # Add endpoint configuration if using openai_endpoint interface
            if model_config.endpoint_base_url:
                model_kwargs["endpoint_base_url"] = model_config.endpoint_base_url
            if model_config.endpoint_api_key:
                model_kwargs["endpoint_api_key"] = model_config.endpoint_api_key

            # Add any extra kwargs if provided (e.g., vendor-specific API keys)
            if model_config.extra_kwargs:
                model_kwargs.update(model_config.extra_kwargs)

            self.llm = init_chat_model_unified(**model_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM for rubric evaluation: {e}") from e

    def evaluate_rubric(
        self, question: str, answer: str, rubric: Rubric
    ) -> tuple[dict[str, int | bool], list[dict[str, Any]]]:
        """
        Evaluate an answer against a rubric's traits (LLM, regex, and callable).

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            rubric: The rubric containing evaluation traits

        Returns:
            Tuple of (results, usage_metadata_list) where:
            - results: Dictionary mapping trait names to their evaluated scores
            - usage_metadata_list: List of usage metadata dicts from LLM calls

        Raises:
            Exception: If evaluation fails completely
        """
        results: dict[str, int | bool] = {}
        usage_metadata_list: list[dict[str, Any]] = []

        # Evaluate regex traits first (fast and deterministic)
        if rubric.regex_traits:
            regex_results = self._evaluate_regex_traits(answer, rubric.regex_traits)
            results.update(regex_results)

        # Evaluate callable traits (deterministic but potentially slower)
        if rubric.callable_traits:
            callable_results = self._evaluate_callable_traits(answer, rubric.callable_traits)
            results.update(callable_results)

        # Evaluate LLM traits if present
        if rubric.llm_traits:
            if self.evaluation_strategy == "batch":
                # Batch evaluation - evaluates all traits in single LLM call
                try:
                    llm_results, usage_metadata = self._evaluate_batch(question, answer, rubric)
                    results.update(llm_results)
                    if usage_metadata:
                        usage_metadata_list.append(usage_metadata)
                except Exception as e:
                    logger.error(f"Batch evaluation failed: {e}")
                    raise RuntimeError(f"Failed to evaluate rubric traits using batch strategy: {e}") from e
            else:  # "sequential"
                # Sequential evaluation - evaluates traits one by one
                try:
                    llm_results, seq_usage_metadata_list = self._evaluate_sequential(question, answer, rubric)
                    results.update(llm_results)
                    usage_metadata_list.extend(seq_usage_metadata_list)
                except Exception as e:
                    logger.error(f"Sequential evaluation failed: {e}")
                    raise RuntimeError(f"Failed to evaluate rubric traits using sequential strategy: {e}") from e

        return results, usage_metadata_list

    def _evaluate_regex_traits(self, answer: str, regex_traits: list[RegexTrait]) -> dict[str, bool]:
        """
        Evaluate regex traits using pattern matching.

        Args:
            answer: The text to evaluate
            regex_traits: List of regex traits to evaluate

        Returns:
            Dictionary mapping trait names to boolean results

        Raises:
            RuntimeError: If evaluation of any trait fails
        """
        results: dict[str, bool] = {}

        for trait in regex_traits:
            try:
                result = trait.evaluate(answer)
                results[trait.name] = result
            except Exception as e:
                logger.warning(f"Failed to evaluate regex trait '{trait.name}': {e}")
                # Mark failed traits as None for consistency with LLM evaluation
                results[trait.name] = None  # type: ignore[assignment]

        return results

    def _evaluate_callable_traits(self, answer: str, callable_traits: list[CallableTrait]) -> dict[str, bool | int]:
        """
        Evaluate callable traits using custom functions.

        Args:
            answer: The text to evaluate
            callable_traits: List of callable traits to evaluate

        Returns:
            Dictionary mapping trait names to boolean or int results (depending on trait kind)

        Raises:
            RuntimeError: If evaluation of any trait fails
        """
        results: dict[str, bool | int] = {}

        for trait in callable_traits:
            try:
                result = trait.evaluate(answer)
                results[trait.name] = result
            except Exception as e:
                logger.warning(f"Failed to evaluate callable trait '{trait.name}': {e}")
                # Mark failed traits as None for consistency with LLM evaluation
                results[trait.name] = None  # type: ignore[assignment]

        return results

    def _evaluate_batch(
        self, question: str, answer: str, rubric: Rubric
    ) -> tuple[dict[str, int | bool], dict[str, Any]]:
        """
        Evaluate all traits in a single LLM call (more efficient).

        Returns:
            Tuple of (results_dict, usage_metadata)
        """
        system_prompt = self._build_batch_system_prompt()
        user_prompt = self._build_batch_user_prompt(question, answer, rubric.llm_traits)

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        # Wrap invoke with usage metadata callback
        with get_usage_metadata_callback() as cb:
            response = self.llm.invoke(messages)
        usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}

        raw_response = response.content if hasattr(response, "content") else str(response)

        results = self._parse_batch_response(raw_response, rubric.llm_traits)
        return results, usage_metadata

    def _evaluate_sequential(
        self, question: str, answer: str, rubric: Rubric
    ) -> tuple[dict[str, int | bool], list[dict[str, Any]]]:
        """
        Evaluate traits one by one (fallback method).

        Returns:
            Tuple of (results_dict, list_of_usage_metadata)
        """
        results = {}
        usage_metadata_list = []

        for trait in rubric.llm_traits:
            try:
                score, usage_metadata = self._evaluate_single_trait(question, answer, trait)
                results[trait.name] = score
                usage_metadata_list.append(usage_metadata)
            except Exception as e:
                logger.warning(f"Failed to evaluate trait '{trait.name}': {e}")
                # Continue with other traits, mark this one as None
                results[trait.name] = None  # type: ignore[assignment]

        return results, usage_metadata_list

    def _evaluate_single_trait(
        self, question: str, answer: str, trait: LLMRubricTrait
    ) -> tuple[int | bool, dict[str, Any]]:
        """
        Evaluate a single trait.

        Returns:
            Tuple of (score, usage_metadata)
        """
        system_prompt = self._build_single_trait_system_prompt(trait)
        user_prompt = self._build_single_trait_user_prompt(question, answer, trait)

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        # Wrap invoke with usage metadata callback
        with get_usage_metadata_callback() as cb:
            response = self.llm.invoke(messages)
        usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}

        raw_response = response.content if hasattr(response, "content") else str(response)

        score = self._parse_single_trait_response(raw_response, trait)
        return score, usage_metadata

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

    def _build_batch_user_prompt(self, question: str, answer: str, traits: list[LLMRubricTrait]) -> str:
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

    def _build_single_trait_system_prompt(self, trait: LLMRubricTrait) -> str:
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

    def _build_single_trait_user_prompt(self, question: str, answer: str, trait: LLMRubricTrait) -> str:
        """Build user prompt for single trait evaluation."""
        return f"""QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

Please evaluate this answer for the trait "{trait.name}": {trait.description or "No description provided"}"""

    def _parse_batch_response(self, response: str, traits: list[LLMRubricTrait]) -> dict[str, int | bool]:
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

    def _parse_single_trait_response(self, response: str, trait: LLMRubricTrait) -> int | bool:
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

    def _validate_score(self, score: Any, trait: LLMRubricTrait) -> int | bool:
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
        Evaluate a single metric trait.

        Args:
            question: The original question
            answer: The answer to evaluate
            trait: The metric trait to evaluate

        Returns:
            Tuple of (confusion_lists, metrics, usage_metadata)
        """
        # Build prompt
        system_prompt = self._build_metric_trait_system_prompt()
        user_prompt = self._build_metric_trait_user_prompt(question, answer, trait)

        # Invoke LLM with usage tracking
        from langchain_core.callbacks import get_usage_metadata_callback

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        with get_usage_metadata_callback() as cb:
            response = self.llm.invoke(messages)
        usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}

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

        return confusion_lists, computed_metrics, usage_metadata

    def _build_metric_trait_system_prompt(self) -> str:
        """Build system prompt for metric trait evaluation."""
        return """You are an expert evaluator performing confusion-matrix analysis on text responses.

Your task is to analyze an answer and categorize its content based on provided instructions.

**CONFUSION MATRIX CATEGORIES:**

- **True Positive (TP)**: Content from the answer that SHOULD be present AND IS present
  - Extract actual excerpts/terms from the answer that match TP instructions

- **False Negative (FN)**: Content that SHOULD be present BUT IS NOT found in the answer
  - List what is missing based on TP instructions

- **True Negative (TN)**: Content that SHOULD NOT be present AND IS correctly absent
  - List instructions that are correctly not mentioned in the answer

- **False Positive (FP)**: Content from the answer that SHOULD NOT be present BUT IS present
  - Extract actual excerpts/terms from the answer that should not be there

**MATCHING CRITERIA:**

1. **Exact matches**: If the instruction specifies an exact term, look for that exact term or very close variants
2. **Semantic matches**: If the instruction describes a concept, accept semantically equivalent expressions
3. **Synonyms**: Accept commonly recognized synonyms (e.g., "disease" and "illness", "tumor" and "neoplasm")
4. **Case insensitive**: Ignore case differences unless specifically instructed otherwise
5. **Partial matches**: If an answer contains a broader or narrower term than instructed, use your judgment:
   - If instruction is "mention cancer" and answer says "lung cancer", count it as TP
   - If instruction is "mention specific organ" and answer just says "organ", it may not fully satisfy the instruction

**CRITICAL RULES:**

- For TP and FP: Extract the ACTUAL text/excerpts FROM THE ANSWER (not the instruction text)
- For FN and TN: Reference the instruction content (what should/shouldn't be there)
- Be thorough but focused - extract key terms or short phrases, not full sentences
- When in doubt, err on the side of being inclusive rather than overly strict
- If something is ambiguous, include it and let the metrics reflect the ambiguity

**OUTPUT FORMAT:**

Return a JSON object with arrays for each category. Each array should contain strings (excerpts or descriptions)."""

    def _build_metric_trait_user_prompt(self, question: str, answer: str, trait: MetricRubricTrait) -> str:
        """Build user prompt for metric trait evaluation (mode-specific)."""
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

    # ========== Deep Judgment Rubric Methods ==========

    def evaluate_rubric_with_deep_judgment(
        self,
        question: str,
        answer: str,
        rubric: Rubric,
        config: Any,  # VerificationConfig
    ) -> dict[str, Any]:
        """
        Evaluate rubric with deep judgment for enabled traits.

        Args:
            question: The original question
            answer: The LLM response to evaluate
            rubric: The rubric containing evaluation traits
            config: VerificationConfig with deep judgment settings

        Returns:
            Dictionary containing:
                - deep_judgment_scores: Scores for deep-judgment-enabled traits
                - standard_scores: Scores for standard traits
                - excerpts: Extracted excerpts per trait
                - reasoning: Reasoning per trait
                - metadata: Per-trait evaluation metadata
                - hallucination_risks: Per-trait hallucination risk (if search enabled)
                - traits_without_valid_excerpts: Traits that failed excerpt extraction
        """
        # Separate deep-judgment vs standard traits
        dj_traits = [t for t in rubric.llm_traits if t.deep_judgment_enabled]
        standard_traits = [t for t in rubric.llm_traits if not t.deep_judgment_enabled]

        # Initialize result containers
        dj_scores: dict[str, int | bool] = {}
        excerpts: dict[str, list[dict[str, Any]]] = {}
        reasoning: dict[str, str] = {}
        metadata: dict[str, dict[str, Any]] = {}
        hallucination_risks: dict[str, dict[str, Any]] = {}
        auto_fail_traits: list[str] = []
        usage_metadata_list: list[dict[str, Any]] = []  # Aggregate usage across all traits

        logger.info(
            f"Evaluating rubric with deep judgment: {len(dj_traits)} DJ traits, {len(standard_traits)} standard traits"
        )

        # Sequential evaluation for deep-judgment traits (one at a time)
        for trait in dj_traits:
            logger.debug(f"Evaluating deep judgment trait: {trait.name}")
            try:
                trait_result = self._evaluate_single_trait_with_deep_judgment(question, answer, trait, config)

                # Collect usage metadata from this trait
                usage_metadata_list.extend(trait_result.get("usage_metadata_list", []))

                # Check for auto-fail
                if trait_result.get("auto_fail"):
                    auto_fail_traits.append(trait.name)
                    metadata[trait.name] = trait_result["metadata"]
                    logger.warning(
                        f"Trait '{trait.name}' auto-failed: no valid excerpts after {trait_result['metadata'].get('excerpt_retry_count', 0)} retries"
                    )
                    continue  # Skip to next trait

                # Store successful evaluation
                dj_scores[trait.name] = trait_result["score"]
                reasoning[trait.name] = trait_result["reasoning"]
                metadata[trait.name] = trait_result["metadata"]

                # Store excerpts if extraction was enabled
                if trait.deep_judgment_excerpt_enabled and "excerpts" in trait_result:
                    excerpts[trait.name] = trait_result["excerpts"]

                    # Store hallucination risk if search was enabled
                    if trait.deep_judgment_search_enabled and "hallucination_risk" in trait_result:
                        hallucination_risks[trait.name] = trait_result["hallucination_risk"]

            except Exception as e:
                logger.error(f"Failed to evaluate deep judgment trait '{trait.name}': {e}")
                # Mark trait as failed
                auto_fail_traits.append(trait.name)
                metadata[trait.name] = {
                    "stages_completed": [],
                    "model_calls": 0,
                    "error": str(e),
                }

        # Evaluate standard traits using existing batch/sequential logic
        standard_scores: dict[str, int | bool] = {}
        standard_usage_metadata_list: list[dict[str, Any]] = []
        if standard_traits:
            logger.debug(f"Evaluating {len(standard_traits)} standard traits")
            standard_rubric = Rubric(llm_traits=standard_traits)
            standard_scores, standard_usage_metadata_list = self.evaluate_rubric(question, answer, standard_rubric)
            usage_metadata_list.extend(standard_usage_metadata_list)

        return {
            "deep_judgment_scores": dj_scores,
            "standard_scores": standard_scores,
            "excerpts": excerpts,
            "reasoning": reasoning,
            "metadata": metadata,
            "hallucination_risks": hallucination_risks,
            "traits_without_valid_excerpts": auto_fail_traits,
            "usage_metadata_list": usage_metadata_list,
        }

    def _evaluate_single_trait_with_deep_judgment(
        self, question: str, answer: str, trait: "LLMRubricTrait", config: Any
    ) -> dict[str, Any]:
        """
        Evaluate a single trait using deep judgment (sequential multi-stage process).

        Returns:
            Dictionary with: score, reasoning, excerpts (optional), metadata, auto_fail flag
        """
        # Initialize metadata
        metadata: dict[str, Any] = {
            "stages_completed": [],
            "model_calls": 0,
            "had_excerpts": trait.deep_judgment_excerpt_enabled,
            "excerpt_retry_count": 0,
            "excerpt_validation_failed": False,
        }

        # Determine flow based on excerpt_enabled
        if trait.deep_judgment_excerpt_enabled:
            # Flow 1: With excerpts (3-4 stages)
            return self._evaluate_trait_with_excerpts(question, answer, trait, config, metadata)
        else:
            # Flow 2: Without excerpts (2 stages)
            return self._evaluate_trait_without_excerpts(question, answer, trait, config, metadata)

    def _evaluate_trait_with_excerpts(
        self, question: str, answer: str, trait: "LLMRubricTrait", config: Any, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Evaluate trait with excerpt extraction (Flow 1: 3-4 stages).

        Stages:
        1. Extract excerpts (with retry on validation failure)
        1.5. Optional: Search-enhanced hallucination assessment
        2. Generate reasoning based on excerpts
        3. Extract final score
        """
        usage_metadata_list = []  # Aggregate usage across all stages

        # Stage 1: Extract excerpts with retry
        excerpt_result = self._extract_excerpts_for_trait(answer, trait, config)
        metadata["model_calls"] += excerpt_result["model_calls"]
        metadata["excerpt_retry_count"] = excerpt_result["retry_count"]
        usage_metadata_list.extend(excerpt_result.get("usage_metadata_list", []))

        # Check for auto-fail
        if excerpt_result.get("auto_fail"):
            metadata["excerpt_validation_failed"] = True
            return {
                "auto_fail": True,
                "metadata": metadata,
                "usage_metadata_list": usage_metadata_list,
            }

        excerpts = excerpt_result["excerpts"]
        metadata["stages_completed"].append("excerpt_extraction")

        # Stage 1.5: Optional search-enhanced hallucination assessment
        hallucination_risk = None
        if trait.deep_judgment_search_enabled and excerpts:
            hallucination_result = self._assess_trait_hallucination(excerpts, trait, config)
            excerpts = hallucination_result["excerpts"]  # Updated with search results
            hallucination_risk = hallucination_result["risk_assessment"]
            metadata["model_calls"] += hallucination_result["model_calls"]
            usage_metadata_list.extend(hallucination_result.get("usage_metadata_list", []))
            metadata["stages_completed"].append("hallucination_assessment")

        # Stage 2: Generate reasoning based on excerpts
        reasoning_result = self._generate_reasoning_for_trait(
            question, answer, trait, excerpts=excerpts, hallucination_risk=hallucination_risk
        )
        reasoning = reasoning_result["reasoning"]
        metadata["model_calls"] += 1
        if reasoning_result.get("usage_metadata"):
            usage_metadata_list.append(reasoning_result["usage_metadata"])
        metadata["stages_completed"].append("reasoning_generation")

        # Stage 3: Extract score
        score_result = self._extract_score_for_trait(question, answer, trait, reasoning)
        score = score_result["score"]
        metadata["model_calls"] += 1
        if score_result.get("usage_metadata"):
            usage_metadata_list.append(score_result["usage_metadata"])
        metadata["stages_completed"].append("score_extraction")

        return {
            "score": score,
            "reasoning": reasoning,
            "excerpts": excerpts,
            "hallucination_risk": hallucination_risk,
            "metadata": metadata,
            "auto_fail": False,
            "usage_metadata_list": usage_metadata_list,
        }

    def _evaluate_trait_without_excerpts(
        self,
        question: str,
        answer: str,
        trait: "LLMRubricTrait",
        config: Any,  # noqa: ARG002
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Evaluate trait without excerpt extraction (Flow 2: 2 stages).

        Stages:
        1. Generate reasoning based on full answer
        2. Extract final score
        """
        usage_metadata_list = []  # Aggregate usage across stages

        # Stage 1: Generate reasoning (no excerpts)
        reasoning_result = self._generate_reasoning_for_trait(
            question, answer, trait, excerpts=None, hallucination_risk=None
        )
        reasoning = reasoning_result["reasoning"]
        metadata["model_calls"] += 1
        if reasoning_result.get("usage_metadata"):
            usage_metadata_list.append(reasoning_result["usage_metadata"])
        metadata["stages_completed"].append("reasoning_generation")

        # Stage 2: Extract score
        score_result = self._extract_score_for_trait(question, answer, trait, reasoning)
        score = score_result["score"]
        metadata["model_calls"] += 1
        if score_result.get("usage_metadata"):
            usage_metadata_list.append(score_result["usage_metadata"])
        metadata["stages_completed"].append("score_extraction")

        return {
            "score": score,
            "reasoning": reasoning,
            "metadata": metadata,
            "auto_fail": False,
            "usage_metadata_list": usage_metadata_list,
        }

    def _extract_excerpts_for_trait(self, answer: str, trait: "LLMRubricTrait", config: Any) -> dict[str, Any]:
        """
        Extract excerpts for a trait with retry on validation failure.

        Returns:
            Dictionary with: excerpts, retry_count, model_calls, auto_fail flag, usage_metadata
        """
        # Get configuration values (per-trait overrides global defaults)
        max_attempts = (
            trait.deep_judgment_excerpt_retry_attempts
            if trait.deep_judgment_excerpt_retry_attempts is not None
            else config.deep_judgment_rubric_excerpt_retry_attempts_default
        )
        fuzzy_threshold = (
            trait.deep_judgment_fuzzy_match_threshold
            if trait.deep_judgment_fuzzy_match_threshold is not None
            else config.deep_judgment_rubric_fuzzy_match_threshold_default
        )
        max_excerpts = (
            trait.deep_judgment_max_excerpts
            if trait.deep_judgment_max_excerpts is not None
            else config.deep_judgment_rubric_max_excerpts_default
        )

        retry_count = 0
        model_calls = 0
        validation_feedback = None
        usage_metadata_list = []  # Track usage across retries

        # Retry loop
        for attempt in range(max_attempts + 1):  # Initial + retries
            # Build prompt (with feedback if retry)
            prompt = self._build_trait_excerpt_prompt(trait, max_excerpts, answer, validation_feedback)

            # Call LLM
            messages = [
                SystemMessage(
                    content="You are an expert at extracting verbatim quotes from text that demonstrate specific qualities."
                ),
                HumanMessage(content=prompt),
            ]

            with get_usage_metadata_callback() as cb:
                response = self.llm.invoke(messages)
            model_calls += 1
            if cb.usage_metadata:
                usage_metadata_list.append(dict(cb.usage_metadata))

            raw_response = response.content if hasattr(response, "content") else str(response)

            # Parse excerpts
            try:
                raw_excerpts = self._parse_excerpt_response(raw_response)
            except Exception as e:
                logger.warning(f"Failed to parse excerpt response for trait '{trait.name}': {e}")
                if attempt < max_attempts:
                    validation_feedback = (
                        f"Failed to parse response: {e}. Please return valid JSON with an 'excerpts' array."
                    )
                    retry_count += 1
                    continue
                else:
                    # All retries exhausted
                    return {
                        "excerpts": [],
                        "retry_count": retry_count,
                        "model_calls": model_calls,
                        "auto_fail": True,
                        "usage_metadata_list": usage_metadata_list,
                    }

            # Validate excerpts with fuzzy matching
            validated, failed = self._validate_trait_excerpts(raw_excerpts, answer, fuzzy_threshold)

            if validated:
                # Success!
                return {
                    "excerpts": validated,
                    "retry_count": retry_count,
                    "model_calls": model_calls,
                    "auto_fail": False,
                    "usage_metadata_list": usage_metadata_list,
                }

            # All excerpts failed validation
            if attempt < max_attempts:
                # Build feedback for retry
                validation_feedback = self._build_retry_feedback(failed, fuzzy_threshold)
                retry_count += 1
                logger.debug(
                    f"Trait '{trait.name}' excerpt validation failed (attempt {attempt + 1}), retrying with feedback"
                )
            else:
                # Exhausted all retries
                logger.warning(f"Trait '{trait.name}' failed excerpt extraction after {retry_count} retries")
                return {
                    "excerpts": [],
                    "retry_count": retry_count,
                    "model_calls": model_calls,
                    "auto_fail": True,
                    "usage_metadata_list": usage_metadata_list,
                }

        # Should not reach here
        return {
            "excerpts": [],
            "retry_count": retry_count,
            "model_calls": model_calls,
            "auto_fail": True,
            "usage_metadata_list": usage_metadata_list,
        }

    def _validate_trait_excerpts(
        self, excerpts: list[dict[str, Any]], answer: str, fuzzy_threshold: float
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Validate excerpts using fuzzy matching.

        Returns:
            Tuple of (valid_excerpts, failed_excerpts)
        """
        from ..tools.fuzzy_match import fuzzy_match_excerpt

        valid = []
        failed = []

        for excerpt in excerpts:
            excerpt_text = excerpt.get("text", "")
            match_found, similarity = fuzzy_match_excerpt(excerpt_text, answer)

            if match_found and similarity >= fuzzy_threshold:
                excerpt["similarity_score"] = similarity
                valid.append(excerpt)
            else:
                excerpt["similarity_score"] = similarity
                failed.append(excerpt)

        return valid, failed

    def _build_retry_feedback(self, failed_excerpts: list[dict[str, Any]], fuzzy_threshold: float) -> str:
        """
        Build feedback message for retry attempt.

        Same format as template Deep Judgment.
        """
        feedback = "The following excerpts failed validation (not found in answer):\n"

        for i, excerpt in enumerate(failed_excerpts, 1):
            feedback += (
                f'{i}. "{excerpt.get("text", "")}" '
                f"(similarity: {excerpt.get('similarity_score', 0):.2f}, "
                f"threshold: {fuzzy_threshold:.2f})\n"
            )

        feedback += "\nPlease provide verbatim quotes that exactly match the answer text."
        return feedback

    def _build_trait_excerpt_prompt(
        self, trait: "LLMRubricTrait", max_excerpts: int, answer: str, feedback: str | None = None
    ) -> str:
        """Build prompt for excerpt extraction (with optional retry feedback)."""
        prompt = f"""Extract verbatim quotes from the answer that demonstrate the following quality trait:

**Trait**: {trait.name}
**Criteria**: {trait.description or "Assess this quality"}

**Answer to analyze**:
{answer}

**Task**:
Extract up to {max_excerpts} verbatim quotes from the answer that demonstrate or relate to this trait.

For each quote, assign a confidence level:
- "high": Strong evidence for the trait
- "medium": Moderate evidence
- "low": Weak or ambiguous evidence

**CRITICAL**: Quotes must be EXACT verbatim text from the answer. Do not paraphrase or modify.
"""

        if feedback:
            prompt += f"\n**RETRY FEEDBACK**:\n{feedback}\n"

        prompt += """
Return your response as JSON:
{
  "excerpts": [
    {"text": "exact quote 1", "confidence": "high"},
    {"text": "exact quote 2", "confidence": "medium"}
  ]
}

JSON Response:"""

        return prompt

    def _parse_excerpt_response(self, response: str) -> list[dict[str, Any]]:
        """Parse the excerpt extraction response."""
        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response[:200]}")

        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}") from e

        excerpts = result.get("excerpts", [])
        if not isinstance(excerpts, list):
            raise ValueError(f"'excerpts' must be a list, got {type(excerpts)}")

        return excerpts

    def _assess_trait_hallucination(
        self, excerpts: list[dict[str, Any]], trait: "LLMRubricTrait", config: Any
    ) -> dict[str, Any]:
        """
        Assess hallucination risk for trait excerpts using search.

        Returns:
            Dictionary with: excerpts (updated with search results), risk_assessment, model_calls, usage_metadata_list
        """
        from ..tools.search_tools import create_search_tool

        # Create search tool
        search_tool = create_search_tool(config.deep_judgment_rubric_search_tool)

        # Batch search for all excerpts
        excerpt_texts = [e.get("text", "") for e in excerpts]
        try:
            search_results = search_tool(excerpt_texts)
            # Ensure it's a list
            if not isinstance(search_results, list):
                search_results = [search_results] * len(excerpt_texts)
        except Exception as e:
            logger.warning(f"Search failed for trait '{trait.name}': {e}")
            search_results = ["Search failed"] * len(excerpt_texts)  # type: ignore[assignment]

        # Assess hallucination risk per excerpt
        model_calls = 0
        per_excerpt_risks = []
        usage_metadata_list = []  # Track usage across hallucination assessments

        for i, excerpt in enumerate(excerpts):
            # Build prompt for hallucination assessment
            prompt = f"""Assess the hallucination risk for this excerpt:

**Excerpt**: {excerpt.get("text", "")}

**Search Results**:
{search_results[i] if i < len(search_results) else "No results"}

Based on the search results, assess the risk that this excerpt contains hallucinated or unverifiable information:
- "none": Strong external evidence supports this
- "low": Some external evidence, likely accurate
- "medium": Weak or ambiguous external evidence
- "high": No external evidence or contradicted by sources

Return only the risk level (none/low/medium/high) and a brief justification.

JSON Response:
{{"risk": "none|low|medium|high", "justification": "brief explanation"}}
"""

            messages = [
                SystemMessage(content="You are an expert at assessing hallucination risk using external evidence."),
                HumanMessage(content=prompt),
            ]

            with get_usage_metadata_callback() as cb:
                response = self.llm.invoke(messages)
            model_calls += 1
            if cb.usage_metadata:
                usage_metadata_list.append(dict(cb.usage_metadata))

            raw_response = response.content if hasattr(response, "content") else str(response)

            # Parse response
            try:
                json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if json_match:
                    risk_data = json.loads(json_match.group())
                    risk = risk_data.get("risk", "medium")
                    justification = risk_data.get("justification", "")
                else:
                    risk = "medium"
                    justification = "Failed to parse response"
            except Exception as e:
                logger.warning(f"Failed to parse hallucination assessment: {e}")
                risk = "medium"
                justification = "Failed to parse"

            # Update excerpt with search data
            excerpt["search_results"] = search_results[i] if i < len(search_results) else None
            excerpt["hallucination_risk"] = risk
            excerpt["hallucination_justification"] = justification
            per_excerpt_risks.append(risk)

        # Overall risk = MAX of per-excerpt risks
        risk_levels = {"none": 0, "low": 1, "medium": 2, "high": 3}
        max_risk_level = max((risk_levels.get(r, 2) for r in per_excerpt_risks), default=2)
        overall_risk = [k for k, v in risk_levels.items() if v == max_risk_level][0]

        return {
            "excerpts": excerpts,
            "risk_assessment": {
                "overall_risk": overall_risk,
                "per_excerpt_risks": per_excerpt_risks,
            },
            "model_calls": model_calls,
            "usage_metadata_list": usage_metadata_list,
        }

    def _generate_reasoning_for_trait(
        self,
        question: str,
        answer: str,
        trait: "LLMRubricTrait",
        excerpts: list[dict[str, Any]] | None = None,
        hallucination_risk: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate reasoning explaining the trait score.

        Returns:
            Dictionary with: reasoning (string), usage_metadata
        """
        if excerpts is not None:
            # With excerpts
            prompt = self._build_trait_reasoning_prompt_with_excerpts(
                question, answer, trait, excerpts, hallucination_risk
            )
        else:
            # Without excerpts
            prompt = self._build_trait_reasoning_prompt_without_excerpts(question, answer, trait)

        messages = [
            SystemMessage(content="You are an expert at analyzing text quality and providing clear reasoning."),
            HumanMessage(content=prompt),
        ]

        with get_usage_metadata_callback() as cb:
            response = self.llm.invoke(messages)

        raw_response = response.content if hasattr(response, "content") else str(response)
        reasoning = raw_response.strip()

        usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}
        return {"reasoning": reasoning, "usage_metadata": usage_metadata}

    def _build_trait_reasoning_prompt_with_excerpts(
        self,
        question: str,
        answer: str,  # noqa: ARG002
        trait: "LLMRubricTrait",
        excerpts: list[dict[str, Any]],
        hallucination_risk: dict[str, Any] | None = None,
    ) -> str:
        """Build reasoning prompt with excerpts."""
        # Format excerpts
        excerpts_formatted = []
        for i, excerpt in enumerate(excerpts, 1):
            conf = excerpt.get("confidence", "unknown")
            text = excerpt.get("text", "")
            risk = excerpt.get("hallucination_risk", "")
            risk_str = f" (hallucination risk: {risk})" if risk else ""
            excerpts_formatted.append(f'{i}. "{text}" [{conf} confidence]{risk_str}')

        excerpts_text = "\n".join(excerpts_formatted) if excerpts_formatted else "No excerpts found."

        risk_context = ""
        if hallucination_risk:
            risk_context = f"\n**Overall Hallucination Risk**: {hallucination_risk.get('overall_risk', 'unknown')}\n"

        return f"""Based on the extracted excerpts, explain how they demonstrate (or fail to demonstrate) the following trait:

**Trait**: {trait.name}
**Criteria**: {trait.description or "Assess this quality"}

**Question**: {question}

**Extracted Excerpts**:
{excerpts_text}
{risk_context}

Provide 2-3 sentences explaining how these excerpts inform your assessment of this trait.
Be specific about what the excerpts reveal about the answer's quality for this trait.

Your reasoning:"""

    def _build_trait_reasoning_prompt_without_excerpts(
        self, question: str, answer: str, trait: "LLMRubricTrait"
    ) -> str:
        """Build reasoning prompt without excerpts (based on full response)."""
        return f"""Analyze the following answer for the quality trait below and explain your assessment:

**Trait**: {trait.name}
**Criteria**: {trait.description or "Assess this quality"}

**Question**: {question}

**Complete Answer**:
{answer}

Provide 2-3 sentences explaining how this answer demonstrates (or fails to demonstrate) this trait.
Base your reasoning on the complete answer text and the trait criteria.

Your reasoning:"""

    def _extract_score_for_trait(
        self,
        question: str,  # noqa: ARG002
        answer: str,  # noqa: ARG002
        trait: "LLMRubricTrait",
        reasoning: str,
    ) -> dict[str, Any]:
        """
        Extract final score for trait based on reasoning.

        Returns:
            Dictionary with: score (int | bool), usage_metadata
        """
        prompt = self._build_trait_scoring_prompt(trait, reasoning)

        messages = [
            SystemMessage(content="You are an expert evaluator providing precise trait scores."),
            HumanMessage(content=prompt),
        ]

        with get_usage_metadata_callback() as cb:
            response = self.llm.invoke(messages)

        raw_response = response.content if hasattr(response, "content") else str(response)

        # Parse score
        score = self._parse_trait_score_response(raw_response, trait)

        usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}
        return {"score": score, "usage_metadata": usage_metadata}

    def _build_trait_scoring_prompt(self, trait: "LLMRubricTrait", reasoning: str) -> str:
        """Build prompt for final scoring."""
        if trait.kind == "boolean":
            return f"""Based on the following reasoning, provide a boolean score for this trait:

**Trait**: {trait.name}
**Criteria**: {trait.description or "Boolean evaluation"}

**Reasoning**:
{reasoning}

Based on this reasoning, does the answer meet the criteria for this trait?

Respond with only "true" or "false".

Your score:"""
        else:
            min_score = trait.min_score or 1
            max_score = trait.max_score or 5
            return f"""Based on the following reasoning, provide a numeric score for this trait:

**Trait**: {trait.name}
**Criteria**: {trait.description or "Score-based evaluation"}
**Scale**: {min_score} (poor/does not meet criteria) to {max_score} (excellent/fully meets criteria)

**Reasoning**:
{reasoning}

Based on this reasoning, what score ({min_score}-{max_score}) should this answer receive?

Respond with only the numeric score.

Your score:"""

    def _parse_trait_score_response(self, response: str, trait: "LLMRubricTrait") -> int | bool:
        """Parse a trait score response."""
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
                    logger.warning(f"Could not parse boolean from: {response}, defaulting to False")
                    return False
        else:
            # Extract numeric score
            numbers = re.findall(r"\d+", response)
            if not numbers:
                logger.warning(f"No numeric score found in: {response}, defaulting to minimum score")
                return trait.min_score or 1

            score = int(numbers[0])
            # Validate and clamp score
            min_score = trait.min_score or 1
            max_score = trait.max_score or 5
            clamped_score = max(min_score, min(max_score, score))

            if clamped_score != score:
                logger.debug(f"Score {score} clamped to valid range [{min_score}, {max_score}]: {clamped_score}")

            return clamped_score
