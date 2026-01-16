"""
Rubric evaluation for qualitative assessment of LLM responses.

This module contains the main RubricEvaluator class which handles evaluation of
LLM responses against rubric traits. It supports multiple trait types:

- LLM traits: Subjective assessment using LLM-as-judge
- Regex traits: Pattern matching for format compliance
- Callable traits: Custom Python functions for deterministic checks
- Metric traits: Precision/recall/F1 metrics using confusion matrix

For deep judgment evaluation (multi-stage excerpt-based assessment), the evaluator
delegates to the RubricDeepJudgmentHandler class in rubric_deep_judgment.py.
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ....infrastructure.llm.interface import init_chat_model_unified
from ....schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait, Rubric
from ....schemas.workflow import INTERFACES_NO_PROVIDER_REQUIRED, ModelConfig
from .rubric_deep_judgment import RubricDeepJudgmentHandler

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

    def _evaluate_deterministic_traits(
        self,
        answer: str,
        traits: list[RegexTrait] | list[CallableTrait],
        trait_type_name: str,
    ) -> dict[str, bool | int]:
        """
        Evaluate deterministic traits (regex or callable) using their evaluate() method.

        This is a generic helper that consolidates the common iteration pattern
        used by both regex and callable trait evaluation.

        Args:
            answer: The text to evaluate
            traits: List of traits to evaluate (RegexTrait or CallableTrait)
            trait_type_name: Human-readable name for logging (e.g., "regex", "callable")

        Returns:
            Dictionary mapping trait names to their evaluated results.
            Failed traits are marked as None for consistency with LLM evaluation.
        """
        results: dict[str, bool | int] = {}

        for trait in traits:
            try:
                result = trait.evaluate(answer)
                results[trait.name] = result
            except Exception as e:
                logger.warning(f"Failed to evaluate {trait_type_name} trait '{trait.name}': {e}")
                # Mark failed traits as None for consistency with LLM evaluation
                results[trait.name] = None  # type: ignore[assignment]

        return results

    def _evaluate_regex_traits(self, answer: str, regex_traits: list[RegexTrait]) -> dict[str, bool]:
        """
        Evaluate regex traits using pattern matching.

        Args:
            answer: The text to evaluate
            regex_traits: List of regex traits to evaluate

        Returns:
            Dictionary mapping trait names to boolean results
        """
        # Type narrowing: regex traits always return bool
        return self._evaluate_deterministic_traits(answer, regex_traits, "regex")  # type: ignore[return-value]

    def _evaluate_callable_traits(self, answer: str, callable_traits: list[CallableTrait]) -> dict[str, bool | int]:
        """
        Evaluate callable traits using custom functions.

        Args:
            answer: The text to evaluate
            callable_traits: List of callable traits to evaluate

        Returns:
            Dictionary mapping trait names to boolean or int results (depending on trait kind)
        """
        return self._evaluate_deterministic_traits(answer, callable_traits, "callable")

    def _evaluate_batch(
        self, question: str, answer: str, rubric: Rubric
    ) -> tuple[dict[str, int | bool], dict[str, Any]]:
        """
        Evaluate all traits in a single LLM call using LangChain strategies.

        Strategy order:
        1. json_schema method (native structured output)
        2. Manual parsing with json-repair

        Returns:
            Tuple of (results_dict, usage_metadata)
        """
        from ....schemas.workflow.rubric_outputs import BatchRubricScores
        from .rubric_parsing import invoke_with_structured_output

        system_prompt = self._build_batch_system_prompt()
        user_prompt = self._build_batch_user_prompt(question, answer, rubric.llm_traits)

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        # Invoke with automatic strategy selection and fallbacks
        parsed_result, usage_metadata = invoke_with_structured_output(self.llm, messages, BatchRubricScores)

        # Validate scores against trait definitions
        results = self._validate_batch_scores(parsed_result.scores, rubric.llm_traits)
        return results, usage_metadata

    def _validate_batch_scores(
        self, scores: dict[str, int | bool], traits: list[LLMRubricTrait]
    ) -> dict[str, int | bool]:
        """Validate and normalize batch scores against trait definitions."""
        validated_results: dict[str, int | bool] = {}
        trait_map = {trait.name: trait for trait in traits}

        for trait_name, score in scores.items():
            if trait_name in trait_map:
                trait = trait_map[trait_name]
                validated_score = self._validate_score(score, trait)
                validated_results[trait_name] = validated_score

        # Add None for missing traits
        for trait in traits:
            if trait.name not in validated_results:
                validated_results[trait.name] = None  # type: ignore[assignment]

        return validated_results

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
        Evaluate a single trait using LangChain strategies.

        Strategy order:
        1. json_schema method (native structured output)
        2. Manual parsing with json-repair

        Returns:
            Tuple of (score, usage_metadata)
        """
        from ....schemas.workflow.rubric_outputs import SingleBooleanScore, SingleNumericScore
        from .rubric_parsing import invoke_with_structured_output

        system_prompt = self._build_single_trait_system_prompt(trait)
        user_prompt = self._build_single_trait_user_prompt(question, answer, trait)

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        # Choose model class based on trait kind
        model_class = SingleBooleanScore if trait.kind == "boolean" else SingleNumericScore

        # Invoke with automatic strategy selection and fallbacks
        parsed_result, usage_metadata = invoke_with_structured_output(self.llm, messages, model_class)

        # Extract score from result
        score: int | bool = parsed_result.result if trait.kind == "boolean" else parsed_result.score  # type: ignore[attr-defined]

        return self._validate_score(score, trait), usage_metadata

    def _build_batch_system_prompt(self) -> str:
        """Build system prompt for batch evaluation."""
        return """You are an expert evaluator assessing the quality of responses using a structured rubric.

Your task is to evaluate a given answer against multiple evaluation traits and return scores in JSON format.

**RESPONSE FORMAT:**
You will receive a JSON Schema specifying the exact output structure. Your response MUST conform to this schema.
Return ONLY a JSON object - no explanations, no markdown, no surrounding text.

**CRITICAL REQUIREMENTS:**
1. **JSON ONLY**: Your entire response must be valid JSON conforming to the provided schema
2. **Schema Compliance**: Follow the JSON Schema exactly - it defines the structure for parsing
3. **Exact Trait Names**: Use the EXACT trait names provided - case-sensitive matching is required
4. **Boolean Traits**: Use JSON `true` or `false` (lowercase, no quotes around the value)
5. **Score Traits**: Use integers within the specified range
6. **Score Clamping**: Scores outside the valid range will be automatically clamped to the nearest boundary
7. **Missing Traits**: Any trait not included will be recorded as `null` (evaluation failure)

**EVALUATION GUIDELINES:**
- Evaluate each trait independently based on its specific criteria
- When uncertain, choose the most conservative/defensive value based on the trait's intent:
  - For positive traits (e.g., "is accurate"), lean toward `false` when uncertain
  - For negative traits (e.g., "contains errors"), lean toward `true` when uncertain
  - For scores, lean toward the middle of the scale when uncertain
- Base assessments solely on the answer content, not assumptions about intent
- Be consistent: similar answers should receive similar scores

**WHAT NOT TO DO:**
- Do NOT wrap JSON in markdown code blocks (no ```)
- Do NOT add explanatory text before or after the JSON
- Do NOT use string values like "true" - use the boolean `true`
- Do NOT skip any traits - include ALL traits in your response"""

    def _build_batch_user_prompt(self, question: str, answer: str, traits: list[LLMRubricTrait]) -> str:
        """Build user prompt for batch evaluation."""
        from ....schemas.workflow.rubric_outputs import BatchRubricScores

        traits_description = []
        trait_names = []

        for trait in traits:
            trait_names.append(trait.name)
            if trait.kind == "boolean":
                trait_desc = f"- **{trait.name}** (boolean): {trait.description or 'Boolean evaluation'}\n  → Return `true` or `false`"
            else:
                min_score = trait.min_score or 1
                max_score = trait.max_score or 5
                trait_desc = f"- **{trait.name}** (score {min_score}-{max_score}): {trait.description or 'Score-based evaluation'}\n  → Return integer from {min_score} to {max_score}"
            traits_description.append(trait_desc)

        # Build example JSON with actual trait names
        example_scores: dict[str, bool | int] = {}
        for trait in traits:
            if trait.kind == "boolean":
                example_scores[trait.name] = True
            else:
                example_scores[trait.name] = (trait.min_score or 1) + 2

        example_json = json.dumps({"scores": example_scores}, indent=2)

        # Get JSON schema from Pydantic model
        json_schema = json.dumps(BatchRubricScores.model_json_schema(), indent=2)

        return f"""Evaluate the following answer against these traits:

**TRAITS TO EVALUATE:**
{chr(10).join(traits_description)}

**QUESTION:**
{question}

**ANSWER TO EVALUATE:**
{answer}

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**REQUIRED OUTPUT FORMAT:**
Return a JSON object with a "scores" key. Use EXACT trait names as shown above.

Example (using YOUR trait names):
{example_json}

**YOUR JSON RESPONSE:**"""

    def _build_single_trait_system_prompt(self, trait: LLMRubricTrait) -> str:
        """Build system prompt for single trait evaluation."""
        if trait.kind == "boolean":
            return f"""You are evaluating responses for the trait: **{trait.name}**

**Criteria:** {trait.description or "Boolean evaluation"}

**RESPONSE FORMAT:**
You will receive a JSON Schema specifying the exact output structure. Your response MUST conform to this schema.
Return valid JSON: {{"result": true}} or {{"result": false}}
Keep your response concise - the JSON is the primary output.

**EVALUATION GUIDELINES:**
- `true`: The criteria IS met - answer clearly satisfies the requirement
- `false`: The criteria IS NOT met - answer fails to satisfy the requirement
- When uncertain, choose the most conservative value based on the trait's nature:
  - For positive traits (e.g., "is accurate"), lean toward `false`
  - For negative traits (e.g., "contains errors"), lean toward `true`

**PARSING NOTES:**
- If JSON parsing fails, we also accept: "true", "yes", "false", "no"
- Do NOT use string values like "true" with quotes - use the boolean
- Avoid wrapping in markdown code blocks"""
        else:
            min_score = trait.min_score or 1
            max_score = trait.max_score or 5
            mid_score = (min_score + max_score) // 2
            return f"""You are evaluating responses for the trait: **{trait.name}**

**Criteria:** {trait.description or "Score-based evaluation"}

**RESPONSE FORMAT:**
You will receive a JSON Schema specifying the exact output structure. Your response MUST conform to this schema.
Return valid JSON: {{"score": N}} where N is an integer from {min_score} to {max_score}
Keep your response concise - the JSON is the primary output.

**SCORING GUIDELINES:**
- {min_score} = Poor - Does not meet criteria at all
- {mid_score} = Average - Partially meets criteria
- {max_score} = Excellent - Fully meets or exceeds criteria

When uncertain about borderline cases, choose conservatively based on the trait's nature:
- For traits where higher is better (e.g., "quality"), lean toward lower scores
- For traits where lower is better (e.g., "error severity"), lean toward higher scores
- When completely uncertain, default to the middle value ({mid_score})

**PARSING NOTES:**
- Scores outside [{min_score}, {max_score}] are automatically clamped to the nearest boundary
- Use integers only (no decimals)
- Avoid wrapping in markdown code blocks"""

    def _build_single_trait_user_prompt(self, question: str, answer: str, trait: LLMRubricTrait) -> str:
        """Build user prompt for single trait evaluation."""
        from ....schemas.workflow.rubric_outputs import SingleBooleanScore, SingleNumericScore

        if trait.kind == "boolean":
            format_hint = '{"result": true} or {"result": false}'
            json_schema = json.dumps(SingleBooleanScore.model_json_schema(), indent=2)
        else:
            format_hint = '{"score": N}'
            json_schema = json.dumps(SingleNumericScore.model_json_schema(), indent=2)

        return f"""**QUESTION:**
{question}

**ANSWER TO EVALUATE:**
{answer}

**TRAIT:** {trait.name}
**CRITERIA:** {trait.description or "No description provided"}

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

Evaluate this answer for the trait above and return your assessment as JSON: {format_hint}"""

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
        Evaluate a single metric trait using LangChain strategies.

        Strategy order:
        1. json_schema method (native structured output)
        2. Manual parsing with json-repair

        Args:
            question: The original question
            answer: The answer to evaluate
            trait: The metric trait to evaluate

        Returns:
            Tuple of (confusion_lists, metrics, usage_metadata)
        """
        from ....schemas.workflow.rubric_outputs import ConfusionMatrixOutput
        from .rubric_parsing import invoke_with_structured_output

        # Build prompt
        system_prompt = self._build_metric_trait_system_prompt()
        user_prompt = self._build_metric_trait_user_prompt(question, answer, trait)

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        # Invoke with automatic strategy selection and fallbacks
        parsed_result, usage_metadata = invoke_with_structured_output(self.llm, messages, ConfusionMatrixOutput)

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
        from ....schemas.workflow.rubric_outputs import ConfusionMatrixOutput

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

        This method delegates to RubricDeepJudgmentHandler for the actual
        multi-stage evaluation process.

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
        # Create handler with the same LLM instance
        handler = RubricDeepJudgmentHandler(self.llm, self.model_config)

        # Delegate to handler, providing a callback for standard trait evaluation
        return handler.evaluate_rubric_with_deep_judgment(
            question=question,
            answer=answer,
            rubric=rubric,
            config=config,
            standard_evaluator_fn=self.evaluate_rubric,
        )
