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
        config: Any,  # noqa: ARG002 - Kept for method signature consistency
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
                    content="""You are an expert at extracting verbatim quotes from text that demonstrate specific qualities.

You will receive a JSON Schema specifying the exact output structure. Your response MUST conform to this schema.
Return ONLY a JSON object - no explanations, no markdown, no surrounding text.

**CRITICAL REQUIREMENTS:**
1. **JSON ONLY**: Your entire response must be valid JSON conforming to the provided schema
2. **VERBATIM QUOTES**: Excerpts must be EXACT text from the answer - do not paraphrase
3. **Confidence Levels**: Assign confidence based on strength of evidence:
   - "high": Direct, explicit evidence for the trait
   - "medium": Reasonable inference or moderate evidence
   - "low": Weak or ambiguous evidence

**WHAT NOT TO DO:**
- Do NOT wrap JSON in markdown code blocks (no ```)
- Do NOT add explanatory text before or after the JSON
- Do NOT paraphrase or modify quotes
- Do NOT invent quotes not present in the text"""
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
        from ....schemas.workflow.rubric_outputs import TraitExcerptsOutput

        json_schema = json.dumps(TraitExcerptsOutput.model_json_schema(), indent=2)

        prompt = f"""Extract verbatim quotes from the answer that demonstrate the following quality trait:

**TRAIT:** {trait.name}
**CRITERIA:** {trait.description or "Assess this quality"}

**ANSWER TO ANALYZE:**
{answer}

**TASK:**
Extract up to {max_excerpts} verbatim quotes from the answer that demonstrate or relate to this trait.

**CONFIDENCE LEVELS:**
- "high": Direct, explicit statement that clearly demonstrates the trait
- "medium": Indirect evidence or reasonable inference supporting the trait
- "low": Weak, ambiguous, or tangential evidence

**IMPORTANT RULES:**
1. Quotes MUST be EXACT verbatim text from the answer above
2. Do not paraphrase, summarize, or modify the text in any way
3. If no relevant excerpts exist, return an empty excerpts array: {{"excerpts": []}}
4. Select the most relevant excerpts - quality over quantity
"""

        if feedback:
            prompt += f"""
**RETRY FEEDBACK (previous excerpts failed validation):**
{feedback}
"""

        prompt += f"""
**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**PARSING NOTES:**
- We validate excerpts using fuzzy matching against the original answer
- Excerpts that don't match will be rejected and may trigger a retry
- Minor whitespace differences are tolerated

**YOUR JSON RESPONSE:**"""

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
            from ....schemas.workflow.rubric_outputs import HallucinationRiskOutput

            json_schema = json.dumps(HallucinationRiskOutput.model_json_schema(), indent=2)

            prompt = f"""Assess the hallucination risk for this excerpt by comparing it against external search results.

**EXCERPT TO VERIFY:**
"{excerpt.get("text", "")}"

**EXTERNAL SEARCH RESULTS:**
{search_results[i] if i < len(search_results) else "No results available"}

**RISK LEVELS (choose one):**
- "none": Strong external evidence supports this - multiple reliable sources confirm
- "low": Some external evidence, likely accurate - at least one source supports
- "medium": Weak or ambiguous evidence - sources unclear or partially contradict
- "high": No supporting evidence or actively contradicted by external sources

**EVALUATION GUIDELINES:**
1. Compare the excerpt's claims against the search results
2. Consider the reliability and specificity of sources
3. When uncertain between adjacent levels, choose the more conservative (higher risk) option
4. Provide a brief justification explaining your reasoning

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**PARSING NOTES:**
- Return ONLY valid JSON - no surrounding text or markdown
- The "risk" field must be exactly one of: "none", "low", "medium", "high"

**YOUR JSON RESPONSE:**"""

            messages = [
                SystemMessage(
                    content="""You are an expert at assessing hallucination risk using external evidence.

You will receive a JSON Schema specifying the exact output structure. Your response MUST conform to this schema.
Return ONLY a JSON object - no explanations, no markdown, no surrounding text.

**YOUR ROLE:**
Compare excerpts against external search results to determine if the information is factually supported.

**CRITICAL REQUIREMENTS:**
1. **JSON ONLY**: Your entire response must be valid JSON conforming to the provided schema
2. **Conservative Assessment**: When uncertain, lean toward higher risk levels
3. **Evidence-Based**: Base your assessment solely on the search results provided

**WHAT NOT TO DO:**
- Do NOT wrap JSON in markdown code blocks (no ```)
- Do NOT add explanatory text before or after the JSON
- Do NOT assume claims are true without supporting evidence"""
                ),
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
        answer: str,  # noqa: ARG002 - Excerpts already contain relevant answer portions
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

        return f"""Analyze the extracted excerpts to explain how they demonstrate (or fail to demonstrate) the following trait.

**Trait**: {trait.name}
**Criteria**: {trait.description or "Assess this quality"}

**Question**: {question}

**Extracted Excerpts**:
{excerpts_text}
{risk_context}

**Your Task**:
Provide 2-3 sentences of reasoning that:
1. Reference specific excerpts and their content
2. Connect each excerpt to the trait criteria
3. Assess whether the excerpts collectively satisfy the trait

This reasoning will be used in a follow-up step to determine the final score.

**Your reasoning:**"""

    def _build_trait_reasoning_prompt_without_excerpts(
        self, question: str, answer: str, trait: "LLMRubricTrait"
    ) -> str:
        """Build reasoning prompt without excerpts (based on full response)."""
        return f"""Analyze the following answer for the quality trait below and provide your reasoning.

**Trait**: {trait.name}
**Criteria**: {trait.description or "Assess this quality"}

**Question**: {question}

**Complete Answer**:
{answer}

**Your Task**:
Provide 2-3 sentences of reasoning that:
1. Identify specific aspects of the answer relevant to this trait
2. Explain how these aspects satisfy or fail to satisfy the criteria
3. Consider both positive and negative evidence

This reasoning will be used in a follow-up step to determine the final score.

**Your reasoning:**"""

    def _extract_score_for_trait(
        self,
        question: str,  # noqa: ARG002 - Kept for API consistency with other trait methods
        answer: str,  # noqa: ARG002 - Kept for API consistency with other trait methods
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
            SystemMessage(
                content="""You are an expert evaluator providing precise trait scores based on prior reasoning.

You will receive a JSON Schema specifying the exact output structure. Your response MUST conform to this schema.
Return ONLY a JSON object - no explanations, no markdown, no surrounding text.

**YOUR ROLE:**
Convert analytical reasoning into a final score that accurately reflects the assessment.

**CRITICAL REQUIREMENTS:**
1. **JSON ONLY**: Your entire response must be valid JSON conforming to the provided schema
2. **Reasoning-Based**: Base your score solely on the reasoning provided
3. **Consistency**: Your score should logically follow from the reasoning's conclusions

**SCORING GUIDELINES:**
- Be consistent: similar reasoning should lead to similar scores
- When uncertain, choose conservatively based on the trait's nature:
  - For positive traits (e.g., "is accurate"), lean toward `false` or lower scores
  - For negative traits (e.g., "contains errors"), lean toward `true` or higher scores

**WHAT NOT TO DO:**
- Do NOT wrap JSON in markdown code blocks (no ```)
- Do NOT add explanatory text before or after the JSON
- Do NOT contradict the reasoning - your score should align with it"""
            ),
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
        from ....schemas.workflow.rubric_outputs import SingleBooleanScore, SingleNumericScore

        if trait.kind == "boolean":
            json_schema = json.dumps(SingleBooleanScore.model_json_schema(), indent=2)
            return f"""Based on the following reasoning, provide a final score for this trait.

**TRAIT:** {trait.name}
**CRITERIA:** {trait.description or "Boolean evaluation"}

**YOUR PREVIOUS REASONING:**
{reasoning}

**SCORE REQUIRED:** true or false

Based on your reasoning above, does the answer meet the criteria?
- `true`: The criteria IS met based on your reasoning
- `false`: The criteria IS NOT met based on your reasoning

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**PARSING NOTES:**
- Return valid JSON: {{"result": true}} or {{"result": false}}
- We also accept plain text: "true", "yes", "false", "no"
- Use lowercase boolean values (not "True" or "False")

**YOUR JSON RESPONSE:**"""
        else:
            min_score = trait.min_score or 1
            max_score = trait.max_score or 5
            mid_score = (min_score + max_score) // 2
            json_schema = json.dumps(SingleNumericScore.model_json_schema(), indent=2)
            return f"""Based on the following reasoning, provide a final score for this trait.

**TRAIT:** {trait.name}
**CRITERIA:** {trait.description or "Score-based evaluation"}

**YOUR PREVIOUS REASONING:**
{reasoning}

**SCORING SCALE:**
- {min_score} = Poor - Does not meet criteria at all
- {mid_score} = Average - Partially meets criteria
- {max_score} = Excellent - Fully meets or exceeds criteria

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**PARSING NOTES:**
- Return valid JSON: {{"score": N}} where N is an integer from {min_score} to {max_score}
- Scores outside [{min_score}, {max_score}] are automatically clamped to boundaries
- Use integers only (no decimals)

**YOUR JSON RESPONSE:**"""

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
