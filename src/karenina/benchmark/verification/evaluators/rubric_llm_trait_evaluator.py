"""LLM-based trait evaluation for rubric assessment.

This module implements the LLMTraitEvaluator class, which handles evaluation
of LLMRubricTrait instances using LLM-as-judge for subjective assessments.

Two evaluation strategies are supported:
- batch: Evaluate all traits in a single LLM call (efficient)
- sequential: Evaluate traits one-by-one (reliable)

Supports three trait kinds:
- boolean: Binary true/false assessment
- score: Numeric rating within a range (e.g., 1-5)
- literal: Categorical classification into predefined classes
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ....schemas.domain import LLMRubricTrait

logger = logging.getLogger(__name__)


class LLMTraitEvaluator:
    """
    Evaluates LLM rubric traits using LLM-as-judge.

    This evaluator handles LLMRubricTrait evaluation by prompting an LLM
    to assess response quality against trait criteria. Supports three kinds:
    - boolean (pass/fail): Binary true/false assessment
    - score (numeric): Rating within a defined range (e.g., 1-5)
    - literal (categorical): Classification into predefined classes

    For literal traits, scores are returned as int indices (0 to len(classes)-1),
    with class labels returned separately for display purposes.

    Example usage:
        evaluator = LLMTraitEvaluator(llm)
        # For boolean/score traits:
        results, usage = evaluator.evaluate_batch(question, answer, traits)
        # For literal traits:
        scores, labels, usage = evaluator.evaluate_literal_batch(question, answer, literal_traits)
    """

    def __init__(self, llm: Any):
        """
        Initialize the LLM trait evaluator.

        Args:
            llm: Initialized LLM instance (LangChain compatible) for evaluation
        """
        self.llm = llm

    def evaluate_batch(
        self, question: str, answer: str, traits: list[LLMRubricTrait]
    ) -> tuple[dict[str, int | bool], dict[str, Any]]:
        """
        Evaluate all traits in a single LLM call using LangChain strategies.

        Strategy order:
        1. json_schema method (native structured output)
        2. Manual parsing with json-repair

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            traits: List of LLM traits to evaluate

        Returns:
            Tuple of (results_dict, usage_metadata)
        """
        from ....schemas.workflow.rubric_outputs import BatchRubricScores
        from .rubric_parsing import invoke_with_structured_output

        system_prompt = self._build_batch_system_prompt()
        user_prompt = self._build_batch_user_prompt(question, answer, traits)

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        # Invoke with automatic strategy selection and fallbacks
        parsed_result, usage_metadata = invoke_with_structured_output(self.llm, messages, BatchRubricScores)

        # Validate scores against trait definitions
        results = self._validate_batch_scores(parsed_result.scores, traits)
        return results, usage_metadata

    def evaluate_sequential(
        self, question: str, answer: str, traits: list[LLMRubricTrait]
    ) -> tuple[dict[str, int | bool], list[dict[str, Any]]]:
        """
        Evaluate traits one by one (fallback method).

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            traits: List of LLM traits to evaluate

        Returns:
            Tuple of (results_dict, list_of_usage_metadata)
        """
        results: dict[str, int | bool] = {}
        usage_metadata_list: list[dict[str, Any]] = []

        for trait in traits:
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

        Args:
            question: The original question
            answer: The answer to evaluate
            trait: The LLM trait to evaluate

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

    def parse_batch_response(self, response: str, traits: list[LLMRubricTrait]) -> dict[str, int | bool]:
        """
        Parse the batch evaluation response.

        This is a fallback method for when structured output fails. Usually
        invoke_with_structured_output handles parsing, but this can be used
        for manual parsing scenarios.

        Args:
            response: Raw LLM response text
            traits: List of traits being evaluated

        Returns:
            Dictionary mapping trait names to scores

        Raises:
            ValueError: If parsing fails
        """
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response}")

        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}") from e

        # Validate and convert the results
        validated_results: dict[str, int | bool] = {}
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

    def parse_single_trait_response(self, response: str, trait: LLMRubricTrait) -> int | bool:
        """
        Parse a single trait evaluation response.

        This is a fallback method for when structured output fails. Usually
        invoke_with_structured_output handles parsing, but this can be used
        for manual parsing scenarios.

        Args:
            response: Raw LLM response text
            trait: The trait being evaluated

        Returns:
            Parsed score (bool for boolean traits, int for numeric)

        Raises:
            ValueError: If parsing fails
        """
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

    # ========== Literal Trait Evaluation Methods ==========

    def evaluate_literal_batch(
        self, question: str, answer: str, traits: list[LLMRubricTrait]
    ) -> tuple[dict[str, int], dict[str, str], dict[str, Any]]:
        """
        Evaluate literal (categorical) traits in a single LLM call.

        Literal traits classify responses into predefined categories. The LLM
        returns class names, which are then converted to integer indices.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            traits: List of literal kind LLM traits to evaluate

        Returns:
            Tuple of (scores, labels, usage_metadata) where:
            - scores: Dict mapping trait names to int indices (0 to N-1, or -1 for error)
            - labels: Dict mapping trait names to class labels (or invalid value for error)
            - usage_metadata: Usage metadata from the LLM call
        """
        from ....schemas.workflow.rubric_outputs import BatchLiteralClassifications
        from .rubric_parsing import invoke_with_structured_output

        # Filter to only literal traits
        literal_traits = [t for t in traits if t.kind == "literal"]
        if not literal_traits:
            return {}, {}, {}

        system_prompt = self._build_literal_batch_system_prompt()
        user_prompt = self._build_literal_batch_user_prompt(question, answer, literal_traits)

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        # Invoke with automatic strategy selection and fallbacks
        parsed_result, usage_metadata = invoke_with_structured_output(self.llm, messages, BatchLiteralClassifications)

        # Validate classifications and convert to scores + labels
        scores, labels = self._validate_literal_classifications(parsed_result.classifications, literal_traits)
        return scores, labels, usage_metadata

    def evaluate_literal_sequential(
        self, question: str, answer: str, traits: list[LLMRubricTrait]
    ) -> tuple[dict[str, int], dict[str, str], list[dict[str, Any]]]:
        """
        Evaluate literal traits one by one (fallback method).

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            traits: List of literal kind LLM traits to evaluate

        Returns:
            Tuple of (scores, labels, usage_metadata_list) where:
            - scores: Dict mapping trait names to int indices (0 to N-1, or -1 for error)
            - labels: Dict mapping trait names to class labels (or invalid value for error)
            - usage_metadata_list: List of usage metadata dicts from LLM calls
        """
        from ....schemas.workflow.rubric_outputs import SingleLiteralClassification
        from .rubric_parsing import invoke_with_structured_output

        scores: dict[str, int] = {}
        labels: dict[str, str] = {}
        usage_metadata_list: list[dict[str, Any]] = []

        # Filter to only literal traits
        literal_traits = [t for t in traits if t.kind == "literal"]

        for trait in literal_traits:
            try:
                system_prompt = self._build_literal_single_system_prompt(trait)
                user_prompt = self._build_literal_single_user_prompt(question, answer, trait)

                messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

                parsed_result, usage_metadata = invoke_with_structured_output(
                    self.llm, messages, SingleLiteralClassification
                )
                usage_metadata_list.append(usage_metadata)

                # Validate and convert classification to score + label
                score, label = self._validate_literal_classification(trait, parsed_result.classification)
                scores[trait.name] = score
                labels[trait.name] = label
            except Exception as e:
                logger.warning(f"Failed to evaluate literal trait '{trait.name}': {e}")
                # Mark as error state: score=-1, label contains error info
                scores[trait.name] = -1
                labels[trait.name] = f"[EVALUATION_ERROR: {e!s}]"

        return scores, labels, usage_metadata_list

    def _validate_literal_classification(self, trait: LLMRubricTrait, class_name: str) -> tuple[int, str]:
        """
        Validate and convert a class name to score index and label.

        Args:
            trait: The literal trait being evaluated
            class_name: The class name returned by the LLM

        Returns:
            Tuple of (score, label) where:
            - score: Int index (0 to N-1) if valid, -1 if invalid class name
            - label: The class name if valid, or the invalid value for debugging
        """
        if trait.kind != "literal" or trait.classes is None:
            return -1, f"[NOT_LITERAL_TRAIT: {class_name}]"

        # Get the index for the class name
        index = trait.get_class_index(class_name)
        if index == -1:
            # Try case-insensitive matching as fallback
            class_names_lower = {name.lower(): name for name in trait.classes}
            matched_name = class_names_lower.get(class_name.lower())
            if matched_name is not None:
                index = trait.get_class_index(matched_name)
                class_name = matched_name  # Use the canonical name
            else:
                # Invalid class name - store the invalid value for debugging
                logger.warning(
                    f"Invalid class '{class_name}' for trait '{trait.name}'. "
                    f"Valid classes: {list(trait.classes.keys())}"
                )
                return -1, class_name  # Return invalid class name for debugging

        return index, class_name

    def _validate_literal_classifications(
        self, classifications: dict[str, str], traits: list[LLMRubricTrait]
    ) -> tuple[dict[str, int], dict[str, str]]:
        """
        Validate and convert batch classifications to scores and labels.

        Args:
            classifications: Dict mapping trait names to class names from LLM
            traits: List of literal traits being evaluated

        Returns:
            Tuple of (scores, labels) dictionaries
        """
        scores: dict[str, int] = {}
        labels: dict[str, str] = {}
        trait_map = {trait.name: trait for trait in traits}

        for trait_name, class_name in classifications.items():
            if trait_name in trait_map:
                trait = trait_map[trait_name]
                score, label = self._validate_literal_classification(trait, class_name)
                scores[trait_name] = score
                labels[trait_name] = label

        # Add error state for missing traits
        for trait in traits:
            if trait.name not in scores:
                scores[trait.name] = -1
                labels[trait.name] = "[MISSING_FROM_RESPONSE]"

        return scores, labels

    def _build_literal_batch_system_prompt(self) -> str:
        """Build system prompt for batch literal trait evaluation."""
        return """You are an expert evaluator classifying responses into predefined categories.

Your task is to classify a given answer into categories for multiple classification traits.

**RESPONSE FORMAT:**
You will receive a JSON Schema specifying the exact output structure. Your response MUST conform to this schema.
Return ONLY a JSON object - no explanations, no markdown, no surrounding text.

**CRITICAL REQUIREMENTS:**
1. **Exact Class Names**: Use the EXACT class names from each trait's categories (case-sensitive)
2. **One Class Per Trait**: Choose exactly one class for each trait
3. **All Traits Required**: Include ALL traits in your response
4. **No Invention**: Do NOT invent new categories - only use the provided class names

**CLASSIFICATION GUIDELINES:**
- Read each trait's class definitions carefully
- Consider the full context of the answer before classifying
- When a response spans multiple categories, choose the most dominant one
- When uncertain, choose the category that best captures the primary intent

**WHAT NOT TO DO:**
- Do NOT wrap JSON in markdown code blocks (no ```)
- Do NOT add explanatory text before or after the JSON
- Do NOT modify or paraphrase class names
- Do NOT skip any traits"""

    def _build_literal_batch_user_prompt(self, question: str, answer: str, traits: list[LLMRubricTrait]) -> str:
        """Build user prompt for batch literal trait evaluation."""
        from ....schemas.workflow.rubric_outputs import BatchLiteralClassifications

        traits_description = []
        example_classifications: dict[str, str] = {}

        for trait in traits:
            if trait.kind != "literal" or trait.classes is None:
                continue

            class_names = list(trait.classes.keys())
            # Build class descriptions
            class_details = []
            for name, description in trait.classes.items():
                class_details.append(f"    - **{name}**: {description}")

            trait_desc = (
                f"- **{trait.name}**: {trait.description or 'Classification trait'}\n"
                f"  Classes: {', '.join(class_names)}\n" + "\n".join(class_details)
            )
            traits_description.append(trait_desc)
            # Use first class as example
            example_classifications[trait.name] = class_names[0]

        example_json = json.dumps({"classifications": example_classifications}, indent=2)
        json_schema = json.dumps(BatchLiteralClassifications.model_json_schema(), indent=2)

        return f"""Classify the following answer for each trait:

**TRAITS TO CLASSIFY:**
{chr(10).join(traits_description)}

**QUESTION:**
{question}

**ANSWER TO CLASSIFY:**
{answer}

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**REQUIRED OUTPUT FORMAT:**
Return a JSON object with a "classifications" key mapping trait names to class names.
Use EXACT trait and class names as shown above.

Example (using YOUR trait and class names):
{example_json}

**YOUR JSON RESPONSE:**"""

    def _build_literal_single_system_prompt(self, trait: LLMRubricTrait) -> str:
        """Build system prompt for single literal trait evaluation."""
        if trait.kind != "literal" or trait.classes is None:
            raise ValueError(f"Trait '{trait.name}' is not a literal kind trait")

        class_names = list(trait.classes.keys())
        class_details = []
        for name, description in trait.classes.items():
            class_details.append(f"  - **{name}**: {description}")

        return f"""You are evaluating responses for the classification trait: **{trait.name}**

**Description:** {trait.description or "Classification trait"}

**Available Classes:**
{chr(10).join(class_details)}

**RESPONSE FORMAT:**
You will receive a JSON Schema specifying the exact output structure. Your response MUST conform to this schema.
Return valid JSON: {{"classification": "<class_name>"}}

**CLASSIFICATION GUIDELINES:**
- You MUST use one of these exact class names: {", ".join(class_names)}
- Choose the single most appropriate class for the answer
- Do NOT invent new classes or modify class names
- When uncertain, choose the class that best captures the primary intent

**PARSING NOTES:**
- Use the exact class name as provided (case-sensitive)
- Do NOT wrap in markdown code blocks
- Do NOT add explanatory text"""

    def _build_literal_single_user_prompt(self, question: str, answer: str, trait: LLMRubricTrait) -> str:
        """Build user prompt for single literal trait evaluation."""
        from ....schemas.workflow.rubric_outputs import SingleLiteralClassification

        if trait.kind != "literal" or trait.classes is None:
            raise ValueError(f"Trait '{trait.name}' is not a literal kind trait")

        class_names = list(trait.classes.keys())
        json_schema = json.dumps(SingleLiteralClassification.model_json_schema(), indent=2)

        return f"""**QUESTION:**
{question}

**ANSWER TO CLASSIFY:**
{answer}

**TRAIT:** {trait.name}
**DESCRIPTION:** {trait.description or "No description provided"}
**AVAILABLE CLASSES:** {", ".join(class_names)}

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

Classify this answer and return your classification as JSON: {{"classification": "<class_name>"}}"""
