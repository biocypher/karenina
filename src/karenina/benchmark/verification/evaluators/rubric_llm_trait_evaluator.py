"""LLM-based trait evaluation for rubric assessment.

This module implements the LLMTraitEvaluator class, which handles evaluation
of LLMRubricTrait instances using LLM-as-judge for subjective assessments.

Two evaluation strategies are supported:
- batch: Evaluate all traits in a single LLM call (efficient)
- sequential: Evaluate traits one-by-one (reliable)
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
    to assess response quality against trait criteria. Supports both
    boolean (pass/fail) and numeric (scored) traits.

    Example usage:
        evaluator = LLMTraitEvaluator(llm)
        results, usage = evaluator.evaluate_batch(question, answer, rubric)
        # or
        results, usage = evaluator.evaluate_sequential(question, answer, rubric)
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
