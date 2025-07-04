"""
Rubric evaluation for qualitative assessment of LLM responses.
"""

import json
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ...llm.interface import init_chat_model_unified
from ...schemas.rubric_class import Rubric, RubricTrait
from ..models import ModelConfiguration


class RubricEvaluator:
    """
    Evaluates LLM responses against a defined rubric using qualitative traits.
    """

    def __init__(self, model_config: ModelConfiguration):
        """
        Initialize the rubric evaluator with an LLM model.
        
        Args:
            model_config: Configuration for the evaluation model
            
        Raises:
            ValueError: If model configuration is invalid
            RuntimeError: If LLM initialization fails
        """
        if not model_config:
            raise ValueError("Model configuration is required")

        if not model_config.model_name:
            raise ValueError("Model name is required in model configuration")

        if not model_config.model_provider:
            raise ValueError("Model provider is required in model configuration")

        self.model_config = model_config

        try:
            self.llm = init_chat_model_unified(
                model=model_config.model_name,
                provider=model_config.model_provider,
                temperature=model_config.temperature,
                interface=model_config.interface,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM for rubric evaluation: {e}") from e

    def evaluate_rubric(
        self,
        question: str,
        answer: str,
        rubric: Rubric
    ) -> dict[str, int | bool]:
        """
        Evaluate an answer against a rubric's traits.
        
        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            rubric: The rubric containing evaluation traits
            
        Returns:
            Dictionary mapping trait names to their evaluated scores
            
        Raises:
            Exception: If evaluation fails completely
        """
        if not rubric.traits:
            return {}

        try:
            # Try batch evaluation first (more efficient)
            return self._evaluate_batch(question, answer, rubric)
        except Exception as batch_error:
            # Fallback to sequential evaluation
            try:
                return self._evaluate_sequential(question, answer, rubric)
            except Exception as seq_error:
                # Log both errors and raise the sequential one
                print(f"Batch evaluation failed: {batch_error}")
                print(f"Sequential evaluation failed: {seq_error}")
                raise seq_error

    def _evaluate_batch(
        self,
        question: str,
        answer: str,
        rubric: Rubric
    ) -> dict[str, int | bool]:
        """
        Evaluate all traits in a single LLM call (more efficient).
        """
        system_prompt = self._build_batch_system_prompt()
        user_prompt = self._build_batch_user_prompt(question, answer, rubric.traits)

        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)
        raw_response = response.content if hasattr(response, "content") else str(response)

        return self._parse_batch_response(raw_response, rubric.traits)

    def _evaluate_sequential(
        self,
        question: str,
        answer: str,
        rubric: Rubric
    ) -> dict[str, int | bool]:
        """
        Evaluate traits one by one (fallback method).
        """
        results = {}

        for trait in rubric.traits:
            try:
                score = self._evaluate_single_trait(question, answer, trait)
                results[trait.name] = score
            except Exception as e:
                print(f"Failed to evaluate trait '{trait.name}': {e}")
                # Continue with other traits, mark this one as None
                results[trait.name] = None

        return results

    def _evaluate_single_trait(
        self,
        question: str,
        answer: str,
        trait: RubricTrait
    ) -> int | bool:
        """
        Evaluate a single trait.
        """
        system_prompt = self._build_single_trait_system_prompt(trait)
        user_prompt = self._build_single_trait_user_prompt(question, answer, trait)

        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

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
            
Description: {trait.description or 'Boolean evaluation'}

Respond with only "true" or "false" based on whether the answer meets this criteria."""
        else:
            min_score = trait.min_score or 1
            max_score = trait.max_score or 5
            return f"""You are evaluating responses for the trait: {trait.name}
            
Description: {trait.description or 'Score-based evaluation'}

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

Please evaluate this answer for the trait "{trait.name}": {trait.description or 'No description provided'}"""

    def _parse_batch_response(self, response: str, traits: list[RubricTrait]) -> dict[str, int | bool]:
        """Parse the batch evaluation response."""
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response}")

        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

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
                validated_results[trait.name] = None

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
            numbers = re.findall(r'\d+', response)
            if not numbers:
                raise ValueError(f"No numeric score found in: {response}")

            score = int(numbers[0])
            return self._validate_score(score, trait)

    def _validate_score(self, score: Any, trait: RubricTrait) -> int | bool:
        """Validate and convert a score for a trait."""
        if trait.kind == "boolean":
            if isinstance(score, bool):
                return score
            elif isinstance(score, (int, str)):
                return bool(score) and str(score).lower() not in ["false", "0", "no"]
            else:
                return bool(score)
        else:
            if not isinstance(score, (int, float)):
                try:
                    score = int(score)
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid score type for trait {trait.name}: {type(score)}")

            min_score = trait.min_score or 1
            max_score = trait.max_score or 5

            # Clamp score to valid range
            score = max(min_score, min(max_score, int(score)))
            return score
