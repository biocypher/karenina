"""Prompt construction for LLM-based trait evaluation (boolean and score kinds).

This module provides the LLMTraitPromptBuilder class for constructing prompts
used in standard LLM trait evaluation, supporting both batch (all traits at once)
and sequential (one at a time) evaluation modes.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from ......schemas.domain import LLMRubricTrait


@dataclass
class LLMTraitPromptBuilder:
    """Builds prompts for LLM-based trait evaluation (boolean and score kinds).

    This class encapsulates prompt construction for standard LLM trait evaluation,
    supporting both batch (all traits at once) and sequential (one at a time) modes.

    Example:
        builder = LLMTraitPromptBuilder()
        system_prompt = builder.build_batch_system_prompt()
        user_prompt = builder.build_batch_user_prompt(
            question="What gene is targeted?",
            answer="BCL2 is the primary target...",
            traits=[trait1, trait2],
            schema_class=BatchRubricScores,
        )
    """

    def build_batch_system_prompt(self) -> str:
        """Build system prompt for batch evaluation of boolean/score traits."""
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

    def build_batch_user_prompt(
        self,
        question: str,
        answer: str,
        traits: list["LLMRubricTrait"],
        schema_class: type[BaseModel],
    ) -> str:
        """Build user prompt for batch evaluation of boolean/score traits.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            traits: List of LLM traits to evaluate
            schema_class: Pydantic model class for JSON schema generation

        Returns:
            Formatted user prompt string
        """
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
        json_schema = json.dumps(schema_class.model_json_schema(), indent=2)

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

    def build_single_trait_system_prompt(self, trait: "LLMRubricTrait") -> str:
        """Build system prompt for single trait evaluation.

        Args:
            trait: The LLM trait to evaluate

        Returns:
            System prompt string appropriate for the trait's kind
        """
        if trait.kind == "boolean":
            return self._build_single_boolean_system_prompt(trait)
        else:
            return self._build_single_score_system_prompt(trait)

    def _build_single_boolean_system_prompt(self, trait: "LLMRubricTrait") -> str:
        """Build system prompt for single boolean trait evaluation."""
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

    def _build_single_score_system_prompt(self, trait: "LLMRubricTrait") -> str:
        """Build system prompt for single score trait evaluation."""
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

    def build_single_trait_user_prompt(
        self,
        question: str,
        answer: str,
        trait: "LLMRubricTrait",
        schema_class: type[BaseModel],
    ) -> str:
        """Build user prompt for single trait evaluation.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            trait: The LLM trait to evaluate
            schema_class: Pydantic model class for JSON schema generation

        Returns:
            Formatted user prompt string
        """
        format_hint = '{"result": true} or {"result": false}' if trait.kind == "boolean" else '{"score": N}'

        json_schema = json.dumps(schema_class.model_json_schema(), indent=2)

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
