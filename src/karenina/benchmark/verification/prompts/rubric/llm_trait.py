"""Prompt construction for LLM-based trait evaluation (boolean and score kinds).

Canonical location for LLM trait prompt building.
Re-exported by evaluators/rubric/prompts/llm_trait.py for backwards compatibility.

This module provides the LLMTraitPromptBuilder class for constructing prompts
used in standard LLM trait evaluation, supporting both batch (all traits at once)
and sequential (one at a time) evaluation modes.

Format-specific content (JSON schema blocks, response format sections, output
format examples) is NOT included here — it is injected by adapter instructions
registered per-interface. This keeps prompt builders format-agnostic.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from karenina.schemas.entities import LLMRubricTrait


@dataclass
class LLMTraitPromptBuilder:
    """Builds prompts for LLM-based trait evaluation (boolean and score kinds).

    This class encapsulates prompt construction for standard LLM trait evaluation,
    supporting both batch (all traits at once) and sequential (one at a time) modes.

    Format-specific content (JSON schema, output format examples) is injected by
    adapter instructions, not included in the base prompts.

    Example:
        builder = LLMTraitPromptBuilder()
        system_prompt = builder.build_batch_system_prompt()
        user_prompt = builder.build_batch_user_prompt(
            question="What gene is targeted?",
            answer="BCL2 is the primary target...",
            traits=[trait1, trait2],
        )
    """

    def build_batch_system_prompt(self) -> str:
        """Build system prompt for batch evaluation of boolean/score traits."""
        return """You are an expert evaluator assessing the quality of responses using a structured rubric.

Your task is to evaluate a given answer against multiple evaluation traits and return scores.

**TRAIT REQUIREMENTS:**
1. **Exact Trait Names**: Use the EXACT trait names provided - case-sensitive matching is required
2. **Boolean Traits**: Use `true` or `false` (lowercase, no quotes around the value)
3. **Score Traits**: Use integers within the specified range
4. **Score Clamping**: Scores outside the valid range will be automatically clamped to the nearest boundary
5. **Missing Traits**: Any trait not included will be recorded as `null` (evaluation failure)

**EVALUATION GUIDELINES:**
- Evaluate each trait independently based on its specific criteria
- When uncertain, choose the most conservative/defensive value based on the trait's intent:
  - For positive traits (e.g., "is accurate"), lean toward `false` when uncertain
  - For negative traits (e.g., "contains errors"), lean toward `true` when uncertain
  - For scores, lean toward the middle of the scale when uncertain
- Base assessments solely on the answer content, not assumptions about intent
- Be consistent: similar answers should receive similar scores"""

    def build_batch_user_prompt(
        self,
        question: str,
        answer: str,
        traits: list["LLMRubricTrait"],
    ) -> str:
        """Build user prompt for batch evaluation of boolean/score traits.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            traits: List of LLM traits to evaluate

        Returns:
            Formatted user prompt string
        """
        traits_description = []

        for trait in traits:
            if trait.kind == "boolean":
                trait_desc = f"- **{trait.name}** (boolean): {trait.description or 'Boolean evaluation'}\n  → Return `true` or `false`"
            else:
                min_score = trait.min_score or 1
                max_score = trait.max_score or 5
                trait_desc = f"- **{trait.name}** (score {min_score}-{max_score}): {trait.description or 'Score-based evaluation'}\n  → Return integer from {min_score} to {max_score}"
            traits_description.append(trait_desc)

        return f"""Evaluate the following answer against these traits:

**TRAITS TO EVALUATE:**
{chr(10).join(traits_description)}

**QUESTION:**
{question}

**ANSWER TO EVALUATE:**
{answer}"""

    def build_batch_example_json(self, traits: list["LLMRubricTrait"]) -> str:
        """Build example JSON string for batch evaluation.

        Used by evaluator callers to pass as instruction_context for adapter
        instructions to include in format-specific prompt sections.

        Args:
            traits: List of LLM traits to build example for

        Returns:
            JSON string with example scores
        """
        example_scores: dict[str, bool | int] = {}
        for trait in traits:
            if trait.kind == "boolean":
                example_scores[trait.name] = True
            else:
                example_scores[trait.name] = (trait.min_score or 1) + 2
        return json.dumps({"scores": example_scores}, indent=2)

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

**EVALUATION GUIDELINES:**
- `true`: The criteria IS met - answer clearly satisfies the requirement
- `false`: The criteria IS NOT met - answer fails to satisfy the requirement
- When uncertain, choose the most conservative value based on the trait's nature:
  - For positive traits (e.g., "is accurate"), lean toward `false`
  - For negative traits (e.g., "contains errors"), lean toward `true`"""

    def _build_single_score_system_prompt(self, trait: "LLMRubricTrait") -> str:
        """Build system prompt for single score trait evaluation."""
        min_score = trait.min_score or 1
        max_score = trait.max_score or 5
        mid_score = (min_score + max_score) // 2
        return f"""You are evaluating responses for the trait: **{trait.name}**

**Criteria:** {trait.description or "Score-based evaluation"}

**SCORING GUIDELINES:**
- {min_score} = Poor - Does not meet criteria at all
- {mid_score} = Average - Partially meets criteria
- {max_score} = Excellent - Fully meets or exceeds criteria

When uncertain about borderline cases, choose conservatively based on the trait's nature:
- For traits where higher is better (e.g., "quality"), lean toward lower scores
- For traits where lower is better (e.g., "error severity"), lean toward higher scores
- When completely uncertain, default to the middle value ({mid_score})"""

    def build_single_trait_user_prompt(
        self,
        question: str,
        answer: str,
        trait: "LLMRubricTrait",
    ) -> str:
        """Build user prompt for single trait evaluation.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            trait: The LLM trait to evaluate

        Returns:
            Formatted user prompt string
        """
        return f"""**QUESTION:**
{question}

**ANSWER TO EVALUATE:**
{answer}

**TRAIT:** {trait.name}
**CRITERIA:** {trait.description or "No description provided"}"""
