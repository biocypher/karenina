"""Prompt construction for LLM-based trait evaluation (boolean and score kinds).

Canonical location for LLM trait prompt building.
Re-exported by evaluators/rubric/prompts/llm_trait.py for backwards compatibility.

This module provides the LLMTraitPromptBuilder class for constructing prompts
used in standard LLM trait evaluation, supporting both batch (all traits at once)
and sequential (one at a time) evaluation modes.

Format-specific content (JSON schema blocks, response format sections, output
format examples) is NOT included here: it is injected by adapter instructions
registered per-interface. This keeps prompt builders format-agnostic.

The user-prompt builders accept a ``task_eval_mode`` flag. When True (set by
TaskEval, where the rubric is evaluated on a logged response without a real
question), the **QUESTION:** block is omitted entirely from the rendered
prompt. When False (the verification path), the QUESTION block is rendered
as before. This is the only place the flag affects rendering; system prompts
do not branch on it.
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
        return """You are a rigorous rubric evaluator. For each trait listed, decide carefully whether the answer satisfies the trait's criteria.

**TRAIT REQUIREMENTS:**
1. **Exact Trait Names**: Use the EXACT trait names provided; case-sensitive matching is required.
2. **Boolean Traits**: Use `true` or `false` (lowercase, no quotes around the value).
3. **Score Traits**: Use integers within the specified range. Scores outside the valid range will be clamped.
4. **Missing Traits**: Any trait not included will be recorded as `null` (evaluation failure).

**EVALUATION GUIDELINES:**
- Read each trait's criteria as written; do not infer requirements that are not stated.
- Ground every judgment in evidence visible in the answer text. If a claim is not visible, do not credit it.
- Consider both supporting and contradicting evidence before concluding.
- If a trait's criteria are ambiguous, state the interpretation you used and apply it consistently.
- Be consistent: similar answers should receive similar scores.
- Score honestly. Do not lean toward a particular value when uncertain; the criteria themselves should resolve the call."""

    def build_batch_user_prompt(
        self,
        question: str,
        answer: str,
        traits: list["LLMRubricTrait"],
        *,
        task_eval_mode: bool = False,
    ) -> str:
        """Build user prompt for batch evaluation of boolean/score traits.

        Args:
            question: The original question asked.
            answer: The LLM's response to evaluate.
            traits: List of LLM traits to evaluate.
            task_eval_mode: When True, omit the **QUESTION:** block entirely.

        Returns:
            Formatted user prompt string.
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

        question_block = "" if task_eval_mode else f"\n\n**QUESTION:**\n{question}"

        return f"""Evaluate the following answer against these traits:

**TRAITS TO EVALUATE:**
{chr(10).join(traits_description)}{question_block}

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
        return f"""You are a rigorous rubric evaluator assessing the trait: **{trait.name}**

**Criteria:** {trait.description or "Boolean evaluation"}

**EVALUATION GUIDELINES:**
- `true`: The criteria IS met; the answer clearly satisfies the requirement.
- `false`: The criteria IS NOT met; the answer fails to satisfy the requirement.
- Ground your judgment in evidence visible in the answer text; do not infer beyond what is stated.
- Consider both supporting and contradicting evidence before concluding.
- If the criteria are ambiguous, state the interpretation you used and apply it consistently."""

    def _build_single_score_system_prompt(self, trait: "LLMRubricTrait") -> str:
        """Build system prompt for single score trait evaluation."""
        min_score = trait.min_score or 1
        max_score = trait.max_score or 5
        mid_score = (min_score + max_score) // 2
        return f"""You are a rigorous rubric evaluator assessing the trait: **{trait.name}**

**Criteria:** {trait.description or "Score-based evaluation"}

**SCORING GUIDELINES:**
- {min_score} = Poor; does not meet the criteria at all.
- {mid_score} = Average; partially meets the criteria.
- {max_score} = Excellent; fully meets or exceeds the criteria.

- Ground your score in evidence visible in the answer text; do not infer beyond what is stated.
- Consider both supporting and contradicting evidence before concluding.
- Be consistent: similar answers should receive similar scores."""

    def build_single_trait_user_prompt(
        self,
        question: str,
        answer: str,
        trait: "LLMRubricTrait",
        *,
        task_eval_mode: bool = False,
    ) -> str:
        """Build user prompt for single trait evaluation.

        Args:
            question: The original question asked.
            answer: The LLM's response to evaluate.
            trait: The LLM trait to evaluate.
            task_eval_mode: When True, omit the **QUESTION:** block entirely.

        Returns:
            Formatted user prompt string.
        """
        question_block = "" if task_eval_mode else f"**QUESTION:**\n{question}\n\n"
        return f"""{question_block}**ANSWER TO EVALUATE:**
{answer}

**TRAIT:** {trait.name}
**CRITERIA:** {trait.description or "No description provided"}"""

    def build_template_system_prompt(self, trait: "LLMRubricTrait") -> str:
        """Build system prompt for template-kind trait evaluation.

        Template-kind traits ask the judge to populate a user-defined Pydantic
        schema based on evidence in the response. The schema is enforced by
        structured output at the adapter layer, so this prompt focuses on
        behaviour: ground every field in the response, populate every field,
        and stay conservative when evidence is missing.
        """
        return f"""You are a rigorous rubric evaluator assessing the trait: **{trait.name}**

**Criteria:** {trait.description or "Structured evaluation"}

**EVALUATION GUIDELINES:**
- Populate EVERY field of the requested output schema.
- Ground each field in evidence observable in the answer text.
- When evidence is missing, choose the most conservative value (empty string, empty list, `false`, or the minimum numeric value) rather than fabricating content.
- Base your assessment solely on the answer content; do not assume facts not present in the response.
- Be consistent: similar answers should produce similar structured outputs."""

    def build_template_user_prompt(
        self,
        question: str,
        answer: str,
        trait: "LLMRubricTrait",
        *,
        task_eval_mode: bool = False,
    ) -> str:
        """Build user prompt for template-kind trait evaluation.

        Args:
            question: The original question asked.
            answer: The LLM's response to evaluate.
            trait: The LLM trait to evaluate.
            task_eval_mode: When True, omit the **QUESTION:** block entirely.
        """
        question_block = "" if task_eval_mode else f"**QUESTION:**\n{question}\n\n"
        return f"""{question_block}**ANSWER TO EVALUATE:**
{answer}

**TRAIT:** {trait.name}
**CRITERIA:** {trait.description or "No description provided"}"""
