"""Prompt construction for literal (categorical) trait evaluation.

Canonical location for literal trait prompt building.
Re-exported by evaluators/rubric/prompts/literal_trait.py for backwards compatibility.

This module provides the LiteralTraitPromptBuilder class for constructing prompts
used in literal trait evaluation, where responses are classified into predefined
categories.

Format-specific content (JSON schema blocks, response format sections, output
format examples) is NOT included here — it is injected by adapter instructions
registered per-interface. This keeps prompt builders format-agnostic.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from karenina.schemas.domain import LLMRubricTrait


@dataclass
class LiteralTraitPromptBuilder:
    """Builds prompts for literal (categorical) trait evaluation.

    This class encapsulates prompt construction for literal trait evaluation,
    where responses are classified into predefined categories.

    Format-specific content (JSON schema, output format examples) is injected by
    adapter instructions, not included in the base prompts.

    Example:
        builder = LiteralTraitPromptBuilder()
        system_prompt = builder.build_batch_system_prompt()
        user_prompt = builder.build_batch_user_prompt(
            question="What gene is targeted?",
            answer="BCL2 is the primary target...",
            traits=[literal_trait1, literal_trait2],
        )
    """

    def build_batch_system_prompt(self) -> str:
        """Build system prompt for batch literal trait evaluation."""
        return """You are an expert evaluator classifying responses into predefined categories.

Your task is to classify a given answer into categories for multiple classification traits.

**CRITICAL REQUIREMENTS:**
1. **Exact Class Names**: Use the EXACT class names from each trait's categories (case-sensitive)
2. **One Class Per Trait**: Choose exactly one class for each trait
3. **All Traits Required**: Include ALL traits in your response
4. **No Invention**: Do NOT invent new categories - only use the provided class names

**CLASSIFICATION GUIDELINES:**
- Read each trait's class definitions carefully
- Consider the full context of the answer before classifying
- When a response spans multiple categories, choose the most dominant one
- When uncertain, choose the category that best captures the primary intent"""

    def build_batch_user_prompt(
        self,
        question: str,
        answer: str,
        traits: list["LLMRubricTrait"],
    ) -> str:
        """Build user prompt for batch literal trait evaluation.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            traits: List of literal kind LLM traits to evaluate

        Returns:
            Formatted user prompt string
        """
        traits_description = []

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

        return f"""Classify the following answer for each trait:

**TRAITS TO CLASSIFY:**
{chr(10).join(traits_description)}

**QUESTION:**
{question}

**ANSWER TO CLASSIFY:**
{answer}"""

    def build_batch_example_json(self, traits: list["LLMRubricTrait"]) -> str:
        """Build example JSON string for batch literal evaluation.

        Used by evaluator callers to pass as instruction_context for adapter
        instructions to include in format-specific prompt sections.

        Args:
            traits: List of literal kind LLM traits

        Returns:
            JSON string with example classifications
        """
        example_classifications: list[dict[str, str]] = []
        for trait in traits:
            if trait.kind != "literal" or trait.classes is None:
                continue
            class_names = list(trait.classes.keys())
            example_classifications.append({"trait_name": trait.name, "class_name": class_names[0]})
        return json.dumps({"classifications": example_classifications}, indent=2)

    def build_single_trait_system_prompt(self, trait: "LLMRubricTrait") -> str:
        """Build system prompt for single literal trait evaluation.

        Args:
            trait: The literal kind LLM trait to evaluate

        Returns:
            System prompt string

        Raises:
            ValueError: If trait is not a literal kind trait
        """
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

**CLASSIFICATION GUIDELINES:**
- You MUST use one of these exact class names: {", ".join(class_names)}
- Choose the single most appropriate class for the answer
- Do NOT invent new classes or modify class names
- When uncertain, choose the class that best captures the primary intent"""

    def build_single_trait_user_prompt(
        self,
        question: str,
        answer: str,
        trait: "LLMRubricTrait",
    ) -> str:
        """Build user prompt for single literal trait evaluation.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            trait: The literal kind LLM trait to evaluate

        Returns:
            Formatted user prompt string

        Raises:
            ValueError: If trait is not a literal kind trait
        """
        if trait.kind != "literal" or trait.classes is None:
            raise ValueError(f"Trait '{trait.name}' is not a literal kind trait")

        class_names = list(trait.classes.keys())

        return f"""**QUESTION:**
{question}

**ANSWER TO CLASSIFY:**
{answer}

**TRAIT:** {trait.name}
**DESCRIPTION:** {trait.description or "No description provided"}
**AVAILABLE CLASSES:** {", ".join(class_names)}"""
