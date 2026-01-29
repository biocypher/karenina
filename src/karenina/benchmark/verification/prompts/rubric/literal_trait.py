"""Prompt construction for literal (categorical) trait evaluation.

Canonical location for literal trait prompt building.
Re-exported by evaluators/rubric/prompts/literal_trait.py for backwards compatibility.

This module provides the LiteralTraitPromptBuilder class for constructing prompts
used in literal trait evaluation, where responses are classified into predefined
categories.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from karenina.schemas.domain import LLMRubricTrait


@dataclass
class LiteralTraitPromptBuilder:
    """Builds prompts for literal (categorical) trait evaluation.

    This class encapsulates prompt construction for literal trait evaluation,
    where responses are classified into predefined categories.

    Example:
        builder = LiteralTraitPromptBuilder()
        system_prompt = builder.build_batch_system_prompt()
        user_prompt = builder.build_batch_user_prompt(
            question="What gene is targeted?",
            answer="BCL2 is the primary target...",
            traits=[literal_trait1, literal_trait2],
            schema_class=BatchLiteralClassifications,
        )
    """

    def build_batch_system_prompt(self) -> str:
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

    def build_batch_user_prompt(
        self,
        question: str,
        answer: str,
        traits: list["LLMRubricTrait"],
        schema_class: type[BaseModel],
    ) -> str:
        """Build user prompt for batch literal trait evaluation.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            traits: List of literal kind LLM traits to evaluate
            schema_class: Pydantic model class for JSON schema generation

        Returns:
            Formatted user prompt string
        """
        traits_description = []
        # Use list format for classifications (required for Anthropic beta.messages.parse compatibility)
        example_classifications: list[dict[str, str]] = []

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
            example_classifications.append({"trait_name": trait.name, "class_name": class_names[0]})

        example_json = json.dumps({"classifications": example_classifications}, indent=2)
        json_schema = json.dumps(schema_class.model_json_schema(), indent=2)

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
Return a JSON object with a "classifications" array containing one object per trait.
Each object has "trait_name" and "class_name" fields.
Use EXACT trait and class names as shown above.

Example (using YOUR trait and class names):
{example_json}

**YOUR JSON RESPONSE:**"""

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

    def build_single_trait_user_prompt(
        self,
        question: str,
        answer: str,
        trait: "LLMRubricTrait",
        schema_class: type[BaseModel],
    ) -> str:
        """Build user prompt for single literal trait evaluation.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            trait: The literal kind LLM trait to evaluate
            schema_class: Pydantic model class for JSON schema generation

        Returns:
            Formatted user prompt string

        Raises:
            ValueError: If trait is not a literal kind trait
        """
        if trait.kind != "literal" or trait.classes is None:
            raise ValueError(f"Trait '{trait.name}' is not a literal kind trait")

        class_names = list(trait.classes.keys())
        json_schema = json.dumps(schema_class.model_json_schema(), indent=2)

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
