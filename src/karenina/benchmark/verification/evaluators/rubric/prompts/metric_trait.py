"""Prompt construction for metric trait evaluation using confusion matrix analysis.

This module provides the MetricTraitPromptBuilder class for constructing prompts
used in metric trait evaluation, which computes precision, recall, F1, specificity,
and accuracy by categorizing answer content into TP/TN/FP/FN buckets.

Two evaluation modes are supported:
- tp_only: Only TP instructions provided; computes precision, recall, F1
- full_matrix: Both TP and TN instructions; computes all metrics
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from ......schemas.domain import MetricRubricTrait


@dataclass
class MetricTraitPromptBuilder:
    """Builds prompts for metric trait evaluation using confusion matrix analysis.

    Metric traits compute precision, recall, F1, specificity, and accuracy
    by categorizing answer content into TP/TN/FP/FN buckets.

    Two evaluation modes are supported:
    - tp_only: Only TP instructions provided; computes precision, recall, F1
    - full_matrix: Both TP and TN instructions; computes all metrics

    Example:
        builder = MetricTraitPromptBuilder()
        system_prompt = builder.build_system_prompt()
        user_prompt = builder.build_user_prompt(
            question="What diseases affect the lungs?",
            answer="Asthma and bronchitis are common lung diseases.",
            trait=metric_trait,
            schema_class=ConfusionMatrixOutput,
        )
    """

    def build_system_prompt(self) -> str:
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

    def build_user_prompt(
        self,
        question: str,
        answer: str,
        trait: "MetricRubricTrait",
        schema_class: type[BaseModel],
    ) -> str:
        """Build user prompt for metric trait evaluation.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            trait: The metric trait to evaluate
            schema_class: Pydantic model class for JSON schema generation

        Returns:
            Formatted user prompt string
        """
        if trait.evaluation_mode == "tp_only":
            return self._build_tp_only_user_prompt(question, answer, trait, schema_class)
        else:
            return self._build_full_matrix_user_prompt(question, answer, trait, schema_class)

    def _build_tp_only_user_prompt(
        self,
        question: str,
        answer: str,
        trait: "MetricRubricTrait",
        schema_class: type[BaseModel],
    ) -> str:
        """Build user prompt for tp_only mode metric evaluation."""
        json_schema = json.dumps(schema_class.model_json_schema(), indent=2)

        # Format TP instructions as numbered list
        tp_instructions_formatted = "\n".join(
            f"  {i}. {instruction}" for i, instruction in enumerate(trait.tp_instructions, 1)
        )

        # Build description line if present
        description_line = f"Description: {trait.description}\n" if trait.description else ""

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

    def _build_full_matrix_user_prompt(
        self,
        question: str,
        answer: str,
        trait: "MetricRubricTrait",
        schema_class: type[BaseModel],
    ) -> str:
        """Build user prompt for full_matrix mode metric evaluation."""
        json_schema = json.dumps(schema_class.model_json_schema(), indent=2)

        # Format TP instructions as numbered list
        tp_instructions_formatted = "\n".join(
            f"  {i}. {instruction}" for i, instruction in enumerate(trait.tp_instructions, 1)
        )

        # Format TN instructions as numbered list
        tn_instructions_formatted = "\n".join(
            f"  {i}. {instruction}" for i, instruction in enumerate(trait.tn_instructions, 1)
        )

        # Build description line if present
        description_line = f"Description: {trait.description}\n" if trait.description else ""

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
