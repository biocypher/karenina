"""Prompt construction for deep judgment evaluation of rubric traits.

Canonical location for DeepJudgmentPromptBuilder.

This module provides the DeepJudgmentPromptBuilder class for constructing prompts
used in the multi-stage deep judgment evaluation process:

1. Extract verbatim excerpts from the answer
2. (Optional) Assess hallucination risk using external search
3. Generate reasoning explaining how excerpts support the trait
4. Extract final score based on reasoning
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from karenina.schemas.domain import LLMRubricTrait


@dataclass
class DeepJudgmentPromptBuilder:
    """Builds prompts for deep judgment evaluation of rubric traits.

    Deep judgment is a multi-stage evaluation process:
    1. Extract verbatim excerpts from the answer
    2. (Optional) Assess hallucination risk using external search
    3. Generate reasoning explaining how excerpts support the trait
    4. Extract final score based on reasoning

    Example:
        builder = DeepJudgmentPromptBuilder()
        system_prompt = builder.build_excerpt_extraction_system_prompt()
        user_prompt = builder.build_excerpt_extraction_user_prompt(
            trait=trait,
            max_excerpts=5,
            answer="The response to analyze...",
            feedback=None,  # Or retry feedback
            schema_class=TraitExcerptsOutput,
        )
    """

    # =========================================================================
    # Stage 1: Excerpt Extraction
    # =========================================================================

    def build_excerpt_extraction_system_prompt(self) -> str:
        """Build system prompt for excerpt extraction."""
        return """You are an expert at extracting verbatim quotes from text that demonstrate specific qualities.

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

    def build_excerpt_extraction_user_prompt(
        self,
        trait: "LLMRubricTrait",
        max_excerpts: int,
        answer: str,
        schema_class: type[BaseModel],
        feedback: str | None = None,
    ) -> str:
        """Build user prompt for excerpt extraction.

        Args:
            trait: The LLM trait to extract excerpts for
            max_excerpts: Maximum number of excerpts to extract
            answer: The answer to extract excerpts from
            schema_class: Pydantic model class for JSON schema generation
            feedback: Optional retry feedback from validation failure

        Returns:
            Formatted user prompt string
        """
        json_schema = json.dumps(schema_class.model_json_schema(), indent=2)

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

    def build_retry_feedback(self, failed_excerpts: list[dict[str, Any]], fuzzy_threshold: float) -> str:
        """Build feedback message for retry attempt after validation failure.

        Args:
            failed_excerpts: List of excerpts that failed validation
            fuzzy_threshold: The similarity threshold used for validation

        Returns:
            Feedback message string
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

    # =========================================================================
    # Stage 1.5: Hallucination Assessment
    # =========================================================================

    def build_hallucination_assessment_system_prompt(self) -> str:
        """Build system prompt for hallucination risk assessment."""
        return """You are an expert at assessing hallucination risk using external evidence.

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

    def build_hallucination_assessment_user_prompt(
        self,
        excerpt_text: str,
        search_results: str,
        schema_class: type[BaseModel],
    ) -> str:
        """Build user prompt for hallucination risk assessment.

        Args:
            excerpt_text: The excerpt text to assess
            search_results: External search results for verification
            schema_class: Pydantic model class for JSON schema generation

        Returns:
            Formatted user prompt string
        """
        json_schema = json.dumps(schema_class.model_json_schema(), indent=2)

        return f"""Assess the hallucination risk for this excerpt by comparing it against external search results.

**EXCERPT TO VERIFY:**
"{excerpt_text}"

**EXTERNAL SEARCH RESULTS:**
{search_results}

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

    # =========================================================================
    # Stage 2: Reasoning Generation
    # =========================================================================

    def build_reasoning_system_prompt(self) -> str:
        """Build system prompt for reasoning generation."""
        return "You are an expert at analyzing text quality and providing clear reasoning."

    def build_reasoning_user_prompt_with_excerpts(
        self,
        question: str,
        trait: "LLMRubricTrait",
        excerpts: list[dict[str, Any]],
        hallucination_risk: dict[str, Any] | None = None,
    ) -> str:
        """Build reasoning prompt when excerpts are available.

        Args:
            question: The original question
            trait: The LLM trait being evaluated
            excerpts: List of validated excerpts with confidence scores
            hallucination_risk: Optional hallucination risk assessment

        Returns:
            Formatted user prompt string
        """
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

    def build_reasoning_user_prompt_without_excerpts(
        self,
        question: str,
        answer: str,
        trait: "LLMRubricTrait",
    ) -> str:
        """Build reasoning prompt when excerpts are not available.

        Args:
            question: The original question
            answer: The complete LLM response
            trait: The LLM trait being evaluated

        Returns:
            Formatted user prompt string
        """
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

    # =========================================================================
    # Stage 3: Score Extraction
    # =========================================================================

    def build_score_extraction_system_prompt(self) -> str:
        """Build system prompt for final score extraction."""
        return """You are an expert evaluator providing precise trait scores based on prior reasoning.

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

    def build_score_extraction_user_prompt(
        self,
        trait: "LLMRubricTrait",
        reasoning: str,
        schema_class: type[BaseModel],
    ) -> str:
        """Build user prompt for final score extraction.

        Args:
            trait: The LLM trait being evaluated
            reasoning: The generated reasoning from the previous stage
            schema_class: Pydantic model class for JSON schema generation

        Returns:
            Formatted user prompt string
        """
        json_schema = json.dumps(schema_class.model_json_schema(), indent=2)

        if trait.kind == "boolean":
            return self._build_boolean_score_user_prompt(trait, reasoning, json_schema)
        else:
            return self._build_numeric_score_user_prompt(trait, reasoning, json_schema)

    def _build_boolean_score_user_prompt(
        self,
        trait: "LLMRubricTrait",
        reasoning: str,
        json_schema: str,
    ) -> str:
        """Build user prompt for boolean score extraction."""
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

    def _build_numeric_score_user_prompt(
        self,
        trait: "LLMRubricTrait",
        reasoning: str,
        json_schema: str,
    ) -> str:
        """Build user prompt for numeric score extraction."""
        min_score = trait.min_score or 1
        max_score = trait.max_score or 5
        mid_score = (min_score + max_score) // 2

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
