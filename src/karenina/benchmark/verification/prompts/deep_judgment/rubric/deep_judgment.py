"""Prompt construction for deep judgment evaluation of rubric traits.

Canonical location for DeepJudgmentPromptBuilder.

This module provides the DeepJudgmentPromptBuilder class for constructing prompts
used in the multi-stage deep judgment evaluation process:

1. Extract verbatim excerpts from the answer
2. (Optional) Assess hallucination risk using external search
3. Generate reasoning explaining how excerpts support the trait
4. Extract final score based on reasoning

Format-specific content (JSON schema blocks, response format sections, parsing
notes) is NOT included here — it is injected by adapter instructions registered
per-interface. This keeps prompt builders format-agnostic.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from karenina.schemas.entities import LLMRubricTrait


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
        return """## Role
You extract verbatim quotes from a candidate response that bear on a single rubric trait, so a downstream reasoning stage can weigh the evidence.

## Principles
- Verbatim only: every quote must appear character-for-character in the response (minor whitespace differences are tolerated, paraphrasing is not).
- Confidence levels:
  - high: direct, explicit statement that clearly bears on the trait.
  - medium: indirect or partial evidence; reasonable inference.
  - low: weak, ambiguous, or tangential evidence.
- Include weak evidence at low confidence rather than returning an empty list. Reserve the empty list for responses with no relevant text at all.
- Prefer focused, sentence-level excerpts. If a longer span is needed, keep it tight around the load-bearing words.
- Within the configured maximum, prioritize the strongest and most distinct excerpts; do not pad.

## Anti-patterns
- Paraphrasing or summarizing the response text.
- Inventing quotes that the response does not contain.
- Returning the entire response as a single excerpt.
- Returning an empty list when ambiguous evidence exists; mark it low instead.
- Selecting near-duplicate excerpts that say the same thing.

## Output handoff
- Your excerpts are fuzzy-matched against the response text; near-misses trigger a retry with feedback identifying the failed quotes.
- The next stage reads excerpts in order and weighs them by their confidence label; ordering and labels carry signal."""

    def build_excerpt_extraction_user_prompt(
        self,
        trait: "LLMRubricTrait",
        max_excerpts: int,
        answer: str,
        feedback: str | None = None,
    ) -> str:
        """Build user prompt for excerpt extraction.

        Args:
            trait: The LLM trait to extract excerpts for.
            max_excerpts: Maximum number of excerpts to extract.
            answer: The answer to extract excerpts from.
            feedback: Optional retry feedback from validation failure.

        Returns:
            Formatted user prompt string.
        """
        criteria = trait.description or "Assess this quality."
        prompt = (
            f"## Trait\n{trait.name}\n\n"
            f"## Criteria\n{criteria}\n\n"
            f"## Response\n{answer}\n\n"
            f"## Maximum excerpts\n{max_excerpts}\n"
        )
        if feedback:
            prompt += f"\n## Retry feedback\n{feedback}\n"
        prompt += "\n## Task\nExtract verbatim excerpts that bear on the trait, with confidence levels."
        return prompt

    def build_retry_feedback(self, failed_excerpts: list[dict[str, Any]], fuzzy_threshold: float) -> str:
        """Build feedback message for retry attempt after validation failure.

        Args:
            failed_excerpts: List of excerpts that failed validation.
            fuzzy_threshold: The similarity threshold used for validation.

        Returns:
            Feedback message string. The verbatim rule lives in the system prompt;
            this message only enumerates the failed quotes.
        """
        lines = ["The following excerpts failed validation (not found in answer):"]
        for i, excerpt in enumerate(failed_excerpts, 1):
            text = excerpt.get("text", "")
            similarity = excerpt.get("similarity_score", 0)
            lines.append(f'{i}. "{text}" (similarity: {similarity:.2f}, threshold: {fuzzy_threshold:.2f})')
        return "\n".join(lines)

    # =========================================================================
    # Stage 1.5: Hallucination Assessment
    # =========================================================================

    def build_hallucination_assessment_system_prompt(self) -> str:
        """Build system prompt for hallucination risk assessment."""
        return """## Role
You assess whether a single excerpt's claims are supported by external search results, so a downstream reasoning stage can weigh the excerpt's reliability.

## Principles
- Risk levels:
  - none: multiple reliable sources clearly confirm the claim.
  - low: at least one source supports the claim; no contradiction.
  - medium: weak, partial, or ambiguous support; sources unclear or partly contradict.
  - high: no supporting evidence, or the claim is actively contradicted.
- Be evidence-based: judge from what the search results actually contain, not from prior beliefs.
- Be conservative under uncertainty: when between two adjacent levels, choose the higher-risk one.
- Provide a brief justification (1-3 sentences) referencing the search results.

## Anti-patterns
- Treating empty or irrelevant results as confirmation that the claim is false.
- Treating a single matching snippet as strong (none-risk) confirmation.
- Assuming search results are current, complete, or authoritative.
- Glossing over disagreement between sources; surface it in the justification.
- Inferring support from the excerpt's tone or specificity rather than from the search results.

## Output handoff
- Your per-excerpt risk is aggregated by maximum severity; one "high" pulls the overall assessment up.
- Your justification is shown to the reasoning stage, which uses it to discount unreliable excerpts; keep it concrete and short."""

    def build_hallucination_assessment_user_prompt(
        self,
        excerpt_text: str,
        search_results: str,
    ) -> str:
        """Build user prompt for hallucination risk assessment.

        Args:
            excerpt_text: The excerpt text to assess.
            search_results: External search results for verification.

        Returns:
            Formatted user prompt string.
        """
        return (
            f'## Excerpt\n"{excerpt_text}"\n\n'
            f"## Search results\n{search_results}\n\n"
            "## Task\nAssess the hallucination risk for the excerpt against the search results."
        )

    # =========================================================================
    # Stage 2: Reasoning Generation
    # =========================================================================

    def build_reasoning_system_prompt(self) -> str:
        """Build system prompt for reasoning generation."""
        return """You are a careful rubric evaluator. Reason about the answer against the trait's criteria using only the evidence visible in the answer text.

GUIDELINES:
- Be thorough; do not infer beyond what is stated.
- Weigh both supporting and contradicting evidence before concluding.
- Cite specific passages or features when they bear on the criteria.
- If the criteria are ambiguous, state the interpretation you used and apply it consistently.
- Use as much room as the evidence requires; avoid both unnecessary verbosity and premature conclusions."""

    def build_reasoning_user_prompt_with_excerpts(
        self,
        question: str,
        trait: "LLMRubricTrait",
        excerpts: list[dict[str, Any]],
        hallucination_risk: dict[str, Any] | None = None,
        *,
        task_eval_mode: bool = False,
    ) -> str:
        """Build reasoning prompt when excerpts are available.

        Args:
            question: The original question.
            trait: The LLM trait being evaluated.
            excerpts: List of validated excerpts with confidence scores.
            hallucination_risk: Optional hallucination risk assessment.
            task_eval_mode: When True, omit the **Question** block entirely.

        Returns:
            Formatted user prompt string.
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

        question_block = "" if task_eval_mode else f"\n**Question**: {question}\n"

        return f"""Analyze the extracted excerpts to explain how they demonstrate (or fail to demonstrate) the following trait.

**Trait**: {trait.name}
**Criteria**: {trait.description or "Assess this quality"}
{question_block}
**Extracted Excerpts**:
{excerpts_text}
{risk_context}

**Your Task**:
Walk through the evidence carefully:
1. Reference specific excerpts and their content.
2. Connect each excerpt to the trait criteria.
3. Consider both supporting and contradicting evidence.
4. State your interpretation when the criteria are ambiguous.
5. End with a one-line conclusion that follows from the evidence.

This reasoning will be used in a follow-up step to determine the final score.

**Your reasoning:**"""

    def build_reasoning_user_prompt_without_excerpts(
        self,
        question: str,
        answer: str,
        trait: "LLMRubricTrait",
        *,
        task_eval_mode: bool = False,
    ) -> str:
        """Build reasoning prompt when excerpts are not available.

        Args:
            question: The original question.
            answer: The complete LLM response.
            trait: The LLM trait being evaluated.
            task_eval_mode: When True, omit the **Question** block entirely.

        Returns:
            Formatted user prompt string.
        """
        question_block = "" if task_eval_mode else f"\n**Question**: {question}\n"
        return f"""Analyze the following answer against the trait's criteria. Walk through the evidence carefully; this reasoning will be used in a follow-up step to determine the final score.

**Trait**: {trait.name}
**Criteria**: {trait.description or "Assess this quality"}
{question_block}
**Answer**:
{answer}

**Your Task**:
Your reasoning should:
1. Identify which parts of the answer bear on the trait's criteria.
2. Cite specific evidence (passages or features) that supports or contradicts the criteria; quote when useful.
3. Consider both supporting and contradicting evidence before concluding.
4. State your interpretation when the criteria are ambiguous.
5. End with a one-line conclusion that follows from the evidence.

Use as much room as the evidence requires; avoid both unnecessary verbosity and premature conclusions.

**Your reasoning:**"""

    # =========================================================================
    # Stage 3: Score Extraction
    # =========================================================================

    def build_score_extraction_system_prompt(self) -> str:
        """Build system prompt for final score extraction."""
        return """You translate a reasoning analysis into a final score for a trait.

GUIDELINES:
- Base the score solely on the reasoning provided.
- The score must follow logically from the reasoning's conclusion.
- Do not introduce new evidence or contradict the reasoning.
- Be consistent: similar reasoning should yield similar scores."""

    def build_score_extraction_user_prompt(
        self,
        trait: "LLMRubricTrait",
        reasoning: str,
    ) -> str:
        """Build user prompt for final score extraction.

        Args:
            trait: The LLM trait being evaluated
            reasoning: The generated reasoning from the previous stage

        Returns:
            Formatted user prompt string
        """
        if trait.kind == "boolean":
            return self._build_boolean_score_user_prompt(trait, reasoning)
        else:
            return self._build_numeric_score_user_prompt(trait, reasoning)

    def _build_boolean_score_user_prompt(
        self,
        trait: "LLMRubricTrait",
        reasoning: str,
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
- `false`: The criteria IS NOT met based on your reasoning"""

    def _build_numeric_score_user_prompt(
        self,
        trait: "LLMRubricTrait",
        reasoning: str,
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
- {max_score} = Excellent - Fully meets or exceeds criteria"""


# =========================================================================
# Module-level helpers
# =========================================================================


_INTEGER_LABEL_LOOKUP: dict[int, list[str]] = {
    3: ["Does not meet", "Partially meets", "Fully meets"],
    5: ["Poor", "Below average", "Adequate", "Strong", "Excellent"],
    7: [
        "Very poor",
        "Poor",
        "Below average",
        "Adequate",
        "Above average",
        "Strong",
        "Excellent",
    ],
    10: [
        "Poor",
        "Very weak",
        "Weak",
        "Below average",
        "Adequate",
        "Above average",
        "Good",
        "Strong",
        "Very strong",
        "Excellent",
    ],
}

_FALLBACK_LADDER: list[str] = [
    "lowest",
    "very low",
    "low",
    "below average",
    "average",
    "above average",
    "high",
    "very high",
    "highest",
]


def build_integer_score_labels(min_score: int, max_score: int) -> list[tuple[int, str]]:
    """Return one (integer, label) per integer in [min_score, max_score].

    For ranges that start at 1 and have a known total length (3, 5, 7, 10),
    use a curated lookup. Otherwise interpolate the fallback ladder onto the
    range linearly so the lowest rung maps to min_score and the highest rung
    maps to max_score.

    Args:
        min_score: Minimum score (inclusive).
        max_score: Maximum score (inclusive). Must be >= min_score.

    Returns:
        List of (integer, label) tuples ordered from min_score to max_score.

    Raises:
        ValueError: If max_score < min_score.
    """
    if max_score < min_score:
        raise ValueError(f"max_score ({max_score}) must be >= min_score ({min_score})")

    span = max_score - min_score + 1
    integers = list(range(min_score, max_score + 1))

    if min_score == 1 and span in _INTEGER_LABEL_LOOKUP:
        labels = _INTEGER_LABEL_LOOKUP[span]
        return list(zip(integers, labels, strict=True))

    last_idx = len(_FALLBACK_LADDER) - 1
    if span == 1:
        return [(integers[0], _FALLBACK_LADDER[last_idx // 2])]

    pairs: list[tuple[int, str]] = []
    for i, value in enumerate(integers):
        rung = round(i * last_idx / (span - 1))
        pairs.append((value, _FALLBACK_LADDER[rung]))
    return pairs
