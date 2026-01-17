"""Deep judgment evaluation for rubric traits.

This module implements the deep judgment feature for rubric trait evaluation,
which extends standard LLM trait evaluation with a multi-stage approach:

1. Stage 1: Extract verbatim excerpts from the answer that demonstrate the trait
   - Excerpts are validated using fuzzy matching to prevent hallucinations
   - Retry logic handles validation failures

2. Stage 1.5 (Optional): Search-enhanced hallucination assessment
   - Uses external search to validate excerpt claims
   - Assigns per-excerpt hallucination risk scores

3. Stage 2: Generate reasoning explaining how excerpts support/refute the trait

4. Stage 3: Extract final score based on reasoning

For traits without excerpt extraction enabled, only stages 2-3 are performed.
"""

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from ....schemas.domain import LLMRubricTrait, Rubric

logger = logging.getLogger(__name__)


class RubricDeepJudgmentHandler:
    """
    Handles deep judgment evaluation for rubric traits.

    Deep judgment is a multi-stage evaluation process that:
    1. Extracts verbatim excerpts from the answer
    2. Optionally validates excerpts against external search
    3. Generates reasoning explaining how excerpts support the trait
    4. Extracts a final score based on the reasoning

    This approach provides more transparent and verifiable trait evaluation
    compared to single-shot LLM judgment.
    """

    def __init__(self, llm: Any, model_config: Any):
        """
        Initialize the deep judgment handler.

        Args:
            llm: Initialized LLM instance for evaluation
            model_config: Configuration for the evaluation model
        """
        self.llm = llm
        self.model_config = model_config

    def evaluate_rubric_with_deep_judgment(
        self,
        question: str,
        answer: str,
        rubric: "Rubric",
        config: Any,  # VerificationConfig
        standard_evaluator_fn: Any,  # Callback for standard trait evaluation
    ) -> dict[str, Any]:
        """
        Evaluate rubric with deep judgment for enabled traits.

        Args:
            question: The original question
            answer: The LLM response to evaluate
            rubric: The rubric containing evaluation traits
            config: VerificationConfig with deep judgment settings
            standard_evaluator_fn: Callback function to evaluate standard (non-DJ) traits
                                   Signature: (question, answer, rubric) -> (scores, usage_metadata_list)

        Returns:
            Dictionary containing:
                - deep_judgment_scores: Scores for deep-judgment-enabled traits
                - standard_scores: Scores for standard traits
                - excerpts: Extracted excerpts per trait
                - reasoning: Reasoning per trait
                - metadata: Per-trait evaluation metadata
                - hallucination_risks: Per-trait hallucination risk (if search enabled)
                - traits_without_valid_excerpts: Traits that failed excerpt extraction
        """
        from ....schemas.domain import Rubric as RubricClass

        # Separate deep-judgment vs standard traits
        dj_traits = [t for t in rubric.llm_traits if t.deep_judgment_enabled]
        standard_traits = [t for t in rubric.llm_traits if not t.deep_judgment_enabled]

        # Initialize result containers
        dj_scores: dict[str, int | bool] = {}
        excerpts: dict[str, list[dict[str, Any]]] = {}
        reasoning: dict[str, str] = {}
        metadata: dict[str, dict[str, Any]] = {}
        hallucination_risks: dict[str, dict[str, Any]] = {}
        auto_fail_traits: list[str] = []
        usage_metadata_list: list[dict[str, Any]] = []  # Aggregate usage across all traits

        logger.info(
            f"Evaluating rubric with deep judgment: {len(dj_traits)} DJ traits, {len(standard_traits)} standard traits"
        )

        # Sequential evaluation for deep-judgment traits (one at a time)
        for trait in dj_traits:
            logger.debug(f"Evaluating deep judgment trait: {trait.name}")
            try:
                trait_result = self._evaluate_single_trait_with_deep_judgment(question, answer, trait, config)

                # Collect usage metadata from this trait
                usage_metadata_list.extend(trait_result.get("usage_metadata_list", []))

                # Check for auto-fail
                if trait_result.get("auto_fail"):
                    auto_fail_traits.append(trait.name)
                    metadata[trait.name] = trait_result["metadata"]
                    logger.warning(
                        f"Trait '{trait.name}' auto-failed: no valid excerpts after "
                        f"{trait_result['metadata'].get('excerpt_retry_count', 0)} retries"
                    )
                    continue  # Skip to next trait

                # Store successful evaluation
                dj_scores[trait.name] = trait_result["score"]
                reasoning[trait.name] = trait_result["reasoning"]
                metadata[trait.name] = trait_result["metadata"]

                # Store excerpts if extraction was enabled
                if trait.deep_judgment_excerpt_enabled and "excerpts" in trait_result:
                    excerpts[trait.name] = trait_result["excerpts"]

                    # Store hallucination risk if search was enabled
                    if trait.deep_judgment_search_enabled and "hallucination_risk" in trait_result:
                        hallucination_risks[trait.name] = trait_result["hallucination_risk"]

            except Exception as e:
                logger.error(f"Failed to evaluate deep judgment trait '{trait.name}': {e}")
                # Mark trait as failed
                auto_fail_traits.append(trait.name)
                metadata[trait.name] = {
                    "stages_completed": [],
                    "model_calls": 0,
                    "error": str(e),
                }

        # Evaluate standard traits using the provided callback
        standard_scores: dict[str, int | bool] = {}
        standard_usage_metadata_list: list[dict[str, Any]] = []
        if standard_traits:
            logger.debug(f"Evaluating {len(standard_traits)} standard traits")
            standard_rubric = RubricClass(llm_traits=standard_traits)
            standard_scores, standard_usage_metadata_list = standard_evaluator_fn(question, answer, standard_rubric)
            usage_metadata_list.extend(standard_usage_metadata_list)

        return {
            "deep_judgment_scores": dj_scores,
            "standard_scores": standard_scores,
            "excerpts": excerpts,
            "reasoning": reasoning,
            "metadata": metadata,
            "hallucination_risks": hallucination_risks,
            "traits_without_valid_excerpts": auto_fail_traits,
            "usage_metadata_list": usage_metadata_list,
        }

    def _evaluate_single_trait_with_deep_judgment(
        self, question: str, answer: str, trait: "LLMRubricTrait", config: Any
    ) -> dict[str, Any]:
        """
        Evaluate a single trait using deep judgment (sequential multi-stage process).

        Returns:
            Dictionary with: score, reasoning, excerpts (optional), metadata, auto_fail flag
        """
        # Initialize metadata
        metadata: dict[str, Any] = {
            "stages_completed": [],
            "model_calls": 0,
            "had_excerpts": trait.deep_judgment_excerpt_enabled,
            "excerpt_retry_count": 0,
            "excerpt_validation_failed": False,
        }

        # Determine flow based on excerpt_enabled
        if trait.deep_judgment_excerpt_enabled:
            # Flow 1: With excerpts (3-4 stages)
            return self._evaluate_trait_with_excerpts(question, answer, trait, config, metadata)
        else:
            # Flow 2: Without excerpts (2 stages)
            return self._evaluate_trait_without_excerpts(question, answer, trait, config, metadata)

    def _evaluate_trait_with_excerpts(
        self, question: str, answer: str, trait: "LLMRubricTrait", config: Any, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Evaluate trait with excerpt extraction (Flow 1: 3-4 stages).

        Stages:
        1. Extract excerpts (with retry on validation failure)
        1.5. Optional: Search-enhanced hallucination assessment
        2. Generate reasoning based on excerpts
        3. Extract final score
        """
        usage_metadata_list = []  # Aggregate usage across all stages

        # Stage 1: Extract excerpts with retry
        excerpt_result = self._extract_excerpts_for_trait(answer, trait, config)
        metadata["model_calls"] += excerpt_result["model_calls"]
        metadata["excerpt_retry_count"] = excerpt_result["retry_count"]
        usage_metadata_list.extend(excerpt_result.get("usage_metadata_list", []))

        # Check for auto-fail
        if excerpt_result.get("auto_fail"):
            metadata["excerpt_validation_failed"] = True
            return {
                "auto_fail": True,
                "metadata": metadata,
                "usage_metadata_list": usage_metadata_list,
            }

        excerpts = excerpt_result["excerpts"]
        metadata["stages_completed"].append("excerpt_extraction")

        # Stage 1.5: Optional search-enhanced hallucination assessment
        hallucination_risk = None
        if trait.deep_judgment_search_enabled and excerpts:
            hallucination_result = self._assess_trait_hallucination(excerpts, trait, config)
            excerpts = hallucination_result["excerpts"]  # Updated with search results
            hallucination_risk = hallucination_result["risk_assessment"]
            metadata["model_calls"] += hallucination_result["model_calls"]
            usage_metadata_list.extend(hallucination_result.get("usage_metadata_list", []))
            metadata["stages_completed"].append("hallucination_assessment")

        # Stage 2: Generate reasoning based on excerpts
        reasoning_result = self._generate_reasoning_for_trait(
            question, answer, trait, excerpts=excerpts, hallucination_risk=hallucination_risk
        )
        reasoning = reasoning_result["reasoning"]
        metadata["model_calls"] += 1
        if reasoning_result.get("usage_metadata"):
            usage_metadata_list.append(reasoning_result["usage_metadata"])
        metadata["stages_completed"].append("reasoning_generation")

        # Stage 3: Extract score
        score_result = self._extract_score_for_trait(question, answer, trait, reasoning)
        score = score_result["score"]
        metadata["model_calls"] += 1
        if score_result.get("usage_metadata"):
            usage_metadata_list.append(score_result["usage_metadata"])
        metadata["stages_completed"].append("score_extraction")

        return {
            "score": score,
            "reasoning": reasoning,
            "excerpts": excerpts,
            "hallucination_risk": hallucination_risk,
            "metadata": metadata,
            "auto_fail": False,
            "usage_metadata_list": usage_metadata_list,
        }

    def _evaluate_trait_without_excerpts(
        self,
        question: str,
        answer: str,
        trait: "LLMRubricTrait",
        config: Any,  # noqa: ARG002 - Kept for method signature consistency
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Evaluate trait without excerpt extraction (Flow 2: 2 stages).

        Stages:
        1. Generate reasoning based on full answer
        2. Extract final score
        """
        usage_metadata_list = []  # Aggregate usage across stages

        # Stage 1: Generate reasoning (no excerpts)
        reasoning_result = self._generate_reasoning_for_trait(
            question, answer, trait, excerpts=None, hallucination_risk=None
        )
        reasoning = reasoning_result["reasoning"]
        metadata["model_calls"] += 1
        if reasoning_result.get("usage_metadata"):
            usage_metadata_list.append(reasoning_result["usage_metadata"])
        metadata["stages_completed"].append("reasoning_generation")

        # Stage 2: Extract score
        score_result = self._extract_score_for_trait(question, answer, trait, reasoning)
        score = score_result["score"]
        metadata["model_calls"] += 1
        if score_result.get("usage_metadata"):
            usage_metadata_list.append(score_result["usage_metadata"])
        metadata["stages_completed"].append("score_extraction")

        return {
            "score": score,
            "reasoning": reasoning,
            "metadata": metadata,
            "auto_fail": False,
            "usage_metadata_list": usage_metadata_list,
        }

    def _extract_excerpts_for_trait(self, answer: str, trait: "LLMRubricTrait", config: Any) -> dict[str, Any]:
        """
        Extract excerpts for a trait with retry on validation failure.

        Returns:
            Dictionary with: excerpts, retry_count, model_calls, auto_fail flag, usage_metadata
        """
        # Get configuration values (per-trait overrides global defaults)
        max_attempts = (
            trait.deep_judgment_excerpt_retry_attempts
            if trait.deep_judgment_excerpt_retry_attempts is not None
            else config.deep_judgment_rubric_excerpt_retry_attempts_default
        )
        fuzzy_threshold = (
            trait.deep_judgment_fuzzy_match_threshold
            if trait.deep_judgment_fuzzy_match_threshold is not None
            else config.deep_judgment_rubric_fuzzy_match_threshold_default
        )
        max_excerpts = (
            trait.deep_judgment_max_excerpts
            if trait.deep_judgment_max_excerpts is not None
            else config.deep_judgment_rubric_max_excerpts_default
        )

        retry_count = 0
        model_calls = 0
        validation_feedback = None
        usage_metadata_list = []  # Track usage across retries

        # Retry loop
        for attempt in range(max_attempts + 1):  # Initial + retries
            # Build prompt (with feedback if retry)
            prompt = self._build_trait_excerpt_prompt(trait, max_excerpts, answer, validation_feedback)

            # Call LLM
            messages = [
                SystemMessage(
                    content="""You are an expert at extracting verbatim quotes from text that demonstrate specific qualities.

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
                ),
                HumanMessage(content=prompt),
            ]

            with get_usage_metadata_callback() as cb:
                response = self.llm.invoke(messages)
            model_calls += 1
            if cb.usage_metadata:
                usage_metadata_list.append(dict(cb.usage_metadata))

            raw_response = response.content if hasattr(response, "content") else str(response)

            # Parse excerpts
            try:
                raw_excerpts = self._parse_excerpt_response(raw_response)
            except Exception as e:
                logger.warning(f"Failed to parse excerpt response for trait '{trait.name}': {e}")
                if attempt < max_attempts:
                    validation_feedback = (
                        f"Failed to parse response: {e}. Please return valid JSON with an 'excerpts' array."
                    )
                    retry_count += 1
                    continue
                else:
                    # All retries exhausted
                    return {
                        "excerpts": [],
                        "retry_count": retry_count,
                        "model_calls": model_calls,
                        "auto_fail": True,
                        "usage_metadata_list": usage_metadata_list,
                    }

            # Validate excerpts with fuzzy matching
            validated, failed = self._validate_trait_excerpts(raw_excerpts, answer, fuzzy_threshold)

            if validated:
                # Success!
                return {
                    "excerpts": validated,
                    "retry_count": retry_count,
                    "model_calls": model_calls,
                    "auto_fail": False,
                    "usage_metadata_list": usage_metadata_list,
                }

            # All excerpts failed validation
            if attempt < max_attempts:
                # Build feedback for retry
                validation_feedback = self._build_retry_feedback(failed, fuzzy_threshold)
                retry_count += 1
                logger.debug(
                    f"Trait '{trait.name}' excerpt validation failed (attempt {attempt + 1}), retrying with feedback"
                )
            else:
                # Exhausted all retries
                logger.warning(f"Trait '{trait.name}' failed excerpt extraction after {retry_count} retries")
                return {
                    "excerpts": [],
                    "retry_count": retry_count,
                    "model_calls": model_calls,
                    "auto_fail": True,
                    "usage_metadata_list": usage_metadata_list,
                }

        # Should not reach here
        return {
            "excerpts": [],
            "retry_count": retry_count,
            "model_calls": model_calls,
            "auto_fail": True,
            "usage_metadata_list": usage_metadata_list,
        }

    def _validate_trait_excerpts(
        self, excerpts: list[dict[str, Any]], answer: str, fuzzy_threshold: float
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Validate excerpts using fuzzy matching.

        Returns:
            Tuple of (valid_excerpts, failed_excerpts)
        """
        from ..utils.trace_fuzzy_match import fuzzy_match_excerpt

        valid = []
        failed = []

        for excerpt in excerpts:
            excerpt_text = excerpt.get("text", "")
            match_found, similarity = fuzzy_match_excerpt(excerpt_text, answer)

            if match_found and similarity >= fuzzy_threshold:
                excerpt["similarity_score"] = similarity
                valid.append(excerpt)
            else:
                excerpt["similarity_score"] = similarity
                failed.append(excerpt)

        return valid, failed

    def _build_retry_feedback(self, failed_excerpts: list[dict[str, Any]], fuzzy_threshold: float) -> str:
        """
        Build feedback message for retry attempt.

        Same format as template Deep Judgment.
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

    def _build_trait_excerpt_prompt(
        self, trait: "LLMRubricTrait", max_excerpts: int, answer: str, feedback: str | None = None
    ) -> str:
        """Build prompt for excerpt extraction (with optional retry feedback)."""
        from ....schemas.workflow.rubric_outputs import TraitExcerptsOutput

        json_schema = json.dumps(TraitExcerptsOutput.model_json_schema(), indent=2)

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

    def _parse_excerpt_response(self, response: str) -> list[dict[str, Any]]:
        """Parse the excerpt extraction response."""
        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response[:200]}")

        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}") from e

        excerpts = result.get("excerpts", [])
        if not isinstance(excerpts, list):
            raise ValueError(f"'excerpts' must be a list, got {type(excerpts)}")

        return excerpts

    def _assess_trait_hallucination(
        self, excerpts: list[dict[str, Any]], trait: "LLMRubricTrait", config: Any
    ) -> dict[str, Any]:
        """
        Assess hallucination risk for trait excerpts using search.

        Returns:
            Dictionary with: excerpts (updated with search results), risk_assessment, model_calls, usage_metadata_list
        """
        from ..utils.search_provider import create_search_tool

        # Create search tool
        search_tool = create_search_tool(config.deep_judgment_rubric_search_tool)

        # Batch search for all excerpts
        excerpt_texts = [e.get("text", "") for e in excerpts]
        try:
            search_results = search_tool(excerpt_texts)
            # Ensure it's a list
            if not isinstance(search_results, list):
                search_results = [search_results] * len(excerpt_texts)
        except Exception as e:
            logger.warning(f"Search failed for trait '{trait.name}': {e}")
            search_results = ["Search failed"] * len(excerpt_texts)  # type: ignore[assignment]

        # Assess hallucination risk per excerpt
        model_calls = 0
        per_excerpt_risks = []
        usage_metadata_list = []  # Track usage across hallucination assessments

        for i, excerpt in enumerate(excerpts):
            # Build prompt for hallucination assessment
            from ....schemas.workflow.rubric_outputs import HallucinationRiskOutput

            json_schema = json.dumps(HallucinationRiskOutput.model_json_schema(), indent=2)

            prompt = f"""Assess the hallucination risk for this excerpt by comparing it against external search results.

**EXCERPT TO VERIFY:**
"{excerpt.get("text", "")}"

**EXTERNAL SEARCH RESULTS:**
{search_results[i] if i < len(search_results) else "No results available"}

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

            messages: list[BaseMessage] = [
                SystemMessage(
                    content="""You are an expert at assessing hallucination risk using external evidence.

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
                ),
                HumanMessage(content=prompt),
            ]

            with get_usage_metadata_callback() as cb:
                response = self.llm.invoke(messages)
            model_calls += 1
            if cb.usage_metadata:
                usage_metadata_list.append(dict(cb.usage_metadata))

            raw_response = response.content if hasattr(response, "content") else str(response)

            # Parse response
            try:
                json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if json_match:
                    risk_data = json.loads(json_match.group())
                    risk = risk_data.get("risk", "medium")
                    justification = risk_data.get("justification", "")
                else:
                    risk = "medium"
                    justification = "Failed to parse response"
            except Exception as e:
                logger.warning(f"Failed to parse hallucination assessment: {e}")
                risk = "medium"
                justification = "Failed to parse"

            # Update excerpt with search data
            excerpt["search_results"] = search_results[i] if i < len(search_results) else None
            excerpt["hallucination_risk"] = risk
            excerpt["hallucination_justification"] = justification
            per_excerpt_risks.append(risk)

        # Overall risk = MAX of per-excerpt risks
        risk_levels = {"none": 0, "low": 1, "medium": 2, "high": 3}
        max_risk_level = max((risk_levels.get(r, 2) for r in per_excerpt_risks), default=2)
        overall_risk = [k for k, v in risk_levels.items() if v == max_risk_level][0]

        return {
            "excerpts": excerpts,
            "risk_assessment": {
                "overall_risk": overall_risk,
                "per_excerpt_risks": per_excerpt_risks,
            },
            "model_calls": model_calls,
            "usage_metadata_list": usage_metadata_list,
        }

    def _generate_reasoning_for_trait(
        self,
        question: str,
        answer: str,
        trait: "LLMRubricTrait",
        excerpts: list[dict[str, Any]] | None = None,
        hallucination_risk: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate reasoning explaining the trait score.

        Returns:
            Dictionary with: reasoning (string), usage_metadata
        """
        if excerpts is not None:
            # With excerpts
            prompt = self._build_trait_reasoning_prompt_with_excerpts(
                question, answer, trait, excerpts, hallucination_risk
            )
        else:
            # Without excerpts
            prompt = self._build_trait_reasoning_prompt_without_excerpts(question, answer, trait)

        messages: list[BaseMessage] = [
            SystemMessage(content="You are an expert at analyzing text quality and providing clear reasoning."),
            HumanMessage(content=prompt),
        ]

        with get_usage_metadata_callback() as cb:
            response = self.llm.invoke(messages)

        raw_response = response.content if hasattr(response, "content") else str(response)
        reasoning = raw_response.strip()

        usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}
        return {"reasoning": reasoning, "usage_metadata": usage_metadata}

    def _build_trait_reasoning_prompt_with_excerpts(
        self,
        question: str,
        answer: str,  # noqa: ARG002 - Excerpts already contain relevant answer portions
        trait: "LLMRubricTrait",
        excerpts: list[dict[str, Any]],
        hallucination_risk: dict[str, Any] | None = None,
    ) -> str:
        """Build reasoning prompt with excerpts."""
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

    def _build_trait_reasoning_prompt_without_excerpts(
        self, question: str, answer: str, trait: "LLMRubricTrait"
    ) -> str:
        """Build reasoning prompt without excerpts (based on full response)."""
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

    def _extract_score_for_trait(
        self,
        question: str,  # noqa: ARG002 - Kept for API consistency with other trait methods
        answer: str,  # noqa: ARG002 - Kept for API consistency with other trait methods
        trait: "LLMRubricTrait",
        reasoning: str,
    ) -> dict[str, Any]:
        """
        Extract final score for trait based on reasoning.

        Returns:
            Dictionary with: score (int | bool), usage_metadata
        """
        prompt = self._build_trait_scoring_prompt(trait, reasoning)

        messages: list[BaseMessage] = [
            SystemMessage(
                content="""You are an expert evaluator providing precise trait scores based on prior reasoning.

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
            ),
            HumanMessage(content=prompt),
        ]

        with get_usage_metadata_callback() as cb:
            response = self.llm.invoke(messages)

        raw_response = response.content if hasattr(response, "content") else str(response)

        # Parse score
        score = self._parse_trait_score_response(raw_response, trait)

        usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}
        return {"score": score, "usage_metadata": usage_metadata}

    def _build_trait_scoring_prompt(self, trait: "LLMRubricTrait", reasoning: str) -> str:
        """Build prompt for final scoring."""
        from ....schemas.workflow.rubric_outputs import SingleBooleanScore, SingleNumericScore

        if trait.kind == "boolean":
            json_schema = json.dumps(SingleBooleanScore.model_json_schema(), indent=2)
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
        else:
            min_score = trait.min_score or 1
            max_score = trait.max_score or 5
            mid_score = (min_score + max_score) // 2
            json_schema = json.dumps(SingleNumericScore.model_json_schema(), indent=2)
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

    def _parse_trait_score_response(self, response: str, trait: "LLMRubricTrait") -> int | bool:
        """Parse a trait score response."""
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
                    logger.warning(f"Could not parse boolean from: {response}, defaulting to False")
                    return False
        else:
            # Extract numeric score
            numbers = re.findall(r"\d+", response)
            if not numbers:
                logger.warning(f"No numeric score found in: {response}, defaulting to minimum score")
                return trait.min_score or 1

            score = int(numbers[0])
            # Validate and clamp score
            min_score = trait.min_score or 1
            max_score = trait.max_score or 5
            clamped_score = max(min_score, min(max_score, score))

            if clamped_score != score:
                logger.debug(f"Score {score} clamped to valid range [{min_score}, {max_score}]: {clamped_score}")

            return clamped_score
