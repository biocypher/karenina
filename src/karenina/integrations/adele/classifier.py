"""
ADeLe Question Classifier.

Classifies questions using ADeLe rubrics via LLM-as-judge.
This module adapts the existing LLMTraitEvaluator infrastructure
to evaluate questions (instead of answers) against ADeLe dimensions.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .schemas import QuestionClassificationResult
from .traits import ADELE_TRAIT_NAMES, get_adele_trait

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class QuestionClassifier:
    """
    Classifies questions using ADeLe rubrics via LLM-as-judge.

    This classifier evaluates questions against ADeLe (Assessment Dimensions
    for Language Evaluation) dimensions to characterize their cognitive
    complexity. Each dimension produces a score from 0-5 corresponding to
    levels: none, very_low, low, intermediate, high, very_high.

    Example usage:
        classifier = QuestionClassifier()
        result = classifier.classify_single("What is 2+2?")
        print(result.scores)  # {"attention_and_scan": 0, "volume": 1, ...}
        print(result.labels)  # {"attention_and_scan": "none", "volume": "very_low", ...}
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        model_name: str = "claude-3-5-haiku-latest",
        provider: str = "anthropic",
        temperature: float = 0.0,
    ):
        """
        Initialize the question classifier.

        Args:
            llm: Optional pre-initialized LLM instance. If not provided,
                 one will be created using model_name and provider.
            model_name: Model name to use if llm not provided.
                       Defaults to claude-3-5-haiku-latest for efficiency.
            provider: Model provider to use if llm not provided.
            temperature: Temperature for LLM calls. Defaults to 0.0 for
                        deterministic classifications.
        """
        self._llm = llm
        self._model_name = model_name
        self._provider = provider
        self._temperature = temperature

    @property
    def llm(self) -> BaseChatModel:
        """Lazily initialize and return the LLM instance."""
        if self._llm is None:
            from karenina.infrastructure.llm.interface import init_chat_model_unified

            self._llm = init_chat_model_unified(
                model=self._model_name,
                provider=self._provider,
                temperature=self._temperature,
            )
        return self._llm

    def classify_single(
        self,
        question_text: str,
        trait_names: list[str] | None = None,
        question_id: str | None = None,
    ) -> QuestionClassificationResult:
        """
        Classify a single question against ADeLe dimensions.

        Args:
            question_text: The question text to classify.
            trait_names: Optional list of ADeLe trait names to evaluate.
                        If None, evaluates all 18 ADeLe traits.
            question_id: Optional ID for the question.

        Returns:
            QuestionClassificationResult with scores, labels, and metadata.
        """
        from karenina.benchmark.verification.evaluators.rubric_parsing import (
            invoke_with_structured_output,
        )
        from karenina.schemas.workflow.rubric_outputs import BatchLiteralClassifications

        # Get traits to evaluate
        if trait_names is None:
            trait_names = ADELE_TRAIT_NAMES
        traits = [get_adele_trait(name) for name in trait_names]

        # Build prompts for question classification
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question_text, traits)

        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Invoke with structured output
        parsed_result, usage_metadata = invoke_with_structured_output(self.llm, messages, BatchLiteralClassifications)

        # Validate and convert classifications
        scores, labels = self._validate_classifications(parsed_result.classifications, traits)

        return QuestionClassificationResult(
            question_id=question_id,
            question_text=question_text,
            scores=scores,
            labels=labels,
            model=self._model_name,
            classified_at=datetime.now(UTC).isoformat(),
            usage_metadata=usage_metadata,
        )

    def classify_batch(
        self,
        questions: list[tuple[str, str]],
        trait_names: list[str] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, QuestionClassificationResult]:
        """
        Classify multiple questions against ADeLe dimensions.

        Args:
            questions: List of (question_id, question_text) tuples.
            trait_names: Optional list of ADeLe trait names to evaluate.
                        If None, evaluates all 18 ADeLe traits.
            on_progress: Optional callback function(completed, total) for
                        progress updates.

        Returns:
            Dictionary mapping question_id to QuestionClassificationResult.
        """
        results: dict[str, QuestionClassificationResult] = {}
        total = len(questions)

        for i, (question_id, question_text) in enumerate(questions):
            try:
                result = self.classify_single(
                    question_text=question_text,
                    trait_names=trait_names,
                    question_id=question_id,
                )
                results[question_id] = result
            except Exception as e:
                logger.error(f"Failed to classify question {question_id}: {e}")
                # Create error result
                results[question_id] = QuestionClassificationResult(
                    question_id=question_id,
                    question_text=question_text,
                    scores={},
                    labels={},
                    model=self._model_name,
                    classified_at=datetime.now(UTC).isoformat(),
                    usage_metadata={"error": str(e)},
                )

            if on_progress:
                on_progress(i + 1, total)

        return results

    def _build_system_prompt(self) -> str:
        """Build system prompt for question classification."""
        return """You are an expert evaluator classifying QUESTIONS (not answers) using the ADeLe framework.

ADeLe (Assessment Dimensions for Language Evaluation) characterizes questions by their cognitive complexity across multiple dimensions. Your task is to analyze the QUESTION ITSELF and determine what level of each dimension would be required to answer it.

**RESPONSE FORMAT:**
You will receive a JSON Schema specifying the exact output structure. Your response MUST conform to this schema.
Return ONLY a JSON object - no explanations, no markdown, no surrounding text.

**CRITICAL REQUIREMENTS:**
1. **Exact Class Names**: Use the EXACT class names from each trait's categories (case-sensitive)
2. **One Class Per Trait**: Choose exactly one class for each trait
3. **All Traits Required**: Include ALL traits in your response
4. **No Invention**: Do NOT invent new categories - only use the provided class names

**CLASSIFICATION GUIDELINES:**
- You are classifying the QUESTION, not evaluating an answer
- Consider what cognitive demands the question places on someone trying to answer it
- Read each trait's class definitions carefully - they describe increasing levels of complexity
- When uncertain, choose the level that best represents the primary cognitive demand
- Consider the question holistically - a simple question in one dimension may be complex in another

**WHAT NOT TO DO:**
- Do NOT wrap JSON in markdown code blocks (no ```)
- Do NOT add explanatory text before or after the JSON
- Do NOT modify or paraphrase class names
- Do NOT skip any traits"""

    def _build_user_prompt(self, question_text: str, traits: list[Any]) -> str:
        """Build user prompt for question classification."""
        from karenina.schemas.workflow.rubric_outputs import BatchLiteralClassifications

        traits_description = []
        example_classifications: dict[str, str] = {}

        for trait in traits:
            if trait.kind != "literal" or trait.classes is None:
                continue

            class_names = list(trait.classes.keys())
            # Build class descriptions
            class_details = []
            for name, description in trait.classes.items():
                # Truncate long descriptions for prompt efficiency
                desc_preview = description[:300] + "..." if len(description) > 300 else description
                class_details.append(f"    - **{name}**: {desc_preview}")

            trait_desc = (
                f"- **{trait.name}**: {trait.description or 'Classification trait'}\n"
                f"  Classes (in order of increasing complexity): {', '.join(class_names)}\n" + "\n".join(class_details)
            )
            traits_description.append(trait_desc)
            # Use middle class as example to avoid bias toward first/last
            mid_idx = len(class_names) // 2
            example_classifications[trait.name] = class_names[mid_idx]

        example_json = json.dumps({"classifications": example_classifications}, indent=2)
        json_schema = json.dumps(BatchLiteralClassifications.model_json_schema(), indent=2)

        return f"""Classify the following QUESTION for each ADeLe dimension:

**QUESTION TO CLASSIFY:**
{question_text}

**ADeLe DIMENSIONS TO CLASSIFY:**
{chr(10).join(traits_description)}

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**REQUIRED OUTPUT FORMAT:**
Return a JSON object with a "classifications" key mapping trait names to class names.
Use EXACT trait and class names as shown above.

Example format (class values are just examples):
{example_json}

**YOUR JSON RESPONSE:**"""

    def _validate_classifications(
        self,
        classifications: dict[str, str],
        traits: list[Any],
    ) -> tuple[dict[str, int], dict[str, str]]:
        """
        Validate and convert classifications to scores and labels.

        Args:
            classifications: Dict mapping trait names to class names from LLM
            traits: List of ADeLe traits being evaluated

        Returns:
            Tuple of (scores, labels) dictionaries
        """
        scores: dict[str, int] = {}
        labels: dict[str, str] = {}
        trait_map = {trait.name: trait for trait in traits}

        for trait_name, class_name in classifications.items():
            if trait_name in trait_map:
                trait = trait_map[trait_name]
                score, label = self._validate_single_classification(trait, class_name)
                scores[trait_name] = score
                labels[trait_name] = label

        # Add error state for missing traits
        for trait in traits:
            if trait.name not in scores:
                scores[trait.name] = -1
                labels[trait.name] = "[MISSING_FROM_RESPONSE]"

        return scores, labels

    def _validate_single_classification(self, trait: Any, class_name: str) -> tuple[int, str]:
        """
        Validate and convert a class name to score index and label.

        Args:
            trait: The ADeLe trait being evaluated
            class_name: The class name returned by the LLM

        Returns:
            Tuple of (score, label) where:
            - score: Int index (0 to N-1) if valid, -1 if invalid class name
            - label: The class name if valid, or the invalid value for debugging
        """
        if trait.kind != "literal" or trait.classes is None:
            return -1, f"[NOT_LITERAL_TRAIT: {class_name}]"

        # Get the index for the class name
        index = trait.get_class_index(class_name)
        if index == -1:
            # Try case-insensitive matching as fallback
            class_names_lower = {name.lower(): name for name in trait.classes}
            matched_name = class_names_lower.get(class_name.lower())
            if matched_name is not None:
                index = trait.get_class_index(matched_name)
                class_name = matched_name  # Use the canonical name
            else:
                # Invalid class name - store the invalid value for debugging
                logger.warning(
                    f"Invalid class '{class_name}' for trait '{trait.name}'. "
                    f"Valid classes: {list(trait.classes.keys())}"
                )
                return -1, class_name  # Return invalid class name for debugging

        return index, class_name
