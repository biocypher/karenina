"""
ADeLe Question Classifier.

Classifies questions using ADeLe rubrics via LLM-as-judge.
This module adapts the existing LLMTraitEvaluator infrastructure
to evaluate questions (instead of answers) against ADeLe dimensions.

All LLM calls use LLMPort.with_structured_output() for consistent backend abstraction.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from karenina.ports import LLMPort
from karenina.ports.messages import Message

from .schemas import QuestionClassificationResult
from .traits import ADELE_TRAIT_NAMES, get_adele_trait

if TYPE_CHECKING:
    from karenina.schemas.workflow.models import ModelConfig

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
        llm: LLMPort | None = None,
        model_name: str = "claude-3-5-haiku-latest",
        provider: str = "anthropic",
        temperature: float = 0.0,
        interface: str = "langchain",
        endpoint_base_url: str | None = None,
        endpoint_api_key: str | None = None,
        trait_eval_mode: str = "batch",
        async_enabled: bool | None = None,
        async_max_workers: int | None = None,
        *,
        model_config: ModelConfig | None = None,
    ):
        """
        Initialize the question classifier.

        Args:
            llm: Optional pre-initialized LLMPort instance. If not provided,
                 one will be created using model_config or individual params.
            model_name: Model name to use if llm not provided.
                       Defaults to claude-3-5-haiku-latest for efficiency.
            provider: Model provider to use if llm not provided.
            temperature: Temperature for LLM calls. Defaults to 0.0 for
                        deterministic classifications.
            interface: The interface to use for model initialization.
                      Supported values: "langchain", "openrouter", "openai_endpoint".
                      Defaults to "langchain".
            endpoint_base_url: Custom base URL for openai_endpoint interface.
                              Required when interface="openai_endpoint".
            endpoint_api_key: API key for openai_endpoint interface.
                             Required when interface="openai_endpoint".
            trait_eval_mode: How to evaluate traits for a single question.
                            "batch" - all traits in one LLM call (faster, cheaper)
                            "sequential" - each trait in separate call (potentially more accurate)
                            Defaults to "batch".
            async_enabled: Whether to run sequential trait evaluations in parallel.
                          If None, reads from KARENINA_ASYNC_ENABLED env var (default: True).
            async_max_workers: Max concurrent workers for parallel execution.
                              If None, reads from KARENINA_ASYNC_MAX_WORKERS env var (default: 2).
            model_config: Optional ModelConfig to use for creating the LLM.
                         Takes precedence over individual model params.
        """
        from karenina.adapters.llm_parallel import read_async_config

        self._llm = llm
        self._model_config = model_config
        self._model_name = model_name
        self._provider = provider
        self._temperature = temperature
        self._interface = interface
        self._endpoint_base_url = endpoint_base_url
        self._endpoint_api_key = endpoint_api_key
        self._trait_eval_mode = trait_eval_mode

        # Read async config with env var fallbacks
        default_enabled, default_workers = read_async_config()
        self._async_enabled = async_enabled if async_enabled is not None else default_enabled
        self._async_max_workers = async_max_workers if async_max_workers is not None else default_workers

    @property
    def llm(self) -> LLMPort:
        """Lazily initialize and return the LLM instance."""
        if self._llm is None:
            from pydantic import SecretStr

            from karenina.adapters.factory import get_llm
            from karenina.schemas.workflow.models import ModelConfig

            # Use provided model_config or create one from individual params
            if self._model_config is not None:
                config = self._model_config
            else:
                # Convert endpoint_api_key to SecretStr if provided
                api_key = SecretStr(self._endpoint_api_key) if self._endpoint_api_key else None
                config = ModelConfig(
                    id="adele-classifier",
                    model_name=self._model_name,
                    model_provider=self._provider,
                    temperature=self._temperature,
                    interface=self._interface,  # type: ignore[arg-type]  # Runtime validated Literal
                    endpoint_base_url=self._endpoint_base_url,
                    endpoint_api_key=api_key,
                )
            self._llm = get_llm(config)
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
        # Get traits to evaluate
        if trait_names is None:
            trait_names = ADELE_TRAIT_NAMES
        traits = [get_adele_trait(name) for name in trait_names]

        # Choose evaluation mode
        if self._trait_eval_mode == "sequential":
            return self._classify_single_sequential(question_text, traits, question_id)
        else:
            return self._classify_single_batch(question_text, traits, question_id)

    def _classify_single_batch(
        self,
        question_text: str,
        traits: list[Any],
        question_id: str | None = None,
    ) -> QuestionClassificationResult:
        """
        Classify a question by evaluating all traits in a single LLM call.

        This is faster and cheaper but may be less accurate for complex questions.
        """
        from karenina.schemas.workflow.rubric_outputs import BatchLiteralClassifications

        # Build prompts for question classification
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question_text, traits)

        messages = [
            Message.system(system_prompt),
            Message.user(user_prompt),
        ]

        # Invoke with structured output using LLMPort
        structured_llm = self.llm.with_structured_output(BatchLiteralClassifications)
        response = structured_llm.invoke(messages)

        # response.raw is the validated Pydantic model
        parsed_result = response.raw
        usage_metadata = asdict(response.usage) if response.usage else {}

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

    def _classify_single_sequential(
        self,
        question_text: str,
        traits: list[Any],
        question_id: str | None = None,
    ) -> QuestionClassificationResult:
        """
        Classify a question by evaluating each trait in a separate LLM call.

        When async_enabled is True, the LLM calls run in parallel using
        LLMParallelInvoker for significant speedup. Otherwise, calls run
        sequentially (legacy behavior).
        """
        from karenina.schemas.workflow.rubric_outputs import SingleLiteralClassification

        scores: dict[str, int] = {}
        labels: dict[str, str] = {}
        combined_usage: dict[str, Any] = {"calls": 0, "total_tokens": 0}

        # Build all tasks upfront using Message types
        tasks: list[tuple[list[Message], type[SingleLiteralClassification]]] = []
        for trait in traits:
            system_prompt = self._build_system_prompt_single_trait()
            user_prompt = self._build_user_prompt_single_trait(question_text, trait)
            messages = [
                Message.system(system_prompt),
                Message.user(user_prompt),
            ]
            tasks.append((messages, SingleLiteralClassification))

        if self._async_enabled:
            # Execute in parallel
            scores, labels, combined_usage = self._execute_parallel_classification(tasks, traits)
        else:
            # Fall back to sequential execution
            scores, labels, combined_usage = self._execute_sequential_classification(tasks, traits)

        return QuestionClassificationResult(
            question_id=question_id,
            question_text=question_text,
            scores=scores,
            labels=labels,
            model=self._model_name,
            classified_at=datetime.now(UTC).isoformat(),
            usage_metadata=combined_usage,
        )

    def _execute_parallel_classification(
        self,
        tasks: list[tuple[list[Message], Any]],
        traits: list[Any],
    ) -> tuple[dict[str, int], dict[str, str], dict[str, Any]]:
        """Execute classification tasks in parallel using LLMParallelInvoker."""
        from karenina.adapters.llm_parallel import LLMParallelInvoker

        invoker = LLMParallelInvoker(self.llm, max_workers=self._async_max_workers)
        results = invoker.invoke_batch_structured(tasks)

        scores: dict[str, int] = {}
        labels: dict[str, str] = {}
        combined_usage: dict[str, Any] = {"calls": 0, "total_tokens": 0, "input_tokens": 0, "output_tokens": 0}

        for i, (result, usage, error) in enumerate(results):
            trait = traits[i]
            if error:
                logger.error(f"Failed to classify trait {trait.name}: {error}")
                scores[trait.name] = -1
                labels[trait.name] = f"[ERROR: {error!s}]"
            else:
                assert result is not None  # mypy: error implies result is None
                score, label = self._validate_single_classification(trait, result.classification)
                scores[trait.name] = score
                labels[trait.name] = label
                combined_usage["calls"] += 1
                if usage:
                    combined_usage["total_tokens"] += usage.get("total_tokens", 0)
                    combined_usage["input_tokens"] += usage.get("input_tokens", 0)
                    combined_usage["output_tokens"] += usage.get("output_tokens", 0)

        return scores, labels, combined_usage

    def _execute_sequential_classification(
        self,
        tasks: list[tuple[list[Message], Any]],
        traits: list[Any],
    ) -> tuple[dict[str, int], dict[str, str], dict[str, Any]]:
        """Execute classification tasks sequentially (legacy behavior)."""
        scores: dict[str, int] = {}
        labels: dict[str, str] = {}
        combined_usage: dict[str, Any] = {"calls": 0, "total_tokens": 0, "input_tokens": 0, "output_tokens": 0}

        for i, (messages, model_class) in enumerate(tasks):
            trait = traits[i]
            try:
                # Use LLMPort.with_structured_output() pattern
                structured_llm = self.llm.with_structured_output(model_class)
                response = structured_llm.invoke(messages)

                # response.raw is the validated Pydantic model
                parsed_result = response.raw
                usage_metadata = asdict(response.usage) if response.usage else {}

                # Validate the classification
                score, label = self._validate_single_classification(trait, parsed_result.classification)
                scores[trait.name] = score
                labels[trait.name] = label

                # Accumulate usage metadata
                combined_usage["calls"] += 1
                if usage_metadata:
                    combined_usage["total_tokens"] += usage_metadata.get("total_tokens", 0)
                    combined_usage["input_tokens"] += usage_metadata.get("input_tokens", 0)
                    combined_usage["output_tokens"] += usage_metadata.get("output_tokens", 0)

            except Exception as e:
                logger.error(f"Failed to classify trait {trait.name}: {e}")
                scores[trait.name] = -1
                labels[trait.name] = f"[ERROR: {e!s}]"

        return scores, labels, combined_usage

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

    def _build_system_prompt_single_trait(self) -> str:
        """Build system prompt for single-trait question classification."""
        return """You are an expert evaluator classifying QUESTIONS (not answers) using the ADeLe framework.

ADeLe (Assessment Dimensions for Language Evaluation) characterizes questions by their cognitive complexity. Your task is to analyze the QUESTION ITSELF and classify it for a SINGLE dimension.

**RESPONSE FORMAT:**
You will receive a JSON Schema specifying the exact output structure. Your response MUST conform to this schema.
Return ONLY a JSON object - no explanations, no markdown, no surrounding text.

**CRITICAL REQUIREMENTS:**
1. **Exact Class Name**: Use the EXACT class name from the trait's categories (case-sensitive)
2. **One Class Only**: Choose exactly one class
3. **No Invention**: Do NOT invent new categories - only use the provided class names

**CLASSIFICATION GUIDELINES:**
- You are classifying the QUESTION, not evaluating an answer
- Consider what cognitive demands the question places on someone trying to answer it
- Read each class definition carefully - they describe increasing levels of complexity
- When uncertain, choose the level that best represents the primary cognitive demand

**WHAT NOT TO DO:**
- Do NOT wrap JSON in markdown code blocks (no ```)
- Do NOT add explanatory text before or after the JSON
- Do NOT modify or paraphrase class names"""

    def _build_user_prompt_single_trait(self, question_text: str, trait: Any) -> str:
        """Build user prompt for single-trait question classification."""
        from karenina.schemas.workflow.rubric_outputs import SingleLiteralClassification

        if trait.kind != "literal" or trait.classes is None:
            raise ValueError(f"Trait {trait.name} is not a literal trait with classes")

        class_names = list(trait.classes.keys())
        class_details = []
        for name, description in trait.classes.items():
            desc_preview = description[:400] + "..." if len(description) > 400 else description
            class_details.append(f"  - **{name}**: {desc_preview}")

        json_schema = json.dumps(SingleLiteralClassification.model_json_schema(), indent=2)
        mid_idx = len(class_names) // 2
        example_json = json.dumps({"classification": class_names[mid_idx]}, indent=2)

        return f"""Classify the following QUESTION for the ADeLe dimension: **{trait.name}**

**QUESTION TO CLASSIFY:**
{question_text}

**DIMENSION: {trait.name}**
{trait.description or "Classification dimension"}

Classes (in order of increasing complexity): {", ".join(class_names)}

{chr(10).join(class_details)}

**JSON SCHEMA (your response MUST conform to this):**
```json
{json_schema}
```

**REQUIRED OUTPUT FORMAT:**
Return a JSON object with a "classification" key containing the exact class name.

Example format (class value is just an example):
{example_json}

**YOUR JSON RESPONSE:**"""

    def _build_system_prompt(self) -> str:
        """Build system prompt for batch question classification."""
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
