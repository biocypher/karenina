"""LLM-based trait evaluation for rubric assessment.

This module implements the LLMTraitEvaluator class, which handles evaluation
of LLMRubricTrait instances using LLM-as-judge for subjective assessments.

Two evaluation strategies are supported:
- batch: Evaluate all traits in a single LLM call (efficient)
- sequential: Evaluate traits one-by-one (reliable)

Supports three trait kinds:
- boolean: Binary true/false assessment
- score: Numeric rating within a range (e.g., 1-5)
- literal: Categorical classification into predefined classes

All LLM calls use LLMPort.with_structured_output() for consistent backend abstraction.
The adapter factory returns the appropriate implementation (LangChainLLMAdapter
or ClaudeSDKLLMAdapter) based on the model interface configuration.
"""

import json
import logging
import re
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from .....ports import LLMPort, Message, PortCapabilities
from .....schemas.domain import LLMRubricTrait
from ...prompts import PromptAssembler, PromptTask
from .prompts import LiteralTraitPromptBuilder, LLMTraitPromptBuilder

if TYPE_CHECKING:
    from .....schemas.verification import PromptConfig
    from .....schemas.workflow.models import ModelConfig

logger = logging.getLogger(__name__)


class LLMTraitEvaluator:
    """
    Evaluates LLM rubric traits using LLM-as-judge.

    This evaluator handles LLMRubricTrait evaluation by prompting an LLM
    to assess response quality against trait criteria. Supports three kinds:
    - boolean (pass/fail): Binary true/false assessment
    - score (numeric): Rating within a defined range (e.g., 1-5)
    - literal (categorical): Classification into predefined classes

    For literal traits, scores are returned as int indices (0 to len(classes)-1),
    with class labels returned separately for display purposes.

    All LLM calls use LLMPort.with_structured_output() for consistent
    backend abstraction.

    Example usage:
        evaluator = LLMTraitEvaluator(llm, model_config=model_config)
        # For boolean/score traits:
        results, usage = evaluator.evaluate_batch(question, answer, traits)
        # For literal traits:
        scores, labels, usage = evaluator.evaluate_literal_batch(question, answer, literal_traits)
    """

    def __init__(
        self,
        llm: LLMPort,
        async_enabled: bool | None = None,
        async_max_workers: int | None = None,
        *,
        model_config: "ModelConfig",
        prompt_config: "PromptConfig | None" = None,
    ):
        """
        Initialize the LLM trait evaluator.

        Args:
            llm: LLMPort adapter for LLM operations.
            async_enabled: Whether to run sequential trait evaluations in parallel.
                          If None, reads from KARENINA_ASYNC_ENABLED env var (default: True).
            async_max_workers: Max concurrent workers for parallel execution.
                              If None, reads from KARENINA_ASYNC_MAX_WORKERS env var (default: 2).
            model_config: Model configuration for reference.
            prompt_config: Optional per-task-type user instructions for prompt assembly.
        """
        from .....adapters.llm_parallel import read_async_config

        self.llm = llm
        self._model_config = model_config
        self._prompt_config = prompt_config

        # Read async config with env var fallbacks
        default_enabled, default_workers = read_async_config()
        self._async_enabled = async_enabled if async_enabled is not None else default_enabled
        self._async_max_workers = async_max_workers if async_max_workers is not None else default_workers

        # Initialize prompt builders
        self._llm_prompt_builder = LLMTraitPromptBuilder()
        self._literal_prompt_builder = LiteralTraitPromptBuilder()

        logger.debug(f"LLMTraitEvaluator: Initialized for interface={model_config.interface}")

    def _get_user_instructions(self, task: PromptTask) -> str | None:
        """Get user instructions for a given prompt task from prompt_config."""
        if self._prompt_config is None:
            return None
        return self._prompt_config.get_for_task(task.value)

    def _assemble_messages(self, task: PromptTask, system_text: str, user_text: str) -> list[Message]:
        """Assemble messages using PromptAssembler with the given task type."""
        assembler = PromptAssembler(
            task=task,
            interface=self._model_config.interface,
            capabilities=PortCapabilities(),
        )
        return assembler.assemble(
            system_text=system_text,
            user_text=user_text,
            user_instructions=self._get_user_instructions(task),
        )

    def evaluate_batch(
        self, question: str, answer: str, traits: list[LLMRubricTrait]
    ) -> tuple[dict[str, int | bool], dict[str, Any]]:
        """
        Evaluate all traits in a single LLM call using structured output.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            traits: List of LLM traits to evaluate

        Returns:
            Tuple of (results_dict, usage_metadata)
        """
        from .....schemas.workflow.rubric_outputs import BatchRubricScores

        system_prompt = self._llm_prompt_builder.build_batch_system_prompt()
        user_prompt = self._llm_prompt_builder.build_batch_user_prompt(question, answer, traits, BatchRubricScores)

        messages = self._assemble_messages(PromptTask.RUBRIC_LLM_TRAIT_BATCH, system_prompt, user_prompt)

        # Use LLMPort.with_structured_output() for parsing
        # The adapter guarantees response.raw is a validated BatchRubricScores instance
        structured_llm = self.llm.with_structured_output(BatchRubricScores)
        response = structured_llm.invoke(messages)

        # Extract usage metadata
        usage_metadata = asdict(response.usage) if response.usage else {}

        # Validate scores against trait definitions (response.raw is already validated by adapter)
        results = self._validate_batch_scores(response.raw.scores, traits)
        return results, usage_metadata

    def evaluate_sequential(
        self, question: str, answer: str, traits: list[LLMRubricTrait]
    ) -> tuple[dict[str, int | bool], list[dict[str, Any]]]:
        """
        Evaluate traits one by one.

        When async_enabled is True, the LLM calls run in parallel using
        LLMParallelInvoker for significant speedup. Otherwise, calls run
        sequentially.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            traits: List of LLM traits to evaluate

        Returns:
            Tuple of (results_dict, list_of_usage_metadata)
        """
        from .....schemas.workflow.rubric_outputs import SingleBooleanScore, SingleNumericScore

        # Build all tasks upfront
        tasks: list[tuple[list[Message], type]] = []
        for trait in traits:
            model_class = SingleBooleanScore if trait.kind == "boolean" else SingleNumericScore
            system_prompt = self._llm_prompt_builder.build_single_trait_system_prompt(trait)
            user_prompt = self._llm_prompt_builder.build_single_trait_user_prompt(question, answer, trait, model_class)
            messages = self._assemble_messages(PromptTask.RUBRIC_LLM_TRAIT_SINGLE, system_prompt, user_prompt)
            tasks.append((messages, model_class))

        if self._async_enabled:
            return self._execute_concurrent(tasks, traits)
        else:
            return self._execute_serial(tasks, traits)

    def _execute_concurrent(
        self,
        tasks: list[tuple[list[Message], type]],
        traits: list[LLMRubricTrait],
    ) -> tuple[dict[str, int | bool], list[dict[str, Any]]]:
        """Execute evaluation tasks concurrently using LLMParallelInvoker.

        Uses the LLMParallelInvoker for concurrent execution with LLMPort.
        """
        from .....adapters import LLMParallelInvoker

        logger.debug(f"LLMParallelInvoker: Executing {len(tasks)} tasks with max_workers={self._async_max_workers}")
        invoker = LLMParallelInvoker(self.llm, max_workers=self._async_max_workers)
        raw_results = invoker.invoke_batch_structured(tasks)

        results: dict[str, int | bool] = {}
        usage_metadata_list: list[dict[str, Any]] = []

        for i, (parsed_result, usage, error) in enumerate(raw_results):
            trait = traits[i]
            if error:
                logger.warning(f"Failed to evaluate trait '{trait.name}': {error}")
                results[trait.name] = None  # type: ignore[assignment]
                usage_metadata_list.append({})
            else:
                # Extract score from result
                assert parsed_result is not None  # mypy: error implies parsed_result is None
                if trait.kind == "boolean":
                    score: int | bool = parsed_result.result
                else:
                    score = parsed_result.score
                results[trait.name] = self._validate_score(score, trait)
                usage_metadata_list.append(usage or {})

        return results, usage_metadata_list

    def _execute_serial(
        self,
        tasks: list[tuple[list[Message], type]],
        traits: list[LLMRubricTrait],
    ) -> tuple[dict[str, int | bool], list[dict[str, Any]]]:
        """Execute evaluation tasks serially (one at a time).

        Uses LLMPort.with_structured_output() for each task one at a time.
        Used when async_enabled=False.
        """
        results: dict[str, int | bool] = {}
        usage_metadata_list: list[dict[str, Any]] = []

        for i, (messages, model_class) in enumerate(tasks):
            trait = traits[i]
            try:
                # Use LLMPort.with_structured_output() for parsing
                structured_llm = self.llm.with_structured_output(model_class)
                response = structured_llm.invoke(messages)

                # Extract usage metadata
                usage_metadata = asdict(response.usage) if response.usage else {}

                # Extract score from parsed result
                parsed_result = response.raw
                if trait.kind == "boolean":
                    score: int | bool = parsed_result.result
                else:
                    score = parsed_result.score
                results[trait.name] = self._validate_score(score, trait)
                usage_metadata_list.append(usage_metadata)
            except Exception as e:
                logger.warning(f"Failed to evaluate trait '{trait.name}': {e}")
                results[trait.name] = None  # type: ignore[assignment]
                usage_metadata_list.append({})

        return results, usage_metadata_list

    def _validate_batch_scores(
        self, scores: dict[str, int | bool], traits: list[LLMRubricTrait]
    ) -> dict[str, int | bool]:
        """Validate and normalize batch scores against trait definitions."""
        validated_results: dict[str, int | bool] = {}
        trait_map = {trait.name: trait for trait in traits}

        for trait_name, score in scores.items():
            if trait_name in trait_map:
                trait = trait_map[trait_name]
                validated_score = self._validate_score(score, trait)
                validated_results[trait_name] = validated_score

        # Add None for missing traits
        for trait in traits:
            if trait.name not in validated_results:
                validated_results[trait.name] = None  # type: ignore[assignment]

        return validated_results

    def _validate_score(self, score: Any, trait: LLMRubricTrait) -> int | bool:
        """Validate and convert a score for a trait."""
        if trait.kind == "boolean":
            if isinstance(score, bool):
                return score
            elif isinstance(score, int | str):
                return bool(score) and str(score).lower() not in ["false", "0", "no"]
            else:
                return bool(score)
        else:
            if not isinstance(score, int | float):
                try:
                    score = int(score)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid score type for trait {trait.name}: {type(score)}") from e

            min_score = trait.min_score or 1
            max_score = trait.max_score or 5

            # Clamp score to valid range
            clamped_score = max(min_score, min(max_score, int(score)))
            return clamped_score

    def parse_batch_response(self, response: str, traits: list[LLMRubricTrait]) -> dict[str, int | bool]:
        """
        Parse the batch evaluation response.

        This is a fallback method for when structured output fails. Usually
        LLMPort.with_structured_output() handles parsing, but this can be used
        for manual parsing scenarios.

        Args:
            response: Raw LLM response text
            traits: List of traits being evaluated

        Returns:
            Dictionary mapping trait names to scores

        Raises:
            ValueError: If parsing fails
        """
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response}")

        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}") from e

        # Validate and convert the results
        validated_results: dict[str, int | bool] = {}
        trait_map = {trait.name: trait for trait in traits}

        for trait_name, score in result.items():
            if trait_name in trait_map:
                trait = trait_map[trait_name]
                validated_score = self._validate_score(score, trait)
                validated_results[trait_name] = validated_score

        # Add None for missing traits
        for trait in traits:
            if trait.name not in validated_results:
                validated_results[trait.name] = None  # type: ignore[assignment]

        return validated_results

    def parse_single_trait_response(self, response: str, trait: LLMRubricTrait) -> int | bool:
        """
        Parse a single trait evaluation response.

        This is a fallback method for when structured output fails. Usually
        LLMPort.with_structured_output() handles parsing, but this can be used
        for manual parsing scenarios.

        Args:
            response: Raw LLM response text
            trait: The trait being evaluated

        Returns:
            Parsed score (bool for boolean traits, int for numeric)

        Raises:
            ValueError: If parsing fails
        """
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
                    raise ValueError(f"Could not parse boolean from: {response}")
        else:
            # Extract numeric score
            numbers = re.findall(r"\d+", response)
            if not numbers:
                raise ValueError(f"No numeric score found in: {response}")

            score = int(numbers[0])
            return self._validate_score(score, trait)

    # ========== Literal Trait Evaluation Methods ==========

    def evaluate_literal_batch(
        self, question: str, answer: str, traits: list[LLMRubricTrait]
    ) -> tuple[dict[str, int], dict[str, str], dict[str, Any]]:
        """
        Evaluate literal (categorical) traits in a single LLM call.

        Literal traits classify responses into predefined categories. The LLM
        returns class names, which are then converted to integer indices.

        Uses LLMPort.with_structured_output() for parsing.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            traits: List of literal kind LLM traits to evaluate

        Returns:
            Tuple of (scores, labels, usage_metadata) where:
            - scores: Dict mapping trait names to int indices (0 to N-1, or -1 for error)
            - labels: Dict mapping trait names to class labels (or invalid value for error)
            - usage_metadata: Usage metadata from the LLM call
        """
        from .....schemas.workflow.rubric_outputs import BatchLiteralClassifications

        # Filter to only literal traits
        literal_traits = [t for t in traits if t.kind == "literal"]
        if not literal_traits:
            return {}, {}, {}

        system_prompt = self._literal_prompt_builder.build_batch_system_prompt()
        user_prompt = self._literal_prompt_builder.build_batch_user_prompt(
            question, answer, literal_traits, BatchLiteralClassifications
        )

        messages = self._assemble_messages(PromptTask.RUBRIC_LITERAL_TRAIT_BATCH, system_prompt, user_prompt)

        # Use LLMPort.with_structured_output() for parsing
        structured_llm = self.llm.with_structured_output(BatchLiteralClassifications)
        response = structured_llm.invoke(messages)

        # Extract usage metadata
        usage_metadata = asdict(response.usage) if response.usage else {}

        # Validate classifications and convert to scores + labels
        # Use to_dict() to convert list format to dict (required for Anthropic beta.messages.parse compatibility)
        parsed_result = response.raw
        scores, labels = self._validate_literal_classifications(parsed_result.to_dict(), literal_traits)
        return scores, labels, usage_metadata

    def evaluate_literal_sequential(
        self, question: str, answer: str, traits: list[LLMRubricTrait]
    ) -> tuple[dict[str, int], dict[str, str], list[dict[str, Any]]]:
        """
        Evaluate literal traits one by one.

        When async_enabled is True, the LLM calls run in parallel using
        LLMParallelInvoker for significant speedup. Otherwise, calls run
        sequentially.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            traits: List of literal kind LLM traits to evaluate

        Returns:
            Tuple of (scores, labels, usage_metadata_list) where:
            - scores: Dict mapping trait names to int indices (0 to N-1, or -1 for error)
            - labels: Dict mapping trait names to class labels (or invalid value for error)
            - usage_metadata_list: List of usage metadata dicts from LLM calls
        """
        from .....schemas.workflow.rubric_outputs import SingleLiteralClassification

        # Filter to only literal traits
        literal_traits = [t for t in traits if t.kind == "literal"]
        if not literal_traits:
            return {}, {}, []

        # Build all tasks upfront
        tasks: list[tuple[list[Message], type[SingleLiteralClassification]]] = []
        for trait in literal_traits:
            system_prompt = self._literal_prompt_builder.build_single_trait_system_prompt(trait)
            user_prompt = self._literal_prompt_builder.build_single_trait_user_prompt(
                question, answer, trait, SingleLiteralClassification
            )
            messages = self._assemble_messages(PromptTask.RUBRIC_LITERAL_TRAIT_SINGLE, system_prompt, user_prompt)
            tasks.append((messages, SingleLiteralClassification))

        if self._async_enabled:
            return self._execute_concurrent_literal(tasks, literal_traits)
        else:
            return self._execute_serial_literal(tasks, literal_traits)

    def _execute_concurrent_literal(
        self,
        tasks: list[tuple[list[Message], type]],
        traits: list[LLMRubricTrait],
    ) -> tuple[dict[str, int], dict[str, str], list[dict[str, Any]]]:
        """Execute literal evaluation tasks concurrently using LLMParallelInvoker.

        Uses the LLMParallelInvoker for concurrent execution with LLMPort.
        """
        from .....adapters import LLMParallelInvoker

        logger.debug(
            f"LLMParallelInvoker: Executing {len(tasks)} literal tasks with max_workers={self._async_max_workers}"
        )
        invoker = LLMParallelInvoker(self.llm, max_workers=self._async_max_workers)
        raw_results = invoker.invoke_batch_structured(tasks)

        scores: dict[str, int] = {}
        labels: dict[str, str] = {}
        usage_metadata_list: list[dict[str, Any]] = []

        for i, (parsed_result, usage, error) in enumerate(raw_results):
            trait = traits[i]
            if error:
                logger.warning(f"Failed to evaluate literal trait '{trait.name}': {error}")
                scores[trait.name] = -1
                labels[trait.name] = f"[EVALUATION_ERROR: {error!s}]"
                usage_metadata_list.append({})
            else:
                assert parsed_result is not None  # mypy: error implies parsed_result is None
                score, label = self._validate_literal_classification(trait, parsed_result.classification)
                scores[trait.name] = score
                labels[trait.name] = label
                usage_metadata_list.append(usage or {})

        return scores, labels, usage_metadata_list

    def _execute_serial_literal(
        self,
        tasks: list[tuple[list[Message], type]],
        traits: list[LLMRubricTrait],
    ) -> tuple[dict[str, int], dict[str, str], list[dict[str, Any]]]:
        """Execute literal evaluation tasks serially (one at a time).

        Uses LLMPort.with_structured_output() for each task one at a time.
        Used when async_enabled=False.
        """
        scores: dict[str, int] = {}
        labels: dict[str, str] = {}
        usage_metadata_list: list[dict[str, Any]] = []

        for i, (messages, model_class) in enumerate(tasks):
            trait = traits[i]
            try:
                # Use LLMPort.with_structured_output() for parsing
                structured_llm = self.llm.with_structured_output(model_class)
                response = structured_llm.invoke(messages)

                # Extract usage metadata
                usage_metadata = asdict(response.usage) if response.usage else {}
                usage_metadata_list.append(usage_metadata)

                # Validate and convert classification to score + label
                parsed_result = response.raw
                score, label = self._validate_literal_classification(trait, parsed_result.classification)
                scores[trait.name] = score
                labels[trait.name] = label
            except Exception as e:
                logger.warning(f"Failed to evaluate literal trait '{trait.name}': {e}")
                scores[trait.name] = -1
                labels[trait.name] = f"[EVALUATION_ERROR: {e!s}]"
                usage_metadata_list.append({})

        return scores, labels, usage_metadata_list

    def _validate_literal_classification(self, trait: LLMRubricTrait, class_name: str) -> tuple[int, str]:
        """
        Validate and convert a class name to score index and label.

        Args:
            trait: The literal trait being evaluated
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

    def _validate_literal_classifications(
        self, classifications: dict[str, str], traits: list[LLMRubricTrait]
    ) -> tuple[dict[str, int], dict[str, str]]:
        """
        Validate and convert batch classifications to scores and labels.

        Args:
            classifications: Dict mapping trait names to class names from LLM
            traits: List of literal traits being evaluated

        Returns:
            Tuple of (scores, labels) dictionaries
        """
        scores: dict[str, int] = {}
        labels: dict[str, str] = {}
        trait_map = {trait.name: trait for trait in traits}

        for trait_name, class_name in classifications.items():
            if trait_name in trait_map:
                trait = trait_map[trait_name]
                score, label = self._validate_literal_classification(trait, class_name)
                scores[trait_name] = score
                labels[trait_name] = label

        # Add error state for missing traits
        for trait in traits:
            if trait.name not in scores:
                scores[trait.name] = -1
                labels[trait.name] = "[MISSING_FROM_RESPONSE]"

        return scores, labels
