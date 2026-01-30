"""
Rubric evaluation for qualitative assessment of LLM responses.

This module contains the main RubricEvaluator class which handles evaluation of
LLM responses against rubric traits. It supports multiple trait types:

- LLM traits: Subjective assessment using LLM-as-judge
- Regex traits: Pattern matching for format compliance
- Callable traits: Custom Python functions for deterministic checks
- Metric traits: Precision/recall/F1 metrics using confusion matrix

Evaluation is delegated to specialized handlers:
- LLM traits: LLMTraitEvaluator (llm_trait.py)
- Metric traits: MetricTraitEvaluator (metric_trait.py)
- Deep judgment: RubricDeepJudgmentHandler (deep_judgment.py)
"""

import logging
from typing import TYPE_CHECKING, Any

from .....adapters import get_llm
from .....ports import LLMPort
from .....schemas.domain import CallableTrait, MetricRubricTrait, RegexTrait, Rubric
from .....schemas.workflow import ModelConfig
from .deep_judgment import RubricDeepJudgmentHandler
from .llm_trait import LLMTraitEvaluator
from .metric_trait import MetricTraitEvaluator

if TYPE_CHECKING:
    from .....schemas.verification import PromptConfig

logger = logging.getLogger(__name__)


class RubricEvaluator:
    """
    Evaluates LLM responses against a defined rubric using qualitative traits.

    This is the main orchestrator for rubric evaluation, delegating to
    specialized evaluators for different trait types:

    - LLMTraitEvaluator: Handles LLM-based subjective assessment
    - MetricTraitEvaluator: Handles precision/recall metrics
    - RubricDeepJudgmentHandler: Handles deep judgment with excerpts
    """

    def __init__(
        self,
        model_config: ModelConfig,
        evaluation_strategy: str = "batch",
        prompt_config: "PromptConfig | None" = None,
    ):
        """
        Initialize the rubric evaluator with an LLM model.

        Args:
            model_config: Configuration for the evaluation model
            evaluation_strategy: Strategy for evaluating LLM traits ("batch" or "sequential")
                - "batch": Evaluate all traits in single LLM call (efficient)
                - "sequential": Evaluate traits one-by-one (reliable)
            prompt_config: Optional per-task-type user instructions for prompt assembly.

        Raises:
            ValueError: If model configuration is invalid (validated by adapter factory)
            RuntimeError: If LLM initialization fails
        """
        self.model_config = model_config
        self.evaluation_strategy = evaluation_strategy
        self._prompt_config = prompt_config

        # Note: ValueError from validate_model_config propagates directly;
        # only runtime errors (adapter unavailable, etc.) are wrapped.
        try:
            # Use the adapter factory to get an LLMPort implementation
            self.llm: LLMPort = get_llm(model_config)
        except ValueError:
            raise  # Let validation errors propagate
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM for rubric evaluation: {e}") from e

        # Initialize specialized evaluators (lazy - only when needed)
        self._llm_trait_evaluator: LLMTraitEvaluator | None = None
        self._metric_trait_evaluator: MetricTraitEvaluator | None = None

    @property
    def llm_trait_evaluator(self) -> LLMTraitEvaluator:
        """Get or create the LLM trait evaluator."""
        if self._llm_trait_evaluator is None:
            self._llm_trait_evaluator = LLMTraitEvaluator(
                self.llm, model_config=self.model_config, prompt_config=self._prompt_config
            )
        return self._llm_trait_evaluator

    @property
    def metric_trait_evaluator(self) -> MetricTraitEvaluator:
        """Get or create the metric trait evaluator."""
        if self._metric_trait_evaluator is None:
            self._metric_trait_evaluator = MetricTraitEvaluator(
                self.llm, model_config=self.model_config, prompt_config=self._prompt_config
            )
        return self._metric_trait_evaluator

    def evaluate_rubric(
        self, question: str, answer: str, rubric: Rubric
    ) -> tuple[dict[str, int | bool], dict[str, str] | None, list[dict[str, Any]]]:
        """
        Evaluate an answer against a rubric's traits (LLM, regex, and callable).

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            rubric: The rubric containing evaluation traits

        Returns:
            Tuple of (results, llm_trait_labels, usage_metadata_list) where:
            - results: Dictionary mapping trait names to their evaluated scores
            - llm_trait_labels: Dictionary mapping literal trait names to class labels (or None if no literal traits)
            - usage_metadata_list: List of usage metadata dicts from LLM calls

        Raises:
            Exception: If evaluation fails completely
        """
        results: dict[str, int | bool] = {}
        llm_trait_labels: dict[str, str] | None = None
        usage_metadata_list: list[dict[str, Any]] = []

        # Evaluate regex traits first (fast and deterministic)
        if rubric.regex_traits:
            regex_results = self._evaluate_regex_traits(answer, rubric.regex_traits)
            results.update(regex_results)

        # Evaluate callable traits (deterministic but potentially slower)
        if rubric.callable_traits:
            callable_results = self._evaluate_callable_traits(answer, rubric.callable_traits)
            results.update(callable_results)

        # Evaluate LLM traits if present - delegate to LLMTraitEvaluator
        if rubric.llm_traits:
            # Separate literal traits from boolean/score traits
            literal_traits = [t for t in rubric.llm_traits if t.kind == "literal"]
            non_literal_traits = [t for t in rubric.llm_traits if t.kind != "literal"]

            # Evaluate non-literal (boolean/score) traits
            if non_literal_traits:
                if self.evaluation_strategy == "batch":
                    try:
                        llm_results, usage_metadata = self.llm_trait_evaluator.evaluate_batch(
                            question, answer, non_literal_traits
                        )
                        results.update(llm_results)
                        if usage_metadata:
                            usage_metadata_list.append(usage_metadata)
                    except Exception as e:
                        logger.error(f"Batch evaluation failed: {e}")
                        raise RuntimeError(f"Failed to evaluate rubric traits using batch strategy: {e}") from e
                else:  # "sequential"
                    try:
                        llm_results, seq_usage_metadata_list = self.llm_trait_evaluator.evaluate_sequential(
                            question, answer, non_literal_traits
                        )
                        results.update(llm_results)
                        usage_metadata_list.extend(seq_usage_metadata_list)
                    except Exception as e:
                        logger.error(f"Sequential evaluation failed: {e}")
                        raise RuntimeError(f"Failed to evaluate rubric traits using sequential strategy: {e}") from e

            # Evaluate literal (categorical) traits
            if literal_traits:
                if self.evaluation_strategy == "batch":
                    try:
                        literal_scores, literal_labels, usage_metadata = (
                            self.llm_trait_evaluator.evaluate_literal_batch(question, answer, literal_traits)
                        )
                        results.update(literal_scores)
                        llm_trait_labels = literal_labels if literal_labels else None
                        if usage_metadata:
                            usage_metadata_list.append(usage_metadata)
                    except Exception as e:
                        logger.error(f"Literal batch evaluation failed: {e}")
                        raise RuntimeError(f"Failed to evaluate literal traits using batch strategy: {e}") from e
                else:  # "sequential"
                    try:
                        literal_scores, literal_labels, seq_usage_metadata_list = (
                            self.llm_trait_evaluator.evaluate_literal_sequential(question, answer, literal_traits)
                        )
                        results.update(literal_scores)
                        llm_trait_labels = literal_labels if literal_labels else None
                        usage_metadata_list.extend(seq_usage_metadata_list)
                    except Exception as e:
                        logger.error(f"Literal sequential evaluation failed: {e}")
                        raise RuntimeError(f"Failed to evaluate literal traits using sequential strategy: {e}") from e

        return results, llm_trait_labels, usage_metadata_list

    def _evaluate_deterministic_traits(
        self,
        answer: str,
        traits: list[RegexTrait] | list[CallableTrait],
        trait_type_name: str,
    ) -> dict[str, bool | int]:
        """
        Evaluate deterministic traits (regex or callable) using their evaluate() method.

        This is a generic helper that consolidates the common iteration pattern
        used by both regex and callable trait evaluation.

        Args:
            answer: The text to evaluate
            traits: List of traits to evaluate (RegexTrait or CallableTrait)
            trait_type_name: Human-readable name for logging (e.g., "regex", "callable")

        Returns:
            Dictionary mapping trait names to their evaluated results.
            Failed traits are marked as None for consistency with LLM evaluation.
        """
        results: dict[str, bool | int] = {}

        for trait in traits:
            try:
                result = trait.evaluate(answer)
                results[trait.name] = result
            except Exception as e:
                logger.warning(f"Failed to evaluate {trait_type_name} trait '{trait.name}': {e}")
                # Mark failed traits as None for consistency with LLM evaluation
                results[trait.name] = None  # type: ignore[assignment]

        return results

    def _evaluate_regex_traits(self, answer: str, regex_traits: list[RegexTrait]) -> dict[str, bool]:
        """
        Evaluate regex traits using pattern matching.

        Args:
            answer: The text to evaluate
            regex_traits: List of regex traits to evaluate

        Returns:
            Dictionary mapping trait names to boolean results
        """
        # Type narrowing: regex traits always return bool
        return self._evaluate_deterministic_traits(answer, regex_traits, "regex")  # type: ignore[return-value]

    def _evaluate_callable_traits(self, answer: str, callable_traits: list[CallableTrait]) -> dict[str, bool | int]:
        """
        Evaluate callable traits using custom functions.

        Args:
            answer: The text to evaluate
            callable_traits: List of callable traits to evaluate

        Returns:
            Dictionary mapping trait names to boolean or int results (depending on trait kind)
        """
        return self._evaluate_deterministic_traits(answer, callable_traits, "callable")

    # ========== Metric Trait Evaluation Methods ==========

    def evaluate_metric_traits(
        self, question: str, answer: str, metric_traits: list[MetricRubricTrait]
    ) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, float]], list[dict[str, Any]]]:
        """
        Evaluate metric traits and return confusion lists and computed metrics.

        This method delegates to MetricTraitEvaluator for the actual evaluation.

        Args:
            question: The original question asked
            answer: The LLM's response to evaluate
            metric_traits: List of metric traits to evaluate

        Returns:
            Tuple of (confusion_lists, metrics, usage_metadata_list) where:
            - confusion_lists: {trait_name: {tp: [...], tn: [...], fp: [...], fn: [...]}}
            - metrics: {trait_name: {precision: 0.85, recall: 0.92, ...}}
            - usage_metadata_list: List of usage metadata dicts from LLM calls

        Raises:
            Exception: If evaluation fails for all traits
        """
        return self.metric_trait_evaluator.evaluate_metric_traits(question, answer, metric_traits)

    # ========== Deep Judgment Rubric Methods ==========

    def evaluate_rubric_with_deep_judgment(
        self,
        question: str,
        answer: str,
        rubric: Rubric,
        config: Any,  # VerificationConfig
    ) -> dict[str, Any]:
        """
        Evaluate rubric with deep judgment for enabled traits.

        This method delegates to RubricDeepJudgmentHandler for the actual
        multi-stage evaluation process.

        Args:
            question: The original question
            answer: The LLM response to evaluate
            rubric: The rubric containing evaluation traits
            config: VerificationConfig with deep judgment settings

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
        # Create handler with the same LLM instance
        handler = RubricDeepJudgmentHandler(self.llm, self.model_config, prompt_config=self._prompt_config)

        # Delegate to handler, providing a callback for standard trait evaluation
        return handler.evaluate_rubric_with_deep_judgment(
            question=question,
            answer=answer,
            rubric=rubric,
            config=config,
            standard_evaluator_fn=self.evaluate_rubric,
        )
