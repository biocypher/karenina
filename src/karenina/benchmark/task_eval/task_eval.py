"""TaskEval implementation for rubric-driven failure characterization.

TaskEval provides a lightweight utility for evaluating task performance using
logged outputs against questions and rubrics. It focuses on quality assessment
rather than generating new responses.

Example:
    task = TaskEval(task_id="math_eval")
    task.add_question({"id": "q1", "question": "What is 2+2?", "raw_answer": "4"})
    task.log("2+2=4")
    result = task.evaluate(config)
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Union

if TYPE_CHECKING:
    from karenina.ports.messages import Message
    from karenina.schemas.entities import Question, Rubric
    from karenina.schemas.entities.rubric import DynamicRubric

from karenina.exceptions import KareninaError
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig

from .helpers import merge_logs_and_traces
from .models import LogEvent, StepEval, TaskEvalResult

logger = logging.getLogger(__name__)


class TaskEval:
    """Lightweight utility for task evaluation using rubric-driven failure characterization.

    TaskEval evaluates logged outputs against questions and rubrics to characterize
    task performance and failure modes. It supports both global and step-specific
    evaluation contexts.

    Core Philosophy:
        - Logs ARE the outputs to evaluate (no answer generation)
        - Questions define what to evaluate against
        - Rubrics define quality dimensions
        - Results reveal WHY responses succeed or fail
    """

    def __init__(
        self,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        callable_registry: dict[str, Callable[[str], bool]] | None = None,
        merge_strategy: Literal["concatenate", "traces_only"] = "concatenate",
    ) -> None:
        """Initialize TaskEval instance.

        Args:
            task_id: Optional task identifier for tracking
            metadata: Optional metadata dictionary
            callable_registry: Registry of callable functions for manual trait evaluation
            merge_strategy: Default strategy for merging logs before evaluation.
                "concatenate" combines text and trace logs; "traces_only" uses
                only logs with trace_messages.
        """
        self.task_id = task_id
        self.metadata = metadata or {}
        self.callable_registry = callable_registry or {}
        self.merge_strategy: Literal["concatenate", "traces_only"] = merge_strategy

        # Storage for logs, questions, rubrics, and dynamic rubrics
        self.global_logs: list[LogEvent] = []
        self.step_logs: dict[str, list[LogEvent]] = {}
        self.global_questions: list[dict[str, Any] | Question] = []
        self.step_questions: dict[str, list[dict[str, Any] | Question]] = {}
        self.global_rubrics: list[Rubric] = []
        self.step_rubrics: dict[str, list[Rubric]] = {}
        self.global_dynamic_rubrics: list[DynamicRubric] = []
        self.step_dynamic_rubrics: dict[str, list[DynamicRubric]] = {}

    # =============================================================================
    # CORE API METHODS
    # =============================================================================

    def log(
        self,
        text: str,
        step_id: str | None = None,
        target: Literal["global", "step", "both"] = "both",
        level: Literal["debug", "info", "warn", "error"] = "info",
        tags: list[str] | None = None,
    ) -> None:
        """Log a text event or response.

        This is the primary method for recording text outputs that will be evaluated.
        All logged text is considered as potential answers for evaluation.

        Args:
            text: Log message text
            step_id: Optional step ID for step-specific logs
            target: Where to log ("global", "step", or "both")
            level: Log level (debug, info, warn, error)
            tags: Optional tags for categorization

        Example:
            task.log("The answer is 42")
            task.log("Step 1 complete", step_id="calculation", level="info")
        """
        log_event = LogEvent(
            level=level,
            text=text,
            tags=tags,
        )

        if target in ("global", "both"):
            self.global_logs.append(log_event)

        if target in ("step", "both") and step_id:
            if step_id not in self.step_logs:
                self.step_logs[step_id] = []
            self.step_logs[step_id].append(log_event)

    def log_trace(
        self,
        messages: "list[Message] | str",
        step_id: str | None = None,
        target: Literal["global", "step", "both"] = "both",
        level: Literal["debug", "info", "warn", "error"] = "info",
        tags: list[str] | None = None,
    ) -> None:
        """Log a structured conversation trace.

        Records Message objects representing a full conversation trace
        (assistant responses, tool calls, tool results). When a string is
        provided, it is automatically wrapped as a single assistant Message.

        Args:
            messages: List of Message objects, or a string (auto-wrapped as
                assistant Message).
            step_id: Optional step ID for step-specific logs.
            target: Where to log ("global", "step", or "both").
            level: Log level (debug, info, warn, error).
            tags: Optional tags for categorization.

        Example:
            from karenina.ports.messages import Message
            task.log_trace([Message.assistant("The answer is 42")])
            task.log_trace("Simple text response")
        """
        from karenina.ports.messages import Message

        if isinstance(messages, str):
            messages = messages.removeprefix("--- AI Message ---\n")
            trace_messages = [Message.assistant(messages)]
        else:
            trace_messages = messages

        log_event = LogEvent(
            level=level,
            tags=tags,
            trace_messages=trace_messages,
        )

        if target in ("global", "both"):
            self.global_logs.append(log_event)

        if target in ("step", "both") and step_id:
            if step_id not in self.step_logs:
                self.step_logs[step_id] = []
            self.step_logs[step_id].append(log_event)

    def add_question(self, question_obj: Union[dict[str, Any], "Question"], step_id: str | None = None) -> None:
        """Add a question for evaluation.

        Questions define what to evaluate logged outputs against. They can be
        dictionaries (from Benchmark) or Question objects.

        Args:
            question_obj: Question dict or Question object
            step_id: Optional step ID for step-specific questions

        Example:
            task.add_question({
                "id": "math_q1",
                "question": "What is 2+2?",
                "raw_answer": "4"
            })
        """
        if step_id:
            if step_id not in self.step_questions:
                self.step_questions[step_id] = []
            self.step_questions[step_id].append(question_obj)
        else:
            self.global_questions.append(question_obj)

    def add_template(
        self,
        template_class: type,
        step_id: str | None = None,
    ) -> None:
        """Add an answer template class for correctness evaluation.

        Accepts a BaseAnswer subclass directly. The class is converted to
        source code and stored as a synthetic question for the pipeline.
        No question text is required.

        Args:
            template_class: A BaseAnswer subclass with fields and verify().
            step_id: Optional step ID for step-specific templates.

        Example:
            class Answer(BaseAnswer):
                target: str = Field(description="The drug target")
                def ground_truth(self):
                    self.correct = {"target": "BCL2"}
                def verify(self) -> bool:
                    return self.target.upper() == self.correct["target"]

            task.add_template(Answer)
        """
        import inspect
        import textwrap

        try:
            source = textwrap.dedent(inspect.getsource(template_class))
        except OSError:
            raise TypeError(
                f"Cannot retrieve source code for {template_class.__name__}. "
                f"add_template() requires a class defined in a .py file or Jupyter notebook. "
                f"For dynamically defined classes, pass template source code as a string "
                f"via add_question() instead."
            ) from None

        # Prepend standard imports so the template is self-contained
        template_code = (
            "from karenina.schemas.entities import BaseAnswer\n"
            "from pydantic import Field\n"
            "from typing import Any\n\n" + source
        )

        question_dict: dict[str, Any] = {
            "id": template_class.__name__.lower(),
            "question": "",
            "raw_answer": "",
            "answer_template": template_code,
        }
        self.add_question(question_dict, step_id=step_id)

    def add_rubric(self, rubric_obj: "Rubric", step_id: str | None = None) -> None:
        """Add a rubric for quality evaluation.

        Rubrics define quality dimensions (traits) to evaluate responses against.

        Args:
            rubric_obj: Rubric object with evaluation traits
            step_id: Optional step ID for step-specific rubrics

        Example:
            rubric = Rubric(llm_traits=[
                LLMRubricTrait(name="accuracy", description="Is answer correct?", kind="boolean")
            ])
            task.add_rubric(rubric)
        """
        if step_id:
            if step_id not in self.step_rubrics:
                self.step_rubrics[step_id] = []
            self.step_rubrics[step_id].append(rubric_obj)
        else:
            self.global_rubrics.append(rubric_obj)

    def add_dynamic_rubric(self, dynamic_rubric: "DynamicRubric", step_id: str | None = None) -> None:
        """Add a dynamic rubric for conditional quality evaluation.

        Dynamic rubrics gate each trait on concept presence in the response.
        Traits whose concept is absent are skipped; present traits are promoted
        into the standard rubric and evaluated normally.

        Args:
            dynamic_rubric: DynamicRubric object with conditional traits
            step_id: Optional step ID for step-specific dynamic rubrics
        """
        if step_id:
            if step_id not in self.step_dynamic_rubrics:
                self.step_dynamic_rubrics[step_id] = []
            self.step_dynamic_rubrics[step_id].append(dynamic_rubric)
        else:
            self.global_dynamic_rubrics.append(dynamic_rubric)

    def register_callable(self, name: str, func: Callable[[str], bool]) -> None:
        """Register a callable function for manual trait evaluation.

        Args:
            name: Name to register the function under
            func: Function that takes a string and returns a boolean

        Raises:
            ValueError: If function doesn't have correct signature
        """
        import inspect

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if len(params) != 1:
            raise ValueError(f"Callable '{name}' must have exactly one parameter, got {len(params)}")

        self.callable_registry[name] = func

    def evaluate(
        self,
        config: VerificationConfig,
        step_id: str | None = None,
        merge_strategy: Literal["concatenate", "traces_only"] | None = None,
        answering_model: ModelConfig | None = None,
        run_name: str | None = None,
    ) -> TaskEvalResult:
        """Evaluate logged outputs against questions and rubrics.

        Args:
            config: Verification configuration (parsing models only)
            step_id: Optional step ID to evaluate specific step (otherwise global)
            merge_strategy: Optional override for the instance merge_strategy.
                If None, uses the instance default.
            answering_model: Optional model identity to record for the answering
                stage. When None, a sentinel with interface="taskeval" is used.
            run_name: Optional run name for result tracking. When None, an
                auto-generated name with prefix "taskeval_" is used.

        Returns:
            TaskEvalResult with evaluation outcomes and failure characterization

        Example:
            config = VerificationConfig(
                parsing_models=[ModelConfig(...)],
                parsing_only=True
            )
            result = task.evaluate(config)
        """
        from uuid import uuid4

        if answering_model is None:
            answering_model = ModelConfig(
                id="taskeval_user_provided",
                model_name="user-provided",
                model_provider="user-provided",
                interface="taskeval",
            )
        if run_name is None:
            run_name = f"taskeval_{uuid4().hex[:8]}"
        if config.is_few_shot_enabled():
            logger.debug("FewShotConfig has no effect in TaskEval mode")

        effective_strategy = merge_strategy or self.merge_strategy

        if step_id:
            return self._evaluate_step(
                config,
                step_id,
                effective_strategy,
                answering_model=answering_model,
                run_name=run_name,
            )
        else:
            return self._evaluate_global(
                config,
                effective_strategy,
                answering_model=answering_model,
                run_name=run_name,
            )

    def _evaluate_global(
        self,
        config: VerificationConfig,
        merge_strategy: Literal["concatenate", "traces_only"],
        answering_model: ModelConfig,
        run_name: str,
    ) -> TaskEvalResult:
        """Evaluate all global logs against global questions and rubrics.

        After completing global evaluation, automatically evaluates all
        available steps as well.

        Args:
            config: Verification configuration with parsing models
            merge_strategy: Strategy for merging logs
            answering_model: Model identity for the answering stage
            run_name: Run name for result tracking

        Returns:
            TaskEvalResult with both global evaluation results and all step evaluations
        """
        step_eval = self._run_evaluation_loop(
            config,
            step_id=None,
            merge_strategy=merge_strategy,
            answering_model=answering_model,
            run_name=run_name,
        )

        task_result = TaskEvalResult(
            task_id=self.task_id,
            metadata=self.metadata,
            global_eval=step_eval,
        )

        # After global evaluation, automatically evaluate all available steps
        for sid in self._get_available_step_ids():
            step_result = self._evaluate_step_internal(
                config,
                sid,
                merge_strategy,
                answering_model=answering_model,
                run_name=run_name,
            )
            task_result.per_step[sid] = step_result

        return task_result

    def _evaluate_step(
        self,
        config: VerificationConfig,
        step_id: str,
        merge_strategy: Literal["concatenate", "traces_only"],
        answering_model: ModelConfig,
        run_name: str,
    ) -> TaskEvalResult:
        """Evaluate step-specific logs against step-specific questions and rubrics.

        Args:
            config: Verification configuration with parsing models
            step_id: ID of the step to evaluate
            merge_strategy: Strategy for merging logs
            answering_model: Model identity for the answering stage
            run_name: Run name for result tracking

        Returns:
            TaskEvalResult with step-specific evaluation results
        """
        step_eval = self._evaluate_step_internal(
            config,
            step_id,
            merge_strategy,
            answering_model=answering_model,
            run_name=run_name,
        )
        return self._build_result(step_eval, step_id=step_id)

    def _evaluate_step_internal(
        self,
        config: VerificationConfig,
        step_id: str,
        merge_strategy: Literal["concatenate", "traces_only"],
        answering_model: ModelConfig,
        run_name: str,
    ) -> StepEval:
        """Internal method to evaluate a single step and return StepEval.

        Args:
            config: Verification configuration with parsing models
            step_id: ID of the step to evaluate
            merge_strategy: Strategy for merging logs
            answering_model: Model identity for the answering stage
            run_name: Run name for result tracking

        Returns:
            StepEval with step-specific evaluation results
        """
        return self._run_evaluation_loop(
            config,
            step_id=step_id,
            merge_strategy=merge_strategy,
            answering_model=answering_model,
            run_name=run_name,
        )

    # =============================================================================
    # DATA PREPARATION METHODS
    # =============================================================================

    def _get_available_step_ids(self) -> set[str]:
        """Get all step IDs that have logs, questions, or rubrics defined.

        Returns:
            Set of step IDs that have any associated data
        """
        step_ids: set[str] = set()
        step_ids.update(self.step_logs.keys())
        step_ids.update(self.step_questions.keys())
        step_ids.update(self.step_rubrics.keys())
        step_ids.update(self.step_dynamic_rubrics.keys())
        return step_ids

    def _get_evaluation_context(self, step_id: str | None = None) -> "EvaluationContext":
        """Get evaluation context for the specified step or global."""
        if step_id:
            questions = self.step_questions.get(step_id, [])
            rubrics = self.step_rubrics.get(step_id, [])
            dynamic_rubrics = self.step_dynamic_rubrics.get(step_id, [])
            logs = self.step_logs.get(step_id, [])
        else:
            questions = self.global_questions
            rubrics = self.global_rubrics
            dynamic_rubrics = self.global_dynamic_rubrics
            logs = self.global_logs

        return EvaluationContext(
            questions=questions,
            logs=logs,
            merged_rubric=self._merge_rubrics(rubrics),
            merged_dynamic_rubric=self._merge_dynamic_rubrics(dynamic_rubrics),
        )

    def _normalize_question(self, question: Union[dict[str, Any], "Question"]) -> dict[str, Any]:
        """Normalize a question to dict format for evaluation.

        Handles both Question objects and dictionary objects from Benchmark.
        """
        from karenina.schemas.entities import Question

        if isinstance(question, Question):
            return {
                "id": question.id,
                "question": question.question,
                "raw_answer": question.raw_answer,
                "keywords": question.keywords,
                "few_shot_examples": question.few_shot_examples,
                "answer_template": getattr(question, "answer_template", None),
                "question_rubric": getattr(question, "question_rubric", None),
                "question_dynamic_rubric": getattr(question, "question_dynamic_rubric", None),
            }
        return question

    def _collect_logs_for_evaluation(
        self,
        logs: list[LogEvent],
        merge_strategy: Literal["concatenate", "traces_only"],
    ) -> "tuple[str, list[Message] | None]":
        """Collect and merge all logged data for evaluation.

        Delegates to merge_logs_and_traces() for the actual merge logic.

        Args:
            logs: List of LogEvent objects.
            merge_strategy: Strategy for merging logs.

        Returns:
            Tuple of (response_text_string, optional_message_list).
        """
        return merge_logs_and_traces(logs, strategy=merge_strategy)

    def _detect_evaluation_mode(self, context: "EvaluationContext") -> str:
        """Auto-detect evaluation mode based on attached questions and rubrics.

        Args:
            context: The evaluation context containing questions and rubrics

        Returns:
            One of: "template_only", "rubric_only", "template_and_rubric"

        Raises:
            ValueError: If neither templates nor rubrics are provided
        """
        has_templates = any(self._normalize_question(q).get("answer_template") for q in context.questions)

        has_rubrics = bool(
            context.merged_rubric
            and (
                context.merged_rubric.llm_traits
                or context.merged_rubric.regex_traits
                or context.merged_rubric.callable_traits
                or context.merged_rubric.metric_traits
                or context.merged_rubric.agentic_traits
            )
        )

        has_dynamic_rubrics = bool(context.merged_dynamic_rubric and not context.merged_dynamic_rubric.is_empty())

        any_rubric = has_rubrics or has_dynamic_rubrics

        if has_templates and any_rubric:
            return "template_and_rubric"
        elif has_templates:
            return "template_only"
        elif any_rubric:
            return "rubric_only"
        else:
            raise ValueError(
                "Must provide either answer templates, rubrics, or both for evaluation. "
                "Add questions with templates via add_question() or add rubrics via add_rubric()."
            )

    # =============================================================================
    # EVALUATION LOOP
    # =============================================================================

    def _run_evaluation_loop(
        self,
        config: VerificationConfig,
        step_id: str | None,
        merge_strategy: Literal["concatenate", "traces_only"],
        answering_model: ModelConfig,
        run_name: str,
    ) -> StepEval:
        """Run the evaluation loop for either global or step-specific context.

        Args:
            config: Verification configuration with parsing models
            step_id: Optional step ID (None for global evaluation)
            merge_strategy: Strategy for merging logs
            answering_model: Model identity for the answering stage
            run_name: Run name for result tracking

        Returns:
            StepEval with evaluation results
        """
        context = self._get_evaluation_context(step_id=step_id)
        evaluation_mode = self._detect_evaluation_mode(context)
        step_eval = StepEval()

        # Collect and merge logs
        concatenated_logs, trace_messages = self._collect_logs_for_evaluation(context.logs, merge_strategy)

        # Auto-extract agent metrics when Message traces are present
        agent_metrics: dict[str, Any] | None = None
        if trace_messages:
            from karenina.benchmark.verification.utils.trace_agent_metrics import (
                extract_agent_metrics_from_messages,
            )

            agent_metrics = extract_agent_metrics_from_messages(trace_messages)

        # Determine how many replicates to run
        replicate_count = getattr(config, "replicate_count", 1)
        if replicate_count < 1:
            replicate_count = 1

        # Extract guard flags from config
        abstention_enabled = config.abstention_enabled
        sufficiency_enabled = config.sufficiency_enabled

        # Build step prefix for synthetic question IDs and error messages
        step_prefix = f"step_{step_id}_" if step_id else ""
        step_suffix = f" in step {step_id}" if step_id else ""

        for rep_idx in range(replicate_count):
            replicate = None if replicate_count == 1 else rep_idx + 1

            # In rubric_only mode with no explicit questions, create a synthetic question
            if evaluation_mode == "rubric_only" and not context.questions and concatenated_logs:
                synthetic_question = {
                    "id": f"{step_prefix}rubric_only_eval",
                    "question": f"Evaluate the logged output{step_suffix}",
                    "raw_answer": "",
                    "answer_template": None,
                }

                self._evaluate_and_store(
                    step_eval=step_eval,
                    question_dict=synthetic_question,
                    response_text=concatenated_logs,
                    parsing_model=config.parsing_models[0],
                    rubric=context.merged_rubric,
                    dynamic_rubric=context.merged_dynamic_rubric,
                    evaluation_mode=evaluation_mode,
                    error_context=f"rubric-only logs{step_suffix}",
                    trace_messages=trace_messages,
                    agent_metrics=agent_metrics,
                    answering_model=answering_model,
                    run_name=run_name,
                    replicate=replicate,
                    abstention_enabled=abstention_enabled,
                    sufficiency_enabled=sufficiency_enabled,
                )

            for question in context.questions:
                question_dict = self._normalize_question(question)
                question_id = question_dict.get("id", "unknown")

                answer_template = question_dict.get("answer_template")

                # In rubric_only mode, templates are optional
                if evaluation_mode != "rubric_only" and not answer_template:
                    continue

                # Merge per-question dynamic rubric with context-level dynamic rubric
                effective_dynamic_rubric = self._resolve_question_dynamic_rubric(
                    question_dict, context.merged_dynamic_rubric
                )

                self._evaluate_and_store(
                    step_eval=step_eval,
                    question_dict=question_dict,
                    response_text=concatenated_logs,
                    parsing_model=config.parsing_models[0],
                    rubric=context.merged_rubric,
                    dynamic_rubric=effective_dynamic_rubric,
                    evaluation_mode=evaluation_mode,
                    error_context=f"question {question_id}{step_suffix}",
                    trace_messages=trace_messages,
                    agent_metrics=agent_metrics,
                    answering_model=answering_model,
                    run_name=run_name,
                    replicate=replicate,
                    abstention_enabled=abstention_enabled,
                    sufficiency_enabled=sufficiency_enabled,
                )

        return step_eval

    def _evaluate_and_store(
        self,
        step_eval: StepEval,
        question_dict: dict[str, Any],
        response_text: str,
        parsing_model: ModelConfig,
        rubric: "Rubric | None",
        dynamic_rubric: "DynamicRubric | None",
        evaluation_mode: str,
        error_context: str,
        trace_messages: "list[Message] | None" = None,
        agent_metrics: dict[str, Any] | None = None,
        answering_model: ModelConfig | None = None,
        run_name: str | None = None,
        replicate: int | None = None,
        abstention_enabled: bool = False,
        sufficiency_enabled: bool = False,
    ) -> None:
        """Evaluate a single question and store the result.

        Args:
            step_eval: StepEval object to store results in
            question_dict: Question dictionary with metadata
            response_text: The logged text to evaluate
            parsing_model: Model to use for parsing/evaluation
            rubric: Rubric with evaluation traits (optional)
            dynamic_rubric: DynamicRubric with conditional traits (optional)
            evaluation_mode: One of "template_only", "rubric_only", "template_and_rubric"
            error_context: Context string for error messages
            trace_messages: Optional list of Message objects for the trace
            agent_metrics: Optional agent execution metrics
            answering_model: Model identity for the answering stage
            run_name: Run name for result tracking
            replicate: Replicate index (1-based), or None for single-replicate runs
            abstention_enabled: Whether abstention detection is enabled
            sufficiency_enabled: Whether sufficiency detection is enabled
        """
        question_id = question_dict.get("id", "unknown")
        assert isinstance(question_id, str), "Question ID must be a string"

        try:
            verification_result = self._evaluate(
                question_dict=question_dict,
                response_text=response_text,
                parsing_model=parsing_model,
                rubric=rubric,
                dynamic_rubric=dynamic_rubric,
                evaluation_mode=evaluation_mode,
                trace_messages=trace_messages,
                agent_metrics=agent_metrics,
                answering_model=answering_model,
                run_name=run_name,
                replicate=replicate,
                abstention_enabled=abstention_enabled,
                sufficiency_enabled=sufficiency_enabled,
            )

            if question_id not in step_eval.verification_results:
                step_eval.verification_results[question_id] = []
            step_eval.verification_results[question_id].append(verification_result)

        except (KareninaError, ValueError, RuntimeError) as e:
            logger.warning("Evaluation failed for %s: %s", error_context, e)
            if question_id not in step_eval.failed_questions:
                step_eval.failed_questions[question_id] = []
            step_eval.failed_questions[question_id].append(str(e))

    # =============================================================================
    # EVALUATION METHODS
    # =============================================================================

    def _evaluate(
        self,
        question_dict: dict[str, Any],
        response_text: str,
        parsing_model: ModelConfig,
        rubric: "Rubric | None",
        dynamic_rubric: "DynamicRubric | None" = None,
        evaluation_mode: str = "template_only",
        trace_messages: "list[Message] | None" = None,
        agent_metrics: dict[str, Any] | None = None,
        answering_model: ModelConfig | None = None,
        run_name: str | None = None,
        replicate: int | None = None,
        abstention_enabled: bool = False,
        sufficiency_enabled: bool = False,
    ) -> Any:
        """Evaluate response using main verification pipeline with cached answer data.

        Args:
            question_dict: Question dictionary with metadata
            response_text: The logged text to evaluate
            parsing_model: Model to use for parsing/evaluation
            rubric: Rubric with evaluation traits (optional)
            dynamic_rubric: DynamicRubric with conditional traits (optional)
            evaluation_mode: One of "template_only", "rubric_only", "template_and_rubric"
            trace_messages: Optional list of Message objects for the trace
            agent_metrics: Optional agent execution metrics
            answering_model: Model identity for the answering stage
            run_name: Run name for result tracking
            replicate: Replicate index (1-based), or None for single-replicate runs
            abstention_enabled: Whether abstention detection is enabled
            sufficiency_enabled: Whether sufficiency detection is enabled

        Returns:
            VerificationResult from the main verification pipeline
        """
        from ..verification.runner import run_single_model_verification

        question_id = question_dict.get("id", "unknown")
        question_text = question_dict.get("question", "")
        answer_template = question_dict.get("answer_template")

        # In rubric_only mode, template is optional
        if evaluation_mode == "rubric_only" and not answer_template:
            answer_template = '''
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Minimal answer template for rubric-only evaluation."""

    raw_response: str = Field(description="The raw response text")

    def verify(self) -> bool:
        # In rubric_only mode, verification always passes (rubrics determine quality)
        return True
'''

        assert isinstance(answer_template, str), "answer_template must be a string"
        assert answering_model is not None, "answering_model must be provided"

        # Prepare cached answer data to inject logged output
        cached_answer_data: dict[str, Any] = {
            "raw_llm_response": response_text,
            "recursion_limit_reached": False,
            "answering_mcp_servers": None,
            "usage_metadata": None,
            "agent_metrics": agent_metrics,
        }

        # Include trace_messages in cached data (serialized as dicts)
        if trace_messages:
            cached_answer_data["trace_messages"] = [m.to_dict() for m in trace_messages]

        verification_result = run_single_model_verification(
            question_id=question_id,
            question_text=question_text,
            template_code=answer_template,
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=rubric,
            dynamic_rubric=dynamic_rubric,
            cached_answer_data=cached_answer_data,
            run_name=run_name,
            replicate=replicate,
            abstention_enabled=abstention_enabled,
            sufficiency_enabled=sufficiency_enabled,
            rubric_evaluation_strategy="batch",
            evaluation_mode=evaluation_mode,
            task_eval_mode=True,
        )

        return verification_result

    # =============================================================================
    # HELPER METHODS
    # =============================================================================

    def _merge_rubrics(self, rubrics: list["Rubric"]) -> "Rubric | None":
        """Merge multiple rubrics into a single rubric, raising error on trait name conflicts."""
        if not rubrics:
            return None

        from karenina.schemas.entities import (
            CallableRubricTrait,
            LLMRubricTrait,
            MetricRubricTrait,
            RegexRubricTrait,
            Rubric,
        )
        from karenina.schemas.entities.rubric import AgenticRubricTrait

        # Check for trait name conflicts first (across all trait types)
        all_trait_names = []
        for rubric in rubrics:
            for trait in rubric.llm_traits:
                all_trait_names.append(trait.name)
            for regex_trait in rubric.regex_traits:
                all_trait_names.append(regex_trait.name)
            for callable_trait in rubric.callable_traits:
                all_trait_names.append(callable_trait.name)
            for metric_trait in rubric.metric_traits:
                all_trait_names.append(metric_trait.name)
            for agentic_trait in rubric.agentic_traits:
                all_trait_names.append(agentic_trait.name)

        # Find duplicates
        seen = set()
        duplicates = set()
        for name in all_trait_names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)

        if duplicates:
            raise ValueError(
                f"Duplicate rubric trait names found across rubrics: {duplicates}. "
                f"Each trait name must be unique across all rubrics in the same context."
            )

        # Combine all traits (now guaranteed to be unique)
        unique_llm_traits: dict[str, LLMRubricTrait] = {}
        unique_regex_traits: dict[str, RegexRubricTrait] = {}
        unique_callable_traits: dict[str, CallableRubricTrait] = {}
        unique_metric_traits: dict[str, MetricRubricTrait] = {}
        unique_agentic_traits: dict[str, AgenticRubricTrait] = {}
        for rubric in rubrics:
            for trait in rubric.llm_traits:
                unique_llm_traits[trait.name] = trait
            for regex_trait in rubric.regex_traits:
                unique_regex_traits[regex_trait.name] = regex_trait
            for callable_trait in rubric.callable_traits:
                unique_callable_traits[callable_trait.name] = callable_trait
            for metric_trait in rubric.metric_traits:
                unique_metric_traits[metric_trait.name] = metric_trait
            for agentic_trait in rubric.agentic_traits:
                unique_agentic_traits[agentic_trait.name] = agentic_trait

        return Rubric(
            llm_traits=list(unique_llm_traits.values()),
            regex_traits=list(unique_regex_traits.values()),
            callable_traits=list(unique_callable_traits.values()),
            metric_traits=list(unique_metric_traits.values()),
            agentic_traits=list(unique_agentic_traits.values()),
        )

    def _merge_dynamic_rubrics(self, dynamic_rubrics: "list[DynamicRubric]") -> "DynamicRubric | None":
        """Merge multiple dynamic rubrics into a single DynamicRubric.

        Delegates to the schema-layer merge function which concatenates traits
        and rejects name collisions.

        Args:
            dynamic_rubrics: List of DynamicRubric objects to merge.

        Returns:
            Merged DynamicRubric, or None if the list is empty.
        """
        if not dynamic_rubrics:
            return None

        from karenina.schemas.entities.rubric import merge_dynamic_rubrics

        result = dynamic_rubrics[0]
        for dr in dynamic_rubrics[1:]:
            result = merge_dynamic_rubrics(result, dr)  # type: ignore[assignment]
        return result

    def _resolve_question_dynamic_rubric(
        self,
        question_dict: dict[str, Any],
        context_dynamic_rubric: "DynamicRubric | None",
    ) -> "DynamicRubric | None":
        """Merge per-question dynamic rubric with context-level dynamic rubric.

        Mirrors the Benchmark path's merge_dynamic_rubrics_for_task behavior:
        deserializes the question-level dict, then merges with the context-level
        dynamic rubric.

        Args:
            question_dict: Normalized question dictionary.
            context_dynamic_rubric: The merged dynamic rubric from the evaluation context.

        Returns:
            Merged DynamicRubric, or the context-level one if no per-question rubric exists.
        """
        question_dr_dict = question_dict.get("question_dynamic_rubric")
        if not question_dr_dict:
            return context_dynamic_rubric

        from karenina.schemas.entities.rubric import DynamicRubric, merge_dynamic_rubrics

        try:
            question_dr = DynamicRubric.model_validate(question_dr_dict)
        except Exception as e:
            question_id = question_dict.get("id", "unknown")
            logger.warning(
                "Failed to parse question dynamic rubric for %s: %s",
                question_id,
                e,
            )
            return context_dynamic_rubric

        try:
            return merge_dynamic_rubrics(context_dynamic_rubric, question_dr)
        except ValueError as e:
            question_id = question_dict.get("id", "unknown")
            logger.error("Error merging dynamic rubrics for %s: %s", question_id, e)
            return context_dynamic_rubric

    def _build_result(self, step_eval: StepEval, step_id: str | None) -> TaskEvalResult:
        """Build the final TaskEvalResult."""
        task_result = TaskEvalResult(
            task_id=self.task_id,
            metadata=self.metadata,
        )

        if step_id:
            task_result.per_step[step_id] = step_eval
        else:
            task_result.global_eval = step_eval

        return task_result


class EvaluationContext:
    """Container for evaluation context data."""

    def __init__(
        self,
        questions: list[Any],
        logs: list[LogEvent],
        merged_rubric: "Rubric | None",
        merged_dynamic_rubric: "DynamicRubric | None" = None,
    ):
        self.questions = questions
        self.logs = logs
        self.merged_rubric = merged_rubric
        self.merged_dynamic_rubric = merged_dynamic_rubric
