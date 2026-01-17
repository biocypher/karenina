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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Union

if TYPE_CHECKING:
    from ...schemas.domain import Question, Rubric

from ...schemas.workflow import ModelConfig, VerificationConfig
from ..verification.evaluators.rubric_evaluator import RubricEvaluator
from .models import LogEvent, StepEval, TaskEvalResult


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
    ) -> None:
        """Initialize TaskEval instance.

        Args:
            task_id: Optional task identifier for tracking
            metadata: Optional metadata dictionary
            callable_registry: Registry of callable functions for manual trait evaluation
        """
        self.task_id = task_id
        self.metadata = metadata or {}
        self.callable_registry = callable_registry or {}

        # Storage for logs, questions, and rubrics
        self.global_logs: list[LogEvent] = []
        self.step_logs: dict[str, list[LogEvent]] = {}
        self.global_questions: list[dict[str, Any] | Question] = []
        self.step_questions: dict[str, list[dict[str, Any] | Question]] = {}
        self.global_rubrics: list[Rubric] = []
        self.step_rubrics: dict[str, list[Rubric]] = {}

    # =============================================================================
    # CORE API METHODS
    # =============================================================================

    def log(
        self,
        text: str | dict[str, str],
        step_id: str | None = None,
        target: Literal["global", "step", "both"] = "both",
        level: Literal["debug", "info", "warn", "error"] = "info",
        tags: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Log an event or response.

        This is the primary method for recording outputs that will be evaluated.
        All logged text is considered as potential answers for evaluation.

        Args:
            text: Log message text (str) or structured dict trace (dict[str, str])
                  When dict is provided, each key-value pair can be evaluated separately
            step_id: Optional step ID for step-specific logs
            target: Where to log ("global", "step", or "both")
            level: Log level (debug, info, warn, error)
            tags: Optional tags for categorization
            payload: Optional additional data

        Example:
            task.log("The answer is 42")
            task.log("Step 1 complete", step_id="calculation", level="info")
            task.log({"reasoning": "...", "action": "...", "result": "..."})
            task.log({"plan": "Create API"}, step_id="planning")
        """
        import json

        # Handle dict traces
        if isinstance(text, dict):
            log_text = json.dumps(text, ensure_ascii=False)
            is_dict = True
            dict_keys_list = list(text.keys())
        else:
            log_text = text
            is_dict = False
            dict_keys_list = None

        log_event = LogEvent(
            level=level,
            text=log_text,
            tags=tags,
            payload=payload,
            is_dict_structured=is_dict,
            dict_keys=dict_keys_list,
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

    def register_callable(self, name: str, func: Callable[[str], bool]) -> None:
        """
        Register a callable function for manual trait evaluation.

        Args:
            name: Name to register the function under
            func: Function that takes a string and returns a boolean

        Raises:
            ValueError: If function doesn't have correct signature
        """
        # Basic validation of function signature
        import inspect

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if len(params) != 1:
            raise ValueError(f"Callable '{name}' must have exactly one parameter, got {len(params)}")

        # Store the function
        self.callable_registry[name] = func

    def evaluate(self, config: VerificationConfig, step_id: str | None = None) -> TaskEvalResult:
        """Evaluate logged outputs against questions and rubrics.

        This method evaluates all logged outputs against the defined questions
        and rubrics to assess quality and characterize failure modes.

        Args:
            config: Verification configuration (parsing models only)
            step_id: Optional step ID to evaluate specific step (otherwise global)

        Returns:
            TaskEvalResult with evaluation outcomes and failure characterization

        Example:
            config = VerificationConfig(
                parsing_models=[ModelConfig(...)],
                parsing_only=True
            )
            result = task.evaluate(config)
        """
        if step_id:
            return self._evaluate_step(config, step_id)
        else:
            return self._evaluate_global(config)

    def _evaluate_global(self, config: VerificationConfig) -> TaskEvalResult:
        """Evaluate all global logs against global questions and rubrics.

        In global evaluation, all global logs are treated as potential answers
        for all global questions, allowing comprehensive evaluation across
        the entire logged output. After completing global evaluation, this
        method automatically evaluates all available steps as well.

        Args:
            config: Verification configuration with parsing models

        Returns:
            TaskEvalResult with both global evaluation results and all step evaluations
        """
        # Run evaluation loop for global context (step_id=None)
        step_eval = self._run_evaluation_loop(config, step_id=None)

        # Build result with global evaluation
        task_result = TaskEvalResult(
            task_id=self.task_id,
            metadata=self.metadata,
            global_eval=step_eval,
        )

        # After global evaluation, automatically evaluate all available steps
        for step_id in self._get_available_step_ids():
            step_result = self._evaluate_step_internal(config, step_id)
            task_result.per_step[step_id] = step_result

        return task_result

    def _evaluate_step(self, config: VerificationConfig, step_id: str) -> TaskEvalResult:
        """Evaluate step-specific logs against step-specific questions and rubrics.

        In step evaluation, only logs from the specified step are evaluated
        against questions and rubrics defined for that step, providing
        focused evaluation for specific workflow stages.

        Args:
            config: Verification configuration with parsing models
            step_id: ID of the step to evaluate

        Returns:
            TaskEvalResult with step-specific evaluation results
        """
        step_eval = self._evaluate_step_internal(config, step_id)
        return self._build_result(step_eval, step_id=step_id)

    def _evaluate_step_internal(self, config: VerificationConfig, step_id: str) -> StepEval:
        """Internal method to evaluate a single step and return StepEval.

        This method performs the actual step evaluation logic and returns
        just the StepEval object, allowing it to be used both for standalone
        step evaluation and as part of global evaluation.

        Args:
            config: Verification configuration with parsing models
            step_id: ID of the step to evaluate

        Returns:
            StepEval with step-specific evaluation results
        """
        return self._run_evaluation_loop(config, step_id=step_id)

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
        return step_ids

    def _get_evaluation_context(self, step_id: str | None = None) -> "EvaluationContext":
        """Get evaluation context for the specified step or global."""
        if step_id:
            questions = self.step_questions.get(step_id, [])
            rubrics = self.step_rubrics.get(step_id, [])
            logs = self.step_logs.get(step_id, [])
        else:
            questions = self.global_questions
            rubrics = self.global_rubrics
            logs = self.global_logs

        return EvaluationContext(questions=questions, logs=logs, merged_rubric=self._merge_rubrics(rubrics))

    def _normalize_question(self, question: Union[dict[str, Any], "Question"]) -> dict[str, Any]:
        """Normalize a question to dict format for evaluation.

        Handles both Question objects and dictionary objects from Benchmark.
        """
        from ...schemas.domain import Question

        if isinstance(question, Question):
            return {
                "id": question.id,
                "question": question.question,
                "raw_answer": question.raw_answer,
                "keywords": question.tags,
                "few_shot_examples": question.few_shot_examples,
                "answer_template": getattr(question, "answer_template", None),
            }
        return question

    def _collect_logs_for_evaluation(self, logs: list[LogEvent]) -> list[str]:
        """Collect all logged text for evaluation.

        Returns all non-empty log texts that could be potential responses.
        This allows TaskEval to work with simple log() calls.
        """
        outputs = []
        for log in logs:
            if log.text and len(log.text.strip()) > 0:
                outputs.append(log.text)
        return outputs

    def _detect_evaluation_mode(self, context: "EvaluationContext") -> str:
        """Auto-detect evaluation mode based on attached questions and rubrics.

        Args:
            context: The evaluation context containing questions and rubrics

        Returns:
            One of: "template_only", "rubric_only", "template_and_rubric"

        Raises:
            ValueError: If neither templates nor rubrics are provided
        """
        # Check if questions have answer templates
        has_templates = any(self._normalize_question(q).get("answer_template") for q in context.questions)

        # Check if rubrics are provided
        has_rubrics = bool(
            context.merged_rubric
            and (
                context.merged_rubric.llm_traits
                or context.merged_rubric.regex_traits
                or context.merged_rubric.callable_traits
                or context.merged_rubric.metric_traits
            )
        )

        if has_templates and has_rubrics:
            return "template_and_rubric"
        elif has_templates:
            return "template_only"
        elif has_rubrics:
            return "rubric_only"
        else:
            raise ValueError(
                "Must provide either answer templates, rubrics, or both for evaluation. "
                "Add questions with templates via add_question() or add rubrics via add_rubric()."
            )

    # =============================================================================
    # EVALUATION LOOP
    # =============================================================================

    def _run_evaluation_loop(self, config: VerificationConfig, step_id: str | None) -> StepEval:
        """Run the evaluation loop for either global or step-specific context.

        This method contains the shared evaluation logic for both global and step
        evaluation, reducing code duplication between _evaluate_global and
        _evaluate_step_internal.

        Args:
            config: Verification configuration with parsing models
            step_id: Optional step ID (None for global evaluation)

        Returns:
            StepEval with evaluation results
        """
        import json

        # Get evaluation context (global or step-specific)
        context = self._get_evaluation_context(step_id=step_id)

        # Detect evaluation mode based on what's attached
        evaluation_mode = self._detect_evaluation_mode(context)

        # Initialize result tracking
        step_eval = StepEval()

        # Collect logs for evaluation (do this once)
        relevant_logs = self._collect_logs_for_evaluation(context.logs)

        # Check for dict-structured logs for per-key evaluation in rubric_only mode
        dict_logs = [log for log in context.logs if log.is_dict_structured]

        # Determine how many replicates to run
        replicate_count = getattr(config, "replicate_count", 1)
        if replicate_count < 1:
            replicate_count = 1

        # Build step prefix for synthetic question IDs and error messages
        step_prefix = f"step_{step_id}_" if step_id else ""
        step_suffix = f" in step {step_id}" if step_id else ""

        # Replicate loop: run entire evaluation N times
        for _ in range(replicate_count):
            # In rubric_only mode with dict logs, create synthetic questions per key
            if evaluation_mode == "rubric_only" and dict_logs:
                # Extract unique keys from all dict logs
                all_dict_keys: set[str] = set()
                for log in dict_logs:
                    if log.dict_keys:
                        all_dict_keys.update(log.dict_keys)

                # Create synthetic questions for each key
                for key in sorted(all_dict_keys):  # Sort for deterministic order
                    # Collect values for this key across all dict logs
                    key_values = []
                    for log in dict_logs:
                        try:
                            log_dict = json.loads(log.text)
                            if key in log_dict:
                                key_values.append(log_dict[key])
                        except json.JSONDecodeError:
                            continue

                    # Concatenate values for this key
                    key_response = "\n\n".join(key_values) if key_values else ""

                    if not key_response:
                        continue

                    # Create synthetic question for this key
                    synthetic_question = {
                        "id": f"{step_prefix}dict_key_{key}",
                        "question": f"Evaluate the '{key}' output{step_suffix}",
                        "raw_answer": "",  # No ground truth for rubric-only
                        "answer_template": None,  # No template in rubric_only mode
                    }

                    self._evaluate_and_store(
                        step_eval=step_eval,
                        question_dict=synthetic_question,
                        response_text=key_response,
                        parsing_model=config.parsing_models[0],
                        rubric=context.merged_rubric,
                        evaluation_mode=evaluation_mode,
                        error_context=f"dict key {key}{step_suffix}",
                    )

            # Process regular questions (works for all modes)
            concatenated_logs = "\n\n".join(relevant_logs) if relevant_logs else ""

            # In rubric_only mode with no explicit questions and no dict logs,
            # create a synthetic question for string logs
            if evaluation_mode == "rubric_only" and not context.questions and not dict_logs and concatenated_logs:
                synthetic_question = {
                    "id": f"{step_prefix}rubric_only_eval",
                    "question": f"Evaluate the logged output{step_suffix}",
                    "raw_answer": "",  # No ground truth for rubric-only
                    "answer_template": None,  # No template in rubric_only mode
                }

                self._evaluate_and_store(
                    step_eval=step_eval,
                    question_dict=synthetic_question,
                    response_text=concatenated_logs,
                    parsing_model=config.parsing_models[0],
                    rubric=context.merged_rubric,
                    evaluation_mode=evaluation_mode,
                    error_context=f"rubric-only string logs{step_suffix}",
                )

            for question in context.questions:
                question_dict = self._normalize_question(question)
                question_id = question_dict.get("id", "unknown")

                # Check if question has answer template
                answer_template = question_dict.get("answer_template")

                # In rubric_only mode, templates are optional
                if evaluation_mode != "rubric_only" and not answer_template:
                    # Skip questions without templates in template modes
                    continue

                self._evaluate_and_store(
                    step_eval=step_eval,
                    question_dict=question_dict,
                    response_text=concatenated_logs,
                    parsing_model=config.parsing_models[0],
                    rubric=context.merged_rubric,
                    evaluation_mode=evaluation_mode,
                    error_context=f"question {question_id}{step_suffix}",
                )

        return step_eval

    def _evaluate_and_store(
        self,
        step_eval: StepEval,
        question_dict: dict[str, Any],
        response_text: str,
        parsing_model: ModelConfig,
        rubric: "Rubric | None",
        evaluation_mode: str,
        error_context: str,
    ) -> None:
        """Evaluate a single question and store the result.

        This helper method handles the common pattern of evaluating a question,
        storing the result, and handling errors gracefully.

        Args:
            step_eval: StepEval object to store results in
            question_dict: Question dictionary with metadata
            response_text: The logged text to evaluate
            parsing_model: Model to use for parsing/evaluation
            rubric: Rubric with evaluation traits (optional)
            evaluation_mode: One of "template_only", "rubric_only", "template_and_rubric"
            error_context: Context string for error messages
        """
        question_id = question_dict.get("id", "unknown")
        assert isinstance(question_id, str), "Question ID must be a string"

        try:
            # Use main verification pipeline
            verification_result = self._evaluate(
                question_dict=question_dict,
                response_text=response_text,
                parsing_model=parsing_model,
                rubric=rubric,
                evaluation_mode=evaluation_mode,
            )

            # Store VerificationResult
            if question_id not in step_eval.verification_results:
                step_eval.verification_results[question_id] = []
            step_eval.verification_results[question_id].append(verification_result)

        except Exception as e:
            # Handle evaluation errors gracefully
            print(f"Warning: Evaluation failed for {error_context}: {e}")

    # =============================================================================
    # EVALUATION METHODS
    # =============================================================================

    def _evaluate_response(
        self, question_dict: dict[str, Any], response_text: str, parsing_model: ModelConfig, rubric: "Rubric | None"
    ) -> dict[str, Any]:
        """Evaluate a single response against a question and rubric using proper verification pipeline.

        This method uses the same verification pipeline as the main benchmark system,
        including answer template parsing, LLM-based parsing, and verify() method calls.

        Args:
            question_dict: Question information with answer_template
            response_text: The logged response to evaluate
            parsing_model: Model configuration for parsing
            rubric: Quality rubric for evaluation

        Returns:
            Dictionary with evaluation results compatible with verification system
        """
        try:
            # Check if we have an answer template for proper verification
            answer_template = question_dict.get("answer_template")
            if not answer_template:
                # Fallback to simple verification if no template
                return self._evaluate_response_fallback(question_dict, response_text, parsing_model, rubric)

            # Use the existing verification pipeline
            result = self._evaluate(question_dict, response_text, parsing_model, rubric)
            return dict(result)  # Ensure we return dict[str, Any]

        except Exception as e:
            return {
                "verify_result": False,
                "verify_granular_result": {"agent_output": response_text, "error_during_evaluation": str(e)},
                "verify_rubric": {},
                "success": False,
                "error": f"Evaluation failed: {str(e)}",
            }

    def _evaluate(
        self,
        question_dict: dict[str, Any],
        response_text: str,
        parsing_model: ModelConfig,
        rubric: "Rubric | None",
        evaluation_mode: str = "template_only",
    ) -> Any:
        """Evaluate response using main verification pipeline with cached answer data.

        Args:
            question_dict: Question dictionary with metadata
            response_text: The logged text to evaluate
            parsing_model: Model to use for parsing/evaluation
            rubric: Rubric with evaluation traits (optional)
            evaluation_mode: One of "template_only", "rubric_only", "template_and_rubric"

        Returns:
            VerificationResult from the main verification pipeline
        """
        from ..verification.runner import run_single_model_verification

        question_id = question_dict.get("id", "unknown")
        question_text = question_dict.get("question", "")
        answer_template = question_dict.get("answer_template")

        # In rubric_only mode, template is optional
        if evaluation_mode == "rubric_only" and not answer_template:
            # Create a minimal template that just captures the raw response
            answer_template = '''
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Minimal answer template for rubric-only evaluation."""

    raw_response: str = Field(description="The raw response text")

    def verify(self) -> bool:
        # In rubric_only mode, verification always passes (rubrics determine quality)
        return True
'''

        # Ensure valid MD5 hash for question ID
        import hashlib

        if not self._is_valid_md5_hash(question_id):
            question_id = hashlib.md5(question_id.encode()).hexdigest()

        # Ensure answer_template is a string at this point
        assert isinstance(answer_template, str), "answer_template must be a string"

        # Create mock answering model (won't be invoked due to cached_answer_data)
        mock_answering_model = ModelConfig(
            id="taskeval_mock",
            model_provider="mock",
            model_name="mock",
            interface="langchain",  # Any interface, won't be called
            system_prompt="Mock model for TaskEval",
        )

        # Prepare cached answer data to inject logged output
        cached_answer_data = {
            "raw_llm_response": response_text,
            "recursion_limit_reached": False,
            "answering_mcp_servers": None,
            "usage_metadata": None,
            "agent_metrics": None,
        }

        # Call main verification pipeline with cached answer
        verification_result = run_single_model_verification(
            question_id=question_id,
            question_text=question_text,
            template_code=answer_template,
            answering_model=mock_answering_model,
            parsing_model=parsing_model,
            rubric=rubric,  # Pass rubric - will be evaluated in RubricEvaluationStage (all 3 types!)
            cached_answer_data=cached_answer_data,  # Skip generation, use our logged output
            abstention_enabled=True,  # Enable abstention detection
            rubric_evaluation_strategy="batch",  # Use batch strategy by default
            evaluation_mode=evaluation_mode,  # Pass detected evaluation mode
        )

        return verification_result  # Return VerificationResult directly

    def _evaluate_response_fallback(
        self, question_dict: dict[str, Any], response_text: str, parsing_model: ModelConfig, rubric: "Rubric | None"
    ) -> dict[str, Any]:
        """Fallback evaluation when no answer template is available."""
        # Use the original simple evaluation logic
        expected_answer = question_dict.get("raw_answer", "")
        correct = self._check_correctness(response_text, expected_answer)

        # Rubric evaluation: use RubricEvaluator
        rubric_scores: dict[str, int | bool] = {}
        if rubric and (rubric.llm_traits or rubric.regex_traits or rubric.callable_traits):
            try:
                evaluator = RubricEvaluator(parsing_model, evaluation_strategy="batch")
                question_text = question_dict.get("question", "")
                rubric_scores, _ = evaluator.evaluate_rubric(
                    question=question_text, answer=response_text, rubric=rubric
                )
            except Exception as e:
                print(f"Warning: RubricEvaluator failed in fallback: {e}")
                rubric_scores = {}

        return {
            "verify_result": correct,
            "verify_granular_result": {
                "agent_output": response_text,
                "expected_answer": expected_answer,
                "evaluation_method": "taskeval_fallback_with_rubric_evaluator",
            },
            "verify_rubric": rubric_scores,
            "success": True,
            "error": None,
        }

    def _check_correctness(self, response_text: str, expected_answer: str) -> bool:
        """Check if response is correct compared to expected answer."""
        if not expected_answer:
            # If no expected answer, consider non-empty response as valid
            return len(response_text.strip()) > 0

        # Check for semantic similarity (case-insensitive contains)
        response_lower = response_text.lower().strip()
        expected_lower = expected_answer.lower().strip()

        return expected_lower in response_lower or response_lower in expected_lower

    # =============================================================================
    # HELPER METHODS
    # =============================================================================

    def _is_valid_md5_hash(self, hash_string: str) -> bool:
        """Check if a string is a valid MD5 hash format."""
        import re

        md5_pattern = re.compile(r"^[a-fA-F0-9]{32}$")
        return bool(md5_pattern.match(hash_string))

    def _merge_rubrics(self, rubrics: list["Rubric"]) -> "Rubric | None":
        """Merge multiple rubrics into a single rubric, raising error on trait name conflicts."""
        if not rubrics:
            return None

        from ...schemas.domain import CallableTrait, LLMRubricTrait, MetricRubricTrait, RegexTrait, Rubric

        # Check for trait name conflicts first (across all trait types)
        all_trait_names = []
        for rubric in rubrics:
            # Add LLM trait names
            for trait in rubric.llm_traits:
                all_trait_names.append(trait.name)
            # Add regex trait names
            for regex_trait in rubric.regex_traits:
                all_trait_names.append(regex_trait.name)
            # Add callable trait names
            for callable_trait in rubric.callable_traits:
                all_trait_names.append(callable_trait.name)
            # Add metric trait names
            for metric_trait in rubric.metric_traits:
                all_trait_names.append(metric_trait.name)

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
        unique_regex_traits: dict[str, RegexTrait] = {}
        unique_callable_traits: dict[str, CallableTrait] = {}
        unique_metric_traits: dict[str, MetricRubricTrait] = {}
        for rubric in rubrics:
            # Combine LLM traits
            for trait in rubric.llm_traits:
                unique_llm_traits[trait.name] = trait
            # Combine regex traits
            for regex_trait in rubric.regex_traits:
                unique_regex_traits[regex_trait.name] = regex_trait
            # Combine callable traits
            for callable_trait in rubric.callable_traits:
                unique_callable_traits[callable_trait.name] = callable_trait
            # Combine metric traits
            for metric_trait in rubric.metric_traits:
                unique_metric_traits[metric_trait.name] = metric_trait

        return Rubric(
            llm_traits=list(unique_llm_traits.values()),
            regex_traits=list(unique_regex_traits.values()),
            callable_traits=list(unique_callable_traits.values()),
            metric_traits=list(unique_metric_traits.values()),
        )

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

    def __init__(self, questions: list[Any], logs: list[LogEvent], merged_rubric: "Rubric | None"):
        self.questions = questions
        self.logs = logs
        self.merged_rubric = merged_rubric
