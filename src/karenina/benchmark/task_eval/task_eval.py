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

from typing import TYPE_CHECKING, Any, Literal, Union

if TYPE_CHECKING:
    from ...schemas.question_class import Question
    from ...schemas.rubric_class import Rubric

from ..models import ModelConfig, VerificationConfig
from ..verification.rubric_evaluator import RubricEvaluator
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

    def __init__(self, task_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        """Initialize TaskEval instance.

        Args:
            task_id: Optional task identifier for tracking
            metadata: Optional metadata dictionary
        """
        self.task_id = task_id
        self.metadata = metadata or {}

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
        text: str,
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
            text: Log message text (the output to evaluate)
            step_id: Optional step ID for step-specific logs
            target: Where to log ("global", "step", or "both")
            level: Log level (debug, info, warn, error)
            tags: Optional tags for categorization
            payload: Optional additional data

        Example:
            task.log("The answer is 42")
            task.log("Step 1 complete", step_id="calculation", level="info")
        """
        log_event = LogEvent(
            level=level,
            text=text,
            tags=tags,
            payload=payload,
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
            rubric = Rubric(traits=[
                RubricTrait(name="accuracy", description="Is answer correct?", kind="boolean")
            ])
            task.add_rubric(rubric)
        """
        if step_id:
            if step_id not in self.step_rubrics:
                self.step_rubrics[step_id] = []
            self.step_rubrics[step_id].append(rubric_obj)
        else:
            self.global_rubrics.append(rubric_obj)

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
        # Get global evaluation context
        context = self._get_evaluation_context(step_id=None)

        # Initialize result tracking
        step_eval = StepEval()

        # Collect all global logs for evaluation (do this once)
        relevant_logs = self._collect_logs_for_evaluation(context.logs)

        if not relevant_logs:
            # No logs to evaluate - handle all questions as having no logs
            for question in context.questions:
                question_dict = self._normalize_question(question)
                question_id = question_dict.get("id", "unknown")
                step_eval.question_verification[question_id] = [
                    {
                        "correct": False,
                        "details": "No logged outputs found for evaluation",
                        "success": False,
                        "error": f"No logs available for question {question_id}",
                    }
                ]
            # No rubric evaluation needed if no logs
            step_eval.rubric_scores = {}
        else:
            # Concatenate all logs into a single response for global evaluation
            concatenated_logs = "\n\n".join(relevant_logs)

            # Collect all rubric traits from standalone rubrics and question-specific rubrics
            all_rubric_traits = {}
            standalone_traits: set[str] = set()
            question_traits: set[str] = set()

            # Collect standalone rubric traits
            if context.merged_rubric and context.merged_rubric.traits:
                for trait in context.merged_rubric.traits:
                    all_rubric_traits[trait.name] = trait
                    standalone_traits.add(trait.name)

            # Collect question-specific rubric traits and check for conflicts
            for question in context.questions:
                question_dict = self._normalize_question(question)
                answer_template = question_dict.get("answer_template")
                if answer_template:
                    # Extract rubric traits from answer template if they exist
                    extracted_traits = self._extract_rubric_traits_from_template(answer_template)
                    # Add to collections
                    for trait in extracted_traits:
                        all_rubric_traits[trait.name] = trait
                        question_traits.add(trait.name)

            # Check for conflicts between standalone and question rubrics
            conflicts = standalone_traits.intersection(question_traits)
            if conflicts:
                raise ValueError(
                    f"Rubric trait name conflicts found: {conflicts}. "
                    f"Standalone rubrics and question rubrics cannot have overlapping trait names."
                )

            # Evaluate standalone rubrics once for all questions
            global_rubric_scores: dict[str, int | bool] = {}
            if context.merged_rubric and context.merged_rubric.traits:
                try:
                    # For manual interface, fall back to simplified evaluation
                    if config.parsing_models[0].interface == "manual":
                        # Use simplified rubric evaluation for manual interface
                        for trait in context.merged_rubric.traits:
                            if trait.kind == "boolean":
                                # For boolean traits, assume true for non-empty logs
                                global_rubric_scores[trait.name] = len(concatenated_logs.strip()) > 0
                            else:  # score trait
                                # For score traits, give reasonable score based on content length
                                global_rubric_scores[trait.name] = 4 if len(concatenated_logs) > 50 else 3
                    else:
                        # Use proper RubricEvaluator for non-manual interfaces
                        from ..verification.rubric_evaluator import RubricEvaluator

                        evaluator = RubricEvaluator(config.parsing_models[0])
                        # Use a generic question for rubric evaluation since it's global
                        global_rubric_scores = evaluator.evaluate_rubric(
                            question="Evaluate the overall quality of the logged outputs.",
                            answer=concatenated_logs,
                            rubric=context.merged_rubric,
                        )
                except Exception as e:
                    print(f"Warning: Global rubric evaluation failed: {e}")
                    global_rubric_scores = {}

            # Initialize combined rubric scores with standalone scores
            combined_rubric_scores = global_rubric_scores.copy()

            # Evaluate each question against the concatenated logs
            for question in context.questions:
                question_dict = self._normalize_question(question)
                question_id = question_dict.get("id", "unknown")

                # Check if this question has a rubric that needs to be evaluated
                question_rubric = None
                answer_template = question_dict.get("answer_template")
                if answer_template:
                    # Extract rubric from answer template if it exists
                    question_rubric_traits = self._extract_rubric_traits_from_template(answer_template)
                    if question_rubric_traits:
                        from ...schemas.rubric_class import Rubric

                        question_rubric = Rubric(traits=question_rubric_traits)

                # Evaluate the concatenated logs (with question-specific rubric if it exists)
                result = self._evaluate_response(
                    question_dict=question_dict,
                    response_text=concatenated_logs,
                    parsing_model=config.parsing_models[0],
                    rubric=question_rubric,  # Only evaluate question-specific rubric here
                )

                # Combine standalone rubric scores with question-specific rubric scores
                question_specific_scores = result.get("verify_rubric", {})
                final_rubric_scores = combined_rubric_scores.copy()
                final_rubric_scores.update(question_specific_scores)

                # Store single evaluation result for all concatenated logs
                question_results = [
                    {
                        "agent_output": concatenated_logs,
                        "correct": result.get("verify_result", False),
                        "details": result.get("verify_granular_result"),
                        "success": result.get("success", False),
                        "error": result.get("error"),
                        "rubric_scores": final_rubric_scores,  # Combined scores
                    }
                ]

                step_eval.question_verification[question_id] = question_results

            # Store the combined rubric scores at the step level
            step_eval.rubric_scores = combined_rubric_scores

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
        # Get step-specific evaluation context
        context = self._get_evaluation_context(step_id=step_id)

        # Initialize result tracking
        step_eval = StepEval()

        # Collect step-specific logs for evaluation (do this once)
        relevant_logs = self._collect_logs_for_evaluation(context.logs)

        if not relevant_logs:
            # No logs to evaluate - handle all questions as having no logs
            for question in context.questions:
                question_dict = self._normalize_question(question)
                question_id = question_dict.get("id", "unknown")
                step_eval.question_verification[question_id] = [
                    {
                        "correct": False,
                        "details": f"No logged outputs found for step {step_id}",
                        "success": False,
                        "error": f"No logs available for question {question_id} in step {step_id}",
                    }
                ]
            # No rubric evaluation needed if no logs
            step_eval.rubric_scores = {}
        else:
            # Concatenate all step logs into a single response for evaluation
            concatenated_logs = "\n\n".join(relevant_logs)

            # Collect rubric traits and check for conflicts (same as global evaluation)
            standalone_traits: set[str] = set()
            question_traits: set[str] = set()

            # Collect standalone rubric traits
            if context.merged_rubric and context.merged_rubric.traits:
                for trait in context.merged_rubric.traits:
                    standalone_traits.add(trait.name)

            # Collect question-specific rubric traits and check for conflicts
            for question in context.questions:
                question_dict = self._normalize_question(question)
                answer_template = question_dict.get("answer_template")
                if answer_template:
                    # Extract rubric traits from answer template if they exist
                    extracted_traits = self._extract_rubric_traits_from_template(answer_template)
                    for trait in extracted_traits:
                        question_traits.add(trait.name)

            # Check for conflicts between standalone and question rubrics
            conflicts = standalone_traits.intersection(question_traits)
            if conflicts:
                raise ValueError(
                    f"Rubric trait name conflicts found: {conflicts}. "
                    f"Standalone rubrics and question rubrics cannot have overlapping trait names."
                )

            # Evaluate standalone rubrics once for all questions in this step
            step_rubric_scores: dict[str, int | bool] = {}
            if context.merged_rubric and context.merged_rubric.traits:
                try:
                    # For manual interface, fall back to simplified evaluation
                    if config.parsing_models[0].interface == "manual":
                        # Use simplified rubric evaluation for manual interface
                        for trait in context.merged_rubric.traits:
                            if trait.kind == "boolean":
                                # For boolean traits, assume true for non-empty logs
                                step_rubric_scores[trait.name] = len(concatenated_logs.strip()) > 0
                            else:  # score trait
                                # For score traits, give reasonable score based on content length
                                step_rubric_scores[trait.name] = 4 if len(concatenated_logs) > 50 else 3
                    else:
                        # Use proper RubricEvaluator for non-manual interfaces
                        from ..verification.rubric_evaluator import RubricEvaluator

                        evaluator = RubricEvaluator(config.parsing_models[0])
                        # Use a generic question for rubric evaluation since it's step-level
                        step_rubric_scores = evaluator.evaluate_rubric(
                            question=f"Evaluate the overall quality of step '{step_id}' outputs.",
                            answer=concatenated_logs,
                            rubric=context.merged_rubric,
                        )
                except Exception as e:
                    print(f"Warning: Step rubric evaluation failed for step {step_id}: {e}")
                    step_rubric_scores = {}

            # Initialize combined rubric scores with standalone scores
            combined_rubric_scores = step_rubric_scores.copy()

            # Evaluate each question against the concatenated logs
            for question in context.questions:
                question_dict = self._normalize_question(question)
                question_id = question_dict.get("id", "unknown")

                # Check if this question has a rubric that needs to be evaluated
                question_rubric = None
                answer_template = question_dict.get("answer_template")
                if answer_template:
                    # Extract rubric from answer template if it exists
                    question_rubric_traits = self._extract_rubric_traits_from_template(answer_template)
                    if question_rubric_traits:
                        from ...schemas.rubric_class import Rubric

                        question_rubric = Rubric(traits=question_rubric_traits)

                # Evaluate the concatenated logs (with question-specific rubric if it exists)
                result = self._evaluate_response(
                    question_dict=question_dict,
                    response_text=concatenated_logs,
                    parsing_model=config.parsing_models[0],
                    rubric=question_rubric,  # Only evaluate question-specific rubric here
                )

                # Combine standalone rubric scores with question-specific rubric scores
                question_specific_scores = result.get("verify_rubric", {})
                final_rubric_scores = combined_rubric_scores.copy()
                final_rubric_scores.update(question_specific_scores)

                # Store single evaluation result for all concatenated logs
                question_results = [
                    {
                        "agent_output": concatenated_logs,
                        "correct": result.get("verify_result", False),
                        "details": result.get("verify_granular_result"),
                        "success": result.get("success", False),
                        "error": result.get("error"),
                        "rubric_scores": final_rubric_scores,  # Combined scores
                    }
                ]

                step_eval.question_verification[question_id] = question_results

            # Store the combined rubric scores at the step level
            step_eval.rubric_scores = combined_rubric_scores

        return step_eval

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
        from ...schemas.question_class import Question

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

    def _extract_rubric_traits_from_template(self, answer_template: str) -> list[Any]:
        """Extract rubric traits from answer template code.

        Args:
            answer_template: The answer template code string

        Returns:
            List of RubricTrait objects found in the template
        """
        try:
            # Prepare minimal execution environment similar to template validation
            from ...schemas.answer_class import BaseAnswer
            from ...schemas.rubric_class import Rubric, RubricTrait

            global_ns = {
                "__builtins__": __builtins__,
                "BaseAnswer": BaseAnswer,
                "Rubric": Rubric,
                "RubricTrait": RubricTrait,
            }
            try:
                from pydantic import Field

                global_ns["Field"] = Field
            except Exception:
                pass
            try:
                from typing import Any, ClassVar, Literal, Optional, Union

                global_ns.update(
                    {
                        "List": list,
                        "Dict": dict,
                        "Optional": Optional,
                        "Union": Union,
                        "Any": Any,
                        "Literal": Literal,
                        "ClassVar": ClassVar,
                    }
                )
            except Exception:
                pass

            local_ns: dict[str, Any] = {}
            exec(answer_template, global_ns, local_ns)

            # Heuristics: check for rubric on Answer class or top-level var
            extracted_traits: list[RubricTrait] = []

            def _coerce_traits(obj: Any) -> list[RubricTrait]:
                traits_list: list[RubricTrait] = []
                if not obj:
                    return traits_list
                # If wrapped in Rubric
                if isinstance(obj, Rubric):
                    for t in obj.traits:
                        if isinstance(t, RubricTrait):
                            traits_list.append(t)
                    return traits_list
                # If already list of RubricTrait
                if isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, RubricTrait):
                            traits_list.append(item)
                        elif isinstance(item, dict) and "name" in item and "kind" in item:
                            try:
                                traits_list.append(RubricTrait(**item))
                            except Exception:
                                continue
                return traits_list

            AnswerCls = local_ns.get("Answer")
            if AnswerCls is not None:
                # Common attribute names that might store rubric traits
                for attr in ("question_rubric", "rubric_traits", "rubric"):
                    if hasattr(AnswerCls, attr):
                        extracted_traits = _coerce_traits(getattr(AnswerCls, attr))
                        if extracted_traits:
                            break

            # Also allow a top-level constant like QUESTION_RUBRIC
            if not extracted_traits and "QUESTION_RUBRIC" in local_ns:
                extracted_traits = _coerce_traits(local_ns.get("QUESTION_RUBRIC"))

            return extracted_traits
        except Exception:
            # Silently ignore rubric extraction errors to keep TaskEval lightweight
            return []

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
            return self._evaluate(question_dict, response_text, parsing_model, rubric)

        except Exception as e:
            return {
                "verify_result": False,
                "verify_granular_result": {"agent_output": response_text, "error_during_evaluation": str(e)},
                "verify_rubric": {},
                "success": False,
                "error": f"Evaluation failed: {str(e)}",
            }

    def _evaluate(
        self, question_dict: dict[str, Any], response_text: str, parsing_model: ModelConfig, rubric: "Rubric | None"
    ) -> dict[str, Any]:
        """Evaluate response using existing verification pipeline with manual interface."""
        from ..verification.runner import run_single_model_verification

        question_id = question_dict.get("id", "unknown")
        question_text = question_dict.get("question", "")
        answer_template = question_dict["answer_template"]

        # Manual interface requires valid MD5 hash, generate one if needed
        import hashlib

        if not self._is_valid_md5_hash(question_id):
            question_id = hashlib.md5(question_id.encode()).hexdigest()

        # Create a mock answering model using manual interface
        mock_answering_model = ModelConfig(
            id="taskeval_mock",
            model_provider="mock",
            model_name="mock",
            interface="manual",
            system_prompt="Mock model for TaskEval",
        )

        # Temporarily add the response to manual traces
        import time

        from ...llm.manual_traces import get_trace_manager

        trace_manager = get_trace_manager()

        try:
            # Add our response to the manual trace system temporarily
            with trace_manager._lock:
                trace_manager._traces[question_id] = response_text
                trace_manager._trace_timestamps[question_id] = time.time()

            # Call the existing verification function (without rubric to avoid double evaluation)
            verification_result = run_single_model_verification(
                question_id=question_id,
                question_text=question_text,
                template_code=answer_template,
                answering_model=mock_answering_model,
                parsing_model=parsing_model,
                rubric=None,  # Don't pass rubric to avoid double evaluation
            )

            # Evaluate rubric separately using RubricEvaluator for standalone evaluation
            rubric_scores: dict[str, int | bool] = {}
            if rubric and rubric.traits:
                try:
                    # For manual interface, we need to use a custom approach since RubricEvaluator
                    # doesn't support question_hash directly. For now, fall back to simplified evaluation.
                    if parsing_model.interface == "manual":
                        # Fallback to simplified rubric evaluation for manual interface
                        # TODO: Enhance RubricEvaluator to support manual interface with question_hash
                        # For now, use basic heuristic evaluation
                        rubric_scores = {}
                        for trait in rubric.traits:
                            if trait.kind == "boolean":
                                # For boolean traits, assume true if verification passed
                                rubric_scores[trait.name] = verification_result.verify_result or False
                            else:  # score trait
                                # For score traits, give reasonable score based on verification
                                rubric_scores[trait.name] = 4 if verification_result.verify_result else 2
                    else:
                        # Use proper RubricEvaluator for non-manual interfaces
                        evaluator = RubricEvaluator(parsing_model)
                        rubric_scores = evaluator.evaluate_rubric(
                            question=question_text, answer=response_text, rubric=rubric
                        )
                except Exception as e:
                    # Don't fail verification if rubric evaluation fails
                    print(f"Warning: Standalone rubric evaluation failed: {e}")
                    rubric_scores = {}

            # Convert VerificationResult to our expected format
            return {
                "verify_result": verification_result.verify_result
                if verification_result.verify_result is not None
                else False,
                "verify_granular_result": {
                    "agent_output": response_text,
                    "parsed_gt_response": verification_result.parsed_gt_response,
                    "parsed_llm_response": verification_result.parsed_llm_response,
                    "evaluation_method": "taskeval_existing_pipeline_with_standalone_rubric",
                    "execution_time": verification_result.execution_time,
                },
                "verify_rubric": rubric_scores,
                "success": verification_result.success,
                "error": verification_result.error,
            }

        finally:
            # Clean up the temporary trace
            with trace_manager._lock:
                trace_manager._traces.pop(question_id, None)
                trace_manager._trace_timestamps.pop(question_id, None)

    def _evaluate_response_fallback(
        self, question_dict: dict[str, Any], response_text: str, parsing_model: ModelConfig, rubric: "Rubric | None"
    ) -> dict[str, Any]:
        """Fallback evaluation when no answer template is available."""
        # Use the original simple evaluation logic
        expected_answer = question_dict.get("raw_answer", "")
        correct = self._check_correctness(response_text, expected_answer)

        # Rubric evaluation: use RubricEvaluator when parsing model is available
        rubric_scores: dict[str, int | bool] = {}
        if rubric and rubric.traits:
            try:
                # For manual interface, fall back to simplified evaluation
                if parsing_model.interface == "manual":
                    # Use simplified rubric evaluation for manual interface
                    rubric_scores = self._evaluate_against_rubric(response_text, correct, rubric)
                else:
                    # Use proper RubricEvaluator with parsing model for other interfaces
                    evaluator = RubricEvaluator(parsing_model)
                    question_text = question_dict.get("question", "")
                    rubric_scores = evaluator.evaluate_rubric(
                        question=question_text, answer=response_text, rubric=rubric
                    )
            except Exception as e:
                # Fallback to simplified rubric evaluation if RubricEvaluator fails
                print(f"Warning: RubricEvaluator failed in fallback, using simplified evaluation: {e}")
                rubric_scores = self._evaluate_against_rubric(response_text, correct, rubric)

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

    def _evaluate_against_rubric(self, response_text: str, is_correct: bool, rubric: "Rubric") -> dict[str, int | bool]:
        """Evaluate response against rubric traits."""
        scores: dict[str, int | bool] = {}

        for trait in rubric.traits:
            if trait.kind == "boolean":
                # Boolean traits: based on correctness and content quality
                has_content = len(response_text.strip()) > 10
                scores[trait.name] = is_correct and has_content
            else:  # score trait
                # Score traits: 1-5 scale based on correctness and content quality
                base_score = 3 if is_correct else 2
                quality_bonus = 1 if len(response_text) > 50 else 0
                scores[trait.name] = int(min(5, max(1, base_score + quality_bonus)))

        return scores

    # =============================================================================
    # HELPER METHODS
    # =============================================================================

    def _is_valid_md5_hash(self, hash_string: str) -> bool:
        """Check if a string is a valid MD5 hash format."""
        import re

        if not isinstance(hash_string, str):
            return False
        md5_pattern = re.compile(r"^[a-fA-F0-9]{32}$")
        return bool(md5_pattern.match(hash_string))

    def _merge_rubrics(self, rubrics: list["Rubric"]) -> "Rubric | None":
        """Merge multiple rubrics into a single rubric, raising error on trait name conflicts."""
        if not rubrics:
            return None

        from ...schemas.rubric_class import Rubric

        # Check for trait name conflicts first
        all_trait_names = []
        for rubric in rubrics:
            for trait in rubric.traits:
                all_trait_names.append(trait.name)

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
        unique_traits = {}
        for rubric in rubrics:
            for trait in rubric.traits:
                unique_traits[trait.name] = trait

        return Rubric(traits=list(unique_traits.values()))

    def _aggregate_rubric_scores(self, result: dict[str, Any], aggregator: dict[str, list[int | bool]]) -> None:
        """Aggregate rubric scores from a single evaluation result."""
        if result.get("verify_rubric"):
            for trait_name, score in result["verify_rubric"].items():
                if trait_name not in aggregator:
                    aggregator[trait_name] = []
                aggregator[trait_name].append(score)

    def _finalize_rubric_scores(self, aggregator: dict[str, list[int | bool]]) -> dict[str, int | bool]:
        """Finalize aggregated rubric scores using appropriate aggregation methods."""
        final_scores: dict[str, int | bool] = {}

        for trait_name, scores in aggregator.items():
            if all(isinstance(s, bool) for s in scores):
                # For boolean traits, use majority vote
                final_scores[trait_name] = sum(scores) > len(scores) / 2
            else:
                # For numeric traits, use average
                final_scores[trait_name] = int(sum(scores) / len(scores))

        return final_scores

    def _extract_failure_modes(self, rubric_scores: dict[str, int | bool]) -> list[str]:
        """Extract failure modes from rubric scores.

        Identifies traits that indicate failures:
        - Boolean traits that are False
        - Score traits below threshold (< 3)
        """
        failure_modes = []

        for trait_name, score in rubric_scores.items():
            if isinstance(score, bool):
                if not score:
                    failure_modes.append(f"Failed trait: {trait_name}")
            elif isinstance(score, int) and score < 3:
                failure_modes.append(f"Low score trait: {trait_name} (score: {score})")

        return failure_modes

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
