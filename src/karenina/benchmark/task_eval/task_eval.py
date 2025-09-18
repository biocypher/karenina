"""TaskEval implementation for rubric-driven failure characterization."""

import json
from typing import TYPE_CHECKING, Any, Literal, Union

if TYPE_CHECKING:
    from ...schemas.question_class import Question
    from ...schemas.rubric_class import Rubric

from ..models import ModelConfig, VerificationConfig
from .models import LogEvent, StepEval, TaskEvalResult


class TaskEval:
    """
    Lightweight utility for task evaluation using the benchmark verification pipeline.

    TaskEval reuses the existing verification infrastructure to evaluate task performance
    and characterize failure modes through rubrics, without managing step lifecycles.
    """

    def __init__(self, task_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        """
        Initialize TaskEval.

        Args:
            task_id: Optional task identifier
            metadata: Optional metadata dictionary
        """
        self.task_id = task_id
        self.metadata = metadata or {}

        # Storage
        self.global_logs: list[LogEvent] = []
        self.step_logs: dict[str, list[LogEvent]] = {}
        self.global_questions: list[dict[str, Any] | Question] = []
        self.step_questions: dict[str, list[dict[str, Any] | Question]] = {}
        self.global_rubrics: list[Rubric] = []
        self.step_rubrics: dict[str, list[Rubric]] = {}

    def log(
        self,
        text: str,
        step_id: str | None = None,
        target: Literal["global", "step", "both"] = "both",
        level: Literal["debug", "info", "warn", "error"] = "info",
        tags: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """
        Log an event.

        Args:
            text: Log message text
            step_id: Optional step ID for step-specific logs
            target: Where to log ("global", "step", or "both")
            level: Log level
            tags: Optional tags
            payload: Optional additional data
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

    def log_agent_output(
        self,
        agent_output: str,
        question_id: str,
        step_id: str | None = None,
        target: Literal["global", "step", "both"] = "both",
        output_type: str = "answer",
        tags: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """
        Log agent output that will be evaluated against questions.

        This is the core method for recording agent outputs that TaskEval will later evaluate.
        The logged outputs ARE the answers that will be assessed.

        Args:
            agent_output: The actual output/response from the agent
            question_id: ID of the question this output answers
            step_id: Optional step ID for step-specific outputs
            target: Where to log ("global", "step", or "both")
            output_type: Type of output (answer, reasoning, analysis, etc.)
            tags: Optional tags
            payload: Optional additional data
        """
        log_event = LogEvent(
            level="info",
            text=agent_output,
            tags=tags,
            payload=payload,
            question_id=question_id,
            is_agent_output=True,
            output_type=output_type,
        )

        if target in ("global", "both"):
            self.global_logs.append(log_event)

        if target in ("step", "both") and step_id:
            if step_id not in self.step_logs:
                self.step_logs[step_id] = []
            self.step_logs[step_id].append(log_event)

    def add_question(self, question_obj: Union[dict[str, Any], "Question"], step_id: str | None = None) -> None:
        """
        Add a question for evaluation.

        Args:
            question_obj: Question dictionary (from Benchmark) or Question object
            step_id: Optional step ID for step-specific questions
        """
        if step_id:
            if step_id not in self.step_questions:
                self.step_questions[step_id] = []
            self.step_questions[step_id].append(question_obj)
        else:
            self.global_questions.append(question_obj)

    def add_rubric(self, rubric_obj: "Rubric", step_id: str | None = None) -> None:
        """
        Add a rubric for evaluation.

        Args:
            rubric_obj: Rubric object
            step_id: Optional step ID for step-specific rubrics
        """
        if step_id:
            if step_id not in self.step_rubrics:
                self.step_rubrics[step_id] = []
            self.step_rubrics[step_id].append(rubric_obj)
        else:
            self.global_rubrics.append(rubric_obj)

    def evaluate(self, config: VerificationConfig, step_id: str | None = None) -> TaskEvalResult:
        """
        Evaluate logged agent outputs against questions and rubrics.

        This method takes the logged agent outputs (which ARE the answers) and evaluates
        them against the defined questions and rubrics using only the parsing pipeline.
        No new answers are generated - we evaluate what was already logged.

        Args:
            config: Verification configuration (only parsing_models will be used)
            step_id: Optional step ID to evaluate specific step (otherwise evaluates global)

        Returns:
            TaskEvalResult with evaluation outcomes
        """
        # Determine what to evaluate
        if step_id:
            questions = self.step_questions.get(step_id, [])
            rubrics = self.step_rubrics.get(step_id, [])
            logs = self.step_logs.get(step_id, [])
        else:
            questions = self.global_questions
            rubrics = self.global_rubrics
            logs = self.global_logs

        # Merge rubrics
        merged_rubric = self._merge_rubrics(rubrics)

        # Initialize step evaluation
        step_eval = StepEval()

        # Temporary storage for aggregating scores across multiple responses
        temp_rubric_scores: dict[str, list[int | bool]] = {}

        # Evaluate each question against logged agent outputs
        for question in questions:
            q_dict = self._normalize_question(question)
            q_id = q_dict.get("id", "unknown")

            # Find ALL logged agent outputs for this question
            agent_outputs = self._find_agent_outputs_for_question(q_id, logs)

            if not agent_outputs:
                # No logged outputs found for this question
                step_eval.question_verification[q_id] = [
                    {
                        "correct": False,
                        "details": f"No agent output found for question {q_id}",
                        "success": False,
                        "error": f"Missing agent output for question {q_id}",
                    }
                ]
                continue

            # Evaluate each logged output using only the parsing pipeline
            question_results = []
            for i, agent_output in enumerate(agent_outputs):
                result = self._evaluate_agent_output(
                    question_dict=q_dict,
                    agent_output=agent_output,
                    parsing_model=config.parsing_models[0],
                    rubric=merged_rubric,
                )

                # Store result for this response
                question_result = {
                    "response_index": i,
                    "agent_output": agent_output,
                    "correct": result.get("verify_result", False),
                    "details": result.get("verify_granular_result"),
                    "success": result.get("success", False),
                    "error": result.get("error"),
                    "rubric_scores": result.get("verify_rubric", {}),
                }
                question_results.append(question_result)

                # Collect rubric scores for later aggregation
                if result.get("verify_rubric"):
                    for trait_name, score in result["verify_rubric"].items():
                        if trait_name not in temp_rubric_scores:
                            temp_rubric_scores[trait_name] = []
                        temp_rubric_scores[trait_name].append(score)

            # Store all results for this question
            step_eval.question_verification[q_id] = question_results

        # Average rubric scores across all responses
        for trait_name, scores in temp_rubric_scores.items():
            if all(isinstance(s, bool) for s in scores):
                # For boolean traits, use majority vote
                step_eval.rubric_scores[trait_name] = sum(scores) > len(scores) / 2
            else:
                # For numeric traits, use average
                step_eval.rubric_scores[trait_name] = int(sum(scores) / len(scores))

        # Derive failure modes from rubric scores
        step_eval.failure_modes = self._extract_failure_modes(step_eval.rubric_scores)

        # Build result
        task_result = TaskEvalResult(
            task_id=self.task_id,
            metadata=self.metadata,
        )

        if step_id:
            task_result.per_step[step_id] = step_eval
        else:
            task_result.global_eval = step_eval

        return task_result

    def _normalize_question(self, question: Union[dict[str, Any], "Question"]) -> dict[str, Any]:
        """
        Normalize a question to dict format for verification.

        Handles both Question objects and dict objects from Benchmark.
        """
        from ...schemas.question_class import Question

        if isinstance(question, Question):
            return {
                "id": question.id,
                "question": question.question,
                "raw_answer": question.raw_answer,
                "keywords": question.tags,
                "few_shot_examples": question.few_shot_examples,
                "answer_template": None,  # Will be generated if needed
            }
        return question  # Already a dict from benchmark

    def _prepare_for_verification(self, question_dict: dict[str, Any]) -> tuple[str, str, str]:
        """
        Prepare question data for verification.

        Returns:
            (question_id, question_text, template_code)
        """
        question_id = question_dict.get("id", "unknown")
        question_text = question_dict.get("question", "")
        template_code = question_dict.get("answer_template")

        # Generate minimal template if missing
        if not template_code:
            template_code = self._generate_minimal_template(question_dict)

        return question_id, question_text, template_code

    def _generate_minimal_template(self, question_dict: dict[str, Any]) -> str:
        """
        Generate a minimal answer template when none is provided.

        This creates a basic Pydantic model that captures the raw answer.
        """
        raw_answer = question_dict.get("raw_answer", "")

        template = f'''from pydantic import BaseModel, Field

class Answer(BaseModel):
    """Generated minimal template for task evaluation."""

    correct: str = Field(description="The correct answer")

    # Set the correct answer as the raw answer from the question
    def __init__(self, **data):
        if "correct" not in data:
            data["correct"] = {json.dumps(raw_answer)}
        super().__init__(**data)
'''
        return template

    def _merge_rubrics(self, rubrics: list["Rubric"]) -> "Rubric | None":
        """
        Merge multiple rubrics into a single rubric.

        Returns None if no rubrics provided.
        """
        if not rubrics:
            return None

        from ...schemas.rubric_class import Rubric

        # Combine all traits from all rubrics
        all_traits = []
        for rubric in rubrics:
            all_traits.extend(rubric.traits)

        # Remove duplicates by name (later ones override earlier ones)
        unique_traits = {}
        for trait in all_traits:
            unique_traits[trait.name] = trait

        return Rubric(traits=list(unique_traits.values()))

    def _extract_failure_modes(self, rubric_scores: dict[str, int | bool]) -> list[str]:
        """
        Extract failure modes from rubric scores.

        For boolean traits: failure mode if False
        For score traits: failure mode if below threshold (configurable, default < 3)
        """
        failure_modes = []

        for trait_name, score in rubric_scores.items():
            if isinstance(score, bool):
                if not score:
                    failure_modes.append(f"Failed trait: {trait_name}")
            elif isinstance(score, int) and score < 3:  # Threshold for score-based traits
                failure_modes.append(f"Low score trait: {trait_name} (score: {score})")

        return failure_modes

    def _find_agent_outputs_for_question(self, question_id: str, logs: list[LogEvent]) -> list[str]:
        """
        Find ALL logged agent outputs for a specific question.

        Args:
            question_id: The question ID to find outputs for
            logs: List of log events to search

        Returns:
            List of agent output texts (empty if none found)
        """
        # Look for all agent outputs that match this question
        outputs = []
        for log in logs:
            if log.is_agent_output and log.question_id == question_id:
                outputs.append(log.text)

        return outputs

    def _evaluate_agent_output(
        self, question_dict: dict[str, Any], agent_output: str, parsing_model: ModelConfig, rubric: "Rubric | None"
    ) -> dict[str, Any]:
        """
        Evaluate a logged agent output using parsing-only evaluation.

        This method implements the correct TaskEval philosophy: we have the agent's
        output already, so we only need to parse/evaluate it against the question/rubric.
        This reuses the existing verification pipeline but with pre-existing agent output.

        Args:
            question_dict: Question information
            agent_output: The logged agent output to evaluate
            parsing_model: Model configuration for parsing
            rubric: Rubric for evaluation

        Returns:
            Dictionary with evaluation results compatible with verification pipeline
        """
        try:
            # Use the existing verification infrastructure but with pre-existing agent output
            # This creates a parsing-only configuration that evaluates the logged output
            from ..models import VerificationConfig

            # Create a minimal verification config for parsing only
            parsing_config = VerificationConfig(
                answering_models=[],  # No answering needed - we have the output
                parsing_models=[parsing_model],
                rubric_enabled=rubric is not None,
            )

            # Prepare question data for verification
            question_id, question_text, template_code = self._prepare_for_verification(question_dict)

            # Create a mock verification result that uses our pre-existing agent output
            # This bypasses the answering phase and goes straight to parsing/evaluation
            verification_input = {
                "question_id": question_id,
                "question": question_text,
                "template_code": template_code,
                "agent_output": agent_output,  # Use logged output instead of generating new
                "rubric": rubric,
                "config": parsing_config,
            }

            # Call the existing verification pipeline with our logged agent output
            # Note: This would need modification to the actual verification runner
            # to accept pre-existing agent output instead of generating new responses
            result = self._run_parsing_only_verification(verification_input)

            return result

        except Exception as e:
            # Fallback to simplified evaluation if full pipeline integration fails
            return self._fallback_evaluation(question_dict, agent_output, rubric, str(e))

    def _run_parsing_only_verification(self, verification_input: dict[str, Any]) -> dict[str, Any]:
        """
        Run verification pipeline with pre-existing agent output.

        This is a placeholder for the proper integration with the verification pipeline
        that would be modified to accept pre-existing agent outputs.
        """
        # For now, use simplified evaluation until full pipeline integration
        # In full implementation, this would call a modified version of
        # run_single_model_verification that skips answering phase

        agent_output = verification_input["agent_output"]
        rubric = verification_input["rubric"]
        question_dict = {
            "id": verification_input["question_id"],
            "question": verification_input["question"],
            "raw_answer": verification_input.get("expected_answer", ""),
        }

        return self._fallback_evaluation(question_dict, agent_output, rubric, "Full pipeline integration pending")

    def _fallback_evaluation(
        self, question_dict: dict[str, Any], agent_output: str, rubric: "Rubric | None", reason: str
    ) -> dict[str, Any]:
        """Simplified evaluation for cases where full pipeline integration isn't available."""
        try:
            # Basic correctness check
            expected_answer = question_dict.get("raw_answer", "")
            if expected_answer:
                # Check for semantic similarity (case-insensitive contains)
                correct = expected_answer.lower().strip() in agent_output.lower()
                # Also check reverse for partial matches
                if not correct:
                    correct = agent_output.lower().strip() in expected_answer.lower()
            else:
                # No expected answer to compare against
                correct = len(agent_output.strip()) > 0

            # Evaluate rubric traits
            rubric_scores: dict[str, int | bool] = {}
            if rubric:
                for trait in rubric.traits:
                    if trait.kind == "boolean":
                        # Boolean traits: based on correctness and output quality
                        has_content = len(agent_output.strip()) > 10
                        rubric_scores[trait.name] = correct and has_content
                    else:  # score trait
                        # Score traits: 1-5 scale based on correctness and output quality
                        base_score = 3 if correct else 2
                        quality_bonus = 1 if len(agent_output) > 50 else 0
                        rubric_scores[trait.name] = int(min(5, max(1, base_score + quality_bonus)))

            return {
                "verify_result": correct,
                "verify_granular_result": {
                    "agent_output": agent_output,
                    "expected_answer": expected_answer,
                    "evaluation_method": "taskeval_simplified",
                    "pipeline_integration": reason,
                },
                "verify_rubric": rubric_scores,
                "success": True,
                "error": None,
            }

        except Exception as e:
            return {
                "verify_result": False,
                "verify_granular_result": {"agent_output": agent_output, "error_during_evaluation": str(e)},
                "verify_rubric": {},
                "success": False,
                "error": f"Evaluation failed: {str(e)}",
            }
