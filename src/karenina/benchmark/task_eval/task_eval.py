"""TaskEval implementation for rubric-driven failure characterization."""

import json
from typing import TYPE_CHECKING, Any, Literal, Union

if TYPE_CHECKING:
    from ...schemas.question_class import Question
    from ...schemas.rubric_class import Rubric

from ..models import VerificationConfig
from ..verification.runner import run_single_model_verification
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
        Run evaluation using the verification pipeline.

        Args:
            config: Verification configuration
            step_id: Optional step ID to evaluate specific step (otherwise evaluates global)

        Returns:
            TaskEvalResult with evaluation outcomes
        """
        # Determine what to evaluate
        if step_id:
            questions = self.step_questions.get(step_id, [])
            rubrics = self.step_rubrics.get(step_id, [])
        else:
            questions = self.global_questions
            rubrics = self.global_rubrics

        # Merge rubrics
        merged_rubric = self._merge_rubrics(rubrics)

        # Initialize step evaluation
        step_eval = StepEval()

        # Evaluate each question
        for question in questions:
            q_dict = self._normalize_question(question)
            q_id, q_text, template = self._prepare_for_verification(q_dict)

            # Use first answering/parsing model pair from config
            # TODO: In future, could iterate through all model combinations
            result = run_single_model_verification(
                question_id=q_id,
                question_text=q_text,
                template_code=template,
                answering_model=config.answering_models[0],
                parsing_model=config.parsing_models[0],
                rubric=merged_rubric,
                keywords=q_dict.get("keywords"),
                few_shot_examples=q_dict.get("few_shot_examples"),
                few_shot_enabled=config.is_few_shot_enabled(),
            )

            # Extract rubric scores
            if result.verify_rubric:
                step_eval.rubric_scores.update(result.verify_rubric)

            # Extract verification results
            step_eval.question_verification = {
                "correct": result.verify_result,
                "details": result.verify_granular_result,
                "success": result.success,
                "error": result.error,
            }

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
