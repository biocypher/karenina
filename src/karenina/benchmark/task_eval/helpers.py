"""Helper functions for TaskEval to reduce code duplication."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...schemas.rubric_class import Rubric

from ..models import ModelConfig
from ..verification.evaluators.rubric_evaluator import RubricEvaluator

# Import the shared function from utils.parsing


def check_rubric_conflicts(
    standalone_rubric: "Rubric | None", questions: list[Any], extract_traits_func: Callable[[str], list[Any]]
) -> tuple[set[str], set[str]]:
    """Check for conflicts between standalone and question-specific rubrics.

    Args:
        standalone_rubric: The standalone rubric
        questions: List of questions to check
        extract_traits_func: Function to extract traits from templates

    Returns:
        Tuple of (standalone_traits, question_traits) sets

    Raises:
        ValueError: If conflicts are found
    """
    standalone_traits: set[str] = set()
    question_traits: set[str] = set()

    # Collect standalone rubric traits
    if standalone_rubric and standalone_rubric.traits:
        for trait in standalone_rubric.traits:
            standalone_traits.add(trait.name)

    # Collect question-specific rubric traits
    for question in questions:
        if isinstance(question, dict):
            question_dict = question
        else:
            question_dict = {"id": question.id, "question": question.question, "raw_answer": question.raw_answer}
            if hasattr(question, "answer_template"):
                question_dict["answer_template"] = question.answer_template

        answer_template = question_dict.get("answer_template")
        if answer_template:
            extracted_traits = extract_traits_func(answer_template)
            for trait in extracted_traits:
                question_traits.add(trait.name)

    # Check for conflicts
    conflicts = standalone_traits.intersection(question_traits)
    if conflicts:
        raise ValueError(
            f"Rubric trait name conflicts found: {conflicts}. "
            f"Standalone rubrics and question rubrics cannot have overlapping trait names."
        )

    return standalone_traits, question_traits


def evaluate_standalone_rubrics(
    parsing_model: ModelConfig,
    merged_rubric: "Rubric | None",
    concatenated_logs: str,
    context: str = "global",
    callable_registry: dict[str, Callable[[str], bool]] | None = None,
) -> dict[str, int | bool]:
    """Evaluate standalone rubrics for a given context.

    Args:
        parsing_model: Model to use for evaluation
        merged_rubric: The merged rubric to evaluate
        concatenated_logs: The concatenated logs to evaluate against
        context: Context string for the evaluation question
        callable_registry: Registry of callable functions for manual trait evaluation

    Returns:
        Dictionary of rubric scores
    """
    rubric_scores: dict[str, int | bool] = {}

    if merged_rubric and (merged_rubric.traits or merged_rubric.manual_traits):
        try:
            evaluator = RubricEvaluator(parsing_model, callable_registry)
            question = f"Evaluate the overall quality of the {context} outputs."
            rubric_scores = evaluator.evaluate_rubric(question=question, answer=concatenated_logs, rubric=merged_rubric)
        except Exception as e:
            print(f"Warning: {context.title()} rubric evaluation failed: {e}")
            rubric_scores = {}

    return rubric_scores


def evaluate_question_with_rubric(
    question_dict: dict[str, Any],
    response_text: str,
    parsing_model: ModelConfig,
    extract_traits_func: Callable[[str], list[Any]],
    evaluate_response_func: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate a single question with its question-specific rubric.

    Args:
        question_dict: Question dictionary
        response_text: Response text to evaluate
        parsing_model: Model to use for parsing
        extract_traits_func: Function to extract rubric traits from templates
        evaluate_response_func: Function to evaluate the response

    Returns:
        Evaluation result dictionary
    """
    # Check if this question has a rubric that needs to be evaluated
    question_rubric = None
    answer_template = question_dict.get("answer_template")
    if answer_template:
        question_rubric_traits = extract_traits_func(answer_template)
        if question_rubric_traits:
            from ...schemas.rubric_class import Rubric

            question_rubric = Rubric(traits=question_rubric_traits)

    # Evaluate the response with question-specific rubric
    return evaluate_response_func(
        question_dict=question_dict, response_text=response_text, parsing_model=parsing_model, rubric=question_rubric
    )


def process_questions_with_concatenated_logs(
    questions: list[Any],
    concatenated_logs: str,
    parsing_model: ModelConfig,
    normalize_question_func: Callable[..., dict[str, Any]],
    extract_traits_func: Callable[[str], list[Any]],
    evaluate_response_func: Callable[..., dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Process all questions against concatenated logs.

    Args:
        questions: List of questions to process
        concatenated_logs: Concatenated logs text
        parsing_model: Model to use for parsing
        normalize_question_func: Function to normalize question format
        extract_traits_func: Function to extract rubric traits
        evaluate_response_func: Function to evaluate responses

    Returns:
        Dictionary mapping question IDs to evaluation results
    """
    question_verification: dict[str, list[dict[str, Any]]] = {}

    for question in questions:
        question_dict = normalize_question_func(question)
        question_id = question_dict.get("id", "unknown")

        # Evaluate with question-specific rubric
        result = evaluate_question_with_rubric(
            question_dict=question_dict,
            response_text=concatenated_logs,
            parsing_model=parsing_model,
            extract_traits_func=extract_traits_func,
            evaluate_response_func=evaluate_response_func,
        )

        # Only include question-specific rubric scores (NOT global standalone rubrics)
        # Global standalone rubrics belong at the step level, not individual questions
        question_specific_scores = result.get("verify_rubric", {})

        # Store single evaluation result for all concatenated logs in TaskEval format
        question_results = [
            {
                "agent_output": concatenated_logs,
                "correct": result.get("verify_result", False),
                "details": result.get("verify_granular_result"),
                "success": result.get("success", False),
                "error": result.get("error"),
                "rubric_scores": question_specific_scores,  # Only question-specific scores
            }
        ]

        question_verification[question_id] = question_results

    return question_verification
