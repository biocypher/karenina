import importlib.util
import os
import sys

from karenina.schemas.entities import Question


def read_questions_from_file(questions_py_path: str, return_dict: bool = False) -> list[Question] | dict[str, Question]:
    """
    Dynamically import all_questions from a questions.py file.

    Args:
        questions_py_path: Path to the questions.py file
        return_dict: If True, return a dictionary with question IDs as keys and Question objects as values.
                    If False, return a list of Question objects. Default is False.

    Returns:
        If return_dict is False: List of Question objects from the all_questions variable in the file
        If return_dict is True: Dictionary with question IDs (hashes) as keys and Question objects as values

    Raises:
        FileNotFoundError: If the questions file doesn't exist
        AttributeError: If the file doesn't contain an 'all_questions' variable
        ImportError: If there's an error importing the module
    """
    if not os.path.exists(questions_py_path):
        raise FileNotFoundError(f"Questions file not found: {questions_py_path}")

    # Get the module name from the file path
    module_name = os.path.splitext(os.path.basename(questions_py_path))[0]

    # Create a module spec from the file location
    spec = importlib.util.spec_from_file_location(module_name, questions_py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {questions_py_path}")

    # Create and load the module
    questions_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = questions_module
    spec.loader.exec_module(questions_module)

    # Extract all_questions from the module
    if not hasattr(questions_module, "all_questions"):
        raise AttributeError(f"Module {module_name} does not contain 'all_questions' variable")

    all_questions: list[Question] = questions_module.all_questions

    if return_dict:
        # Return a dictionary with question IDs as keys and Question objects as values
        return {question.id: question for question in all_questions}
    else:
        # Return the original list
        return all_questions
