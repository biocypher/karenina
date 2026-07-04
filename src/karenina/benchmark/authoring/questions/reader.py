"""Question file reader for dynamically loading questions from Python files.

Loads generated questions.py files via importlib, extracting the all_questions
list. Uses unique module names and cleans up sys.modules after loading.
"""

import importlib.util
import os
import sys

from karenina.schemas.entities import Question


def read_questions_from_file(questions_py_path: str, return_dict: bool = False) -> list[Question] | dict[str, Question]:
    """Dynamically import all_questions from a questions.py file.

    Uses a unique module name to avoid collisions between files with the same
    basename, and cleans up sys.modules after extraction.

    Args:
        questions_py_path: Path to the questions.py file.
        return_dict: If True, return a dictionary with question IDs as keys
            and Question objects as values. Default is False.

    Returns:
        If return_dict is False: list of Question objects from all_questions.
        If return_dict is True: dict mapping question IDs to Question objects.

    Raises:
        FileNotFoundError: If the questions file doesn't exist.
        AttributeError: If the file doesn't contain an 'all_questions' variable.
        ImportError: If there's an error importing the module.
    """
    if not os.path.exists(questions_py_path):
        raise FileNotFoundError(f"Questions file not found: {questions_py_path}")

    # Use a unique module name to avoid collisions between files with the same basename
    module_name = f"_karenina_questions_{abs(hash(questions_py_path))}"

    # Create a module spec from the file location
    spec = importlib.util.spec_from_file_location(module_name, questions_py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {questions_py_path}")

    # Create and load the module, cleaning up sys.modules after extraction
    questions_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = questions_module
    try:
        spec.loader.exec_module(questions_module)

        # Extract all_questions from the module
        if not hasattr(questions_module, "all_questions"):
            raise AttributeError(f"Module {module_name} does not contain 'all_questions' variable")

        all_questions: list[Question] = questions_module.all_questions

        if return_dict:
            return {question.id: question for question in all_questions}
        else:
            return all_questions
    finally:
        sys.modules.pop(module_name, None)
