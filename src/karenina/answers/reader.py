import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from karenina.answers.generator import inject_question_id_into_answer_class
from karenina.schemas.answer_class import BaseAnswer


def read_answer_templates(answers_json_path: str | Path) -> dict[str, type]:
    """
    Read answer templates from a JSON file and return a dictionary of answer templates.

    Args:
        answers_json_path: The path to the JSON file containing the answer templates.

    Returns:
        A dictionary of answer templates. Keys are hashes of the questions, values are answer templates
        in the form of Answer classes (pydantic models)
    """

    namespace = {"BaseAnswer": BaseAnswer, "Field": Field, "Literal": Literal, "List": list}

    answer_dict = {}

    with open(answers_json_path) as f:
        answer_templates = json.load(f)

    idx = 1
    for key, value in answer_templates.items():
        exec(re.sub(r"^class Answer", f"class Answer{idx}", value), namespace)
        Answer = namespace["Answer" + str(idx)]
        # Inject the question ID programmatically
        AnswerWithID = inject_question_id_into_answer_class(Answer, key)  # type: ignore[arg-type]
        answer_dict[key] = AnswerWithID
        idx += 1

    return answer_dict
