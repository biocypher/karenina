import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from karenina.benchmark.authoring.answers.generator import inject_question_id_into_answer_class
from karenina.schemas.entities import BaseAnswer


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
        exec(re.sub(r"^class Answer", f"class Answer{idx}", value, flags=re.MULTILINE), namespace)
        Answer = namespace["Answer" + str(idx)]

        # Store the template code for exec-created classes
        if hasattr(Answer, "_source_code"):
            Answer._source_code = value

        # Inject the question ID programmatically
        AnswerWithID = inject_question_id_into_answer_class(Answer, key)  # type: ignore[arg-type]
        answer_dict[key] = AnswerWithID
        idx += 1

    return answer_dict
