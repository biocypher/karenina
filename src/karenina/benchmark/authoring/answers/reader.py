"""Answer template reader for loading templates from JSON files.

Reads Answer class definitions stored as Python source code in JSON,
exec's them in a comprehensive namespace, and returns ready-to-use classes.
"""

import json
import re
from pathlib import Path

from karenina.benchmark.authoring.answers.generator import inject_question_id_into_answer_class
from karenina.benchmark.verification.utils.template_validation import _build_exec_namespace


def read_answer_templates(answers_json_path: str | Path) -> dict[str, type]:
    """Read answer templates from a JSON file and return a dictionary of answer templates.

    Uses the comprehensive exec namespace from template_validation, which
    includes BaseAnswer, VerifiedField, all primitives, typing utilities,
    and Pydantic Field.

    Args:
        answers_json_path: The path to the JSON file containing the answer templates.

    Returns:
        A dictionary of answer templates. Keys are hashes of the questions, values
        are answer templates in the form of Answer classes (pydantic models).
    """
    namespace = _build_exec_namespace()

    answer_dict = {}

    with open(answers_json_path) as f:
        answer_templates = json.load(f)

    idx = 1
    for key, value in answer_templates.items():
        exec(re.sub(r"^class Answer", f"class Answer{idx}", value, flags=re.MULTILINE), namespace)  # noqa: S102
        Answer = namespace["Answer" + str(idx)]

        # Store the template code for exec-created classes
        if hasattr(Answer, "_source_code"):
            Answer._source_code = value

        # Inject the question ID programmatically
        AnswerWithID = inject_question_id_into_answer_class(Answer, key)
        answer_dict[key] = AnswerWithID
        idx += 1

    return answer_dict
