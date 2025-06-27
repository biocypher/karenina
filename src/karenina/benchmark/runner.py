import warnings

from langchain_core.messages import HumanMessage, SystemMessage

from karenina.llm.interface import init_chat_model_unified
from karenina.prompts.answer_evaluation import ANSWER_EVALUATION_SYS, ANSWER_EVALUATION_USER


def run_benchmark(question_dict: dict, response_dict: dict, answer_templates: dict):
    """
    Run the benchmark for a given model and questions.

    Args:
        question_dict: A dictionary of questions. Must be formatted as {question_id (the hash of the question): question in string format}.
        response_dict: A dictionary of responses. Must be formatted as {question_id (the hash of the question): response from the model in string format}.
        answer_templates: A dictionary of answer templates. Must be formatted as {question_id (the hash of the question): answer template in string format}.

    Returns:
        A dictionary of results.
    """

    # get the intersection of the keys common to the three dictionaries
    common_keys = set(question_dict.keys()) & set(response_dict.keys()) & set(answer_templates.keys())

    if len(common_keys) != max(len(question_dict.keys()), len(response_dict.keys()), len(answer_templates.keys())):
        warnings.warn(
            "The question_dict, response_dict, and answer_templates dictionaries have different keys. Using the intersection of the keys."
        )

    qdict = {k: v for k, v in question_dict.items() if k in common_keys}
    rdict = {k: v for k, v in response_dict.items() if k in common_keys}
    atdict = {k: v for k, v in answer_templates.items() if k in common_keys}

    llm = init_chat_model_unified(
        model="gemini-2.5-flash-preview-05-20", provider="google_genai", interface="langchain"
    )

    response_checks = {}

    for question_id, question in qdict.items():
        messages = [
            SystemMessage(content=ANSWER_EVALUATION_SYS),
            HumanMessage(
                content=ANSWER_EVALUATION_USER.format(
                    question=question,
                    response=rdict[question_id],
                )
            ),
        ]

        response = llm.with_structured_output(atdict[question_id]).invoke(messages)

        response_checks[question_id] = response

    return response_checks
