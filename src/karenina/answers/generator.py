import json
from typing import Dict, List, Literal, Tuple, Union  # noqa: F401

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field  # noqa: F401
from tqdm import tqdm

from karenina.llm.interface import init_chat_model_unified

# TODO: probably here we should import more from typing to makesure that the template definition always runs smoothly
from karenina.prompts.answer_generation import ANSWER_GENERATION_SYS, ANSWER_GENERATION_USER
from karenina.questions.reader import read_questions_from_file
from karenina.schemas.answer_class import BaseAnswer  # noqa: F401
from karenina.utils.code_parser import extract_and_combine_codeblocks


def generate_answer_template(
    question: str,
    question_json: str,
    model: str = "gemini-2.0-flash",
    model_provider: str = "google_genai",
    temperature: float = 0,
    custom_system_prompt: str = None,
    interface: str = "langchain",
) -> str:
    """
    Generate a answer template for a given question and question json.

    Args:
        question: The question to generate an answer template for.
        question_json: The json representation of the question.
        model: The model to use for the answer template.
        model_provider: The provider of the model.
        temperature: The temperature of the model.
        custom_system_prompt: Optional custom system prompt to use instead of default.
        interface: The interface to use ('langchain' or 'openrouter').

    Returns:
        The python code for the answer template.
    """
    llm = init_chat_model_unified(model=model, provider=model_provider, interface=interface, temperature=temperature)

    # Use custom system prompt if provided, otherwise use default
    system_prompt = custom_system_prompt if custom_system_prompt else ANSWER_GENERATION_SYS

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=ANSWER_GENERATION_USER.format(question=question, question_json=question_json)),
    ]

    return llm.invoke(messages).content


def generate_answer_templates_from_questions_file(
    questions_py_path: str,
    model: str = "gemini-2.0-flash",
    model_provider: str = "google_genai",
    interface: str = "langchain",
    return_blocks: bool = False,
) -> dict:
    """
    Given a path to a questions.py file, dynamically import all_questions from it,
    generate answer templates for each question using the specified model and provider,
    and return a dictionary mapping question.id to the generated Answer class.
    """
    # Use the reader function to get all questions
    all_questions = read_questions_from_file(questions_py_path)

    answer_templates = {}
    all_code_blocks = {}
    for i, question in tqdm(enumerate(all_questions)):
        answer_template = generate_answer_template(
            question.question,
            question.model_dump_json(),
            model=model,
            model_provider=model_provider,
            interface=interface,
        )
        code_blocks = extract_and_combine_codeblocks(answer_template)
        # define the class in a local namespace
        local_ns = {}
        exec(code_blocks, globals(), local_ns)
        Answer = local_ns["Answer"]
        answer_templates[question.id] = Answer

        if return_blocks:
            all_code_blocks[question.id] = code_blocks

    if return_blocks:
        return answer_templates, all_code_blocks
    else:
        return answer_templates


def load_answer_templates_from_json(
    json_file_path: str, return_blocks: bool = False
) -> Union[Dict[str, type], Tuple[Dict[str, type], Dict[str, str]]]:
    """
    Load answer templates from a JSON file containing code blocks.

    Args:
        json_file_path: Path to the JSON file containing code blocks
        return_blocks: Whether to also return the code blocks dictionary

    Returns:
        If return_blocks is False: Dictionary mapping question IDs to Answer classes
        If return_blocks is True: Tuple of (answer_templates, code_blocks)
    """
    # Read the JSON file
    with open(json_file_path) as f:
        all_code_blocks = json.load(f)

    answer_templates = {}
    for question_id, code_blocks in all_code_blocks.items():
        # Define the class in a local namespace
        local_ns = {}
        exec(code_blocks, globals(), local_ns)
        Answer = local_ns["Answer"]
        answer_templates[question_id] = Answer

    if return_blocks:
        return answer_templates, all_code_blocks
    else:
        return answer_templates
