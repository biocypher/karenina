"""Helper functions for TaskEval."""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from karenina.ports.messages import Message
    from karenina.schemas.entities import Rubric

from .models import LogEvent

logger = logging.getLogger(__name__)


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
    if standalone_rubric and standalone_rubric.llm_traits:
        for trait in standalone_rubric.llm_traits:
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


def convert_string_logs_to_messages(texts: list[str]) -> "list[Message]":
    """Wrap each text string as an assistant Message.

    Args:
        texts: List of text strings to convert.

    Returns:
        List of Message objects, one per input text.
    """
    from karenina.ports.messages import Message

    return [Message.assistant(text) for text in texts if text]


def merge_logs_and_traces(logs: list[LogEvent], strategy: str = "concatenate") -> "tuple[str, list[Message] | None]":
    """Merge LogEvent entries into a response string and optional Message list.

    This is the core merge logic for TaskEval evaluation. It combines text logs
    and structured trace_messages from LogEvents into the formats needed by the
    verification pipeline.

    Args:
        logs: List of LogEvent objects to merge.
        strategy: Merge strategy.
            "concatenate" (default): text logs converted to Messages plus
                trace_messages combined; string produced via messages_to_raw_trace().
            "traces_only": only LogEvents with trace_messages are used;
                text-only logs are ignored.

    Returns:
        Tuple of (response_text_string, optional_message_list).
        The string is always non-None (may be empty).
        The message list is None when no Message objects are available.
    """
    from karenina.benchmark.verification.utils.trace_formatting import messages_to_raw_trace
    from karenina.ports.messages import Message

    all_messages: list[Message] = []

    if strategy == "traces_only":
        for log in logs:
            if log.trace_messages:
                all_messages.extend(log.trace_messages)
    else:
        # "concatenate": combine text logs (as Messages) + trace_messages
        for log in logs:
            if log.trace_messages:
                all_messages.extend(log.trace_messages)
            elif log.text and log.text.strip():
                all_messages.append(Message.assistant(log.text))

    if not all_messages:
        return "", None

    response_text = messages_to_raw_trace(all_messages)
    return response_text, all_messages
