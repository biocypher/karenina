"""Helper functions for TaskEval."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from karenina.ports.messages import Message

from .models import LogEvent

logger = logging.getLogger(__name__)


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
