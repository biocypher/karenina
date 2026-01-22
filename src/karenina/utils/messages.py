"""Message manipulation utilities for LLM conversations.

This module provides helper functions for manipulating message lists,
including adding retry feedback for validation failures.
"""

from karenina.ports import Message

__all__ = ["append_error_feedback"]


def append_error_feedback(messages: list[Message], error: str) -> list[Message]:
    """Append validation error feedback to messages for retry.

    Creates a new message list with a user message appended that contains
    the validation error, giving the LLM context to fix issues on retry.

    Args:
        messages: Original list of messages.
        error: The validation error message to include as feedback.

    Returns:
        New message list with error feedback appended. The original list
        is not modified.

    Example:
        >>> msgs = [Message.user("Parse this: foo")]
        >>> new_msgs = append_error_feedback(msgs, "Field 'value' is required")
        >>> len(new_msgs)
        2
        >>> "PREVIOUS ATTEMPT FAILED" in new_msgs[-1].content[0].text
        True
    """
    feedback = Message.user(
        f"PREVIOUS ATTEMPT FAILED with error: {error}\nPlease fix the validation issues and try again."
    )
    return list(messages) + [feedback]
