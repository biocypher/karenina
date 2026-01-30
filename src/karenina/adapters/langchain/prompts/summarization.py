"""Conversation summarization prompt for agent middleware.

Variables:
    {question_context} - Optional question context (built by build_question_context())
    {messages_text} - Formatted conversation messages
"""

PROMPT = """You are summarizing a conversation to preserve context for an ongoing task.
{question_context}CONVERSATION TO SUMMARIZE:
{messages_text}

INSTRUCTIONS:
Create a concise but information-rich summary that preserves:
1. The original question/goal being addressed
2. Key data and results from any tool calls (specific values, IDs, names)
3. Important reasoning steps and conclusions reached
4. Any errors encountered and how they were handled
5. The current state of progress toward answering the question

Keep the summary focused and factual. Do not include unnecessary pleasantries or meta-commentary.

SUMMARY:"""


def build_question_context(original_question: str | None) -> str:
    """Build optional question context section for summarization prompt.

    Args:
        original_question: The original user question, or None.

    Returns:
        Formatted context string, or empty string if no question.
    """
    if original_question:
        return f"""
ORIGINAL QUESTION: {original_question}

"""
    return ""
