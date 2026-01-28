"""Adapter-agnostic trace parsing utilities.

This module provides utilities for parsing harmonized trace strings.
These functions work on the standardized "--- Message Type ---" format
produced by both LangChain and Claude SDK adapters.

The trace format is:
    --- AI Message ---
    Content here...

    --- Tool Message (call_id: xyz) ---
    Tool result...

    --- AI Message ---
    Final response...
"""

from __future__ import annotations


def extract_final_ai_message(harmonized_trace: str) -> tuple[str | None, str | None]:
    """Extract only the final AI text response from a harmonized agent trace string.

    This is an adapter-agnostic function that works on the standardized
    "--- AI Message ---" string format produced by both LangChain and
    Claude SDK adapters.

    For plain text responses (non-agent LLM calls without MCP), the input will
    not have message block format. In this case, the function returns the input
    as-is since it represents the direct AI response.

    Args:
        harmonized_trace: The full agent trace string, or plain text from a
            non-agent LLM response. Expected format uses "--- Message Type ---"
            delimiters.

    Returns:
        Tuple of (extracted_message, error_message) where:
        - extracted_message: The final AI text content, or None if extraction failed
        - error_message: Error description if extraction failed, or None if successful

    Example:
        >>> trace = '''--- AI Message ---
        ... Let me search for that.
        ...
        ... Tool Calls:
        ...   search (call_abc)
        ...
        ... --- Tool Message (call_id: call_abc) ---
        ... Results found
        ...
        ... --- AI Message ---
        ... The answer is 42.'''
        >>> message, error = extract_final_ai_message(trace)
        >>> print(message)  # "The answer is 42."

        >>> # Plain text (non-agent) is returned as-is
        >>> plain, error = extract_final_ai_message("The answer is 42.")
        >>> print(plain)  # "The answer is 42."
    """
    # Check for empty or whitespace-only trace
    if not harmonized_trace or not harmonized_trace.strip():
        return None, "Empty or whitespace-only trace"

    trace = harmonized_trace.strip()

    # Split trace into message blocks based on the separator pattern
    # Message blocks are separated by "--- <Type> Message ---" headers
    message_blocks: list[dict[str, str]] = []
    current_block_type: str | None = None
    current_block_content: list[str] = []

    lines = trace.split("\n")

    for line in lines:
        # Check if this is a message header
        if line.startswith("--- ") and line.endswith(" ---"):
            # Save previous block if exists
            if current_block_type is not None:
                message_blocks.append({"type": current_block_type, "content": "\n".join(current_block_content).strip()})

            # Start new block
            current_block_type = line
            current_block_content = []
        else:
            # Add line to current block content
            if current_block_type is not None:
                current_block_content.append(line)

    # Save last block
    if current_block_type is not None:
        message_blocks.append({"type": current_block_type, "content": "\n".join(current_block_content).strip()})

    # Check if we found any message blocks
    if not message_blocks:
        # No message blocks found - this is likely a plain text response from a non-agent LLM
        # (e.g., openai_endpoint or openrouter without MCP). In this case, the entire trace
        # is the AI's response, so return it directly.
        return trace, None

    # Get the last message block
    last_block = message_blocks[-1]

    # Check if the last block is an AI message
    if not last_block["type"].startswith("--- AI Message"):
        return None, "Last message in trace is not an AI message"

    # Extract content, excluding tool call information if present
    # Tool calls in AI messages appear after the main content with "Tool Calls:" header
    content = last_block["content"]

    # If there are tool calls in the message, extract only the text before them
    if "\nTool Calls:" in content:
        content = content.split("\nTool Calls:")[0].strip()
    # Also check if content starts with "Tool Calls:" (no text before)
    elif content.startswith("Tool Calls:"):
        content = ""

    # Final check: ensure we have non-empty content
    if not content:
        return None, "Final AI message has no text content (only tool calls)"

    return content, None


def prepare_evaluation_input(
    raw_response: str,
    use_full_trace: bool,
) -> tuple[str, str | None]:
    """Prepare input for template/rubric evaluation by optionally filtering the trace.

    This is a convenience wrapper around extract_final_ai_message that handles
    the common pattern of conditionally extracting only the final AI message
    based on a configuration flag.

    Args:
        raw_response: The raw LLM response (full trace or plain text)
        use_full_trace: If True, return raw_response as-is.
            If False, extract only the final AI message.

    Returns:
        Tuple of (evaluation_input, error_message) where:
        - evaluation_input: The filtered input to use for evaluation.
            Returns raw_response if use_full_trace=True or if extraction fails.
        - error_message: Error description if extraction failed, or None if successful.
            Note: If extraction fails, the raw_response is still returned as fallback
            but the error is surfaced for the caller to decide how to handle.

    Example:
        >>> # Using full trace
        >>> input, error = prepare_evaluation_input(trace, use_full_trace=True)
        >>> assert input == trace
        >>> assert error is None

        >>> # Extracting final message
        >>> input, error = prepare_evaluation_input(trace, use_full_trace=False)
        >>> # If successful, input contains only the final AI message
        >>> # If failed, input is the full trace and error explains what went wrong
    """
    if use_full_trace:
        return raw_response, None

    extracted_message, error = extract_final_ai_message(raw_response)

    if error is not None:
        # Extraction failed - return the original response with the error
        # Caller decides whether to fail or continue with full trace
        return raw_response, f"Failed to extract final AI message: {error}"

    if extracted_message is None:
        # Should not happen since extract_final_ai_message returns either
        # (content, None) or (None, error), but handle defensively
        return raw_response, "Failed to extract final AI message: no message found"

    return extracted_message, None
