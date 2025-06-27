"""Code block parsing utilities for Karenina.

This module provides functions for extracting and processing code blocks from
text content, particularly useful for handling LLM-generated code responses
and template generation.
"""

import re

# Improved regex: matches triple backtick code blocks with or without language identifiers, optional whitespace, and both Unix/Windows line endings
BACKTICK_PATTERN = r"```(?:[a-zA-Z0-9_+-]*)[ \t]*\r?\n([\s\S]*?)```"


def extract_and_combine_codeblocks(text: str) -> str:
    """Extract and combine code blocks from text content.

    Args:
        text: A string containing zero or more code blocks, where each code block is
            surrounded by triple backticks (```).

    Returns:
        A string containing the combined code from all code blocks, with each block
        separated by a newline.

    Example:
        >>> text = '''Here's some code:
        ... ```python
        ... print('hello')
        ... ```
        ... And more:
        ... ```
        ... print('world')
        ... ```'''
        >>> result = extract_and_combine_codeblocks(text)
        >>> print(result)
        print('hello')
        print('world')
    """
    # Find all code blocks in the text using regex
    # Pattern matches anything between triple backticks, with or without a language identifier
    code_blocks = re.findall(BACKTICK_PATTERN, text, re.DOTALL)

    if not code_blocks:
        return ""

    # Process each codeblock
    processed_blocks = []
    for block in code_blocks:
        # Strip leading and trailing whitespace
        block = block.strip()
        processed_blocks.append(block)

    # Combine all codeblocks with newlines between them
    combined_code = "\n\n".join(processed_blocks)
    return combined_code
