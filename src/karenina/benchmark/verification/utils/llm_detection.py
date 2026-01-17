"""LLM detection utilities for verification operations.

This module provides functions for detecting LLM types and capabilities
to inform behavior decisions during verification.

Functions:
    is_openai_endpoint_llm: Check if LLM is a custom OpenAI-compatible endpoint
"""

from typing import Any

__all__ = [
    "is_openai_endpoint_llm",
]


def is_openai_endpoint_llm(llm: Any) -> bool:
    """Check if the LLM is a ChatOpenAIEndpoint (custom OpenAI-compatible endpoint).

    These endpoints often don't support native structured output (json_schema method)
    and can hang indefinitely when attempting to use it. This function helps callers
    decide whether to skip structured output attempts.

    Detection methods:
    1. Class name is "ChatOpenAIEndpoint"
    2. Class hierarchy includes ChatOpenAIEndpoint
    3. Module path suggests interface wrapper with ChatOpenAI
    4. Has custom base_url not pointing to api.openai.com

    Args:
        llm: LangChain chat model instance

    Returns:
        True if the LLM appears to be an OpenAI-compatible endpoint, False otherwise

    Example:
        >>> if is_openai_endpoint_llm(llm):
        ...     # Skip json_schema method, use fallback parsing
        ...     pass
    """
    # Check by class name to avoid circular imports
    llm_class_name = type(llm).__name__
    # Also check the module path for more robust detection
    llm_module = type(llm).__module__

    is_endpoint = (
        llm_class_name == "ChatOpenAIEndpoint"
        or "ChatOpenAIEndpoint" in str(type(llm).__mro__)
        or (llm_module and "interface" in llm_module and llm_class_name == "ChatOpenAI")
    )

    # Also check if it has a custom base_url that's not OpenAI's
    if hasattr(llm, "openai_api_base") and llm.openai_api_base:
        base_url = str(llm.openai_api_base)
        if base_url and not base_url.startswith("https://api.openai.com"):
            is_endpoint = True

    return bool(is_endpoint)
