"""LLM infrastructure for Karenina.

This package provides LLM-related utilities and exceptions.

For new code, use the port-based interface:
- get_llm() / get_agent() from karenina.adapters
- LLMPort / AgentPort protocols from karenina.ports
"""

from .exceptions import LLMError, LLMNotAvailableError, SessionError

__all__ = [
    "LLMError",
    "LLMNotAvailableError",
    "SessionError",
]
