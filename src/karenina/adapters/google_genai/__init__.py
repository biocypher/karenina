"""google_genai adapter: Gemini through the google-genai SDK.

Implements all three ports on Vertex AI (application default credentials,
the default) or the Gemini API (API key plus optional base URL):

- GoogleGenAIAgentAdapter: karenina-owned tool loop. Function and MCP tools
  run in the adapter, Gemini's server-side tools (Google Search, URL
  reading) run inside each call, with a per-run Google Search budget.
- GoogleGenAILLMAdapter: single calls with structured output and streaming.
- GoogleGenAIParserAdapter: structured parsing over the LLM adapter.

Configuration lives in ``ModelConfig`` and its ``extra_kwargs`` (see
``GoogleGenAISettings``). Design: ``docs/adapters/google-genai-adapter-design.md``.

Example:
    >>> from karenina.schemas.config import ModelConfig
    >>> from karenina.adapters.factory import get_agent
    >>> config = ModelConfig(
    ...     id="argon-judge",
    ...     model_name="gemini-4-argon",
    ...     interface="google_genai",
    ...     extra_kwargs={"builtin_tools": ["google_search", "url_context"], "max_search_queries": 5},
    ... )
    >>> agent = get_agent(config)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "GeminiAPIError",
    "GeminiAuthError",
    "GeminiFinishReasonError",
    "GeminiIncompleteResponseError",
    "GoogleGenAIAgentAdapter",
    "GoogleGenAILLMAdapter",
    "GoogleGenAIParserAdapter",
    "GoogleGenAISettings",
]

if TYPE_CHECKING:
    from .agent import GoogleGenAIAgentAdapter
    from .errors import GeminiAPIError, GeminiAuthError, GeminiFinishReasonError, GeminiIncompleteResponseError
    from .llm import GoogleGenAILLMAdapter
    from .parser import GoogleGenAIParserAdapter
    from .settings import GoogleGenAISettings

_LAZY_IMPORTS = {
    "GeminiAPIError": "karenina.adapters.google_genai.errors",
    "GeminiAuthError": "karenina.adapters.google_genai.errors",
    "GeminiFinishReasonError": "karenina.adapters.google_genai.errors",
    "GeminiIncompleteResponseError": "karenina.adapters.google_genai.errors",
    "GoogleGenAIAgentAdapter": "karenina.adapters.google_genai.agent",
    "GoogleGenAILLMAdapter": "karenina.adapters.google_genai.llm",
    "GoogleGenAIParserAdapter": "karenina.adapters.google_genai.parser",
    "GoogleGenAISettings": "karenina.adapters.google_genai.settings",
}


def __getattr__(name: str) -> Any:
    """Import the adapter classes lazily, so the package loads without the SDK."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'karenina.adapters.google_genai' has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)
