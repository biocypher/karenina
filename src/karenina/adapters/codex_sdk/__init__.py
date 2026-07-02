"""Codex SDK adapter for natively agentic evaluation.

Wraps OpenAI's Codex Python SDK (openai-codex), which drives a bundled
``codex app-server`` binary over JSON-RPC. Agent-only: LLM and parser
duties fall back to the langchain interface.

Requires: pip install openai-codex (or pip install 'karenina[codex]')

Adapter classes:
    - CodexSDKAgentAdapter: One codex thread/turn per run with OS-level
      sandboxing, built-in shell and patch tools, and timeout salvage.

Utilities:
    - CodexMessageConverter: Convert between unified Message and codex items
    - check_codex_available: Check SDK and binary availability
    - codex_items_to_raw_trace / codex_items_to_trace_messages: Trace formats
    - extract_codex_usage: Extract UsageMetadata from ThreadTokenUsage
    - wrap_codex_error: Map SDK exceptions to port errors
    - EndpointShim: Local request rewriter for strict /v1/responses endpoints
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "CodexSDKAgentAdapter",
    "CodexMessageConverter",
    "EndpointShim",
    "check_codex_available",
    "codex_items_to_raw_trace",
    "codex_items_to_trace_messages",
    "convert_mcp_to_codex_config",
    "extract_codex_usage",
    "wrap_codex_error",
]

if TYPE_CHECKING:
    from karenina.adapters.codex_sdk.agent import CodexSDKAgentAdapter
    from karenina.adapters.codex_sdk.availability import check_codex_available
    from karenina.adapters.codex_sdk.endpoint_shim import EndpointShim
    from karenina.adapters.codex_sdk.errors import wrap_codex_error
    from karenina.adapters.codex_sdk.mcp import convert_mcp_to_codex_config
    from karenina.adapters.codex_sdk.messages import CodexMessageConverter
    from karenina.adapters.codex_sdk.trace import (
        codex_items_to_raw_trace,
        codex_items_to_trace_messages,
    )
    from karenina.adapters.codex_sdk.usage import extract_codex_usage


def __getattr__(name: str) -> Any:
    """Lazy import adapter classes to avoid circular imports."""
    _imports = {
        "CodexSDKAgentAdapter": "karenina.adapters.codex_sdk.agent",
        "CodexMessageConverter": "karenina.adapters.codex_sdk.messages",
        "EndpointShim": "karenina.adapters.codex_sdk.endpoint_shim",
        "check_codex_available": "karenina.adapters.codex_sdk.availability",
        "codex_items_to_raw_trace": "karenina.adapters.codex_sdk.trace",
        "codex_items_to_trace_messages": "karenina.adapters.codex_sdk.trace",
        "convert_mcp_to_codex_config": "karenina.adapters.codex_sdk.mcp",
        "extract_codex_usage": "karenina.adapters.codex_sdk.usage",
        "wrap_codex_error": "karenina.adapters.codex_sdk.errors",
    }

    if name in _imports:
        import importlib

        module = importlib.import_module(_imports[name])
        return getattr(module, name)

    raise AttributeError(f"module 'karenina.adapters.codex_sdk' has no attribute '{name}'")
