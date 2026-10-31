"""MCP configuration conversion for the Codex SDK adapter (stub).

Codex configures MCP servers through its CLI config schema (a top-level
``mcp_servers`` table reached via ``--config`` overrides or the per-thread
``config`` dict), not through a per-call API. Wiring karenina's
MCPServerConfig into that channel is planned but not implemented yet, so
this module is a converter stub: it warns and skips.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from karenina.ports import MCPServerConfig

logger = logging.getLogger(__name__)


def convert_mcp_to_codex_config(
    mcp_servers: dict[str, MCPServerConfig] | None,
) -> dict[str, Any]:
    """Convert karenina MCP server configs to a codex config overlay.

    Not implemented yet. Codex expects MCP servers under a ``mcp_servers``
    config table (keys such as command, args, env, url), reachable through
    ``CodexConfig.config_overrides`` or ``thread_start(config=...)``. Until
    that mapping is validated end to end, requested MCP servers are
    skipped with a warning rather than silently misconfigured.

    Args:
        mcp_servers: Karenina MCP server configuration, or None.

    Returns:
        Empty dict (no codex config overlay is produced).
    """
    if mcp_servers:
        logger.warning(
            "MCP servers are not yet supported by the codex_sdk adapter. Skipping %d configured server(s): %s",
            len(mcp_servers),
            ", ".join(sorted(mcp_servers)),
        )
    return {}
