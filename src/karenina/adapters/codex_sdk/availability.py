"""Availability check for the Codex SDK adapter.

The adapter needs the openai-codex Python package plus a codex CLI binary.
The package normally pins openai-codex-cli-bin, which bundles the binary,
so a plain ``pip install openai-codex`` satisfies both. A codex binary on
PATH also works when the bundled one is absent.
"""

from __future__ import annotations

import shutil

from karenina.adapters.registry import AdapterAvailability

_INSTALL_HINT = "Install with: pip install 'karenina[codex]' or: pip install openai-codex"


def check_codex_available() -> AdapterAvailability:
    """Check if the Codex SDK and a codex binary are available.

    Returns:
        AdapterAvailability with status, reason, and a langchain fallback
        when unavailable.
    """
    try:
        import openai_codex  # noqa: F401
    except ImportError:
        return AdapterAvailability(
            available=False,
            reason=f"openai-codex package not installed. {_INSTALL_HINT}",
            fallback_interface="langchain",
        )

    binary_path: str | None = None
    try:
        import codex_cli_bin

        binary_path = str(codex_cli_bin.bundled_codex_path())
    except (ImportError, OSError):
        binary_path = shutil.which("codex")

    if binary_path is None:
        return AdapterAvailability(
            available=False,
            reason=(
                "openai-codex is installed but no codex CLI binary was found "
                f"(neither codex-cli-bin nor 'codex' on PATH). {_INSTALL_HINT}"
            ),
            fallback_interface="langchain",
        )

    return AdapterAvailability(
        available=True,
        reason=f"Codex SDK available with binary at: {binary_path}",
    )
