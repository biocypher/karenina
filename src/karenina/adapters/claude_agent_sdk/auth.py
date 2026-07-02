"""Authentication env helpers for the Claude Agent SDK adapter.

When the parent ``ModelConfig`` provides no ``anthropic_api_key``, this module
forwards subscription credentials such as ``CLAUDE_CODE_OAUTH_TOKEN`` from the
host environment to the SDK subprocess. This lets users authenticate the SDK
through their Claude subscription instead of an Anthropic API key.

The forwarded variables are only added when no explicit API key is configured,
to keep behavior predictable for callers that already pass credentials through
the configuration object.
"""

from __future__ import annotations

import os

SUBSCRIPTION_AUTH_ENV_VARS: tuple[str, ...] = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_AUTH_TOKEN",
)


def subscription_auth_env() -> dict[str, str]:
    """Return subscription auth env vars present in the host environment.

    Returns:
        Mapping of env variable name to value for every variable in
        :data:`SUBSCRIPTION_AUTH_ENV_VARS` that is set on the host. The mapping
        is empty when none are set.
    """

    return {name: os.environ[name] for name in SUBSCRIPTION_AUTH_ENV_VARS if name in os.environ}
