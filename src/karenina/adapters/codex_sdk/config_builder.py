"""Pure configuration builders for the Codex SDK adapter.

These helpers translate karenina's ModelConfig into the configuration
channels the Codex SDK exposes:

- ``CodexConfig.config_overrides``: tuples of ``key=value`` strings that
  become ``codex app-server --config key=value`` CLI arguments. Values are
  parsed as TOML by the codex CLI, so string values need embedded double
  quotes. This is the channel for custom model provider setup.
- ``CodexConfig.env``: environment variables for the spawned app-server
  process. API secrets travel ONLY through this channel, referenced from
  the provider config via ``env_key``. Secrets must never appear inside a
  ``--config`` string.

Everything in this module is SDK-free and side-effect free so it can be
unit tested without the openai-codex package installed.

Custom endpoint note: the codex CLI (0.132.0) only supports
``wire_api="responses"`` (``wire_api="chat"`` was removed). Stock vLLM
serves /v1/responses but rejects codex's request shape, so the adapter
routes custom endpoints through a local rewriting shim (see
``endpoint_shim.py``). The ``base_url`` passed here is therefore usually
the shim's localhost URL, not the raw endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from karenina.adapters.agent_runtime import get_agent_runtime_access_mode

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig

# Provider table key used for custom OpenAI-compatible endpoints.
CODEX_PROVIDER_KEY = "karenina"

# Environment variable that carries the endpoint API key into the codex
# app-server process. Referenced from the provider config via env_key so the
# secret itself never appears in a CLI argument.
CODEX_API_KEY_ENV_VAR = "KARENINA_CODEX_API_KEY"

# Codex turn retries against the model provider. Kept low so endpoint
# failures surface quickly instead of stalling the verification pipeline.
CODEX_REQUEST_MAX_RETRIES = 2
CODEX_STREAM_MAX_RETRIES = 2

# Generous stream idle timeout for slow local models. Codex aborts and
# re-sends the whole request when a stream stays idle past this limit, so
# large models that pause between SSE chunks need ample headroom.
CODEX_STREAM_IDLE_TIMEOUT_MS = 600_000


def resolve_model_provider(model_config: ModelConfig) -> str | None:
    """Return the codex model_provider key for this configuration.

    Returns the custom provider key when a custom endpoint is configured,
    or None for the native path (codex defaults to its built-in OpenAI
    provider and authentication).
    """
    if model_config.endpoint_base_url:
        return CODEX_PROVIDER_KEY
    return None


def resolve_sandbox_mode(model_config: ModelConfig) -> str:
    """Return the codex sandbox mode name for this configuration.

    Maps karenina's agent_runtime access_mode to the codex Sandbox enum
    member name: read_only access maps to ``read_only``, everything else
    maps to ``workspace_write``. ``full_access`` is intentionally never
    produced.
    """
    if get_agent_runtime_access_mode(model_config) == "read_only":
        return "read_only"
    return "workspace_write"


def build_codex_config_overrides(
    model_config: ModelConfig,
    base_url: str | None = None,
) -> tuple[str, ...]:
    """Build CodexConfig.config_overrides for this model configuration.

    Args:
        model_config: The karenina model configuration.
        base_url: The URL codex should send requests to. When the adapter
            runs its endpoint shim this is the shim's localhost URL.
            Defaults to ``model_config.endpoint_base_url``.

    Returns:
        Tuple of ``key=value`` override strings defining a custom model
        provider, or an empty tuple for the native OpenAI path.
    """
    if not model_config.endpoint_base_url:
        return ()

    effective_base_url = base_url or model_config.endpoint_base_url
    provider = CODEX_PROVIDER_KEY
    overrides = [
        f'model_providers.{provider}.name="Karenina custom endpoint"',
        f'model_providers.{provider}.base_url="{effective_base_url}"',
        f'model_providers.{provider}.wire_api="responses"',
        f"model_providers.{provider}.request_max_retries={CODEX_REQUEST_MAX_RETRIES}",
        f"model_providers.{provider}.stream_max_retries={CODEX_STREAM_MAX_RETRIES}",
        f"model_providers.{provider}.stream_idle_timeout_ms={CODEX_STREAM_IDLE_TIMEOUT_MS}",
    ]
    if model_config.endpoint_api_key is not None:
        # Reference the env var by name only. The secret value goes through
        # build_codex_env(), never through a --config string.
        overrides.append(f'model_providers.{provider}.env_key="{CODEX_API_KEY_ENV_VAR}"')
    return tuple(overrides)


def build_codex_env(model_config: ModelConfig) -> dict[str, str] | None:
    """Build the environment dict for the codex app-server process.

    Returns the endpoint API key under CODEX_API_KEY_ENV_VAR when one is
    configured for a custom endpoint, otherwise None. Keyless endpoints
    must not set env_key at all: codex would then require the variable and
    inject an Authorization header.
    """
    if model_config.endpoint_base_url and model_config.endpoint_api_key is not None:
        return {CODEX_API_KEY_ENV_VAR: model_config.endpoint_api_key.get_secret_value()}
    return None


def build_thread_start_kwargs(
    model_config: ModelConfig,
    *,
    base_instructions: str | None,
    cwd: str,
) -> dict[str, Any]:
    """Build SDK-free keyword arguments for AsyncCodex.thread_start().

    The returned dict carries plain values only. The agent adapter maps
    ``sandbox`` (a mode name from resolve_sandbox_mode) to the SDK Sandbox
    enum and adds the approval mode at call time.

    Args:
        model_config: The karenina model configuration.
        base_instructions: System prompt for the thread, or None.
        cwd: Working directory for the thread (workspace boundary).

    Returns:
        Dict with model, model_provider, cwd, sandbox (mode name string),
        and base_instructions (only when set).
    """
    kwargs: dict[str, Any] = {
        "model": model_config.model_name,
        "model_provider": resolve_model_provider(model_config),
        "cwd": cwd,
        "sandbox": resolve_sandbox_mode(model_config),
    }
    if base_instructions:
        kwargs["base_instructions"] = base_instructions
    return kwargs
