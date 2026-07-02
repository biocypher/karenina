"""Tests for the SDK-free codex configuration builders."""

from __future__ import annotations

from karenina.adapters.codex_sdk.config_builder import (
    CODEX_API_KEY_ENV_VAR,
    CODEX_PROVIDER_KEY,
    build_codex_config_overrides,
    build_codex_env,
    build_thread_start_kwargs,
    resolve_model_provider,
    resolve_sandbox_mode,
)
from karenina.schemas.config import ModelConfig


def _keyed_config() -> ModelConfig:
    return ModelConfig(
        id="keyed",
        model_name="qwen3.5-122b-a10b",
        interface="codex_sdk",
        endpoint_base_url="http://example-endpoint:8000/v1",
        endpoint_api_key="super-secret-token",
    )


class TestBuildCodexConfigOverrides:
    def test_endpoint_overrides_use_responses_wire_api(self, endpoint_model_config: ModelConfig) -> None:
        overrides = build_codex_config_overrides(endpoint_model_config, "http://127.0.0.1:9999/v1")
        joined = "\n".join(overrides)
        assert f'model_providers.{CODEX_PROVIDER_KEY}.wire_api="responses"' in overrides
        assert f'model_providers.{CODEX_PROVIDER_KEY}.base_url="http://127.0.0.1:9999/v1"' in overrides
        assert 'wire_api="chat"' not in joined

    def test_base_url_defaults_to_endpoint(self, endpoint_model_config: ModelConfig) -> None:
        overrides = build_codex_config_overrides(endpoint_model_config)
        assert any("http://example-endpoint:8000/v1" in entry for entry in overrides)

    def test_generous_stream_idle_timeout_emitted(self, endpoint_model_config: ModelConfig) -> None:
        overrides = build_codex_config_overrides(endpoint_model_config)
        assert f"model_providers.{CODEX_PROVIDER_KEY}.stream_idle_timeout_ms=600000" in overrides

    def test_native_path_emits_no_overrides(self, native_model_config: ModelConfig) -> None:
        assert build_codex_config_overrides(native_model_config) == ()

    def test_secret_never_in_overrides(self) -> None:
        config = _keyed_config()
        overrides = build_codex_config_overrides(config, "http://127.0.0.1:9999/v1")
        joined = "\n".join(overrides)
        assert "super-secret-token" not in joined
        # The env var is referenced by name only.
        assert f'model_providers.{CODEX_PROVIDER_KEY}.env_key="{CODEX_API_KEY_ENV_VAR}"' in overrides

    def test_keyless_endpoint_sets_no_env_key(self, endpoint_model_config: ModelConfig) -> None:
        overrides = build_codex_config_overrides(endpoint_model_config)
        assert not any("env_key" in entry for entry in overrides)


class TestBuildCodexEnv:
    def test_keyless_endpoint_returns_none(self, endpoint_model_config: ModelConfig) -> None:
        assert build_codex_env(endpoint_model_config) is None

    def test_native_path_returns_none(self, native_model_config: ModelConfig) -> None:
        assert build_codex_env(native_model_config) is None

    def test_keyed_endpoint_carries_secret_in_env_only(self) -> None:
        env = build_codex_env(_keyed_config())
        assert env == {CODEX_API_KEY_ENV_VAR: "super-secret-token"}


class TestResolvers:
    def test_model_provider_for_endpoint(self, endpoint_model_config: ModelConfig) -> None:
        assert resolve_model_provider(endpoint_model_config) == CODEX_PROVIDER_KEY

    def test_model_provider_native_is_none(self, native_model_config: ModelConfig) -> None:
        assert resolve_model_provider(native_model_config) is None

    def test_sandbox_defaults_to_workspace_write(self, endpoint_model_config: ModelConfig) -> None:
        assert resolve_sandbox_mode(endpoint_model_config) == "workspace_write"

    def test_sandbox_read_only_access_mode(self) -> None:
        config = ModelConfig(
            id="ro",
            model_name="m",
            interface="codex_sdk",
            extra_kwargs={"agent_runtime": {"access_mode": "read_only"}},
        )
        assert resolve_sandbox_mode(config) == "read_only"


class TestBuildThreadStartKwargs:
    def test_endpoint_kwargs(self, endpoint_model_config: ModelConfig) -> None:
        kwargs = build_thread_start_kwargs(
            endpoint_model_config,
            base_instructions="Be helpful",
            cwd="/tmp/workspace",
        )
        assert kwargs == {
            "model": "qwen3.5-122b-a10b",
            "model_provider": CODEX_PROVIDER_KEY,
            "cwd": "/tmp/workspace",
            "sandbox": "workspace_write",
            "base_instructions": "Be helpful",
        }

    def test_no_system_prompt_omits_base_instructions(self, native_model_config: ModelConfig) -> None:
        kwargs = build_thread_start_kwargs(native_model_config, base_instructions=None, cwd="/tmp/w")
        assert "base_instructions" not in kwargs
        assert kwargs["model_provider"] is None
