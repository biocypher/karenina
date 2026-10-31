"""Tests for GoogleGenAISettings: ModelConfig and extra_kwargs parsing."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from karenina.adapters.google_genai.settings import (
    DEFAULT_MAX_TOOL_RESULT_CHARS,
    DEFAULT_REQUEST_TIMEOUT,
    GoogleGenAISettings,
)
from tests.unit.adapters.google_genai._fakes import make_config


def _settings(**overrides: object) -> GoogleGenAISettings:
    return GoogleGenAISettings.from_model_config(make_config(**overrides))


@pytest.mark.unit
class TestBackendDefaults:
    def test_vertex_is_the_default(self) -> None:
        settings = _settings()
        assert settings.backend == "vertex"
        assert settings.location == "global"
        assert settings.project is None
        assert settings.include_thoughts is False
        assert settings.stream is False

    def test_endpoint_base_url_selects_gemini_api(self) -> None:
        settings = _settings(endpoint_base_url="https://poc-ai-gateway.example.org")
        assert settings.backend == "gemini_api"
        assert settings.base_url == "https://poc-ai-gateway.example.org"
        # Thought summaries keep the gateway's silence cutoff away (design D7).
        assert settings.include_thoughts is True
        assert settings.stream is True

    def test_explicit_backend_and_flags_win(self) -> None:
        settings = _settings(extra_kwargs={"backend": "gemini_api", "stream": False, "include_thoughts": False})
        assert settings.backend == "gemini_api"
        assert settings.stream is False
        assert settings.include_thoughts is False

    def test_project_and_location_from_extra_kwargs_and_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _settings(extra_kwargs={"project": "p1", "location": "europe-west4"}).project == "p1"
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        settings = _settings()
        assert settings.project == "env-project"
        assert settings.location == "us-central1"

    def test_api_key_with_vertex_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="endpoint_api_key"):
            _settings(endpoint_api_key=SecretStr("k"))

    def test_api_key_is_kept_out_of_repr(self) -> None:
        settings = _settings(endpoint_base_url="https://gw.example.org", endpoint_api_key=SecretStr("secret-key"))
        assert settings.api_key == "secret-key"
        assert "secret-key" not in repr(settings)


@pytest.mark.unit
class TestModelConfigFields:
    def test_fields_are_mapped(self) -> None:
        settings = _settings(temperature=0.3, max_tokens=2048, system_prompt="Be terse.", request_timeout=45.0)
        assert settings.model_name == "gemini-3.8-flash"
        assert settings.temperature == 0.3
        assert settings.max_output_tokens == 2048
        assert settings.system_prompt == "Be terse."
        assert settings.request_timeout == 45.0

    def test_request_timeout_defaults_when_unset(self) -> None:
        assert _settings(request_timeout=None).request_timeout == DEFAULT_REQUEST_TIMEOUT

    def test_mcp_fields_are_mapped(self) -> None:
        settings = _settings(
            mcp_tool_filter=["a"],
            mcp_tool_description_overrides={"a": "new"},
            mcp_http_timeout=12.0,
            mcp_sse_read_timeout=60.0,
        )
        assert settings.mcp_tool_filter == ("a",)
        assert settings.mcp_tool_description_overrides == {"a": "new"}
        assert settings.mcp_http_timeout == 12.0
        assert settings.mcp_sse_read_timeout == 60.0


@pytest.mark.unit
class TestExtraKwargsValidation:
    def test_unknown_extra_kwargs_keys_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="builtin_tool"):
            _settings(extra_kwargs={"builtin_tool": ["google_search"]})

    def test_bad_backend_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="backend"):
            _settings(extra_kwargs={"backend": "openai"})

    @pytest.mark.parametrize("value", ["google_search", ["web_search"], [1]])
    def test_bad_builtin_tools_are_rejected(self, value: object) -> None:
        with pytest.raises(ValueError, match="builtin_tools"):
            _settings(extra_kwargs={"builtin_tools": value})

    def test_builtin_tools_keep_order_and_drop_repeats(self) -> None:
        settings = _settings(extra_kwargs={"builtin_tools": ["url_context", "google_search", "url_context"]})
        assert settings.builtin_tools == ("url_context", "google_search")

    @pytest.mark.parametrize("value", [-1, True, "5", 2.5])
    def test_bad_search_budget_is_rejected(self, value: object) -> None:
        with pytest.raises(ValueError, match="max_search_queries"):
            _settings(extra_kwargs={"builtin_tools": ["google_search"], "max_search_queries": value})

    def test_thinking_level_is_normalised_and_validated(self) -> None:
        assert _settings(extra_kwargs={"thinking_level": "high"}).thinking_level == "HIGH"
        with pytest.raises(ValueError, match="thinking_level"):
            _settings(extra_kwargs={"thinking_level": "extreme"})

    def test_thinking_level_and_budget_together_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="thinking_level or thinking_budget"):
            _settings(extra_kwargs={"thinking_level": "LOW", "thinking_budget": 1024})

    def test_non_bool_flags_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="stream"):
            _settings(extra_kwargs={"stream": "yes"})

    def test_generate_content_config_passthrough(self) -> None:
        settings = _settings(extra_kwargs={"generate_content_config": {"seed": 7, "top_p": 0.9}})
        assert settings.generate_content_config == {"seed": 7, "top_p": 0.9}

    def test_generate_content_config_reserved_keys_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="tools"):
            _settings(extra_kwargs={"generate_content_config": {"tools": []}})

    def test_generate_content_config_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown GenerateContentConfig fields"):
            _settings(extra_kwargs={"generate_content_config": {"sead": 7}})

    def test_missing_model_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="model_name"):
            _settings(model_name=None)


@pytest.mark.unit
class TestRunOptions:
    def test_defaults_come_from_settings(self) -> None:
        settings = _settings(extra_kwargs={"builtin_tools": ["google_search"], "max_search_queries": 5})
        options = settings.run_options({})
        assert options.builtin_tools == ("google_search",)
        assert options.max_search_queries == 5
        assert options.max_tool_result_chars == DEFAULT_MAX_TOOL_RESULT_CHARS

    def test_agent_config_extra_overrides(self) -> None:
        settings = _settings(extra_kwargs={"builtin_tools": ["google_search"], "max_search_queries": 5})
        options = settings.run_options(
            {"builtin_tools": [], "max_search_queries": None, "max_tool_result_chars": 100, "cache_tools": True}
        )
        assert options.builtin_tools == ()
        assert options.max_search_queries is None
        assert options.max_tool_result_chars == 100

    def test_bad_override_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_tool_result_chars"):
            _settings().run_options({"max_tool_result_chars": 0})


@pytest.mark.unit
class TestThinkingConfig:
    def test_omitted_when_nothing_is_requested(self) -> None:
        assert _settings().thinking_config_kwargs() is None

    def test_includes_requested_fields(self) -> None:
        settings = _settings(extra_kwargs={"include_thoughts": True, "thinking_budget": 512})
        assert settings.thinking_config_kwargs() == {"include_thoughts": True, "thinking_budget": 512}
