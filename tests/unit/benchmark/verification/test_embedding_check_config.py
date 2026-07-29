"""Tests for embedding check using VerificationConfig parameters instead of env vars."""

import pytest

from karenina.benchmark.verification.utils.embedding_check import perform_embedding_check
from karenina.schemas.config.models import ModelConfig


@pytest.mark.unit
class TestEmbeddingCheckConfig:
    """perform_embedding_check should accept explicit config parameters."""

    def test_disabled_via_parameter(self) -> None:
        """When enabled=False is passed, check should not run regardless of env vars."""
        parsing_model = ModelConfig(id="gpt-4", interface="langchain", model_name="gpt-4")
        result = perform_embedding_check(
            ground_truth_data={"answer": "yes"},
            llm_response_data={"answer": "yes"},
            parsing_model=parsing_model,
            enabled=False,
        )
        should_override, score, model, performed = result
        assert performed is False
        assert should_override is False

    def test_enabled_false_ignores_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When enabled=False is passed explicitly, EMBEDDING_CHECK env var is ignored."""
        monkeypatch.setenv("EMBEDDING_CHECK", "true")
        parsing_model = ModelConfig(id="gpt-4", interface="langchain", model_name="gpt-4")
        result = perform_embedding_check(
            ground_truth_data={"answer": "yes"},
            llm_response_data={"answer": "yes"},
            parsing_model=parsing_model,
            enabled=False,
        )
        _, _, _, performed = result
        assert performed is False

    def test_explicit_threshold_and_model_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``threshold`` and ``model`` overrides short-circuit before env-var lookups.

        With ``enabled=False`` the function must return ``performed=False``
        without ever reading the EMBEDDING_CHECK_MODEL / EMBEDDING_CHECK_THRESHOLD
        env vars (which would otherwise try to load a sentence-transformer
        model). The overrides are accepted as parameters and never reach the
        env-var fallback path.
        """
        monkeypatch.setenv("EMBEDDING_CHECK_MODEL", "should-not-be-used")
        monkeypatch.setenv("EMBEDDING_CHECK_THRESHOLD", "0.999")
        parsing_model = ModelConfig(id="gpt-4", interface="langchain", model_name="gpt-4")
        should_override, score, model_name, performed = perform_embedding_check(
            ground_truth_data={"answer": "yes"},
            llm_response_data={"answer": "yes"},
            parsing_model=parsing_model,
            enabled=False,
            threshold=0.5,
            model="explicit-model",
        )
        assert performed is False
        # No model was loaded and no score was computed on the disabled path.
        assert score is None
        assert model_name is None
        assert should_override is False
