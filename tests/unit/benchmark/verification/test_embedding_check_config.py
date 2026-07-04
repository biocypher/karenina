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

    def test_enabled_parameter_exists(self) -> None:
        import inspect

        sig = inspect.signature(perform_embedding_check)
        assert "enabled" in sig.parameters

    def test_threshold_parameter_exists(self) -> None:
        import inspect

        sig = inspect.signature(perform_embedding_check)
        assert "threshold" in sig.parameters

    def test_model_parameter_exists(self) -> None:
        import inspect

        sig = inspect.signature(perform_embedding_check)
        assert "model" in sig.parameters

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
