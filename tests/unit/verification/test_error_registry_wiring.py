"""Tests for ErrorRegistry wiring through VerificationContext.

Verifies that custom error patterns from VerificationConfig are built into
an ErrorRegistry and made available on VerificationContext for pipeline stages.
"""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.schemas.config import ModelConfig
from karenina.utils.errors import ErrorCategory, ErrorRegistry
from karenina.utils.retry_policy import ErrorPatternConfig


def _minimal_model() -> ModelConfig:
    """Create a minimal ModelConfig for testing."""
    return ModelConfig(id="test-model", model_name="test-model", model_provider="openai")


def _minimal_context(**overrides: object) -> VerificationContext:
    """Create a VerificationContext with minimal required fields."""
    defaults = {
        "question_id": "abc123",
        "template_id": "tpl456",
        "question_text": "What is 2+2?",
        "template_code": "class Answer(BaseAnswer): result: str",
        "answering_model": _minimal_model(),
        "parsing_model": _minimal_model(),
    }
    defaults.update(overrides)
    return VerificationContext(**defaults)


@pytest.mark.unit
class TestVerificationContextErrorRegistry:
    """VerificationContext carries an ErrorRegistry for pipeline stages."""

    def test_default_registry_is_present(self) -> None:
        """Context created without explicit registry has a default ErrorRegistry."""
        ctx = _minimal_context()
        assert isinstance(ctx.error_registry, ErrorRegistry)

    def test_default_registry_classifies_builtins(self) -> None:
        """Default registry still recognizes built-in error patterns."""
        ctx = _minimal_context()
        assert ctx.error_registry.classify(ConnectionError("fail")) == ErrorCategory.CONNECTION
        assert ctx.error_registry.classify(ValueError("bad")) == ErrorCategory.PERMANENT

    def test_custom_registry_is_used(self) -> None:
        """Context accepts an explicit ErrorRegistry with custom patterns."""
        registry = ErrorRegistry()
        registry.register_pattern("my_custom_error", ErrorCategory.RATE_LIMIT)
        ctx = _minimal_context(error_registry=registry)
        assert ctx.error_registry.classify(Exception("my_custom_error hit")) == ErrorCategory.RATE_LIMIT

    def test_mark_error_uses_registry_category(self) -> None:
        """Stages can use context.error_registry to classify before mark_error."""
        registry = ErrorRegistry()
        registry.register_pattern("vllm_overload", ErrorCategory.RATE_LIMIT)
        ctx = _minimal_context(error_registry=registry)

        exc = Exception("vllm_overload detected")
        ctx.mark_error(str(exc), category=ctx.error_registry.classify(exc))

        assert ctx.error_category == ErrorCategory.RATE_LIMIT
        assert ctx.error == "vllm_overload detected"


@pytest.mark.unit
class TestBuildRegistryFromConfig:
    """ErrorRegistry is correctly built from ErrorPatternConfig list."""

    def test_build_from_empty_patterns(self) -> None:
        """Empty pattern list produces a default registry."""
        from karenina.benchmark.verification.runner import _build_error_registry

        registry = _build_error_registry([])
        assert isinstance(registry, ErrorRegistry)
        # Built-in rules still work
        assert registry.classify(ConnectionError("fail")) == ErrorCategory.CONNECTION

    def test_build_from_message_substring_pattern(self) -> None:
        """Message substring patterns are registered correctly."""
        from karenina.benchmark.verification.runner import _build_error_registry

        patterns = [
            ErrorPatternConfig(
                pattern="vllm_backend_error",
                category="server_error",
                match_type="message_substring",
            ),
        ]
        registry = _build_error_registry(patterns)
        assert registry.classify(Exception("vllm_backend_error occurred")) == ErrorCategory.SERVER_ERROR

    def test_build_from_type_name_pattern(self) -> None:
        """Type name patterns are registered correctly."""
        from karenina.benchmark.verification.runner import _build_error_registry

        patterns = [
            ErrorPatternConfig(
                pattern="CustomGPUError",
                category="connection",
                match_type="type_name",
            ),
        ]
        registry = _build_error_registry(patterns)
        exc_class = type("CustomGPUError", (Exception,), {})
        assert registry.classify(exc_class("boom")) == ErrorCategory.CONNECTION

    def test_build_from_multiple_patterns(self) -> None:
        """Multiple patterns of different types are all registered."""
        from karenina.benchmark.verification.runner import _build_error_registry

        patterns = [
            ErrorPatternConfig(pattern="quota_exceeded", category="rate_limit"),
            ErrorPatternConfig(pattern="MyTimeoutError", category="timeout", match_type="type_name"),
        ]
        registry = _build_error_registry(patterns)

        assert registry.classify(Exception("quota_exceeded!")) == ErrorCategory.RATE_LIMIT
        exc_class = type("MyTimeoutError", (Exception,), {})
        assert registry.classify(exc_class("slow")) == ErrorCategory.TIMEOUT
