"""Tests for the shared ModelConfig stamping helpers (T16b).

These helpers replaced the near-identical private copies in batch_runner
(_apply_request_timeout / _apply_retry_config) and the benchmark facade
(_apply_timeout / _apply_retry). Semantics under test: only-when-unset
stamping, no-op on None pipeline values, identity preservation when nothing
changes, and no mutation of the original model.
"""

import pytest

from karenina.benchmark.verification.model_stamping import (
    stamp_pipeline_defaults,
    stamp_request_timeout,
    stamp_retry_policy,
)
from karenina.schemas.config import ModelConfig
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy


def _make_model(**overrides) -> ModelConfig:
    return ModelConfig(
        id="m1",
        model_name="m1",
        model_provider="openai",
        interface="langchain",
        **overrides,
    )


def _make_policy(max_attempts: int = 7) -> RetryPolicy:
    return RetryPolicy(timeout=CategoryRetryConfig(max_attempts=max_attempts, backoff_min=1.0, backoff_max=2.0))


@pytest.mark.unit
class TestStampRequestTimeout:
    def test_stamped_when_unset(self) -> None:
        model = _make_model()
        assert model.request_timeout is None
        stamped = stamp_request_timeout(model, 300.0)
        assert stamped.request_timeout == 300.0
        assert stamped is not model

    def test_preserved_when_set(self) -> None:
        model = _make_model(request_timeout=42.0)
        stamped = stamp_request_timeout(model, 300.0)
        assert stamped is model
        assert stamped.request_timeout == 42.0

    def test_none_pipeline_value_is_noop(self) -> None:
        model = _make_model()
        assert stamp_request_timeout(model, None) is model

    def test_original_not_mutated(self) -> None:
        model = _make_model()
        stamp_request_timeout(model, 300.0)
        assert model.request_timeout is None


@pytest.mark.unit
class TestStampRetryPolicy:
    def test_stamped_when_unset(self) -> None:
        model = _make_model()
        policy = _make_policy()
        stamped = stamp_retry_policy(model, policy)
        assert stamped is not model
        assert stamped.retry_policy is policy

    def test_preserved_when_set(self) -> None:
        own_policy = _make_policy(max_attempts=9)
        model = _make_model(retry_policy=own_policy)
        stamped = stamp_retry_policy(model, _make_policy(max_attempts=3))
        assert stamped is model
        assert stamped.retry_policy is own_policy

    def test_none_pipeline_value_is_noop(self) -> None:
        model = _make_model()
        assert stamp_retry_policy(model, None) is model

    def test_original_not_mutated(self) -> None:
        model = _make_model()
        stamp_retry_policy(model, _make_policy())
        assert model.retry_policy is None


@pytest.mark.unit
class TestStampPipelineDefaults:
    def test_both_stamped_when_unset(self) -> None:
        model = _make_model()
        policy = _make_policy()
        stamped = stamp_pipeline_defaults(model, request_timeout=120.0, retry_policy=policy)
        assert stamped.request_timeout == 120.0
        assert stamped.retry_policy is policy
        # Original untouched
        assert model.request_timeout is None
        assert model.retry_policy is None

    def test_partial_stamping(self) -> None:
        """A model with its own timeout still receives the retry policy."""
        model = _make_model(request_timeout=42.0)
        policy = _make_policy()
        stamped = stamp_pipeline_defaults(model, request_timeout=120.0, retry_policy=policy)
        assert stamped.request_timeout == 42.0
        assert stamped.retry_policy is policy

    def test_identity_when_nothing_to_stamp(self) -> None:
        """Both pipeline values None: the exact original instance is returned."""
        model = _make_model()
        assert stamp_pipeline_defaults(model, request_timeout=None, retry_policy=None) is model

    def test_identity_when_all_fields_set(self) -> None:
        model = _make_model(request_timeout=42.0, retry_policy=_make_policy(max_attempts=9))
        stamped = stamp_pipeline_defaults(model, request_timeout=120.0, retry_policy=_make_policy(max_attempts=3))
        assert stamped is model


@pytest.mark.unit
class TestBackCompatAliases:
    """batch_runner keeps the old private names as aliases for existing imports."""

    def test_aliases_point_at_shared_helpers(self) -> None:
        from karenina.benchmark.verification import batch_runner

        assert batch_runner._apply_request_timeout is stamp_request_timeout
        assert batch_runner._apply_retry_config is stamp_retry_policy
