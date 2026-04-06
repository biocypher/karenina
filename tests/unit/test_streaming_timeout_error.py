"""Tests for StreamingTimeoutError exception class."""

from __future__ import annotations

import pytest

from karenina.exceptions import KareninaError, StreamingTimeoutError
from karenina.utils.errors import ErrorCategory, ErrorRegistry


@pytest.mark.unit
class TestStreamingTimeoutError:
    """Verify StreamingTimeoutError behavior and classification."""

    def test_is_subclass_of_karenina_error(self) -> None:
        assert issubclass(StreamingTimeoutError, KareninaError)

    def test_is_subclass_of_timeout_error(self) -> None:
        assert issubclass(StreamingTimeoutError, TimeoutError)

    def test_stores_message(self) -> None:
        err = StreamingTimeoutError("timed out after 30s")
        assert err.message == "timed out after 30s"
        assert str(err) == "timed out after 30s"

    def test_stores_partial_content(self) -> None:
        err = StreamingTimeoutError("timeout", partial_content="partial response")
        assert err.partial_content == "partial response"

    def test_default_partial_content_is_empty_string(self) -> None:
        err = StreamingTimeoutError("timeout")
        assert err.partial_content == ""

    def test_caught_by_except_timeout_error(self) -> None:
        with pytest.raises(TimeoutError):
            raise StreamingTimeoutError("timed out")

    def test_caught_by_except_karenina_error(self) -> None:
        with pytest.raises(KareninaError):
            raise StreamingTimeoutError("timed out")

    def test_error_registry_classifies_as_timeout(self) -> None:
        registry = ErrorRegistry()
        err = StreamingTimeoutError("streaming timed out")
        category = registry.classify(err)
        assert category == ErrorCategory.TIMEOUT
