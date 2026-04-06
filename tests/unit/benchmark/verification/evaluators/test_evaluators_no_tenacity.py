"""Tests verifying abstention and sufficiency evaluators have no tenacity decorators.

After the retry-error-harmonization refactor, the adapter (created via
get_llm()) handles retry internally. The evaluator functions should not
have their own tenacity retry decorators or nested retry wrapper functions.
"""

import inspect

import pytest

from karenina.benchmark.verification.evaluators.trace import abstention, sufficiency


@pytest.mark.unit
class TestAbstentionNoTenacity:
    """Verify abstention evaluator has no tenacity retry logic."""

    def test_no_tenacity_imports(self) -> None:
        """The abstention module should not import from tenacity."""
        source = inspect.getsource(abstention)
        assert "from tenacity" not in source
        assert "import tenacity" not in source

    def test_no_retry_decorator_in_detect_abstention(self) -> None:
        """detect_abstention should not contain a @retry decorated inner function."""
        source = inspect.getsource(abstention.detect_abstention)
        assert "@retry" not in source
        assert "_detect_with_retry" not in source

    def test_no_is_retryable_error_import(self) -> None:
        """The abstention module should not import is_retryable_error."""
        source = inspect.getsource(abstention)
        assert "is_retryable_error" not in source

    def test_no_log_retry_import(self) -> None:
        """The abstention module should not import log_retry."""
        source = inspect.getsource(abstention)
        assert "log_retry" not in source


@pytest.mark.unit
class TestSufficiencyNoTenacity:
    """Verify sufficiency evaluator has no tenacity retry logic."""

    def test_no_tenacity_imports(self) -> None:
        """The sufficiency module should not import from tenacity."""
        source = inspect.getsource(sufficiency)
        assert "from tenacity" not in source
        assert "import tenacity" not in source

    def test_no_retry_decorator_in_detect_sufficiency(self) -> None:
        """detect_sufficiency should not contain a @retry decorated inner function."""
        source = inspect.getsource(sufficiency.detect_sufficiency)
        assert "@retry" not in source
        assert "_detect_with_retry" not in source

    def test_no_is_retryable_error_import(self) -> None:
        """The sufficiency module should not import is_retryable_error."""
        source = inspect.getsource(sufficiency)
        assert "is_retryable_error" not in source

    def test_no_log_retry_import(self) -> None:
        """The sufficiency module should not import log_retry."""
        source = inspect.getsource(sufficiency)
        assert "log_retry" not in source
