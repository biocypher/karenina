"""Tests for with_llm_semaphore decorator in _parallel_base.py.

The semaphore getter is hoisted to _parallel_base module scope (imported
from the async_lifecycle leaf module), so the monkeypatch target is
karenina.adapters._parallel_base.get_global_llm_semaphore: the name the
wrapper actually resolves at call time.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from karenina.adapters._parallel_base import with_llm_semaphore


@pytest.mark.unit
class TestWithLlmSemaphore:
    """Tests for the with_llm_semaphore decorator."""

    def test_passthrough_when_no_semaphore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no global semaphore is set, the function runs directly."""
        monkeypatch.setattr(
            "karenina.adapters._parallel_base.get_global_llm_semaphore",
            lambda: None,
        )

        @with_llm_semaphore
        def my_func(x: int, y: int) -> int:
            return x + y

        assert my_func(3, 4) == 7

    def test_acquires_and_releases_semaphore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When a global semaphore is active, it is acquired before and released after the call."""
        sem = MagicMock(spec=threading.Semaphore)
        monkeypatch.setattr(
            "karenina.adapters._parallel_base.get_global_llm_semaphore",
            lambda: sem,
        )

        @with_llm_semaphore
        def my_func() -> str:
            return "ok"

        result = my_func()

        assert result == "ok"
        sem.acquire.assert_called_once()
        sem.release.assert_called_once()

    def test_releases_semaphore_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Semaphore is released even when the wrapped function raises."""
        sem = MagicMock(spec=threading.Semaphore)
        monkeypatch.setattr(
            "karenina.adapters._parallel_base.get_global_llm_semaphore",
            lambda: sem,
        )

        @with_llm_semaphore
        def my_func() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            my_func()

        sem.acquire.assert_called_once()
        sem.release.assert_called_once()

    def test_backpressure_with_limited_permits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a real semaphore of 1 permit, concurrent calls are serialized."""
        sem = threading.Semaphore(1)
        monkeypatch.setattr(
            "karenina.adapters._parallel_base.get_global_llm_semaphore",
            lambda: sem,
        )

        execution_log: list[tuple[str, str]] = []
        lock = threading.Lock()

        @with_llm_semaphore
        def slow_func(name: str) -> str:
            with lock:
                execution_log.append((name, "start"))
            time.sleep(0.1)
            with lock:
                execution_log.append((name, "end"))
            return name

        threads = []
        results: dict[str, str] = {}

        def run(name: str) -> None:
            results[name] = slow_func(name)

        for n in ("A", "B"):
            t = threading.Thread(target=run, args=(n,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert results["A"] == "A"
        assert results["B"] == "B"

        # Verify serialization: the second task must start after the first ends.
        # With a semaphore of 1, we should never see two "start" events
        # without an intervening "end".
        active = 0
        max_active = 0
        for _name, event in execution_log:
            if event == "start":
                active += 1
            elif event == "end":
                active -= 1
            max_active = max(max_active, active)

        assert max_active == 1, f"Expected max concurrency of 1, got {max_active}"

    def test_preserves_function_metadata(self) -> None:
        """The decorator preserves the wrapped function's name and docstring."""

        @with_llm_semaphore
        def documented_func() -> None:
            """A documented function."""

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "A documented function."

    def test_passes_args_and_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Arguments and keyword arguments are forwarded correctly."""
        monkeypatch.setattr(
            "karenina.adapters._parallel_base.get_global_llm_semaphore",
            lambda: None,
        )

        @with_llm_semaphore
        def my_func(a: int, b: int, *, multiplier: int = 1) -> int:
            return (a + b) * multiplier

        assert my_func(2, 3, multiplier=10) == 50
