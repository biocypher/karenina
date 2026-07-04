"""Tests for global LLM semaphore helpers in the executor module."""

import threading

import pytest

from karenina.benchmark.verification.executor import (
    get_global_llm_semaphore,
    set_global_llm_semaphore,
)


@pytest.mark.unit
class TestGlobalLLMSemaphore:
    """Tests for the global LLM semaphore getter/setter helpers."""

    def teardown_method(self) -> None:
        """Clear the global semaphore after each test."""
        set_global_llm_semaphore(None)

    def test_default_is_none(self) -> None:
        """The semaphore should be None when not explicitly set."""
        assert get_global_llm_semaphore() is None

    def test_set_and_get(self) -> None:
        """Setting a semaphore should make it retrievable via the getter."""
        sem = threading.Semaphore(5)
        set_global_llm_semaphore(sem)
        assert get_global_llm_semaphore() is sem

    def test_clear(self) -> None:
        """Setting to None should clear the semaphore."""
        sem = threading.Semaphore(3)
        set_global_llm_semaphore(sem)
        assert get_global_llm_semaphore() is sem

        set_global_llm_semaphore(None)
        assert get_global_llm_semaphore() is None

    def test_visible_across_threads(self) -> None:
        """The semaphore must be visible from other threads (not thread-local)."""
        sem = threading.Semaphore(2)
        set_global_llm_semaphore(sem)

        observed: list[threading.Semaphore | None] = []
        barrier = threading.Barrier(2)

        def reader() -> None:
            barrier.wait(timeout=5.0)
            observed.append(get_global_llm_semaphore())

        t = threading.Thread(target=reader)
        t.start()
        barrier.wait(timeout=5.0)
        t.join(timeout=5.0)

        assert len(observed) == 1
        assert observed[0] is sem
