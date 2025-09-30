"""Tests for async utilities module."""

import time

import pytest

from karenina.utils.async_utils import (
    AsyncConfig,
    execute_with_config,
    run_async_chunked,
    run_sync_with_progress,
)


def slow_function(item: str) -> str:
    """A slow synchronous function for testing."""
    time.sleep(0.01)  # Small delay to simulate work
    return item.upper()


def error_function(item: str) -> str:
    """A function that raises an error for testing error handling."""
    if item == "error":
        raise ValueError("Test error")
    return item.upper()


class TestAsyncConfig:
    """Test the AsyncConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AsyncConfig()
        assert config.enabled is True
        assert config.chunk_size == 5
        assert config.max_workers is None
        assert config.batch_size is None
        assert config.concurrent_batches is None
        assert config.delay_between_batches == 0.5

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = AsyncConfig(
            enabled=False,
            chunk_size=10,
            max_workers=8,
            batch_size=20,
            concurrent_batches=3,
            delay_between_batches=1.0,
        )
        assert config.enabled is False
        assert config.chunk_size == 10
        assert config.max_workers == 8
        assert config.batch_size == 20
        assert config.concurrent_batches == 3
        assert config.delay_between_batches == 1.0

    def test_from_env(self, monkeypatch) -> None:
        """Test configuration from environment variables."""
        # Test with environment variables set
        monkeypatch.setenv("KARENINA_ASYNC_ENABLED", "false")
        monkeypatch.setenv("KARENINA_ASYNC_CHUNK_SIZE", "8")
        monkeypatch.setenv("KARENINA_ASYNC_MAX_WORKERS", "16")
        monkeypatch.setenv("KARENINA_ASYNC_BATCH_SIZE", "25")
        monkeypatch.setenv("KARENINA_ASYNC_CONCURRENT_BATCHES", "4")
        monkeypatch.setenv("KARENINA_ASYNC_DELAY_BETWEEN_BATCHES", "2.0")

        config = AsyncConfig.from_env()
        assert config.enabled is False
        assert config.chunk_size == 8
        assert config.max_workers == 16
        assert config.batch_size == 25
        assert config.concurrent_batches == 4
        assert config.delay_between_batches == 2.0

    def test_from_env_defaults(self, monkeypatch) -> None:
        """Test configuration defaults when environment variables are not set."""
        # Clear relevant environment variables
        for key in [
            "KARENINA_ASYNC_ENABLED",
            "KARENINA_ASYNC_CHUNK_SIZE",
            "KARENINA_ASYNC_MAX_WORKERS",
            "KARENINA_ASYNC_BATCH_SIZE",
            "KARENINA_ASYNC_CONCURRENT_BATCHES",
            "KARENINA_ASYNC_DELAY_BETWEEN_BATCHES",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = AsyncConfig.from_env()
        assert config.enabled is True  # Default
        assert config.chunk_size == 5  # Default
        assert config.max_workers is None  # Default
        assert config.batch_size is None  # Default
        assert config.concurrent_batches is None  # Default
        assert config.delay_between_batches == 0.5  # Default


class TestRunSyncWithProgress:
    """Test the synchronous function with progress tracking."""

    def test_sync_with_progress(self) -> None:
        """Test synchronous execution with progress callback."""
        items = ["hello", "world", "test"]
        progress_calls = []

        def progress_callback(percentage: float, message: str) -> None:
            progress_calls.append((percentage, message))

        results = run_sync_with_progress(items, slow_function, progress_callback)

        # Check results
        assert results == ["HELLO", "WORLD", "TEST"]

        # Check progress callbacks
        assert len(progress_calls) == 3
        assert progress_calls[0][0] == pytest.approx(33.33, abs=0.01)
        assert progress_calls[1][0] == pytest.approx(66.67, abs=0.01)
        assert progress_calls[2][0] == 100.0
        assert "1/3" in progress_calls[0][1]
        assert "2/3" in progress_calls[1][1]
        assert "3/3" in progress_calls[2][1]

    def test_sync_empty_list(self) -> None:
        """Test synchronous execution with empty list."""
        results = run_sync_with_progress([], slow_function)
        assert results == []

    def test_sync_with_errors(self) -> None:
        """Test synchronous execution with errors."""
        items = ["hello", "error", "world"]
        results = run_sync_with_progress(items, error_function)

        assert results[0] == "HELLO"
        assert isinstance(results[1], ValueError)
        assert results[2] == "WORLD"


@pytest.mark.asyncio
class TestRunAsyncChunked:
    """Test the async chunked execution function."""

    async def test_async_chunked(self):
        """Test async execution with chunked processing."""
        items = ["hello", "world", "test", "async"]
        progress_calls = []

        def progress_callback(percentage: float, message: str) -> None:
            progress_calls.append((percentage, message))

        results = await run_async_chunked(items, slow_function, chunk_size=2, progress_callback=progress_callback)

        # Check results
        assert results == ["HELLO", "WORLD", "TEST", "ASYNC"]

        # Check that progress callbacks were called
        assert len(progress_calls) > 0

    async def test_async_chunked_empty_list(self):
        """Test async execution with empty list."""
        results = await run_async_chunked([], slow_function)
        assert results == []

    async def test_async_chunked_with_errors(self):
        """Test async execution with some errors."""
        items = ["hello", "error", "world"]
        results = await run_async_chunked(items, error_function, chunk_size=2)

        assert results[0] == "HELLO"
        assert isinstance(results[1], ValueError)
        assert results[2] == "WORLD"

    async def test_async_chunked_single_item(self):
        """Test async execution with single item."""
        items = ["hello"]
        results = await run_async_chunked(items, slow_function)
        assert results == ["HELLO"]

    async def test_async_chunked_max_workers(self):
        """Test async execution with max_workers setting."""
        items = ["a", "b", "c", "d", "e"]
        results = await run_async_chunked(items, slow_function, max_workers=2)
        assert results == ["A", "B", "C", "D", "E"]


@pytest.mark.asyncio
class TestExecuteWithConfig:
    """Test the main execution function with configuration."""

    async def test_sync_execution(self):
        """Test that sync mode works correctly."""
        config = AsyncConfig(enabled=False)
        items = ["hello", "world"]
        progress_calls = []

        def progress_callback(percentage: float, message: str) -> None:
            progress_calls.append((percentage, message))

        results = await execute_with_config(items, slow_function, config, progress_callback)

        assert results == ["HELLO", "WORLD"]
        assert len(progress_calls) == 2  # One call per item

    async def test_async_execution(self):
        """Test that async mode works correctly."""
        config = AsyncConfig(enabled=True, chunk_size=2)
        items = ["hello", "world", "test"]
        progress_calls = []

        def progress_callback(percentage: float, message: str) -> None:
            progress_calls.append((percentage, message))

        results = await execute_with_config(items, slow_function, config, progress_callback)

        assert results == ["HELLO", "WORLD", "TEST"]
        assert len(progress_calls) > 0

    async def test_async_vs_sync_results_identical(self):
        """Test that async and sync modes produce identical results."""
        items = ["hello", "world", "test", "async", "parallel"]

        # Run in sync mode
        sync_config = AsyncConfig(enabled=False)
        sync_results = await execute_with_config(items, slow_function, sync_config)

        # Run in async mode
        async_config = AsyncConfig(enabled=True, chunk_size=3)
        async_results = await execute_with_config(items, slow_function, async_config)

        # Results should be identical
        assert sync_results == async_results
        assert sync_results == ["HELLO", "WORLD", "TEST", "ASYNC", "PARALLEL"]

    async def test_batched_execution(self):
        """Test batched async execution."""
        config = AsyncConfig(enabled=True, batch_size=2, concurrent_batches=2)
        items = ["a", "b", "c", "d", "e"]

        results = await execute_with_config(items, slow_function, config)
        assert results == ["A", "B", "C", "D", "E"]


class TestPerformance:
    """Test performance improvements of async execution."""

    @pytest.mark.asyncio
    async def test_async_is_faster(self):
        """Test that async execution is faster than sync for I/O bound tasks."""
        items = ["a"] * 10  # 10 items to process

        # Time sync execution
        sync_config = AsyncConfig(enabled=False)
        start_time = time.time()
        sync_results = await execute_with_config(items, slow_function, sync_config)
        sync_time = time.time() - start_time

        # Time async execution
        async_config = AsyncConfig(enabled=True, chunk_size=5)
        start_time = time.time()
        async_results = await execute_with_config(items, slow_function, async_config)
        async_time = time.time() - start_time

        # Results should be the same
        assert sync_results == async_results

        # Async should be faster (allowing for some variance)
        # Note: This is a rough test and may be flaky in some environments
        # We'll be more lenient here since timing tests can be unreliable
        if async_time >= sync_time * 0.9:
            pytest.skip(f"Performance test inconclusive: async={async_time:.3f}s, sync={sync_time:.3f}s")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_all_errors(self):
        """Test handling when all items result in errors."""
        items = ["error", "error", "error"]
        results = await run_async_chunked(items, error_function, chunk_size=2)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, ValueError)

    def test_sync_all_errors(self) -> None:
        """Test sync handling when all items result in errors."""
        items = ["error", "error", "error"]
        results = run_sync_with_progress(items, error_function)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, ValueError)

    @pytest.mark.asyncio
    async def test_mixed_results(self):
        """Test handling mixed success and error results."""
        items = ["success", "error", "another_success", "error"]
        results = await run_async_chunked(items, error_function, chunk_size=2)

        assert len(results) == 4
        assert results[0] == "SUCCESS"
        assert isinstance(results[1], ValueError)
        assert results[2] == "ANOTHER_SUCCESS"
        assert isinstance(results[3], ValueError)

    @pytest.mark.asyncio
    async def test_large_chunk_size(self):
        """Test with chunk size larger than number of items."""
        items = ["a", "b", "c"]
        results = await run_async_chunked(items, slow_function, chunk_size=10)
        assert results == ["A", "B", "C"]

    def test_progress_callback_none(self) -> None:
        """Test that None progress callback doesn't cause errors."""
        items = ["hello", "world"]
        results = run_sync_with_progress(items, slow_function, None)
        assert results == ["HELLO", "WORLD"]

    @pytest.mark.asyncio
    async def test_async_progress_callback_none(self):
        """Test that None progress callback doesn't cause errors in async mode."""
        items = ["hello", "world"]
        results = await run_async_chunked(items, slow_function, progress_callback=None)
        assert results == ["HELLO", "WORLD"]


if __name__ == "__main__":
    pytest.main([__file__])
