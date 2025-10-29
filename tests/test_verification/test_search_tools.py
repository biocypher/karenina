"""Unit tests for search tool abstraction layer."""

import pytest

from karenina.benchmark.verification.search_tools import create_search_tool
from karenina.schemas import SearchResultItem


class TestFactoryFunction:
    """Test create_search_tool factory function."""

    def test_factory_with_unknown_string(self) -> None:
        """Test factory raises error for unknown tool name."""
        with pytest.raises(ValueError, match="Unknown built-in search tool"):
            create_search_tool("unknown_tool")

    def test_factory_with_callable(self) -> None:
        """Test factory with simple callable function."""

        def my_search(query: str | list[str]) -> str | list[str]:
            if isinstance(query, list):
                return [f"Results for: {q}" for q in query]
            return f"Results for: {query}"

        search_tool = create_search_tool(my_search)

        # Test single query - now returns list[SearchResultItem]
        result = search_tool("test query")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], SearchResultItem)
        assert result[0].content == "Results for: test query"

        # Test batch queries - now returns list[list[SearchResultItem]]
        results = search_tool(["q1", "q2", "q3"])
        assert isinstance(results, list)
        assert len(results) == 3
        assert isinstance(results[0], list)
        assert results[0][0].content == "Results for: q1"
        assert results[1][0].content == "Results for: q2"
        assert results[2][0].content == "Results for: q3"

    def test_factory_with_langchain_tool_invoke(self) -> None:
        """Test factory with langchain tool that has invoke() method."""

        class MockLangChainTool:
            name = "mock_search"

            def invoke(self, query: str) -> str:
                return f"Search results for: {query}"

        tool = MockLangChainTool()
        search_tool = create_search_tool(tool)

        # Test single query - returns list[SearchResultItem]
        result = search_tool("Python")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "Search results for: Python" in result[0].content

        # Test batch queries - returns list[list[SearchResultItem]]
        results = search_tool(["q1", "q2"])
        assert isinstance(results, list)
        assert len(results) == 2
        assert isinstance(results[0], list)
        assert "Search results for: q1" in results[0][0].content

    def test_factory_with_langchain_tool_run(self) -> None:
        """Test factory with legacy langchain tool that has run() method."""

        class MockLegacyTool:
            name = "legacy_search"

            def run(self, query: str) -> str:
                return f"Legacy search: {query}"

        tool = MockLegacyTool()
        search_tool = create_search_tool(tool)

        result = search_tool("test")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "Legacy search: test" in result[0].content

    def test_factory_with_invalid_tool(self) -> None:
        """Test factory raises error for invalid tool."""

        class InvalidTool:
            pass

        with pytest.raises(ValueError, match="must have 'invoke', 'run', 'ainvoke', 'arun' method, or be callable"):
            create_search_tool(InvalidTool())

    def test_wrapped_tool_handles_exceptions(self) -> None:
        """Test wrapped tool handles exceptions gracefully."""

        class FailingTool:
            def invoke(self, query: str) -> str:  # noqa: ARG002
                raise RuntimeError("API error")

        tool = FailingTool()
        search_tool = create_search_tool(tool)

        # Single query should return empty list on error
        result = search_tool("test")
        assert isinstance(result, list)
        assert len(result) == 0

        # Batch queries should return empty lists for each query
        results = search_tool(["q1", "q2"])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, list) and len(r) == 0 for r in results)

    def test_wrapped_tool_handles_non_string_results(self) -> None:
        """Test wrapped tool handles non-string results."""

        class StructuredResultTool:
            def invoke(self, query: str) -> dict:  # noqa: ARG002
                # Return dict in unrecognized format - should return empty list
                return {"results": ["item1", "item2"]}

        tool = StructuredResultTool()
        search_tool = create_search_tool(tool)

        # Dict is not in recognized format (not list of dicts), should return empty list
        result = search_tool("test")
        assert isinstance(result, list)
        # Unknown format returns empty list
        assert len(result) == 0

    def test_factory_preserves_kwargs_for_builtin_tools(self) -> None:
        """Test factory passes kwargs to built-in tool constructors."""
        # Skip if tavily not available
        try:
            pytest.importorskip("langchain_community.tools.tavily_search")
        except pytest.skip.Exception:
            pytest.skip("Tavily dependencies not available")
            return

        # This should pass kwargs to create_tavily_search_tool
        # If no API key in environment, should raise ValueError
        # If API key present, tool should be created successfully
        try:
            search_tool = create_search_tool("tavily", max_results=5, search_depth="advanced")
            assert callable(search_tool)  # Tool created successfully
        except ValueError as e:
            # Expected if no API key
            assert "API key" in str(e)


class TestSearchToolInterface:
    """Test search tool interface behavior."""

    def test_search_tool_single_query(self) -> None:
        """Test search tool with single query string."""

        def mock_search(query: str | list[str]) -> str | list[str]:
            if isinstance(query, list):
                return [f"Result: {q}" for q in query]
            return f"Result: {query}"

        search_tool = create_search_tool(mock_search)

        result = search_tool("single query")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], SearchResultItem)
        assert result[0].content == "Result: single query"

    def test_search_tool_batch_queries(self) -> None:
        """Test search tool with list of queries."""

        def mock_search(query: str | list[str]) -> str | list[str]:
            if isinstance(query, list):
                return [f"Result: {q}" for q in query]
            return f"Result: {query}"

        search_tool = create_search_tool(mock_search)

        results = search_tool(["q1", "q2", "q3"])
        assert isinstance(results, list)
        assert len(results) == 3
        # Each result is a list of SearchResultItem
        assert isinstance(results[0], list)
        assert results[0][0].content == "Result: q1"
        assert results[1][0].content == "Result: q2"
        assert results[2][0].content == "Result: q3"

    def test_search_tool_with_empty_list(self) -> None:
        """Test search tool with empty query list."""

        def mock_search(query: str | list[str]) -> str | list[str]:
            if isinstance(query, list):
                return [f"Result: {q}" for q in query]
            return f"Result: {query}"

        search_tool = create_search_tool(mock_search)

        results = search_tool([])
        assert isinstance(results, list)
        assert len(results) == 0


class TestTavilySearchTool:
    """Test Tavily search tool (if available)."""

    def test_tavily_tool_creation(self) -> None:
        """Test creating Tavily tool (requires API key)."""
        # Skip if tavily not available
        try:
            pytest.importorskip("langchain_community.tools.tavily_search")
        except pytest.skip.Exception:
            pytest.skip("Tavily dependencies not available")
            return

        # If no API key, should raise error
        # If API key present, tool should be created
        try:
            search_tool = create_search_tool("tavily")
            assert callable(search_tool)  # Tool created successfully with API key from env
        except ValueError as e:
            # Expected if no API key
            assert "API key" in str(e)

    def test_tavily_tool_with_api_key(self) -> None:
        """Test Tavily tool with dummy API key (for interface testing only)."""
        # Skip if tavily not available
        pytest.importorskip("langchain_community.tools.tavily_search")

        # We can't actually test search without a real API key,
        # but we can test that the tool is created correctly
        try:
            search_tool = create_search_tool("tavily", api_key="dummy_key")
            # Tool should be a callable
            assert callable(search_tool)
        except ImportError:
            pytest.skip("Tavily dependencies not available")
