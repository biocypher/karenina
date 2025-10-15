"""Unit tests for search tool abstraction layer."""

import pytest

from karenina.benchmark.verification.search_tools import (
    LangChainSearchToolAdapter,
    SearchResult,
    create_search_tool,
)


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """Test creating a SearchResult."""
        result = SearchResult(
            query="What is Python?",
            results_summary="Python is a programming language...",
        )

        assert result.query == "What is Python?"
        assert "programming language" in result.results_summary
        assert result.raw_results is None

    def test_search_result_with_raw_data(self) -> None:
        """Test SearchResult with raw data."""
        raw_data = {"hits": [{"title": "Python", "url": "https://python.org"}]}
        result = SearchResult(
            query="Python docs",
            results_summary="Found Python documentation",
            raw_results=raw_data,
        )

        assert result.raw_results == raw_data

    def test_search_result_is_frozen(self) -> None:
        """Test that SearchResult is immutable."""
        result = SearchResult(
            query="test",
            results_summary="summary",
        )

        with pytest.raises(AttributeError):  # FrozenInstanceError
            result.results_summary = "new summary"


class TestLangChainSearchToolAdapter:
    """Test LangChainSearchToolAdapter."""

    def test_adapter_with_invoke_method(self) -> None:
        """Test adapter with langchain tool that has invoke() method."""

        class MockLangChainTool:
            name = "mock_search"

            def invoke(self, query: str) -> str:
                return f"Search results for: {query}"

        tool = MockLangChainTool()
        adapter = LangChainSearchToolAdapter(tool)

        result = adapter.search("Python programming")

        assert result.query == "Python programming"
        assert "Search results for: Python programming" in result.results_summary

    def test_adapter_with_run_method(self) -> None:
        """Test adapter with legacy langchain tool that has run() method."""

        class MockLegacyTool:
            name = "legacy_search"

            def run(self, query: str) -> str:
                return f"Legacy search: {query}"

        tool = MockLegacyTool()
        adapter = LangChainSearchToolAdapter(tool)

        result = adapter.search("test query")

        assert result.query == "test query"
        assert "Legacy search: test query" in result.results_summary

    def test_adapter_with_callable(self) -> None:
        """Test adapter with callable tool."""

        def mock_search(query: str) -> str:
            return f"Callable search: {query}"

        adapter = LangChainSearchToolAdapter(mock_search)

        result = adapter.search("test")

        assert result.query == "test"
        assert "Callable search: test" in result.results_summary

    def test_adapter_with_invalid_tool(self) -> None:
        """Test adapter raises error for invalid tool."""

        class InvalidTool:
            pass

        with pytest.raises(ValueError, match="must have 'invoke', 'run' method, or be callable"):
            LangChainSearchToolAdapter(InvalidTool())

    def test_adapter_handles_exceptions(self) -> None:
        """Test adapter handles tool exceptions gracefully."""

        class FailingTool:
            def invoke(self, query: str) -> str:  # noqa: ARG002
                raise RuntimeError("API error")

        tool = FailingTool()
        adapter = LangChainSearchToolAdapter(tool)

        result = adapter.search("test")

        assert result.query == "test"
        assert "Search failed" in result.results_summary

    def test_adapter_handles_non_string_results(self) -> None:
        """Test adapter handles non-string tool results."""

        class StructuredResultTool:
            def invoke(self, query: str) -> dict:  # noqa: ARG002
                return {"results": ["item1", "item2"]}

        tool = StructuredResultTool()
        adapter = LangChainSearchToolAdapter(tool)

        result = adapter.search("test")

        assert result.query == "test"
        assert "results" in result.results_summary.lower()
        assert result.raw_results == {"results": ["item1", "item2"]}


class TestCreateSearchTool:
    """Test create_search_tool factory function."""

    def test_factory_with_string_tavily(self) -> None:
        """Test factory with 'tavily' string (requires actual dependencies)."""
        # This test will be skipped if tavily not installed
        pytest.importorskip("langchain_community.tools.tavily_search")

        # Note: This will raise ValueError if TAVILY_API_KEY not set
        # We'll catch that as expected behavior
        try:
            tool = create_search_tool("tavily")
            # If we have the API key, tool should be created
            assert hasattr(tool, "search")
        except ValueError as e:
            # Expected if API key not set
            assert "API key" in str(e)

    def test_factory_with_unknown_string(self) -> None:
        """Test factory raises error for unknown tool name."""
        with pytest.raises(ValueError, match="Unknown built-in search tool"):
            create_search_tool("unknown_tool")

    def test_factory_with_langchain_tool(self) -> None:
        """Test factory with langchain tool instance."""

        class MockTool:
            name = "custom"

            def invoke(self, query: str) -> str:  # noqa: ARG002
                return "Custom: test"

        custom_tool = MockTool()
        adapter = create_search_tool(custom_tool)

        assert hasattr(adapter, "search")

        result = adapter.search("test")
        assert result.query == "test"
        assert "Custom" in result.results_summary

    def test_factory_with_callable(self) -> None:
        """Test factory with callable function."""

        def my_search(query: str) -> str:
            return f"Function search: {query}"

        adapter = create_search_tool(my_search)

        result = adapter.search("test query")
        assert "Function search: test query" in result.results_summary

    def test_factory_preserves_kwargs_for_builtin_tools(self) -> None:
        """Test factory passes kwargs to built-in tool constructors."""
        # Skip if tavily not available
        pytest.importorskip("langchain_community.tools.tavily_search")

        # This should pass kwargs to TavilySearchTool constructor
        # Will fail if no API key, but that's expected
        with pytest.raises(ValueError, match="API key"):
            create_search_tool("tavily", max_results=5, search_depth="advanced")


class TestProtocolCompliance:
    """Test that implementations conform to SearchTool protocol."""

    def test_adapter_conforms_to_protocol(self) -> None:
        """Test that LangChainSearchToolAdapter conforms to SearchTool protocol."""

        class MockTool:
            def invoke(self, query: str) -> str:  # noqa: ARG002
                return "result"

        tool = MockTool()
        adapter = LangChainSearchToolAdapter(tool)

        # Should have search method
        assert hasattr(adapter, "search")
        assert callable(adapter.search)

        # Should return SearchResult
        result = adapter.search("test")
        assert isinstance(result, SearchResult)

    def test_tavily_tool_conforms_to_protocol(self) -> None:
        """Test that TavilySearchTool conforms to SearchTool protocol (if available)."""
        # Skip if tavily not available
        pytest.importorskip("langchain_community.tools.tavily_search")

        from karenina.benchmark.verification.search_tools_tavily import TavilySearchTool

        # Create tool (will fail without API key, but we're just testing interface)
        try:
            tool = TavilySearchTool(api_key="dummy_key_for_testing")

            # Should have search method
            assert hasattr(tool, "search")
            assert callable(tool.search)

            # We can't test actual search without real API key
            # but we've verified the interface exists

        except ImportError:
            pytest.skip("Tavily dependencies not available")
