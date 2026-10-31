"""Tests for google_genai tool conversion and execution."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from karenina.adapters.google_genai.tools import (
    Toolset,
    builtin_tools,
    mcp_tool_bindings,
    sanitize_tool_name,
    static_tool_bindings,
    truncate_tool_result,
)
from karenina.ports import AgentExecutionError, Tool


def _mcp_tool(name: str, description: str = "desc", schema: dict[str, Any] | None = None) -> Any:
    return SimpleNamespace(name=name, description=description, inputSchema=schema or {"type": "object"})


def _text(text: str) -> Any:
    return SimpleNamespace(type="text", text=text)


@pytest.mark.unit
class TestSanitizeToolName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("europepmc_search", "europepmc_search"),
            ("fetch.url-v2", "fetch.url-v2"),
            ("search docs", "search_docs"),
            ("ns:tool/x", "ns_tool_x"),
            ("9lives", "_9lives"),
            ("-dash", "_-dash"),
            ("", "_"),
        ],
    )
    def test_names_match_gemini_pattern(self, name: str, expected: str) -> None:
        assert sanitize_tool_name(name) == expected

    def test_long_names_are_cut_to_64(self) -> None:
        assert len(sanitize_tool_name("x" * 100)) == 64


@pytest.mark.unit
class TestToolset:
    def test_reverse_map_and_declarations(self) -> None:
        toolset = Toolset(
            static_tool_bindings([Tool(name="search docs", description="Search", input_schema={"type": "object"})])
        )
        assert toolset.original_name("search_docs") == "search docs"
        assert toolset.original_name("unknown") == "unknown"
        declarations = toolset.function_declaration_tool().function_declarations
        assert [d.name for d in declarations] == ["search_docs"]
        assert declarations[0].parameters_json_schema == {"type": "object"}

    def test_empty_toolset_has_no_declaration_tool(self) -> None:
        toolset = Toolset([])
        assert not toolset
        assert toolset.function_declaration_tool() is None

    def test_duplicate_sanitised_names_are_rejected(self) -> None:
        bindings = static_tool_bindings(
            [
                Tool(name="get data", description="a", input_schema={}),
                Tool(name="get:data", description="b", input_schema={}),
            ]
        )
        with pytest.raises(AgentExecutionError, match="get_data"):
            Toolset(bindings)

    def test_empty_schema_becomes_an_object_schema(self) -> None:
        (binding,) = static_tool_bindings([Tool(name="ping", description="p", input_schema={})])
        assert binding.parameters_json_schema == {"type": "object", "properties": {}}


@pytest.mark.unit
class TestExecute:
    async def test_mcp_call_uses_the_original_name(self) -> None:
        session = SimpleNamespace(
            call_tool=AsyncMock(return_value=SimpleNamespace(content=[_text("a"), _text("b")], isError=False))
        )
        toolset = Toolset(mcp_tool_bindings([("epmc", session, _mcp_tool("search docs"))]))
        text, is_error = await toolset.execute("search_docs", {"q": "x"})
        session.call_tool.assert_awaited_once_with("search docs", {"q": "x"})
        assert (text, is_error) == ("a\nb", False)

    async def test_mcp_is_error_becomes_an_error_result(self) -> None:
        session = SimpleNamespace(
            call_tool=AsyncMock(return_value=SimpleNamespace(content=[_text("not found")], isError=True))
        )
        toolset = Toolset(mcp_tool_bindings([("s", session, _mcp_tool("get"))]))
        assert await toolset.execute("get", {}) == ("not found", True)

    async def test_mcp_exception_becomes_an_error_result(self) -> None:
        session = SimpleNamespace(call_tool=AsyncMock(side_effect=RuntimeError("server gone")))
        toolset = Toolset(mcp_tool_bindings([("s", session, _mcp_tool("get"))]))
        text, is_error = await toolset.execute("get", None)
        assert is_error is True
        assert "server gone" in text

    async def test_non_text_blocks_and_structured_content(self) -> None:
        image = SimpleNamespace(type="image", data="...")
        session = SimpleNamespace(
            call_tool=AsyncMock(
                side_effect=[
                    SimpleNamespace(content=[image], isError=False),
                    SimpleNamespace(content=[], structuredContent={"n": 1}, isError=False),
                ]
            )
        )
        toolset = Toolset(mcp_tool_bindings([("s", session, _mcp_tool("get"))]))
        assert (await toolset.execute("get", {}))[0] == "[image content omitted]"
        assert json.loads((await toolset.execute("get", {}))[0]) == {"n": 1}

    async def test_unknown_tool_is_an_error_result(self) -> None:
        text, is_error = await Toolset([]).execute("nope", {})
        assert is_error is True
        assert "unknown tool" in text

    async def test_static_tool_returns_a_placeholder(self) -> None:
        toolset = Toolset(static_tool_bindings([Tool(name="calc", description="c", input_schema={})]))
        text, is_error = await toolset.execute("calc", {"x": 1})
        assert is_error is False
        assert "calc" in text and "{'x': 1}" in text


@pytest.mark.unit
class TestMcpBindings:
    def test_filter_and_description_overrides(self) -> None:
        session = object()
        bindings = mcp_tool_bindings(
            [
                ("s", session, _mcp_tool("keep", "old")),
                ("s", session, _mcp_tool("drop")),
                ("s", session, _mcp_tool("x", "")),
            ],
            tool_filter=["keep", "x"],
            description_overrides={"keep": "new description"},
        )
        assert [b.original_name for b in bindings] == ["keep", "x"]
        assert bindings[0].description == "new description"
        assert bindings[1].description == "Tool from s"
        assert bindings[0].session is session
        assert bindings[0].server_name == "s"


@pytest.mark.unit
class TestHelpers:
    def test_truncate_tool_result(self) -> None:
        assert truncate_tool_result("short", 10) == "short"
        cut = truncate_tool_result("x" * 25, 10)
        assert cut.startswith("x" * 10)
        assert "[... truncated 15 characters]" in cut

    def test_builtin_tools(self) -> None:
        search, urls = builtin_tools(["google_search", "url_context"])
        assert search.google_search is not None
        assert urls.url_context is not None
        with pytest.raises(ValueError):
            builtin_tools(["code_execution"])
