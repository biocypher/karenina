"""Tests for schema-gated MCP tool argument coercion."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from karenina.adapters.langchain.mcp import (
    _coerce_json_string_tool_args,
    _wrap_json_string_args_tool,
)


class SearchArgs(BaseModel):
    query_strings: list[str]
    query_string: str
    variables: dict[str, Any]


def test_coerces_json_strings_for_schema_container_fields() -> None:
    result = _coerce_json_string_tool_args(
        {
            "query_strings": '["atopic eczema", "Atopic Dermatitides"]',
            "query_string": '["this remains a string"]',
            "variables": '{"efoId": "EFO_0000274"}',
        },
        SearchArgs,
        "search_entities",
    )

    assert result == {
        "query_strings": ["atopic eczema", "Atopic Dermatitides"],
        "query_string": '["this remains a string"]',
        "variables": {"efoId": "EFO_0000274"},
    }


def test_leaves_correct_and_invalid_values_unchanged() -> None:
    original = {
        "query_strings": ["atopic eczema"],
        "query_string": "atopic eczema",
        "variables": "{not json}",
    }

    result = _coerce_json_string_tool_args(original, SearchArgs, "search_entities")

    assert result == original
    assert result is not original


@pytest.mark.asyncio
async def test_wrapped_tool_coerces_before_pydantic_validation() -> None:
    async def search_entities(
        query_strings: list[str],
        query_string: str,
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "query_strings": query_strings,
            "query_string": query_string,
            "variables": variables,
        }

    tool = StructuredTool.from_function(
        coroutine=search_entities,
        name="search_entities",
        description="Search entities.",
        args_schema=SearchArgs,
    )
    wrapped = _wrap_json_string_args_tool(tool)

    assert wrapped.response_format == "content"
    result = await wrapped.ainvoke(
        {
            "query_strings": '["atopic eczema"]',
            "query_string": '["literal string"]',
            "variables": '{"efoId": "EFO_0000274"}',
        }
    )

    assert result == {
        "query_strings": ["atopic eczema"],
        "query_string": '["literal string"]',
        "variables": {"efoId": "EFO_0000274"},
    }
