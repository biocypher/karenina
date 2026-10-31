"""Tests for grounding metadata collection and its synthetic trace pairs."""

from __future__ import annotations

import json
from typing import Any

import pytest
from google.genai import types

from karenina.adapters.google_genai import grounding
from karenina.adapters.google_genai.grounding import (
    GroundingInfo,
    grounding_to_trace,
    merge_candidate_metadata,
    resolve_source_urls,
)
from karenina.ports import Role, ToolResultContent, ToolUseContent
from tests.unit.adapters.google_genai._fakes import search_grounding, url_context


@pytest.mark.unit
class TestMerge:
    def test_single_metadata_keeps_every_query(self) -> None:
        info = merge_candidate_metadata(None, search_grounding(["a", "a", "b"], [("T", "u1")]), None)
        assert info is not None
        assert info.queries == ["a", "a", "b"]
        assert info.sources == [{"title": "T", "uri": "u1"}]

    def test_repeated_chunk_metadata_is_not_counted_twice(self) -> None:
        info = merge_candidate_metadata(None, search_grounding(["a", "b"], [("T", "u1")]), None)
        info = merge_candidate_metadata(info, search_grounding(["a", "b", "c"], [("T", "u1"), ("U", "u2")]), None)
        assert info is not None
        assert info.queries == ["a", "b", "c"]
        assert [s["uri"] for s in info.sources] == ["u1", "u2"]
        assert info.query_count == 3

    def test_supported_segments_are_counted_once(self) -> None:
        support = types.GroundingSupport(segment=types.Segment(start_index=0, end_index=4, text="Text"))
        metadata = types.GroundingMetadata(web_search_queries=["q"], grounding_supports=[support])
        info = merge_candidate_metadata(merge_candidate_metadata(None, metadata, None), metadata, None)
        assert info is not None
        assert info.supported_segments == 1

    def test_url_context_statuses(self) -> None:
        info = merge_candidate_metadata(
            None,
            None,
            url_context(("https://a", "URL_RETRIEVAL_STATUS_SUCCESS"), ("https://b", "URL_RETRIEVAL_STATUS_ERROR")),
        )
        assert info is not None
        assert info.urls == [
            {"url": "https://a", "status": "URL_RETRIEVAL_STATUS_SUCCESS"},
            {"url": "https://b", "status": "URL_RETRIEVAL_STATUS_ERROR"},
        ]

    def test_nothing_reported(self) -> None:
        assert merge_candidate_metadata(None, None, None) is None
        assert merge_candidate_metadata(None, types.GroundingMetadata(), None) is None


@pytest.mark.unit
class TestGroundingToTrace:
    def test_search_and_url_pairs(self) -> None:
        info = GroundingInfo(
            queries=["q1", "q2"],
            sources=[{"title": "Paper", "uri": "https://src"}],
            supported_segments=2,
            urls=[{"url": "https://page", "status": "URL_RETRIEVAL_STATUS_SUCCESS"}],
        )
        search_use, search_result, url_use, url_result = grounding_to_trace(info, turn=3)

        assert search_use.role == Role.ASSISTANT
        assert search_use.content == [
            ToolUseContent(id="google_search_3", name="google_search", input={"queries": ["q1", "q2"]})
        ]
        result_block = search_result.content[0]
        assert isinstance(result_block, ToolResultContent)
        assert result_block.tool_use_id == "google_search_3"
        assert json.loads(result_block.content) == {
            "server_side": True,
            "sources": [{"title": "Paper", "uri": "https://src"}],
            "supported_segments": 2,
        }
        assert url_use.content == [
            ToolUseContent(id="url_context_3", name="url_context", input={"urls": ["https://page"]})
        ]
        assert json.loads(url_result.content[0].content)["urls"][0]["status"] == "URL_RETRIEVAL_STATUS_SUCCESS"

    def test_empty_info_gives_no_messages(self) -> None:
        assert grounding_to_trace(None, 1) == []
        assert grounding_to_trace(GroundingInfo(), 1) == []


@pytest.mark.unit
class TestResolveSourceUrls:
    async def test_redirect_links_are_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        heads: list[str] = []

        class FakeAsyncClient:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> FakeAsyncClient:
                return self

            async def __aexit__(self, *_exc: Any) -> None:
                return None

            async def head(self, url: str) -> Any:
                heads.append(url)
                if url.endswith("broken"):
                    raise ConnectionError("down")
                return type("R", (), {"headers": {"location": "https://real.example/paper"}})()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
        redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc"
        info = GroundingInfo(
            sources=[
                {"title": "A", "uri": redirect},
                {"title": "B", "uri": f"{redirect}broken"},
                {"title": "C", "uri": "https://direct.example"},
            ]
        )
        await resolve_source_urls(info)
        assert heads == [redirect, f"{redirect}broken"]
        assert info.sources[0]["resolved_uri"] == "https://real.example/paper"
        assert "resolved_uri" not in info.sources[1]
        assert "resolved_uri" not in info.sources[2]
        assert grounding._REDIRECT_MARKER in redirect
