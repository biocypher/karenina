"""Server-side tool activity: grounding and URL-reading metadata.

Gemini runs Google Search and URL reading inside a single call and reports
them on the candidate (``grounding_metadata`` and
``url_context_metadata``). This module collects that metadata, merging it
across stream chunks, and renders it as synthetic tool-use and tool-result
pairs named ``google_search`` and ``url_context`` (design D8), so the
queries, sources and URL statuses stay auditable in both trace formats.
Page opens are taken from ``url_context_metadata`` only, never from model
text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from karenina.ports import Message, Role, ToolUseContent

from .settings import BUILTIN_GOOGLE_SEARCH, BUILTIN_URL_CONTEXT

logger = logging.getLogger(__name__)

_REDIRECT_MARKER = "grounding-api-redirect"
_RESOLVE_TIMEOUT_SECONDS = 10.0


@dataclass
class GroundingInfo:
    """Server-side tool activity of one model call.

    Attributes:
        queries: Google Search queries, in the order reported.
        sources: Grounding sources as ``{"title", "uri"}`` dicts (plus
            ``resolved_uri`` when redirect links were resolved).
        supported_segments: Number of answer segments backed by a source.
        urls: URL reads as ``{"url", "status"}`` dicts.
    """

    queries: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    supported_segments: int = 0
    urls: list[dict[str, str]] = field(default_factory=list)
    _segment_keys: set[tuple[Any, ...]] = field(default_factory=set, repr=False)

    @property
    def query_count(self) -> int:
        """Number of Google Search queries the call made."""
        return len(self.queries)

    @property
    def is_empty(self) -> bool:
        """True when the call used no server-side tool."""
        return not (self.queries or self.sources or self.urls)


def _enum_name(value: Any) -> str:
    """Name of an SDK enum value, or its string form."""
    name = getattr(value, "name", None)
    return str(name) if name else str(value)


def merge_candidate_metadata(
    info: GroundingInfo | None,
    grounding_metadata: Any,
    url_context_metadata: Any,
) -> GroundingInfo | None:
    """Merge one candidate's (or stream chunk's) metadata into ``info``.

    A stream can repeat metadata on several chunks, so queries already seen
    on an earlier chunk, sources with a known URI, segments with a known
    span and URLs already listed are not added again. Within one metadata
    object every query is kept.

    Args:
        info: The metadata collected so far, or None.
        grounding_metadata: ``candidate.grounding_metadata`` (may be None).
        url_context_metadata: ``candidate.url_context_metadata`` (may be None).

    Returns:
        The merged info, or None when neither object carries anything.
    """
    if grounding_metadata is None and url_context_metadata is None:
        return info
    merged = info if info is not None else GroundingInfo()

    if grounding_metadata is not None:
        seen_queries = set(merged.queries)
        for query in getattr(grounding_metadata, "web_search_queries", None) or []:
            if query not in seen_queries:
                merged.queries.append(query)
        known_uris = {s.get("uri") for s in merged.sources}
        for chunk in getattr(grounding_metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", None) if web is not None else None
            if uri and uri not in known_uris:
                known_uris.add(uri)
                merged.sources.append({"title": getattr(web, "title", None) or "", "uri": uri})
        for support in getattr(grounding_metadata, "grounding_supports", None) or []:
            segment = getattr(support, "segment", None)
            key = (
                getattr(segment, "start_index", None),
                getattr(segment, "end_index", None),
                getattr(segment, "text", None),
            )
            if key not in merged._segment_keys:
                merged._segment_keys.add(key)
                merged.supported_segments += 1

    if url_context_metadata is not None:
        known_urls = {u["url"]: u for u in merged.urls}
        for item in getattr(url_context_metadata, "url_metadata", None) or []:
            url = getattr(item, "retrieved_url", None)
            if not url:
                continue
            status = _enum_name(getattr(item, "url_retrieval_status", None))
            if url in known_urls:
                known_urls[url]["status"] = status
            else:
                entry = {"url": url, "status": status}
                known_urls[url] = entry
                merged.urls.append(entry)

    return None if merged.is_empty and info is None else merged


def grounding_to_trace(info: GroundingInfo | None, turn: int) -> list[Message]:
    """Render server-side tool activity as synthetic tool pairs.

    Args:
        info: The call's grounding info, or None.
        turn: The 1-based turn number, used in the synthetic ids.

    Returns:
        A ``google_search`` pair when the call searched, then a
        ``url_context`` pair when it read URLs. Empty when neither happened.
    """
    if info is None or info.is_empty:
        return []
    messages: list[Message] = []
    if info.queries or info.sources:
        tool_id = f"{BUILTIN_GOOGLE_SEARCH}_{turn}"
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseContent(id=tool_id, name=BUILTIN_GOOGLE_SEARCH, input={"queries": list(info.queries)})],
            )
        )
        payload = {"server_side": True, "sources": info.sources, "supported_segments": info.supported_segments}
        messages.append(Message.tool_result(tool_use_id=tool_id, content=json.dumps(payload)))
    if info.urls:
        tool_id = f"{BUILTIN_URL_CONTEXT}_{turn}"
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=[
                    ToolUseContent(id=tool_id, name=BUILTIN_URL_CONTEXT, input={"urls": [u["url"] for u in info.urls]})
                ],
            )
        )
        payload_urls = {"server_side": True, "urls": info.urls}
        messages.append(Message.tool_result(tool_use_id=tool_id, content=json.dumps(payload_urls)))
    return messages


async def resolve_source_urls(info: GroundingInfo | None) -> None:
    """Follow grounding redirect links to the real source URL, in place.

    One ``HEAD`` request per redirect link, without following redirects.
    The ``Location`` header becomes ``resolved_uri``. Failures are logged and
    leave the source unchanged.

    Args:
        info: The grounding info to update (no-op when None).
    """
    if info is None:
        return
    pending = [s for s in info.sources if _REDIRECT_MARKER in s.get("uri", "") and "resolved_uri" not in s]
    if not pending:
        return
    import httpx

    async with httpx.AsyncClient(follow_redirects=False, timeout=_RESOLVE_TIMEOUT_SECONDS) as http:
        for source in pending:
            try:
                response = await http.head(source["uri"])
            except Exception as e:
                logger.debug("Could not resolve grounding URL %s: %s", source["uri"], e)
                continue
            location = response.headers.get("location")
            if location:
                source["resolved_uri"] = location
