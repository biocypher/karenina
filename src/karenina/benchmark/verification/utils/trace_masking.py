"""Mask selected tool-result bodies in a harmonized trace string.

Harmonized traces separate messages with marker lines such as
``--- AI Message ---`` and ``--- Tool Message (call_id: xyz) ---``.
Judge-side analyses can hide bulky machine payloads while retaining the
model's own words and every nonmatching message exactly as stored.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import NamedTuple

_MESSAGE_MARKER_RE = re.compile(r"--- (Tool Message \(call_id: [^\n]*?\)|AI Message|Human Message)[^\n]*---")
_TYPE_DECLARATION_RE = re.compile(r"\btype\s+\w+\s*\{")
_SCHEMA_PLACEHOLDER = "[masked: GraphQL schema introspection response (~{nchars} chars, {ntypes} type declarations)]"


class MaskStats(NamedTuple):
    """Masked trace plus counts describing what changed."""

    text: str
    messages_masked: int
    chars_removed: int


def is_graphql_schema_payload(body: str) -> bool:
    """Return whether a tool-message body looks like a GraphQL schema."""
    if len(_TYPE_DECLARATION_RE.findall(body)) >= 3:
        return True
    return '"__schema"' in body or '"__type"' in body


def _describe_graphql_schema(body: str) -> str:
    """Build a compact description of a GraphQL schema payload."""
    return _SCHEMA_PLACEHOLDER.format(
        nchars=len(body),
        ntypes=len(_TYPE_DECLARATION_RE.findall(body)),
    )


def mask_tool_messages(
    trace: str,
    should_mask: Callable[[str], bool],
    describe: Callable[[str], str],
) -> MaskStats:
    """Replace matching Tool Message bodies with short descriptions.

    Args:
        trace: Harmonized trace text.
        should_mask: Predicate applied to each tool-message body.
        describe: Function that builds the replacement text.

    Returns:
        The transformed trace and counts of masked messages and removed
        characters.
    """
    markers = list(_MESSAGE_MARKER_RE.finditer(trace))
    if not markers:
        return MaskStats(trace, 0, 0)

    output: list[str] = []
    cursor = 0
    messages_masked = 0
    chars_removed = 0
    for index, marker in enumerate(markers):
        body_start = marker.end()
        body_end = markers[index + 1].start() if index + 1 < len(markers) else len(trace)
        body = trace[body_start:body_end]
        output.append(trace[cursor:body_start])
        if marker.group(1).startswith("Tool Message") and should_mask(body):
            placeholder = describe(body)
            trailing = "\n" if body.endswith("\n") else ""
            output.append(placeholder + trailing)
            messages_masked += 1
            chars_removed += len(body) - len(placeholder) - len(trailing)
        else:
            output.append(body)
        cursor = body_end

    return MaskStats("".join(output), messages_masked, chars_removed)


def mask_graphql_schema_messages(trace: str) -> MaskStats:
    """Mask GraphQL schema introspection responses in tool messages."""
    return mask_tool_messages(
        trace,
        should_mask=is_graphql_schema_payload,
        describe=_describe_graphql_schema,
    )
