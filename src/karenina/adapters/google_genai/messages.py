"""Message conversion between karenina ``Message`` and google-genai ``Content``.

Mapping (design Section 10):

- System messages become ``system_instruction`` (joined when several).
- User text becomes ``Content(role="user", parts=[Part(text=...)])``.
- Assistant ``TextContent`` becomes a ``model`` text part.
- ``ThinkingContent(thinking, signature)`` becomes
  ``Part(text, thought=True, thought_signature=b64decode(signature))``.
- ``ToolUseContent`` becomes a ``function_call`` part with the sanitised name.
- Tool results become one ``user`` Content of ``function_response`` parts.
- The synthetic ``google_search`` / ``url_context`` pairs are dropped.

Inside an agent run the loop sends the SDK's own ``Content`` objects back,
never re-converted ones, so thought signatures stay exact. This converter
serves incoming messages (prompt assembly and replayed conversations) and
builds the trace from model responses.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from karenina.ports import (
    Content,
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)

from .settings import BUILTIN_TOOL_NAMES
from .tools import sanitize_tool_name

logger = logging.getLogger(__name__)

_UNKNOWN_TOOL_NAME = "unknown_tool"


@dataclass(frozen=True)
class FunctionCallRef:
    """A function call of one model turn, with its trace id.

    Attributes:
        tool_use_id: The id used in the trace (the API's id, or a generated one).
        name: The declared (sanitised) function name the model called.
        args: The call arguments.
        api_id: The API's call id, echoed in the function response when set.
    """

    tool_use_id: str
    name: str
    args: dict[str, Any]
    api_id: str | None


def extract_system_instruction(messages: list[Message]) -> str | None:
    """Join the text of all system messages.

    Args:
        messages: The conversation messages.

    Returns:
        The system texts joined by blank lines, or None when there are none.
    """
    texts = [m.text for m in messages if m.role == Role.SYSTEM and m.text]
    return "\n\n".join(texts) if texts else None


def encode_signature(signature: bytes | None) -> str | None:
    """Base64-encode a thought signature for ``ThinkingContent.signature``."""
    if not signature:
        return None
    return base64.b64encode(signature).decode("ascii")


def decode_signature(signature: str | None) -> bytes | None:
    """Decode a base64 ``ThinkingContent.signature`` back into bytes.

    Returns None (and logs) for a signature that is not valid base64, for
    example one produced by another provider.
    """
    if not signature:
        return None
    try:
        return base64.b64decode(signature, validate=True)
    except (binascii.Error, ValueError):
        logger.debug("Dropping a thinking signature that is not base64")
        return None


def _is_synthetic_tool_use(block: ToolUseContent) -> bool:
    """True for the adapter's synthetic server-side tool entries."""
    return block.name in BUILTIN_TOOL_NAMES and block.id.startswith(f"{block.name}_")


def to_contents(messages: list[Message]) -> list[Any]:
    """Convert karenina messages into google-genai ``Content`` objects.

    System messages are skipped (see ``extract_system_instruction``).
    Consecutive tool messages are merged into one ``user`` Content, as
    Gemini expects all function responses of a turn together.

    Args:
        messages: The conversation messages.

    Returns:
        A list of ``types.Content``.
    """
    from google.genai import types

    contents: list[Any] = []
    tool_names: dict[str, str] = {}
    synthetic_ids: set[str] = set()
    pending_responses: list[Any] = []

    def flush_responses() -> None:
        if pending_responses:
            contents.append(types.Content(role="user", parts=list(pending_responses)))
            pending_responses.clear()

    for msg in messages:
        if msg.role == Role.SYSTEM:
            continue
        if msg.role == Role.TOOL:
            for block in msg.content:
                if not isinstance(block, ToolResultContent) or block.tool_use_id in synthetic_ids:
                    continue
                key = "error" if block.is_error else "result"
                pending_responses.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=block.tool_use_id,
                            name=tool_names.get(block.tool_use_id, _UNKNOWN_TOOL_NAME),
                            response={key: block.content},
                        )
                    )
                )
            continue

        flush_responses()
        if msg.role == Role.USER:
            user_parts = [types.Part(text=b.text) for b in msg.content if isinstance(b, TextContent) and b.text]
            if user_parts:
                contents.append(types.Content(role="user", parts=user_parts))
        elif msg.role == Role.ASSISTANT:
            model_parts: list[Any] = []
            for item in msg.content:
                if isinstance(item, ThinkingContent):
                    model_parts.append(
                        types.Part(text=item.thinking, thought=True, thought_signature=decode_signature(item.signature))
                    )
                elif isinstance(item, TextContent) and item.text:
                    model_parts.append(types.Part(text=item.text))
                elif isinstance(item, ToolUseContent):
                    if _is_synthetic_tool_use(item):
                        synthetic_ids.add(item.id)
                        continue
                    name = sanitize_tool_name(item.name)
                    tool_names[item.id] = name
                    model_parts.append(
                        types.Part(function_call=types.FunctionCall(id=item.id, name=name, args=dict(item.input)))
                    )
            if model_parts:
                contents.append(types.Content(role="model", parts=model_parts))

    flush_responses()
    return contents


def model_parts_to_message(
    parts: list[Any],
    *,
    turn: int,
    name_for: Callable[[str], str] | None = None,
    usage_metadata: dict[str, Any] | None = None,
) -> tuple[Message | None, list[FunctionCallRef]]:
    """Build the trace message of one model turn from its parts.

    Thought parts are joined into one ``ThinkingContent`` (its signature is
    the last one carried by a thought part) and text parts into one
    ``TextContent``, both without separators, as the SDK's ``response.text``
    joins them. Stream chunks deliver fragments of one part, so a separator
    would split words. Function calls become ``ToolUseContent`` with the
    API's id, or ``call_<turn>_<index>`` when the API gives none.

    Args:
        parts: The model's parts for the turn (all stream chunks together).
        turn: The 1-based turn number, used in generated ids.
        name_for: Maps a declared function name back to the original tool
            name for the trace. Identity when None.
        usage_metadata: Per-call usage to attach to the message.

    Returns:
        Tuple of (assistant message or None when the turn has no content,
        function calls in order).
    """
    thoughts: list[str] = []
    signature: bytes | None = None
    texts: list[str] = []
    calls: list[FunctionCallRef] = []
    tool_uses: list[ToolUseContent] = []

    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            api_id = getattr(function_call, "id", None) or None
            declared = getattr(function_call, "name", None) or _UNKNOWN_TOOL_NAME
            args = dict(getattr(function_call, "args", None) or {})
            tool_use_id = api_id or f"call_{turn}_{len(calls)}"
            calls.append(FunctionCallRef(tool_use_id=tool_use_id, name=declared, args=args, api_id=api_id))
            original = name_for(declared) if name_for is not None else declared
            tool_uses.append(ToolUseContent(id=tool_use_id, name=original, input=args))
            continue
        text = getattr(part, "text", None)
        if getattr(part, "thought", None):
            if text:
                thoughts.append(text)
            if getattr(part, "thought_signature", None):
                signature = part.thought_signature
        elif text:
            texts.append(text)

    content: list[Content] = []
    if thoughts:
        content.append(ThinkingContent(thinking="".join(thoughts), signature=encode_signature(signature)))
    if texts:
        content.append(TextContent(text="".join(texts)))
    content.extend(tool_uses)
    if not content:
        return None, calls
    return Message(role=Role.ASSISTANT, content=content, usage_metadata=usage_metadata), calls


def from_contents(contents: list[Any]) -> list[Message]:
    """Convert google-genai ``Content`` objects back into karenina messages.

    The inverse of ``to_contents`` for round trips: model contents become
    assistant messages, function responses become tool messages (matched by
    id, or by the oldest unanswered call of the same name), and user text
    becomes user messages.

    Args:
        contents: ``types.Content`` objects.

    Returns:
        The karenina messages.
    """
    messages: list[Message] = []
    open_calls: list[tuple[str, str]] = []
    for turn, content in enumerate(contents, start=1):
        parts = list(getattr(content, "parts", None) or [])
        if getattr(content, "role", None) == "model":
            message, calls = model_parts_to_message(parts, turn=turn)
            if message is not None:
                messages.append(message)
            open_calls.extend((c.tool_use_id, c.name) for c in calls)
            continue
        user_texts: list[str] = []
        for part in parts:
            response = getattr(part, "function_response", None)
            if response is None:
                if getattr(part, "text", None):
                    user_texts.append(part.text)
                continue
            payload = dict(getattr(response, "response", None) or {})
            is_error = "error" in payload and "result" not in payload
            value = payload.get("error") if is_error else payload.get("result", payload)
            tool_use_id = getattr(response, "id", None) or _match_open_call(open_calls, response.name)
            messages.append(Message.tool_result(tool_use_id=tool_use_id, content=_as_text(value), is_error=is_error))
        if user_texts:
            messages.append(Message.user("".join(user_texts)))
    return messages


def _match_open_call(open_calls: list[tuple[str, str]], name: str | None) -> str:
    """Pop the oldest open call with this name and return its id."""
    for index, (call_id, call_name) in enumerate(open_calls):
        if call_name == name:
            del open_calls[index]
            return call_id
    return f"{name or _UNKNOWN_TOOL_NAME}_response"


def _as_text(value: Any) -> str:
    """Render a function response value as text."""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)
