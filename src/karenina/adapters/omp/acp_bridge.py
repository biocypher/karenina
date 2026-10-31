"""ACP v1 client callbacks and projection into Karenina agent concepts.

This module is the semantic bridge. It deliberately depends on ACP concepts,
not OMP-native events, so its mapping remains auditable against the protocol:
message/thought chunks become assistant content, tool lifecycle updates become
paired tool-use/results, prompt usage becomes UsageMetadata, and permission
requests enforce Karenina's requested access mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from acp import RequestError
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AllowedOutcome,
    DeclineElicitationResponse,
    DeniedOutcome,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
)

from karenina.ports import (
    Message,
    TextContent,
    ThinkingContent,
    ToolUseContent,
    UsageMetadata,
)


@dataclass
class _AssistantState:
    """Mutable stream state for one ACP assistant message id."""

    message: Message
    text: str = ""
    thinking: str = ""

    def refresh(self) -> None:
        content: list[Any] = []
        if self.thinking:
            content.append(ThinkingContent(thinking=self.thinking))
        if self.text:
            content.append(TextContent(text=self.text))
        self.message.content = content


def _jsonable_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(by_alias=True, exclude_none=True, mode="json")
        if isinstance(dumped, dict):
            return dumped
    if value is None:
        return {}
    return {"value": value}


def _serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True, exclude_none=True, mode="json")
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _content_text(content: list[Any] | None) -> str:
    parts: list[str] = []
    for item in content or []:
        nested = getattr(item, "content", None)
        if isinstance(nested, TextContentBlock):
            parts.append(nested.text)
            continue
        if getattr(item, "type", None) == "diff":
            path = getattr(item, "path", "")
            parts.append(f"Updated {path}".strip())
        elif getattr(item, "type", None) == "terminal":
            parts.append(f"Terminal {getattr(item, 'terminal_id', '')}".strip())
    return "\n".join(part for part in parts if part)


class OmpAcpClient:
    """Headless ACP client that records a prompt turn as Karenina messages."""

    def __init__(self, *, access_mode: str) -> None:
        self.access_mode = access_mode
        self.connection: Any = None
        self.messages: list[Message] = []
        self.cost_usd: float | None = None
        self._assistant: dict[str, _AssistantState] = {}
        self._anonymous_message_id = 0
        self._current_anonymous_key: str | None = None
        self._tool_results_emitted: set[str] = set()

    def on_connect(self, connection: Any) -> None:
        self.connection = connection

    async def request_permission(
        self,
        session_id: str,  # noqa: ARG002
        tool_call: Any,
        options: list[Any],
        **kwargs: Any,  # noqa: ARG002
    ) -> RequestPermissionResponse:
        """Allow unattended work according to the configured access mode."""
        kind = getattr(tool_call, "kind", None)
        read_only_kind = kind in {"read", "search", "think", "fetch"}
        may_allow = self.access_mode == "read_write" or read_only_kind
        if may_allow:
            selected = next((option for option in options if option.kind.startswith("allow_")), None)
            if selected is not None:
                return RequestPermissionResponse(
                    outcome=AllowedOutcome(outcome="selected", option_id=selected.option_id)
                )
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Project one ACP session notification into the ordered trace."""
        if isinstance(update, AgentMessageChunk):
            if isinstance(update.content, TextContentBlock):
                state = self._assistant_state(update.message_id)
                state.text += update.content.text
                state.refresh()
            return
        if isinstance(update, AgentThoughtChunk):
            if isinstance(update.content, TextContentBlock):
                state = self._assistant_state(update.message_id)
                state.thinking += update.content.text
                state.refresh()
            return
        if isinstance(update, ToolCallStart):
            self._current_anonymous_key = None
            name = update.kind or "other"
            tool_use = ToolUseContent(
                id=update.tool_call_id,
                name=name,
                input=_jsonable_dict(update.raw_input),
            )
            self.messages.append(Message.assistant(tool_calls=[tool_use]))
            if update.status in {"completed", "failed"}:
                self._append_tool_result(update)
            return
        if isinstance(update, ToolCallProgress):
            if update.status in {"completed", "failed"}:
                self._append_tool_result(update)
            return
        if isinstance(update, UsageUpdate) and update.cost is not None and str(update.cost.currency).upper() == "USD":
            self.cost_usd = float(update.cost.amount)

    def _assistant_state(self, message_id: str | None) -> _AssistantState:
        if message_id:
            key = message_id
        elif self._current_anonymous_key is not None:
            key = self._current_anonymous_key
        else:
            self._anonymous_message_id += 1
            key = f"anonymous-{self._anonymous_message_id}"
            self._current_anonymous_key = key
        state = self._assistant.get(key)
        if state is None:
            message = Message.assistant()
            state = _AssistantState(message=message)
            self._assistant[key] = state
            self.messages.append(message)
        return state

    def _append_tool_result(self, update: Any) -> None:
        call_id = str(update.tool_call_id)
        if call_id in self._tool_results_emitted:
            return
        self._tool_results_emitted.add(call_id)
        rendered = _serialize(getattr(update, "raw_output", None))
        if not rendered:
            rendered = _content_text(getattr(update, "content", None))
        if not rendered:
            rendered = f"Tool {getattr(update, 'status', 'completed')}"
        self.messages.append(
            Message.tool_result(
                tool_use_id=call_id,
                content=rendered,
                is_error=getattr(update, "status", None) == "failed",
            )
        )

    async def write_text_file(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RequestError.method_not_found("fs/read_text_file")

    async def create_terminal(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/kill")

    async def create_elicitation(self, *_args: Any, **_kwargs: Any) -> Any:
        return DeclineElicitationResponse(action="decline")

    async def complete_elicitation(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def ext_method(self, method: str, _params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, _method: str, _params: dict[str, Any]) -> None:
        return None


def usage_from_prompt_response(response: Any, *, model: str, cost_usd: float | None) -> UsageMetadata:
    """Map ACP prompt usage into Karenina's accounting conventions.

    ACP's total includes cache buckets. Karenina consistently defines
    `total_tokens` as input + output and records cache buckets separately.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return UsageMetadata(cost_usd=cost_usd, model=model)
    input_tokens = int(usage.input_tokens)
    output_tokens = int(usage.output_tokens)
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=cost_usd,
        cache_read_tokens=int(usage.cached_read_tokens) if usage.cached_read_tokens is not None else None,
        cache_creation_tokens=int(usage.cached_write_tokens) if usage.cached_write_tokens is not None else None,
        model=model,
    )
