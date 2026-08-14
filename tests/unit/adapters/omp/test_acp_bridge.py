from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("acp", reason="OMP optional dependency not installed")

from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    PermissionOption,
    PromptResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    Usage,
)

from karenina.adapters.omp.acp_bridge import OmpAcpClient, usage_from_prompt_response
from karenina.ports import Role, ThinkingContent, ToolResultContent


def _text(value: str) -> TextContentBlock:
    return TextContentBlock(type="text", text=value)


def test_acp_updates_coalesce_messages_and_pair_tools() -> None:
    client = OmpAcpClient(access_mode="read_write")

    async def collect() -> None:
        await client.session_update(
            "session",
            AgentThoughtChunk(session_update="agent_thought_chunk", message_id="m1", content=_text("plan ")),
        )
        await client.session_update(
            "session",
            AgentThoughtChunk(session_update="agent_thought_chunk", message_id="m1", content=_text("file")),
        )
        await client.session_update(
            "session",
            ToolCallStart(
                session_update="tool_call",
                tool_call_id="call-1",
                title="read: input.txt",
                kind="read",
                status="pending",
                raw_input={"path": "input.txt"},
            ),
        )
        await client.session_update(
            "session",
            ToolCallProgress(
                session_update="tool_call_update",
                tool_call_id="call-1",
                status="completed",
                raw_output={"content": "marker"},
            ),
        )
        await client.session_update(
            "session",
            AgentMessageChunk(session_update="agent_message_chunk", message_id="m2", content=_text("OMP_")),
        )
        await client.session_update(
            "session",
            AgentMessageChunk(session_update="agent_message_chunk", message_id="m2", content=_text("OK")),
        )

    asyncio.run(collect())

    assert len(client.messages) == 4
    assert isinstance(client.messages[0].content[0], ThinkingContent)
    assert client.messages[0].content[0].thinking == "plan file"
    assert client.messages[1].tool_calls[0].name == "read"
    assert client.messages[1].tool_calls[0].input == {"path": "input.txt"}
    assert client.messages[2].role == Role.TOOL
    assert isinstance(client.messages[2].content[0], ToolResultContent)
    assert client.messages[2].content[0].tool_use_id == "call-1"
    assert client.messages[3].text == "OMP_OK"


def test_usage_maps_cache_separately_from_karenina_total() -> None:
    response = PromptResponse(
        stop_reason="end_turn",
        usage=Usage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=115,
            cached_read_tokens=100,
        ),
    )

    usage = usage_from_prompt_response(response, model="provider/model", cost_usd=0.25)

    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.total_tokens == 15
    assert usage.cache_read_tokens == 100
    assert usage.cost_usd == 0.25


def test_permissions_enforce_read_only_mode() -> None:
    client = OmpAcpClient(access_mode="read_only")
    allow = PermissionOption(option_id="allow", name="Allow", kind="allow_once")
    reject = PermissionOption(option_id="reject", name="Reject", kind="reject_once")

    read = asyncio.run(client.request_permission("s", type("Call", (), {"kind": "read"})(), [allow, reject]))
    execute = asyncio.run(client.request_permission("s", type("Call", (), {"kind": "execute"})(), [allow, reject]))

    assert read.outcome.outcome == "selected"
    assert read.outcome.option_id == "allow"
    assert execute.outcome.outcome == "cancelled"
