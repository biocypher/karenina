"""Adapt AgentPort / LLMPort to the QcAgent protocol used by QcLoop."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from karenina.ports import AgentConfig, Message
from karenina.ports.agent import AgentPort, MCPServerConfig
from karenina.ports.llm import LLMPort
from karenina.ports.messages import Role

from .models import QcTurn, QcUsage
from .timing import ActiveTimeTracker

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig

    from .config import RoleModelConfig

EventSink = Callable[[dict[str, Any]], Awaitable[None] | None]


def mcp_urls_to_servers(mcp_urls_dict: dict[str, str] | None) -> dict[str, MCPServerConfig] | None:
    """Convert ModelConfig.mcp_urls_dict into AgentPort mcp_servers."""
    if not mcp_urls_dict:
        return None
    servers: dict[str, MCPServerConfig] = {}
    for name, url in mcp_urls_dict.items():
        servers[name] = {"type": "http", "url": url}
    return servers


def _usage_from_agent(usage: Any) -> QcUsage | None:
    if usage is None:
        return None
    if hasattr(usage, "to_dict"):
        data = usage.to_dict()
    elif isinstance(usage, dict):
        data = usage
    else:
        data = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
            "cost_usd": getattr(usage, "cost_usd", None),
            "model": getattr(usage, "model", None),
        }
    return QcUsage(
        input_tokens=int(data.get("input_tokens") or 0),
        output_tokens=int(data.get("output_tokens") or 0),
        total_tokens=int(data.get("total_tokens") or 0),
        cost_usd=data.get("cost_usd"),
        model=data.get("model"),
    )


def _trace_messages_to_dicts(messages: list[Any] | None) -> list[dict[str, Any]]:
    if not messages:
        return []
    out: list[dict[str, Any]] = []
    for index, msg in enumerate(messages):
        if hasattr(msg, "to_dict"):
            d = msg.to_dict()
            d["block_index"] = index
            out.append(d)
        elif isinstance(msg, dict):
            out.append(msg)
        else:
            out.append({"role": "unknown", "content": str(msg), "block_index": index})
    return out


def merge_turns(base: QcTurn | None, nxt: QcTurn) -> QcTurn:
    """Merge sequential stages into one role turn (preserve all traces)."""
    if base is None:
        return nxt
    usage = base.usage.merge(nxt.usage) if base.usage else nxt.usage
    wall = None
    if base.wall_time_seconds is not None or nxt.wall_time_seconds is not None:
        wall = (base.wall_time_seconds or 0.0) + (nxt.wall_time_seconds or 0.0)
    active = None
    if base.active_time_seconds is not None or nxt.active_time_seconds is not None:
        active = (base.active_time_seconds or 0.0) + (nxt.active_time_seconds or 0.0)
    tool_t = None
    if base.tool_time_seconds is not None or nxt.tool_time_seconds is not None:
        tool_t = (base.tool_time_seconds or 0.0) + (nxt.tool_time_seconds or 0.0)
    return QcTurn(
        text=nxt.text or base.text,
        error=nxt.error or base.error,
        prompt=base.prompt or nxt.prompt,
        stop_reason=nxt.stop_reason,
        raw_trace="\n\n".join(p for p in (base.raw_trace, nxt.raw_trace) if p),
        trace_messages=[*base.trace_messages, *nxt.trace_messages],
        usage=usage,
        turns=base.turns + nxt.turns,
        limit_reached=base.limit_reached or nxt.limit_reached,
        stage=nxt.stage or base.stage,
        wall_time_seconds=wall,
        active_time_seconds=active,
        tool_time_seconds=tool_t,
        steered=base.steered or nxt.steered,
    )


class AgentPortQcAgent:
    """Wrap an AgentPort as a QC agent with session history and traces."""

    def __init__(
        self,
        agent: AgentPort,
        *,
        system_prompt: str | None = None,
        max_turns: int = 25,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        tools: list[Any] | None = None,
        tool_time_buffer_seconds: float = 600.0,
        exclude_tool_name_substrings: list[str] | None = None,
    ) -> None:
        self._agent = agent
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._mcp_servers = mcp_servers
        self._tools = tools
        self._tool_time_buffer_seconds = tool_time_buffer_seconds
        self._exclude_tool_name_substrings = exclude_tool_name_substrings
        self._history: list[Message] = []
        self._current_task: asyncio.Task[Any] | None = None
        self._last_turn: QcTurn | None = None

    def reset_session(self) -> None:
        self._history = []
        self._last_turn = None
        self._current_task = None

    def snapshot_turn(self) -> QcTurn | None:
        return self._last_turn

    async def cancel_turn(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            await asyncio.gather(self._current_task, return_exceptions=True)

    async def run_turn(
        self,
        prompt: str,
        *,
        active_budget_seconds: float | None = None,
        exclude_tool_time: bool = True,
        event_sink: EventSink | None = None,
        stage: str | None = None,
    ) -> QcTurn:
        self._history.append(Message.user(prompt))
        messages = list(self._history)
        if self._system_prompt:
            # Prefer system as first message if adapter honors it via AgentConfig too
            pass

        wall_timeout: float | None = active_budget_seconds
        if active_budget_seconds is not None and exclude_tool_time:
            wall_timeout = active_budget_seconds + self._tool_time_buffer_seconds

        config = AgentConfig(
            max_turns=self._max_turns,
            system_prompt=self._system_prompt,
            timeout=wall_timeout,
        )

        tracker = ActiveTimeTracker(
            exclude_tool_time=exclude_tool_time,
            tool_name_substrings=self._exclude_tool_name_substrings,
            downstream=event_sink,
        )

        started = time.perf_counter()

        async def _invoke() -> Any:
            return await self._agent.arun(
                messages=messages,
                tools=self._tools,
                mcp_servers=self._mcp_servers,
                config=config,
            )

        task = asyncio.create_task(_invoke())
        self._current_task = task
        finished_in_budget = True
        try:
            if active_budget_seconds is not None:
                # Active budget pauses while tool_activity events are in-flight.
                finished_in_budget = await tracker.wait(task, active_budget_seconds)
                if not finished_in_budget and not task.done() and exclude_tool_time:
                    # Allow extra wall-clock for in-flight / long tool calls.
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(task),
                            timeout=self._tool_time_buffer_seconds,
                        )
                        finished_in_budget = True
                    except TimeoutError:
                        finished_in_budget = False
                if not finished_in_budget and not task.done():
                    await self.cancel_turn()
                    await asyncio.gather(task, return_exceptions=True)
                    wall = time.perf_counter() - started
                    turn = QcTurn(
                        text=(self._last_turn.text if self._last_turn else ""),
                        error="agent did not conclude within active-time budget",
                        prompt=prompt,
                        stop_reason="timeout",
                        raw_trace=self._last_turn.raw_trace if self._last_turn else "",
                        trace_messages=list(self._last_turn.trace_messages) if self._last_turn else [],
                        stage=stage,
                        wall_time_seconds=wall,
                        active_time_seconds=active_budget_seconds,
                        tool_time_seconds=tracker.tool_time_seconds,
                        steered=stage in ("wrap_up", "final_conclusion"),
                    )
                    self._last_turn = turn
                    return turn
            result = await task
        except asyncio.CancelledError:
            wall = time.perf_counter() - started
            turn = QcTurn(
                text="",
                error="agent turn cancelled",
                prompt=prompt,
                stop_reason="cancelled",
                stage=stage,
                wall_time_seconds=wall,
                active_time_seconds=active_budget_seconds,
                tool_time_seconds=tracker.tool_time_seconds,
            )
            self._last_turn = turn
            return turn
        except Exception as exc:  # noqa: BLE001
            wall = time.perf_counter() - started
            turn = QcTurn(
                text="",
                error=str(exc),
                prompt=prompt,
                stop_reason="error",
                stage=stage,
                wall_time_seconds=wall,
                tool_time_seconds=tracker.tool_time_seconds,
            )
            self._last_turn = turn
            return turn
        finally:
            self._current_task = None

        wall = time.perf_counter() - started
        # Emit synthetic tool activity from completed trace (for tool_time accounting)
        for msg in result.trace_messages or []:
            if hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    name = getattr(tc, "name", "") or ""
                    await tracker.emit(
                        {"kind": "tool_activity", "tool_id": getattr(tc, "id", name), "name": name, "status": "completed"}
                    )

        # Extend session history with assistant/tool messages from this run
        # Keep user prompt we already added; append non-user trail from this invocation.
        for msg in result.trace_messages or []:
            if getattr(msg, "role", None) == Role.USER:
                continue
            if isinstance(msg, Message):
                self._history.append(msg)

        usage = _usage_from_agent(result.usage)
        turn = QcTurn(
            text=result.final_response or "",
            prompt=prompt,
            stop_reason="timeout" if (not finished_in_budget or result.limit_reached) else "completed",
            raw_trace=result.raw_trace or "",
            trace_messages=_trace_messages_to_dicts(result.trace_messages),
            usage=usage,
            turns=int(result.turns or 0),
            limit_reached=bool(result.limit_reached),
            stage=stage,
            wall_time_seconds=wall,
            active_time_seconds=max(0.0, wall - tracker.tool_time_seconds)
            if exclude_tool_time
            else wall,
            tool_time_seconds=tracker.tool_time_seconds,
            steered=stage in ("wrap_up", "final_conclusion"),
            error=None,
        )
        if result.limit_reached and not turn.text:
            turn = turn.model_copy(update={"error": "agent hit recursion/turn limit"})
        self._last_turn = turn
        return turn


class LlmPortQcAgent:
    """Wrap an LLMPort as a single-turn QC agent (no tools)."""

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm
        self._history: list[Message] = []
        self._last_turn: QcTurn | None = None

    def reset_session(self) -> None:
        self._history = []
        self._last_turn = None

    def snapshot_turn(self) -> QcTurn | None:
        return self._last_turn

    async def cancel_turn(self) -> None:
        return None

    async def run_turn(
        self,
        prompt: str,
        *,
        active_budget_seconds: float | None = None,
        exclude_tool_time: bool = True,
        event_sink: EventSink | None = None,
        stage: str | None = None,
    ) -> QcTurn:
        del exclude_tool_time, event_sink  # no tools on LLM path
        self._history.append(Message.user(prompt))
        started = time.perf_counter()
        try:
            coro = self._llm.ainvoke(list(self._history))
            if active_budget_seconds is not None:
                response = await asyncio.wait_for(coro, timeout=active_budget_seconds)
            else:
                response = await coro
            text = response.content or ""
            self._history.append(Message.assistant(text))
            usage = _usage_from_agent(getattr(response, "usage", None))
            wall = time.perf_counter() - started
            turn = QcTurn(
                text=text,
                prompt=prompt,
                stop_reason="completed",
                raw_trace=text,
                trace_messages=[
                    Message.user(prompt).to_dict(),
                    Message.assistant(text).to_dict(),
                ],
                usage=usage,
                turns=1,
                stage=stage,
                wall_time_seconds=wall,
                active_time_seconds=wall,
                tool_time_seconds=0.0,
                steered=stage in ("wrap_up", "final_conclusion"),
            )
            self._last_turn = turn
            return turn
        except TimeoutError:
            wall = time.perf_counter() - started
            turn = QcTurn(
                text="",
                error="agent did not conclude within active-time budget",
                prompt=prompt,
                stop_reason="timeout",
                stage=stage,
                wall_time_seconds=wall,
                active_time_seconds=active_budget_seconds,
                tool_time_seconds=0.0,
                steered=stage in ("wrap_up", "final_conclusion"),
            )
            self._last_turn = turn
            return turn
        except Exception as exc:  # noqa: BLE001
            wall = time.perf_counter() - started
            turn = QcTurn(
                text="",
                error=str(exc),
                prompt=prompt,
                stop_reason="error",
                stage=stage,
                wall_time_seconds=wall,
            )
            self._last_turn = turn
            return turn


def build_qc_agent(
    role_config: RoleModelConfig,
    *,
    tool_time_buffer_seconds: float = 600.0,
    exclude_tool_name_substrings: list[str] | None = None,
) -> AgentPortQcAgent | LlmPortQcAgent:
    """Create a QC agent for a role from ModelConfig.

    Uses AgentPort when MCP tools are configured; otherwise LLMPort for
    simpler structured JSON roles without tools.
    """
    from karenina.adapters import get_agent, get_llm

    model: ModelConfig = role_config.model
    use_agent = bool(model.mcp_urls_dict)
    system = role_config.system_prompt_override or model.system_prompt

    if use_agent:
        agent = get_agent(model)
        return AgentPortQcAgent(
            agent,
            system_prompt=system,
            mcp_servers=mcp_urls_to_servers(model.mcp_urls_dict),
            tool_time_buffer_seconds=tool_time_buffer_seconds,
            exclude_tool_name_substrings=exclude_tool_name_substrings,
        )
    llm = get_llm(model)
    return LlmPortQcAgent(llm)
