"""Oh My Pi AgentPort adapter using the standard ACP v1 transport."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from karenina.adapters.agent_runtime import get_agent_runtime_access_mode
from karenina.ports import (
    AdapterUnavailableError,
    AgentConfig,
    AgentExecutionError,
    AgentResponseError,
    AgentResult,
    MCPServerConfig,
    Message,
    Role,
    Tool,
)
from karenina.ports.capabilities import PortCapabilities

from .messages import build_omp_prompt, extract_final_response, omp_messages_to_raw_trace

if TYPE_CHECKING:
    from karenina.schemas.config import ModelConfig

logger = logging.getLogger(__name__)

_NO_FINAL_RESPONSE = "[No final response extracted]"
_STOPPED_RESPONSE = "[Agent stopped before producing a final response]"
_DEFAULT_CANCEL_GRACE_SECONDS = 10.0
_PROCESS_EXIT_GRACE_SECONDS = 5.0


class OmpAgentAdapter:
    """Run one isolated OMP ACP session for each AgentPort invocation."""

    def __init__(self, model_config: ModelConfig) -> None:
        self._config = model_config

    @property
    def capabilities(self) -> PortCapabilities:
        access_mode = get_agent_runtime_access_mode(self._config)
        return PortCapabilities(
            supports_system_prompt=True,
            supports_file_tools=True,
            supports_code_execution=access_mode == "read_write",
            uses_sandboxed_execution=False,
        )

    def _extra(self) -> dict[str, Any]:
        return dict(self._config.extra_kwargs or {})

    def _model_selector(self) -> str:
        model_name = str(self._config.model_name)
        if "/" in model_name:
            return model_name
        if not self._config.model_provider:
            raise AgentExecutionError("OMP requires model_provider to build a provider/model selector")
        return f"{self._config.model_provider}/{model_name}"

    def _command(self, access_mode: str) -> list[str]:
        extra = self._extra()
        configured = extra.get("omp_cli_path")
        executable = str(configured) if configured else shutil.which("omp")
        if not executable:
            raise AdapterUnavailableError(
                "Oh My Pi CLI not found. Install it from https://omp.sh/.",
                reason="omp executable not in PATH",
                fallback_interface="langchain",
            )
        command = [executable, "acp"]
        command.extend(["--approval-mode", "yolo" if access_mode == "read_write" else "always-ask"])
        if not bool(extra.get("enable_skills", False)):
            command.append("--no-skills")
        if not bool(extra.get("enable_rules", False)):
            command.append("--no-rules")
        if not bool(extra.get("enable_extensions", False)):
            command.append("--no-extensions")
        extra_args = extra.get("omp_args", [])
        if isinstance(extra_args, str):
            raise AgentExecutionError("OMP extra_kwargs['omp_args'] must be a list of argument strings")
        command.extend(str(arg) for arg in extra_args)
        return command

    async def arun(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Execute an OMP prompt turn through ACP and return its projected trace."""
        if tools:
            raise AgentExecutionError(
                "OMP's ACP v1 adapter cannot accept Karenina Tool definitions; use OMP built-ins or MCP servers"
            )
        config = config or AgentConfig()
        access_mode = get_agent_runtime_access_mode(self._config)
        workspace_owner: tempfile.TemporaryDirectory[str] | None = None
        if config.workspace_path is None:
            workspace_owner = tempfile.TemporaryDirectory(prefix="karenina-omp-")
            workspace = Path(workspace_owner.name)
        else:
            workspace = Path(config.workspace_path).expanduser().resolve()
            if not workspace.is_dir():
                raise AgentExecutionError(f"OMP workspace does not exist or is not a directory: {workspace}")

        try:
            return await self._run_acp(
                messages=messages,
                mcp_servers=mcp_servers,
                config=config,
                access_mode=access_mode,
                workspace=workspace,
            )
        finally:
            if workspace_owner is not None:
                workspace_owner.cleanup()

    async def _run_acp(
        self,
        *,
        messages: list[Message],
        mcp_servers: dict[str, MCPServerConfig] | None,
        config: AgentConfig,
        access_mode: str,
        workspace: Path,
    ) -> AgentResult:
        try:
            from acp import PROTOCOL_VERSION, connect_to_agent, text_block
            from acp.schema import ClientCapabilities, Implementation
        except ImportError as exc:
            raise AdapterUnavailableError(
                "ACP Python client not installed. Install karenina with the 'omp' extra.",
                reason="agent-client-protocol package missing",
                fallback_interface="langchain",
            ) from exc

        from .acp_bridge import OmpAcpClient, usage_from_prompt_response
        from .mcp import convert_mcp_servers

        command = self._command(access_mode)
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(workspace),
            env=os.environ.copy(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            await self._stop_process(process)
            raise AgentExecutionError("OMP ACP subprocess did not expose stdio pipes")

        stderr_task = asyncio.create_task(process.stderr.read())
        client = OmpAcpClient(access_mode=access_mode)
        connection: Any = None
        session_id: str | None = None
        prompt_task: asyncio.Task[Any] | None = None
        response: Any = None
        timeout_reached = False
        run_error: Exception | None = None

        try:
            connection = connect_to_agent(client, process.stdin, process.stdout)
            initialized = await connection.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
                client_info=Implementation(name="karenina", title="Karenina", version="0.1.0"),
            )
            if initialized.protocol_version != PROTOCOL_VERSION:
                raise AgentResponseError(
                    f"ACP protocol mismatch: client supports {PROTOCOL_VERSION}, OMP selected {initialized.protocol_version}"
                )

            session = await connection.new_session(
                cwd=str(workspace),
                mcp_servers=convert_mcp_servers(mcp_servers),
            )
            session_id = session.session_id
            await connection.set_config_option(
                config_id="model",
                session_id=session_id,
                value=self._model_selector(),
            )
            thinking = self._extra().get("thinking") or self._extra().get("thinking_level")
            if thinking:
                await connection.set_config_option(
                    config_id="thinking",
                    session_id=session_id,
                    value=str(thinking),
                )

            prompt = build_omp_prompt(messages, config.system_prompt or self._config.system_prompt)
            if not prompt:
                raise AgentResponseError("OMP prompt is empty after converting Karenina messages")
            prompt_task = asyncio.create_task(connection.prompt(session_id=session_id, prompt=[text_block(prompt)]))
            if config.timeout is None:
                response = await prompt_task
            else:
                try:
                    response = await asyncio.wait_for(asyncio.shield(prompt_task), timeout=config.timeout)
                except TimeoutError:
                    timeout_reached = True
                    await connection.cancel(session_id=session_id)
                    cancel_grace = float(self._extra().get("timeout_cancel_grace", _DEFAULT_CANCEL_GRACE_SECONDS))
                    try:
                        response = await asyncio.wait_for(asyncio.shield(prompt_task), timeout=cancel_grace)
                    except TimeoutError:
                        prompt_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await prompt_task
                        response = SimpleNamespace(stop_reason="cancelled", usage=None)
        except Exception as exc:
            run_error = exc
        finally:
            if connection is not None and session_id is not None:
                with contextlib.suppress(Exception):
                    await connection.close_session(session_id=session_id)
            if connection is not None:
                with contextlib.suppress(Exception):
                    await connection.close()
            await self._stop_process(process)

        stderr = (await stderr_task).decode(errors="replace").strip()
        if run_error is not None:
            if isinstance(run_error, AgentExecutionError | AgentResponseError | AdapterUnavailableError):
                raise run_error
            diagnostic = stderr[-4_000:] if stderr else None
            raise AgentExecutionError(f"OMP ACP execution failed: {run_error}", stderr=diagnostic) from run_error
        if response is None:
            raise AgentResponseError("OMP ACP prompt completed without a response")

        stop_reason = str(response.stop_reason)
        limit_reached = stop_reason in {"max_tokens", "max_turn_requests"}
        trace_messages = [message for message in client.messages if message.role not in (Role.USER, Role.SYSTEM)]
        final_response = extract_final_response(trace_messages)
        if final_response is None:
            final_response = _STOPPED_RESPONSE if (limit_reached or timeout_reached) else _NO_FINAL_RESPONSE
        raw_trace = omp_messages_to_raw_trace(trace_messages).strip()
        if limit_reached:
            raw_trace = f"{raw_trace}\n\n[Note: Agent limit reached - partial response shown]".strip()
        if timeout_reached:
            raw_trace = f"{raw_trace}\n\n[Note: Agent timed out - partial trace shown]".strip()
        turns = sum(
            1 for message in trace_messages if message.role == Role.ASSISTANT and (message.text or message.tool_calls)
        )
        selector = self._model_selector()
        return AgentResult(
            final_response=final_response,
            raw_trace=raw_trace,
            trace_messages=trace_messages,
            usage=usage_from_prompt_response(response, model=selector, cost_usd=client.cost_usd),
            turns=turns,
            limit_reached=limit_reached,
            session_id=session_id,
            actual_model=selector,
            timeout_reached=timeout_reached,
        )

    async def _stop_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=_PROCESS_EXIT_GRACE_SECONDS)
        except TimeoutError:
            process.kill()
            await process.wait()

    def run(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Synchronous wrapper around :meth:`arun`."""
        return asyncio.run(self.arun(messages, tools=tools, mcp_servers=mcp_servers, config=config))

    async def aclose(self) -> None:
        """No-op: each invocation owns and closes its subprocess."""
        return None
