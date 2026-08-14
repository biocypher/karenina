"""Standalone Claude Code invocation and trace parsing."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_MARKER = "--- ADVERSARIAL SAMPLE ---"


@dataclass(frozen=True, slots=True)
class ClaudeCodeResult:
    """Captured result of one standalone Claude Code session."""

    success: bool
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool = False


def build_claude_command(prompt: str, *, model: str | None) -> list[str]:
    """Build the non-interactive Claude Code command."""
    command = ["claude"]
    if model:
        command.extend(["--model", model])
    command.extend(
        [
            "--dangerously-skip-permissions",
            "--verbose",
            "--output-format",
            "stream-json",
            "-p",
            prompt,
        ]
    )
    return command


def invoke_claude_code(
    prompt: str,
    *,
    model: str | None,
    workdir: Path,
    timeout_seconds: int,
) -> ClaudeCodeResult:
    """Run one independent Claude Code process and capture its JSONL trace."""
    if shutil.which("claude") is None:
        raise FileNotFoundError("The `claude` executable is required for fresh adversarial generation")
    try:
        completed = subprocess.run(
            build_claude_command(prompt, model=model),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=workdir,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or ""
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or ""
        return ClaudeCodeResult(
            success=False,
            stdout=stdout,
            stderr=stderr,
            returncode=None,
            timed_out=True,
        )
    return ClaudeCodeResult(
        success=completed.returncode == 0,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def trace_metadata(raw_trace: str) -> tuple[str | None, str | None, list[dict[str, object]]]:
    """Return the initialized model, session ID, and MCP server records."""
    for line in raw_trace.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            servers = event.get("mcp_servers")
            server_records = (
                [server for server in servers if isinstance(server, dict)]
                if isinstance(servers, list)
                else []
            )
            return (
                str(event["model"]) if event.get("model") else None,
                str(event["session_id"]) if event.get("session_id") else None,
                server_records,
            )
    return None, None, []


def validate_open_targets_mcp(raw_trace: str) -> None:
    """Require a connected server record or an Open Targets MCP tool call."""
    _model, _session_id, servers = trace_metadata(raw_trace)
    for server in servers:
        name = str(server.get("name", "")).casefold()
        if ("open targets" in name or "open-targets" in name) and server.get("status") == "connected":
            return
    for line in raw_trace.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        for block in message.get("content") or []:
            name = str(block.get("name", "")).casefold()
            if block.get("type") == "tool_use" and (
                "open-targets" in name or "open_targets" in name
            ):
                return
    raise RuntimeError(
        "Claude Code trace did not report a connected Open Targets MCP server "
        "or an Open Targets MCP tool call"
    )


def extract_sample_text(raw_trace: str) -> str | None:
    """Recover structured sample text from a Write call or plain output."""
    for line in raw_trace.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        for block in message.get("content") or []:
            if block.get("type") != "tool_use" or "Write" not in str(block.get("name", "")):
                continue
            content = (block.get("input") or {}).get("content") or ""
            if SAMPLE_MARKER in content:
                return str(content)
    if SAMPLE_MARKER in raw_trace:
        return raw_trace[raw_trace.index(SAMPLE_MARKER) :]
    return None


def trace_to_markdown(raw_trace: str) -> str:
    """Render a stream-JSON trace as readable Markdown."""
    sections = ["# Session Trace\n"]
    step = 0
    for line in raw_trace.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if block.get("type") == "text" and str(block.get("text", "")).strip():
                    step += 1
                    sections.append(f"## Step {step}: Assistant\n\n{str(block['text']).strip()}\n")
                elif block.get("type") == "tool_use":
                    rendered = json.dumps(block.get("input") or {}, indent=2, ensure_ascii=False)
                    sections.append(f"### Tool Call: `{block.get('name', 'unknown')}`\n\n```json\n{rendered}\n```\n")
        elif event.get("type") == "tool_result":
            content = event.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, indent=2, ensure_ascii=False)
            if len(content) > 3_000:
                content = content[:3_000] + "\n\n... (truncated, see trace_raw.jsonl for full output)"
            sections.append(f"### Tool Result\n\n```\n{content}\n```\n")
        elif event.get("type") == "result":
            sections.append(
                "\n---\n\n"
                f"**Model**: {event.get('model', 'N/A')} | "
                f"**Cost**: ${event.get('cost_usd', 'N/A')} | "
                f"**Duration**: {event.get('duration_ms', 'N/A')}ms | "
                f"**Turns**: {event.get('num_turns', 'N/A')}\n"
            )
    if step == 0:
        sections.append("## Raw Output\n\n```\n" + raw_trace[:5_000] + "\n```\n")
    return "\n".join(sections)


__all__ = [
    "ClaudeCodeResult",
    "build_claude_command",
    "extract_sample_text",
    "invoke_claude_code",
    "trace_metadata",
    "trace_to_markdown",
    "validate_open_targets_mcp",
]
