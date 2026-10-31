"""Live gate for the codex_sdk adapter against a real vLLM endpoint.

Exercises the full karenina path: get_agent(ModelConfig(interface="codex_sdk"))
with the per-run endpoint shim, a real codex app-server subprocess, and the
live model. Two gates:

1. End-to-end: ask the agent to create a file and list the directory in a
   scratch workspace. PASS requires a shell tool call in raw_trace with
   canonical delimiters, a use/result pair in trace_messages,
   usage.input_tokens > 0, and a non-empty final_response.
2. Timeout salvage: a short-timeout run must return (not raise) with
   timeout_reached=True after interrupting the in-flight turn.

Run from the worktree:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/codex/codex_adapter_gate.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from karenina.adapters.factory import get_agent
from karenina.ports import AgentConfig, Message, Role
from karenina.schemas.config import ModelConfig

BASE_URL = os.environ.get("CODEX_GATE_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("CODEX_GATE_MODEL", "qwen3.5-122b-a10b")
SCRATCH = Path("/tmp/codex-adapter-gate")

PROMPT = (
    "Create a file named hello.txt containing 'hello from the codex adapter' "
    "and then list the directory contents using the shell. Finish with a one "
    "sentence summary of what you did."
)


def _check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" ({detail})" if detail else ""))
    return condition


def _build_model_config() -> ModelConfig:
    return ModelConfig(
        id="codex-gate",
        model_name=MODEL,
        model_provider="openai",
        interface="codex_sdk",
        endpoint_base_url=BASE_URL,
    )


def gate_1_end_to_end() -> bool:
    print("\n=== Gate 1: end-to-end run (create file + list directory) ===")
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    agent = get_agent(_build_model_config())
    t0 = time.time()
    result = agent.run(
        [Message.system("You are a helpful coding agent."), Message.user(PROMPT)],
        config=AgentConfig(timeout=300, workspace_path=SCRATCH),
    )
    print(f"  wall time: {time.time() - t0:.1f}s")
    print(f"  final_response: {result.final_response[:200]!r}")
    print(f"  usage: {result.usage.to_dict()}")
    print(f"  turns: {result.turns}  limit_reached: {result.limit_reached}  timeout_reached: {result.timeout_reached}")
    print("  --- raw_trace (first 1200 chars) ---")
    print("  " + result.raw_trace[:1200].replace("\n", "\n  "))

    use_pairs = []
    for msg in result.trace_messages:
        for tc in msg.tool_calls:
            results = [
                m
                for m in result.trace_messages
                if m.role == Role.TOOL and any(getattr(c, "tool_use_id", None) == tc.id for c in m.content)
            ]
            use_pairs.append((tc.name, tc.id, bool(results)))
    shell_pairs = [p for p in use_pairs if p[0] == "shell" and p[2]]

    hello = SCRATCH / "hello.txt"
    ok = True
    ok &= _check("final_response non-empty", bool(result.final_response.strip()))
    ok &= _check(
        "raw_trace has canonical AI Message delimiter",
        "--- AI Message ---" in result.raw_trace,
    )
    ok &= _check(
        "raw_trace has shell tool call",
        "shell (call_" in result.raw_trace and "Tool Calls:" in result.raw_trace,
    )
    ok &= _check(
        "raw_trace has Tool Message delimiter",
        "--- Tool Message (call_id:" in result.raw_trace,
    )
    ok &= _check(
        "trace_messages has shell use/result pair",
        bool(shell_pairs),
        f"{len(shell_pairs)} pair(s)",
    )
    ok &= _check("usage.input_tokens > 0", result.usage.input_tokens > 0, str(result.usage.input_tokens))
    ok &= _check("usage.output_tokens > 0", result.usage.output_tokens > 0, str(result.usage.output_tokens))
    ok &= _check("hello.txt created in workspace", hello.exists())
    ok &= _check("not flagged as timeout", result.timeout_reached is False)
    return ok


def gate_2_timeout_salvage() -> bool:
    print("\n=== Gate 2: short-timeout run (interrupt + partial trace) ===")
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    agent = get_agent(_build_model_config())
    t0 = time.time()
    # The shell sleep keeps the turn genuinely in flight when the wall-clock
    # timeout fires, forcing the interrupt + partial-salvage path.
    result = agent.run(
        [
            Message.user(
                "First run the shell command 'sleep 60' and wait for it to finish. "
                "Then create a file named done.txt and summarize what you did."
            )
        ],
        config=AgentConfig(timeout=8, workspace_path=SCRATCH),
    )
    wall = time.time() - t0
    print(f"  wall time: {wall:.1f}s")
    print(f"  final_response: {result.final_response[:120]!r}")
    print(f"  raw_trace tail: ...{result.raw_trace[-200:]!r}")
    print(f"  trace_messages: {len(result.trace_messages)}  usage: {result.usage.to_dict()}")

    ok = True
    ok &= _check("timeout_reached is True", result.timeout_reached is True)
    ok &= _check("timeout note in raw_trace", "[Note: Agent timed out" in result.raw_trace)
    ok &= _check("returned within grace window", wall < 60, f"{wall:.1f}s")
    return ok


def main() -> int:
    print(f"endpoint: {BASE_URL}  model: {MODEL}")
    ok1 = gate_1_end_to_end()
    ok2 = gate_2_timeout_salvage()
    print(f"\nRESULT: gate1={'PASS' if ok1 else 'FAIL'} gate2={'PASS' if ok2 else 'FAIL'}")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
