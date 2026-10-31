"""Step-0 smoke test: OpenAI Codex Python SDK against a local vLLM endpoint.

Validates that the bundled `codex` app-server can drive a non-interactive
coding turn through a custom OpenAI-compatible model provider (vLLM) before we
build a karenina adapter around it.

Run from the worktree:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/codex/codex_sdk_smoke.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

VLLM_BASE_URL = os.environ.get("CODEX_SMOKE_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = "qwen3.5-122b-a10b"
PROVIDER = "vllm"
SCRATCH = Path("/tmp/codex-smoke")

# CLI -c overrides. Values are parsed as TOML by the codex CLI, so strings need
# embedded double quotes. The original CLI (0.132.0) rejected wire_api="chat";
# it requires wire_api="responses". vLLM serves /v1/responses, so this works.
CONFIG_OVERRIDES = (
    f'model_providers.{PROVIDER}.name="vLLM (local)"',
    f'model_providers.{PROVIDER}.base_url="{VLLM_BASE_URL}"',
    f'model_providers.{PROVIDER}.wire_api="responses"',
    # No env_key => codex will not require/inject an Authorization header.
    # request_max_retries low so failures surface fast during smoke.
    f"model_providers.{PROVIDER}.request_max_retries=2",
    f"model_providers.{PROVIDER}.stream_max_retries=2",
)

PROMPT = (
    "Create a file named hello.txt containing 'hello from codex' and then list the directory contents using the shell."
)


def _reset_scratch() -> None:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True, exist_ok=True)


def _dump(model) -> object:
    if model is None:
        return None
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=True, exclude_none=True, mode="json")
    return model


def _summarize_item(item) -> dict:
    """Item is a ThreadItem RootModel; unwrap .root and pull key fields."""
    inner = item.root if hasattr(item, "root") else item
    out: dict = {"type": getattr(inner, "type", type(inner).__name__)}
    for field in (
        "text",
        "phase",
        "command",
        "aggregated_output",
        "exit_code",
        "status",
        "cwd",
        "duration_ms",
        "summary",
        "content",
        "server",
        "tool",
        "query",
        "changes",
    ):
        if hasattr(inner, field):
            val = getattr(inner, field)
            if val is None:
                continue
            if hasattr(val, "value"):  # enum
                val = val.value
            if isinstance(val, list) and val and hasattr(val[0], "model_dump"):
                val = [_dump(v) for v in val]
            out[field] = val
    return out


def _print_result(label: str, result) -> None:
    print(f"\n===== {label} =====")
    print("turn id:        ", result.id)
    print("status:         ", getattr(result.status, "value", result.status))
    print("error:          ", _dump(result.error))
    print("duration_ms:    ", result.duration_ms)
    print("final_response: ", repr(result.final_response))
    print("n_items:        ", len(result.items))
    for i, item in enumerate(result.items):
        print(f"  item[{i}]: {json.dumps(_summarize_item(item), default=str)[:600]}")
    print("usage:          ", json.dumps(_dump(result.usage), default=str))


def main() -> int:
    _reset_scratch()
    print(f"scratch dir reset: {SCRATCH}")
    print("config_overrides:\n  " + "\n  ".join(CONFIG_OVERRIDES))

    config = CodexConfig(
        config_overrides=CONFIG_OVERRIDES,
        # Try keyless first: no env at all. RUST_LOG kept off unless debugging.
        env=None,
    )

    rc = 0
    codex = Codex(config)
    try:
        print("\nclient metadata:", _dump(codex.metadata))

        thread = codex.thread_start(
            model=MODEL,
            model_provider=PROVIDER,
            sandbox=Sandbox.workspace_write,
            cwd=str(SCRATCH),
            base_instructions="You are a helpful coding agent.",
            approval_mode=ApprovalMode.deny_all,  # non-interactive
        )
        print("thread started:", thread.id)

        # ---- Smoke 1: thread.run() (blocking collect) -------------------
        t0 = time.time()
        result = thread.run(PROMPT)
        dt = time.time() - t0
        print(f"\n[smoke1] thread.run() wall time: {dt:.1f}s")
        _print_result("SMOKE 1 (thread.run)", result)

        hello = SCRATCH / "hello.txt"
        print(f"\nhello.txt exists after smoke1: {hello.exists()}")
        if hello.exists():
            print("hello.txt contents:", repr(hello.read_text()))

        # ---- Smoke 2: thread.turn() + manual .stream() iteration --------
        # Clean file to prove smoke2 re-creates it.
        if hello.exists():
            hello.unlink()
        print("\n[smoke2] starting turn() and iterating .stream() ...")
        t0 = time.time()
        handle = thread.turn(PROMPT)
        print("[smoke2] turn handle id:", handle.id)
        print("[smoke2] handle has .interrupt:", hasattr(handle, "interrupt"))
        notif_methods: list[str] = []
        from openai_codex._run import _collect_turn_result  # noqa: PLC0415

        stream = handle.stream()

        # Tee the stream so we can both observe notifications and collect a
        # TurnResult from the SAME stream (calling handle.run() after manually
        # exhausting .stream() would block: notifications are consumed once).
        def _observing_stream():
            for ev in stream:
                notif_methods.append(ev.method)
                yield ev

        result2 = _collect_turn_result(_observing_stream(), turn_id=handle.id)
        dt2 = time.time() - t0
        print(f"[smoke2] wall time: {dt2:.1f}s")
        print(f"[smoke2] streamed notification methods ({len(notif_methods)}):")
        # Collapse repeated deltas for readability.
        collapsed: list[str] = []
        for m in notif_methods:
            if collapsed and collapsed[-1].split(" x")[0] == m:
                base = collapsed[-1].split(" x")[0]
                n = int(collapsed[-1].split(" x")[1]) if " x" in collapsed[-1] else 1
                collapsed[-1] = f"{base} x{n + 1}"
            else:
                collapsed.append(m)
        for m in collapsed:
            print("   ", m)
        _print_result("SMOKE 2 (turn + stream)", result2)

        print(f"\nhello.txt exists after smoke2: {hello.exists()}")
        if hello.exists():
            print("hello.txt contents:", repr(hello.read_text()))

        ok1 = result.status.value == "completed"
        ok2 = result2.status.value == "completed"
        ok_file = hello.exists()
        print(f"\nRESULT: smoke1_completed={ok1} smoke2_completed={ok2} file_present={ok_file}")
        if not (ok1 and ok2 and ok_file):
            rc = 1
    except Exception as exc:  # noqa: BLE001
        rc = 2
        print(f"\nSMOKE FAILED: {type(exc).__name__}: {exc}")
        # Surface the SDK stderr buffer for diagnosis.
        try:
            tail = codex._client._stderr_tail(80)  # type: ignore[attr-defined]
            print("\n----- codex app-server stderr tail -----")
            print(tail)
        except Exception:  # noqa: BLE001
            pass
        import traceback

        traceback.print_exc()
    finally:
        codex.close()

    print(f"\nexit code: {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
