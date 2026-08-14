---
name: testing-gate
description: >
  Run the karenina live test gate under karenina/tests/live/: the async
  adapter battery (B1–B11), QA progressive save/resume, scenario progressive
  save/resume, and the docker claude-cli image. Opt in with
  KARENINA_LIVE_TESTS=1; the docker-backed test additionally needs
  KARENINA_LIVE_DOCKER_TESTS=1 and a reachable daemon. Use before merging any
  change to the verification engine, adapters, sinks, or scenario manager, or
  whenever asked to gate/verify that the workflows still run. The live tests
  use only the registered `live` marker. Neutral (non-biological) fixtures.
---

# Testing Gate: the live battery

A **fail-fast gate**, not exhaustive coverage. Each test is the smallest run
that exercises one operation the framework depends on, in a neutral domain
(capital-city trivia / a guardrail-only sycophancy slice) so the gate proves
the machinery, not the biology.

## Offline check before the live gate

Before opting into the live battery, run the zero-cost offline gate: the
`unit`-marked suite in `tests/unit` (marker `unit` = pure logic — no I/O, no
LLM calls):

```bash
cd karenina
uv run pytest tests/unit -q
```

Current scale: ~5300 tests in roughly two minutes (round-1 baseline: 5287
passed, 2 skipped, ~130 s). Most regressions surface here without touching a
live endpoint; reach for `KARENINA_LIVE_TESTS=1` only when the change genuinely
needs generation or judging against a reachable endpoint.

## What is on disk

`karenina/tests/live/` contains exactly:

| Path | Role |
|---|---|
| `test_async_behavior_live.py` | Async adapter battery (B1–B11) against vLLM + dead-port retry telemetry. |
| `test_progressive_save_live.py` | QA progressive save / resume: fresh finalize, triple-level resume skip, multi-model fanout. |
| `test_progressive_save_scenario_live.py` | Scenario progressive save / resume: fresh finalize, combo-level resume skip. |
| `_async_live_helpers.py` | Model builders, retry policies, in-flight counter, teardown markers, and the two gates. |
| `docker/Dockerfile.claude-cli` + `docker/README.md` | Pinned Claude Code CLI image for the claude_agent_sdk container backend. |

There is **no** `conftest.py` in `tests/live/`, **no** `scripts/live_gate.sh`,
and **no** capability-grouped files (`test_generation.py`, `test_offline_gate.py`,
`test_rubric.py`, `test_scenario.py`, `test_save_resume.py`,
`test_engine_features.py`, `test_extension.py`, `test_agentic.py` do **not**
exist). `__pycache__/` still holds stale `.pyc` files from removed tests —
ignore them; they are not importable source.

## The battery

### `test_async_behavior_live.py` — async adapter battery (B1–B11)

Every test here calls `require_adapter(interface)` first, so a missing adapter
dependency skips rather than fails. Fixtures load a 3-question QA subset and a
2-scenario guardrail benchmark; the full runs (B7/B8) assert results complete
*and* that no async-teardown debris ("Event loop is closed", "was never
awaited", "Task was destroyed but it is pending") appears in captured warnings
or logs.

| Test | Proves |
|---|---|
| `test_b1_invoke_and_ainvoke` (param: `openai_endpoint`, `langchain_deep_agents`) | Sync and async invocation both return content and non-zero usage — invoke/ainvoke parity. |
| `test_b2_claude_tool_invoke_and_ainvoke` | `claude_tool` over `anthropic_base_url` returns text despite vLLM emitting thinking blocks. |
| `test_b3_streaming_with_usage` (param) | `astream` accumulates content and (where supported) usage. |
| `test_b3_mid_stream_timeout_exception_shape` | A mid-stream wall-clock timeout raises `StreamingTimeoutError` with a string `partial_content`. |
| `test_b3_mid_stream_timeout_claude_tool_partial_usage` | claude_tool partial usage survives a mid-stream timeout (`input_tokens > 0`). |
| `test_b4_structured_output` (param) | `with_structured_output` returns schema-conformant JSON; adapters vLLM can't support assert a `ParseError` instead. |
| `test_b5_parse_to_pydantic` (param) | `parse_to_pydantic` / `aparse_to_pydantic` extract structured data. |
| `test_b6_concurrency_cap_scenario_batch` | Scenario batch stays within `max_concurrent_requests` (limiter is the bound). |
| `test_b6_concurrency_cap_qa_batch` | QA batch stays within `async_max_workers` (worker pool is the bound). |
| `test_b6_concurrency_cap_qa_batch_limiter` | QA batch where the limiter, not the pool, is the bound — `max_observed <= 2`. |
| `test_b7_qa_run_verification_clean_teardown` | 3 questions through `run_verification` with parallel workers; no teardown debris. |
| `test_b8_scenario_run_clean_teardown` | 2 scenarios end-to-end with the same teardown hygiene assertions. |
| `test_b9_retry_telemetry_dead_port` (param) | Dead-port (`http://127.0.0.1:1`) invocation raises, and retry telemetry matches the documented baseline — no GPU needed. |
| `test_b9_retry_telemetry_claude_sdk_parser_dead_port` | `claude_agent_sdk` parser against a dead port records retries. |
| `test_b10_claude_sdk_container_agent` | Containerized `claude_agent_sdk` agent against vLLM `/v1/messages` (docker-gated). |
| `test_b11_deep_agents_filesystem_agent` | `langchain_deep_agents` agent with the default `FilesystemBackend`, no container; `usage.total_tokens > 0`. |

B6's in-flight caps are observed with `InFlightCounter` + `counted_async()`
from the helpers; B9 uses `dead_port_model(interface)` with a tight
`RetryPolicy` so each attempt costs milliseconds.

### `test_progressive_save_live.py` — QA progressive save / resume

Runs a 3-question `paper_examples/QA` subset against small qwen models
(qwen3.5-a3b on port 8002, qwen3.6-a3b on port 8103).

| Test | Proves |
|---|---|
| `test_live_fresh_run_clean_finalize` | Fresh `run_verification(sink=ProgressiveFileSink(...))` writes the final JSON and deletes sidecars (`state_path` / `jsonl_path` gone, 3 results). |
| `test_live_resume_skips_completed_triples` | A second pass with `skip_triples` populated drops the completed triples and runs nothing (`len(second_set) == 0`). |
| `test_live_multi_model_fanout_resume` | Cross-model triple accounting: complete qwen3.5 first, add qwen3.6, and only the 3 qwen3.6 triples run live. |

### `test_progressive_save_scenario_live.py` — scenario progressive save / resume

Slices 2 sycophancy scenarios from the adversarial checkpoint
(`adversarial/experiment/checkpoints/OT_sycophancy_easy_casual.jsonld`),
rebuilt with a plain guardrail node, run against qwen3.5-a3b.

| Test | Proves |
|---|---|
| `test_scenario_fresh_run_clean_finalize` | Fresh scenario run writes a final export, deletes sidecars, and the sink sees one task per combo (`total_tasks == 2`, `completed_count == 2`). |
| `test_scenario_resume_skips_completed_combos` | A second pass with `skip_triples` populated skips both completed combos (`len(second_set) == 0`). |

### `_async_live_helpers.py`

Shared builders and gates (import from here; do not hand-roll):

- **Model builders** — `openai_model()`, `claude_tool_model()`,
  `deep_agents_model()`, `claude_sdk_container_model()` all target the vLLM
  endpoint (`KARENINA_LIVE_VLLM_URL` / `KARENINA_LIVE_VLLM_MODEL`);
  `dead_port_model(interface)` points at `http://127.0.0.1:1` with a tight
  retry policy for the no-GPU B9 tests.
- **Retry policies** — `tight_retry_policy()` (small budgets, zero backoff) and
  `zero_retry_policy()` (all budgets zero, single attempt).
- **Concurrency** — `InFlightCounter` (thread-safe in-flight counter) and
  `counted_async(fn, counter)` to wrap an adapter method and observe overlap.
- **Teardown hygiene** — `find_teardown_problems(texts)` checks
  `TEARDOWN_ERROR_MARKERS` against captured warnings/logs.
- **Gates** — `require_adapter(interface)` (skips when
  `check_adapter_available(interface)` says the adapter's dependencies are
  missing) and `docker_gate_reason()` (see below).

### `docker/` — the claude-cli image

`Dockerfile.claude-cli` builds `karenina-live-claude-cli:2.1.146`, a minimal
`node:20-slim` image with Claude Code CLI pinned to 2.1.146. The pin matters:
2.1.170+ injects system-role messages that vLLM's `/v1/messages` rejects with
HTTP 400. Bump only with a deliberate, revalidated CLI upgrade. `README.md`
documents the build, the colima daemon fallback, and a `claude --version`
validation step.

## Markers

Markers are registered in `karenina/pyproject.toml`
(`[tool.pytest.ini_options] markers`): `unit`, `integration`, `e2e`, `slow`,
`pipeline`, `rubric`, `storage`, `cli`, `deep_judgment`, `live`,
`paper_archive`. The live tests use **only** `live`. `smoke`, `nogpu`,
`agentic`, and `live_docker` are **not** registered markers — do not reference
them.

## Gating

- **Master opt-in.** Every live module sets
  `pytestmark = [pytest.mark.live, pytest.mark.skipif(os.getenv("KARENINA_LIVE_TESTS") != "1", reason="set KARENINA_LIVE_TESTS=1 to run")]`.
  Without `KARENINA_LIVE_TESTS=1` the whole battery skips.
- **Per-adapter.** `require_adapter(interface)` skips a test when the
  interface's dependencies aren't installed/configured
  (`check_adapter_available`).
- **Docker.** `docker_gate_reason()` (only B10) returns a skip reason unless
  `KARENINA_LIVE_DOCKER_TESTS=1`, the `docker` binary is on PATH, the daemon is
  reachable (`docker info`), and the pinned image is present
  (`docker image inspect`). It returns `None` when the gate is open.

There is **no** endpoint reachability preflight: live tests hit the vLLM
endpoint directly and will *fail* (not skip) if it is down. Only the dead-port
tests (B9) are immune, since they target `http://127.0.0.1:1`.

## Environment

| Var | Default | Purpose |
|---|---|---|
| `KARENINA_LIVE_TESTS` | unset | Must be `1` to run any live test. |
| `KARENINA_LIVE_VLLM_URL` | `http://codon-gpu-001:8000` | Base URL of the vLLM server (async battery). |
| `KARENINA_LIVE_VLLM_MODEL` | `qwen3.5-122b-a10b` | Served model name (async battery). |
| `KARENINA_LIVE_DOCKER_TESTS` | unset | Must be `1` for B10 (plus a daemon and the pinned image). |
| `KARENINA_LIVE_CLAUDE_CLI_IMAGE` | `karenina-live-claude-cli:2.1.146` | Pinned container image for B10. |
| `KARENINA_LIVE_QA_BENCHMARK` | `paper_examples/QA/qa_benchmark.jsonld` | QA benchmark for the async fixtures and save/resume tests. Missing file self-skips. |
| `KARENINA_LIVE_SCENARIO_CHECKPOINT` | `adversarial/experiment/checkpoints/OT_sycophancy_easy_casual.jsonld` | Scenario checkpoint for the scenario fixtures. Missing file self-skips. |
| `KARENINA_ADVERSARIAL_LIB` | `adversarial/experiment` | Directory providing the `rebuild_with_plain_guardrail` scenario helper. |
| `VLLM_QWEN35_URL` / `VLLM_QWEN35_MODEL` | `http://codon-gpu-001:8002` / `qwen3.5-a3b` | Small-model endpoint used by the save/resume tests. |
| `VLLM_QWEN36_URL` / `VLLM_QWEN36_MODEL` | `http://codon-gpu-001:8103` / `qwen3.6-a3b` | Second small-model endpoint for the multi-model fanout. |

## How to run

Always `cd karenina` first (never run `uv` from the monorepo root).

```bash
cd karenina
KARENINA_LIVE_TESTS=1 uv run pytest tests/live -m live -q
```

Per-file iteration:

```bash
cd karenina
KARENINA_LIVE_TESTS=1 uv run pytest tests/live/test_async_behavior_live.py -q
KARENINA_LIVE_TESTS=1 uv run pytest tests/live/test_progressive_save_live.py -q
KARENINA_LIVE_TESTS=1 uv run pytest tests/live/test_progressive_save_scenario_live.py -q
```

A single test:

```bash
cd karenina
KARENINA_LIVE_TESTS=1 uv run pytest tests/live/test_async_behavior_live.py::test_b7_qa_run_verification_clean_teardown -q
```

Docker-backed B10 (needs a daemon + the pinned image — see
`tests/live/docker/README.md` for build and colima):

```bash
cd karenina
KARENINA_LIVE_TESTS=1 KARENINA_LIVE_DOCKER_TESTS=1 \
  uv run pytest tests/live/test_async_behavior_live.py::test_b10_claude_sdk_container_agent -q
```

## Reading a failure

- **B1/B2**: a run that should produce content and `usage.total_tokens > 0`
  returning zero means the invoke path (or usage capture) regressed.
- **B3 sub-cases**: `StreamingTimeoutError` must carry a string
  `partial_content`; claude_tool must still report `input_tokens > 0` after a
  mid-stream timeout. A failure means stream timeout shape or partial-usage
  capture regressed.
- **B4/B5**: schema-conformant extraction regressed, or an adapter vLLM can't
  support is no longer surfacing the expected `ParseError`.
- **B6**: `max_observed > 2` means the concurrency cap (limiter or worker pool)
  is no longer being honored.
- **B7/B8/B10**: teardown debris in warnings/logs means an event loop, portal
  pool, or container cleanup regressed.
- **B9**: a dead-port invocation that does not raise, or a recorded-retry count
  that diverges from the baseline flag in the parametrization, means
  `RetryExecutor` routing or telemetry propagation regressed.
- **B11**: `usage.total_tokens == 0` means the deep-agents generation fell off
  the agent path.
- **save/resume (QA)**: `len(result_set) == 3`, `output.exists()`, and both
  `state_path` / `jsonl_path` absent on a fresh run; `len(second_set) == 0` on
  resume. A failure means sidecar finalize or `skip_triples` merge regressed.
  In the fanout, `len(second_set) == 3` with every row
  `model_name == VLLM_QWEN36_MODEL` — anything else means cross-model triple
  accounting broke.
- **save/resume (scenario)**: `sink.total_tasks == 2` and
  `sink.completed_count == 2` on a fresh run; `len(second_set) == 0` on resume.
  A failure means combo-level `on_start`/`on_finalize` or scenario triple skip
  regressed.

## Extending the gate

Add the smallest run for a genuinely new operation and keep it neutral (no
biological ground truth). **Prefer the offline paths**: if a behavior can be
proven without a live endpoint, put it in the ordinary `tests/unit/` suite
(replay entries, fixture-backed LLM) or extend the dead-port pattern (`B9`,
`dead_port_model` + `tight_retry_policy`) so it never needs the GPU. Reserve
`tests/live/` for behavior that truly requires generation or judging against a
reachable endpoint, and drop the new test into the existing file it belongs to
rather than starting a new capability module. Markers: live tests take only
`live`; do not invent new markers outside the registered set.
