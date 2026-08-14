"""Live acceptance tests for the codex_sdk agent adapter.

Runs against a real vLLM endpoint through the full adapter stack: per-run
endpoint shim, codex app-server subprocess, and the live model. Opt-in via
``KARENINA_LIVE_TESTS=1``. Endpoint and model are configurable so the suite
stays portable:

    KARENINA_CODEX_LIVE_BASE_URL: OpenAI-compatible base URL
        (default http://hl-codon-gpu-020:8000/v1)
    KARENINA_CODEX_LIVE_MODEL: served model name
        (default qwen3.5-122b-a10b)

Coverage:
    1. End-to-end shell run with canonical trace and usage assertions.
    2. Short-timeout run against an in-flight shell command, confirming
       interrupt plus partial-trace salvage instead of an exception or hang.
    3. Two-turn scenario through the real scenario machinery, proving the
       serialized conversation history reaches the model.
    4. Concurrency: 4 parallel arun calls (asyncio.gather) and 4 parallel
       run() calls (ThreadPoolExecutor) on one adapter instance, with
       cross-contamination and process/thread leak checks.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

from karenina.adapters import get_agent
from karenina.ports import AgentConfig, Message, Role
from karenina.schemas.config import ModelConfig

_LIVE_OPT_IN = os.getenv("KARENINA_LIVE_TESTS") == "1"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _LIVE_OPT_IN, reason="set KARENINA_LIVE_TESTS=1 to run"),
]

LIVE_CODEX_BASE_URL = os.getenv("KARENINA_CODEX_LIVE_BASE_URL", "http://hl-codon-gpu-020:8000/v1")
LIVE_CODEX_MODEL = os.getenv("KARENINA_CODEX_LIVE_MODEL", "qwen3.5-122b-a10b")

END_TO_END_PROMPT = (
    "Create a file named hello.txt containing 'hello from the codex adapter' "
    "and then list the directory contents using the shell. Finish with a one "
    "sentence summary of what you did."
)

TIMEOUT_PROMPT = (
    "First run the shell command 'sleep 60' and wait for it to finish. "
    "Then create a file named done.txt and summarize what you did."
)


def _codex_model(role_id: str = "codex-live") -> ModelConfig:
    """ModelConfig for the codex_sdk adapter against the live endpoint."""
    return ModelConfig(
        id=role_id,
        model_name=LIVE_CODEX_MODEL,
        model_provider="openai",
        interface="codex_sdk",
        endpoint_base_url=LIVE_CODEX_BASE_URL,
    )


def _shell_use_result_pairs(result) -> list[str]:
    """Return tool_use ids of shell calls that have a paired tool result."""
    tool_result_ids = {
        getattr(block, "tool_use_id", None)
        for message in result.trace_messages
        if message.role == Role.TOOL
        for block in message.content
    }
    return [
        tc.id
        for message in result.trace_messages
        for tc in message.tool_calls
        if tc.name == "shell" and tc.id in tool_result_ids
    ]


def test_codex_end_to_end_shell_run(tmp_path: Path) -> None:
    """Full agent run: shell tool round trip, canonical trace, real usage."""
    agent = get_agent(_codex_model())

    result = agent.run(
        [Message.system("You are a helpful coding agent."), Message.user(END_TO_END_PROMPT)],
        config=AgentConfig(timeout=300, workspace_path=tmp_path),
    )

    assert result.final_response.strip()
    assert result.timeout_reached is False

    # Canonical delimited raw trace with a shell tool call round trip.
    assert "--- AI Message ---" in result.raw_trace
    assert "Tool Calls:" in result.raw_trace
    assert "shell (call_" in result.raw_trace
    assert "--- Tool Message (call_id:" in result.raw_trace

    # Structured trace carries at least one shell use/result pair.
    assert _shell_use_result_pairs(result), "no shell use/result pair in trace_messages"

    # Real token accounting from the live endpoint.
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.usage.model == LIVE_CODEX_MODEL

    # The agent actually wrote into the pinned workspace.
    assert (tmp_path / "hello.txt").exists()


def test_codex_timeout_interrupt_salvages_partial_trace(tmp_path: Path) -> None:
    """Wall-clock timeout against an in-flight shell command.

    The 'sleep 60' keeps the turn running when the timeout fires, forcing
    the interrupt path. The adapter must return a partial result (not raise
    and not hang) with timeout_reached=True and the timeout note appended.
    """
    agent = get_agent(_codex_model(role_id="codex-live-timeout"))

    result = agent.run(
        [Message.user(TIMEOUT_PROMPT)],
        config=AgentConfig(timeout=8, workspace_path=tmp_path),
    )

    assert result.timeout_reached is True
    assert "[Note: Agent timed out" in result.raw_trace
    assert result.final_response  # placeholder or partial text, never empty


# ---------------------------------------------------------------------------
# Two-turn scenario: history must reach the model
# ---------------------------------------------------------------------------

_FRUITS = ("durian", "rambutan", "persimmon", "quince")

_FRUIT_TEMPLATE = """
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    fruit: str = Field(description="The fruit named in the response")

    def model_post_init(self, __context):
        self.correct = {"fruit": ""}

    def verify(self) -> bool:
        return bool(self.fruit.strip())
"""


def _parsing_model(role_id: str = "qwen-parser") -> ModelConfig:
    """Plain langchain ChatOpenAI judge against the same vLLM endpoint."""
    return ModelConfig(
        id=role_id,
        model_name=LIVE_CODEX_MODEL,
        model_provider="openai",
        interface="openai_endpoint",
        endpoint_base_url=LIVE_CODEX_BASE_URL,
        endpoint_api_key="EMPTY",
        temperature=0.0,
        extra_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )


def test_codex_scenario_history_reaches_model() -> None:
    """Two-turn scenario where turn 2 is only answerable from turn 1's answer.

    Turn 1 asks the model to pick one fruit from a list. Turn 2 asks which
    fruit it picked earlier, without naming any fruit. A correct turn 2
    answer therefore proves the serialized conversation history (the
    role-labeled transcript built by CodexMessageConverter) actually
    reached the model through the real scenario machinery.
    """
    from karenina.benchmark import Benchmark
    from karenina.scenario.builder import Scenario
    from karenina.schemas.entities import Question
    from karenina.schemas.scenario.types import END
    from karenina.schemas.verification import VerificationConfig

    scenario = Scenario("codex-fruit-recall")
    scenario.add_node(
        "pick",
        question=Question(
            question=(
                f"Pick exactly one fruit from this list: {', '.join(_FRUITS)}. State clearly which fruit you picked."
            ),
            raw_answer="any fruit from the list",
            answer_template=_FRUIT_TEMPLATE,
        ),
    )
    scenario.add_node(
        "recall",
        question=Question(
            question="Which fruit did you pick earlier? Answer with just the fruit name.",
            raw_answer="the fruit picked in the previous turn",
            answer_template=_FRUIT_TEMPLATE,
        ),
    )
    scenario.add_edge("pick", "recall")
    scenario.add_edge("recall", END)
    scenario.set_entry("pick")

    benchmark = Benchmark.create(
        name="codex-live-scenario",
        description="codex_sdk 2-turn history recall scenario",
        workspace_root=Path(tempfile.mkdtemp(prefix="karenina_codex_scenario_ws_")),
    )
    benchmark.add_scenario(scenario.validate())

    config = VerificationConfig(
        answering_models=[_codex_model(role_id="codex-scenario")],
        parsing_models=[_parsing_model()],
        evaluation_mode="template_only",
        async_enabled=False,
    )

    result_set = benchmark.run_verification(config=config, run_name="codex-live-scenario")
    assert len(result_set.results) == 2, f"expected 2 turn results, got {len(result_set.results)}"

    by_node: dict[str, object] = {}
    for result in result_set.results:
        if "Pick exactly one fruit" in result.metadata.question_text:
            by_node["pick"] = result
        else:
            by_node["recall"] = result
    assert set(by_node) == {"pick", "recall"}

    pick_template = by_node["pick"].template
    recall_template = by_node["recall"].template
    assert pick_template is not None and recall_template is not None
    assert by_node["pick"].metadata.failure is None
    assert by_node["recall"].metadata.failure is None

    picked = (pick_template.parsed_llm_response or {}).get("fruit", "").strip().lower()
    assert picked in _FRUITS, f"turn 1 did not pick a listed fruit: {picked!r}"

    # Evidence that history reached the model: turn 2 names turn 1's fruit
    # even though turn 2's question mentions no fruit at all.
    recalled = (recall_template.parsed_llm_response or {}).get("fruit", "").strip().lower()
    recall_blob = f"{recalled} {recall_template.raw_llm_response}".lower()
    assert picked in recall_blob, f"turn 2 did not recall {picked!r}: parsed={recalled!r}"
    # And no OTHER listed fruit was recalled instead.
    if recalled in _FRUITS:
        assert recalled == picked


# ---------------------------------------------------------------------------
# Concurrency: gather, threadpool, leak checks
# ---------------------------------------------------------------------------

_CONCURRENCY = 4


def _count_codex_processes() -> int:
    """Count live codex app-server subprocesses spawned from this venv.

    Matches on the bundled binary path segment (codex_cli_bin) so other
    codex installs on the machine (IDE integrations) are excluded.
    """
    proc = subprocess.run(["pgrep", "-f", "codex_cli_bin"], capture_output=True, text=True, check=False)
    return len([line for line in proc.stdout.splitlines() if line.strip()])


def _count_shim_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name.startswith("karenina-codex-endpoint-shim"))


def _concurrency_prompt(token: str) -> str:
    return f"Run the shell command `echo {token}` using the shell tool, then state exactly what it printed."


def _tokens() -> list[str]:
    return [
        f"CODEX-CONC-{i}-{suffix}"
        for i, suffix in zip(range(_CONCURRENCY), ("ALPHA", "BRAVO", "CHARLIE", "DELTA"), strict=False)
    ]


def _assert_no_cross_contamination(results: list, tokens: list[str]) -> None:
    for i, (result, token) in enumerate(zip(results, tokens, strict=False)):
        assert token in result.raw_trace, f"run {i} trace missing its own token {token}"
        for other in tokens:
            if other != token:
                assert other not in result.raw_trace, f"run {i} trace contaminated with {other}"
        assert result.usage.input_tokens > 0


@pytest.fixture(scope="module")
def single_run_seconds(tmp_path_factory: pytest.TempPathFactory) -> float:
    """Wall-clock of one solo run, the baseline for concurrency speedup."""
    if not _LIVE_OPT_IN:
        pytest.skip("set KARENINA_LIVE_TESTS=1 to run")
    agent = get_agent(_codex_model(role_id="codex-baseline"))
    workspace = tmp_path_factory.mktemp("codex-baseline")
    t0 = time.time()
    result = agent.run(
        [Message.user(_concurrency_prompt("CODEX-BASELINE-0"))],
        config=AgentConfig(timeout=180, workspace_path=workspace),
    )
    elapsed = time.time() - t0
    assert "CODEX-BASELINE-0" in result.raw_trace
    return elapsed


def test_codex_concurrent_arun_gather(tmp_path: Path, single_run_seconds: float) -> None:
    """4 concurrent arun calls on one adapter instance, asyncio.gather."""
    adapter = get_agent(_codex_model(role_id="codex-gather"))
    tokens = _tokens()
    baseline_procs = _count_codex_processes()
    baseline_threads = _count_shim_threads()

    async def run_all() -> list:
        tasks = []
        for i, token in enumerate(tokens):
            workspace = tmp_path / f"ws-{i}"
            workspace.mkdir()
            tasks.append(
                adapter.arun(
                    [Message.user(_concurrency_prompt(token))],
                    config=AgentConfig(timeout=180, workspace_path=workspace),
                )
            )
        return await asyncio.gather(*tasks)

    t0 = time.time()
    results = asyncio.run(run_all())
    elapsed = time.time() - t0

    assert len(results) == _CONCURRENCY
    _assert_no_cross_contamination(results, tokens)
    # Meaningfully better than serial: well under 4x a single run.
    assert elapsed < 3 * single_run_seconds, f"gather took {elapsed:.1f}s vs single {single_run_seconds:.1f}s"

    time.sleep(1)
    assert _count_codex_processes() == baseline_procs, "codex app-server process leaked"
    assert _count_shim_threads() == baseline_threads, "endpoint shim thread leaked"


def test_codex_concurrent_run_threadpool(tmp_path: Path, single_run_seconds: float) -> None:
    """4 concurrent sync run() calls, the pipeline's parallelization shape."""
    adapter = get_agent(_codex_model(role_id="codex-threadpool"))
    tokens = _tokens()
    baseline_procs = _count_codex_processes()
    baseline_threads = _count_shim_threads()

    def one(i: int, token: str):
        workspace = tmp_path / f"ws-{i}"
        workspace.mkdir()
        return adapter.run(
            [Message.user(_concurrency_prompt(token))],
            config=AgentConfig(timeout=180, workspace_path=workspace),
        )

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=_CONCURRENCY) as pool:
        results = list(pool.map(one, range(_CONCURRENCY), tokens))
    elapsed = time.time() - t0

    assert len(results) == _CONCURRENCY
    _assert_no_cross_contamination(results, tokens)
    assert elapsed < 3 * single_run_seconds, f"threadpool took {elapsed:.1f}s vs single {single_run_seconds:.1f}s"

    time.sleep(1)
    assert _count_codex_processes() == baseline_procs, "codex app-server process leaked"
    assert _count_shim_threads() == baseline_threads, "endpoint shim thread leaked"
