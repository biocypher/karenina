"""Live acceptance suite for async job handling (spec: async-consistency).

Runs against the vLLM server on codon-gpu-001 (OpenAI side and Anthropic
side), plus dead-port tests that need no GPU. Opt-in via
``KARENINA_LIVE_TESTS=1``. Container-backed tests additionally require
``KARENINA_LIVE_DOCKER_TESTS=1`` and a passing docker preflight.

Test-to-task map (which spec task each test gates):

    B1  invoke/ainvoke parity (openai_endpoint, deep_agents)  -> T1, T13
    B2  invoke/ainvoke via claude_tool + anthropic_base_url   -> T2, T5
    B3  streaming with usage capture + mid-stream timeout      -> T7, T8
    B4  with_structured_output across adapters                 -> T1, T2
    B5  parse_to_pydantic / aparse_to_pydantic across parsers  -> T1, T2
    B6  concurrency cap observed at the adapter leaf           -> T13
    B7  full QA run_verification, clean teardown               -> T4, T6, T10, T12, T14
    B8  scenario batch end-to-end, clean teardown              -> T11, T14
    B9  retry + telemetry against a dead port                  -> T1, T2, T3
    B10 claude_agent_sdk agent via docker container wrapper    -> T5, T0a
    B11 langchain_deep_agents agent run (FilesystemBackend)    -> T1, T14

Baseline notes (unmodified library, recorded in
ralph/async_consistency/BASELINE.md): B9 parametrizations carry an
``expect_recorded`` flag that encodes the documented current behavior.
The fix tasks (T1/T2/T3) flip those flags as deliberate test edits.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from karenina.adapters import get_agent, get_llm, get_parser
from karenina.adapters.langchain.llm import LangChainLLMAdapter
from karenina.benchmark import Benchmark
from karenina.exceptions import StreamingTimeoutError
from karenina.ports import AgentConfig, Message, ParseError
from karenina.schemas.verification import VerificationConfig
from karenina.utils.retry_policy import track_retries

from ._async_live_helpers import (
    InFlightCounter,
    claude_sdk_container_model,
    claude_tool_model,
    counted_async,
    dead_port_model,
    deep_agents_model,
    docker_gate_reason,
    find_teardown_problems,
    openai_model,
    require_adapter,
    reset_langchain_openai_client_cache,
    tight_retry_policy,
    zero_retry_policy,
)

_LIVE_OPT_IN = os.getenv("KARENINA_LIVE_TESTS") == "1"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _LIVE_OPT_IN, reason="set KARENINA_LIVE_TESTS=1 to run"),
]

logger = logging.getLogger(__name__)

SIMPLE_QUESTION = "What is the capital of France? Answer with just the city name."
TRACE_TEXT = "After weighing several candidates, I am confident the capital of France is Paris."

# Parser prompts must spell out what to extract. The thinking-enabled qwen
# (deep_agents cannot disable thinking via extra_body) ruminates past the
# request timeout on an underspecified "extract the requested fields"
# instruction, observed live while authoring this suite.
PARSE_SYSTEM_TEXT = (
    "You extract structured data. From the response text, identify the city "
    "named as the capital and fill the requested field with that city name."
)
PARSE_USER_TEXT = f"Response text: {TRACE_TEXT}"


def _parse_messages() -> list[Message]:
    """Pre-assembled parser prompt shared by B5 and the B9 parser case."""
    return [
        Message.system(PARSE_SYSTEM_TEXT),
        Message.user(PARSE_USER_TEXT),
    ]


DEFAULT_QA_BENCHMARK = Path(
    os.getenv(
        "KARENINA_LIVE_QA_BENCHMARK",
        "/Users/carli/Projects/karenina-salvage/paper_examples/QA/qa_benchmark.jsonld",
    )
)

# The adversarial experiment ships the scenario rebuild helpers used by the
# scenario fixtures (mirrors test_progressive_save_scenario_live.py).
_ADVERSARIAL_LIB = Path(
    os.getenv(
        "KARENINA_ADVERSARIAL_LIB",
        "/Users/carli/Projects/karenina-salvage/adversarial/experiment",
    )
)
_SCENARIO_CHECKPOINT = Path(
    os.getenv(
        "KARENINA_LIVE_SCENARIO_CHECKPOINT",
        str(_ADVERSARIAL_LIB / "checkpoints" / "OT_sycophancy_easy_casual.jsonld"),
    )
)


class CapitalAnswer(BaseModel):
    """Tiny structured-output schema shared by B4/B5/B9."""

    city: str = Field(description="The city named in the response")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qa_subset() -> Benchmark:
    """Load 3 QA benchmark questions into a fresh in-memory Benchmark.

    Mirrors test_progressive_save_live.py. Skips when the checkpoint is
    missing on this machine.
    """
    if not DEFAULT_QA_BENCHMARK.exists():
        pytest.skip(f"QA benchmark not found at {DEFAULT_QA_BENCHMARK}")
    full = Benchmark.load(DEFAULT_QA_BENCHMARK)
    all_templates = full.get_finished_templates()
    subset_ids = {t.question_id for t in all_templates[:3]}

    subset = Benchmark.create(name="live-async-subset", description="live async test subset", version="1.0.0")
    for template in all_templates:
        if template.question_id not in subset_ids:
            continue
        q_data = full._questions_cache[template.question_id]
        subset.add_question(
            question=q_data["question"],
            raw_answer=q_data.get("raw_answer", ""),
            answer_template=q_data["answer_template"],
            question_id=template.question_id,
            finished=True,
        )
    return subset


@pytest.fixture(scope="module")
def scenario_benchmark() -> Benchmark:
    """Build a 2-scenario benchmark with a plain guardrail node.

    Mirrors test_progressive_save_scenario_live.py but targets the
    async-suite vLLM endpoint. Skips when the adversarial checkpoint or
    helpers are missing.
    """
    if not _SCENARIO_CHECKPOINT.exists():
        pytest.skip(f"scenario checkpoint not found: {_SCENARIO_CHECKPOINT}")
    if str(_ADVERSARIAL_LIB) not in sys.path:
        sys.path.insert(0, str(_ADVERSARIAL_LIB))
    try:
        from lib import rebuild_with_plain_guardrail
    except ImportError as exc:
        pytest.skip(f"adversarial/experiment helpers not importable: {exc}")

    full = Benchmark.load(_SCENARIO_CHECKPOINT)
    all_scenarios = full.get_scenarios()
    if len(all_scenarios) < 2:
        pytest.skip(f"checkpoint has only {len(all_scenarios)} scenarios, need >=2")

    guardrail = openai_model(role_id="guardrail")
    benchmark = Benchmark.create(
        name="live-async-scenarios",
        description="live async acceptance: 2 sycophancy scenarios with plain guardrail",
        workspace_root=Path(tempfile.mkdtemp(prefix="karenina_live_async_ws_")),
    )
    for i, orig in enumerate(all_scenarios[:2]):
        benchmark.add_scenario(rebuild_with_plain_guardrail(orig, guardrail, index=i))
    return benchmark


def _qa_config(*, max_workers: int = 3, max_concurrent_requests: int | None = None) -> VerificationConfig:
    """VerificationConfig for QA runs against the live endpoint."""
    fields: dict = {
        "answering_models": [openai_model(role_id="answerer")],
        "parsing_models": [openai_model(role_id="parser")],
        "replicate_count": 1,
        "evaluation_mode": "template_only",
        "async_enabled": True,
        "async_max_workers": max_workers,
    }
    if max_concurrent_requests is not None:
        fields["max_concurrent_requests"] = max_concurrent_requests
    return VerificationConfig(**fields)


def _scenario_config(*, max_workers: int = 2, max_concurrent_requests: int | None = None) -> VerificationConfig:
    """VerificationConfig for scenario runs against the live endpoint."""
    fields: dict = {
        "answering_models": [openai_model(role_id="answerer")],
        "parsing_models": [openai_model(role_id="parser")],
        "evaluation_mode": "template_only",
        "async_enabled": True,
        "async_max_workers": max_workers,
    }
    if max_concurrent_requests is not None:
        fields["max_concurrent_requests"] = max_concurrent_requests
    return VerificationConfig(**fields)


# ---------------------------------------------------------------------------
# B1: invoke/ainvoke parity, direct to vLLM (gates T1, T13)
# ---------------------------------------------------------------------------


B1_MODELS = [
    pytest.param("openai_endpoint", openai_model, id="openai_endpoint"),
    pytest.param("langchain_deep_agents", deep_agents_model, id="deep_agents"),
]


@pytest.mark.parametrize(("interface", "make_model"), B1_MODELS)
def test_b1_invoke_and_ainvoke(interface: str, make_model) -> None:
    """B1: sync and async invocation produce content and usage.

    Gates T1 (deep_agents retry routing must not change happy-path
    behavior) and T13 (the global limiter wraps ainvoke without breaking it).
    """
    require_adapter(interface)
    llm = get_llm(make_model(), auto_fallback=False)
    messages = [Message.user(SIMPLE_QUESTION)]

    # The baseline needed a langchain-openai client-cache reset here for
    # deep_agents (cross-loop pooled connection bug). Removed at T1: the
    # resulting APIConnectionError is now retried by RetryExecutor.
    sync_response = llm.invoke(messages)
    assert sync_response.content.strip()
    assert "paris" in sync_response.content.lower()
    assert sync_response.usage.total_tokens > 0

    async_response = asyncio.run(llm.ainvoke(messages))
    assert async_response.content.strip()
    assert "paris" in async_response.content.lower()
    assert async_response.usage.total_tokens > 0


# ---------------------------------------------------------------------------
# B2: claude_tool against the Anthropic side of vLLM (gates T2, T5)
# ---------------------------------------------------------------------------


def test_b2_claude_tool_invoke_and_ainvoke() -> None:
    """B2: claude_tool via anthropic_base_url returns text despite thinking.

    The adapter joins text blocks only, so thinking blocks are tolerated.
    Gates T2 (retry routing through RetryExecutor must keep this path
    working) and T5 (request_timeout wall-clock parity).
    """
    require_adapter("claude_tool")
    llm = get_llm(claude_tool_model(), auto_fallback=False)
    messages = [Message.user(SIMPLE_QUESTION)]

    sync_response = llm.invoke(messages)
    assert sync_response.content.strip()
    assert "paris" in sync_response.content.lower()
    assert sync_response.usage.total_tokens > 0

    async_response = asyncio.run(llm.ainvoke(messages))
    assert async_response.content.strip()
    assert "paris" in async_response.content.lower()
    assert async_response.usage.total_tokens > 0


# ---------------------------------------------------------------------------
# B3: streaming with usage capture (gates T7, T8)
# ---------------------------------------------------------------------------


B3_MODELS = [
    # stream_usage=True asks ChatOpenAI for stream_options.include_usage so
    # the final chunk carries token counts.
    pytest.param("openai_endpoint", lambda: openai_model(stream_usage=True), True, id="openai_endpoint"),
    # claude_tool extracts usage from get_final_message() on the success path.
    pytest.param("claude_tool", claude_tool_model, True, id="claude_tool"),
    # deep_agents create_chat_model does not plumb stream_usage, so usage on
    # the streaming path is not captured on the baseline. Content only.
    pytest.param("langchain_deep_agents", deep_agents_model, False, id="deep_agents"),
]


@pytest.mark.parametrize(("interface", "make_model", "expect_usage"), B3_MODELS)
def test_b3_streaming_with_usage(interface: str, make_model, expect_usage: bool) -> None:
    """B3: astream accumulates content and (where supported) usage.

    Gates T7 (streaming usage-capture parity) and T8. The deep_agents
    usage gap is documented baseline behavior, not asserted away.
    """
    require_adapter(interface)
    llm = get_llm(make_model(), auto_fallback=False)
    messages = [Message.user(SIMPLE_QUESTION)]

    # Still required for deep_agents streaming after T1: the bare astream
    # context manager has no retry layer, so the cross-loop client cache
    # bug is not absorbed by RetryExecutor here (it is on the ainvoke and
    # parser paths, which dropped this reset at T1).
    reset_langchain_openai_client_cache()

    async def drive():
        async with llm.astream(messages) as streaming_response:
            async for _chunk in streaming_response:
                pass
        return streaming_response

    streamed = asyncio.run(drive())
    assert streamed.accumulated_content.strip()
    assert "paris" in streamed.accumulated_content.lower()
    if expect_usage:
        assert streamed.usage.total_tokens > 0


def test_b3_mid_stream_timeout_exception_shape() -> None:
    """B3 sub-case: a mid-stream wall-clock timeout raises StreamingTimeoutError.

    Baseline asserts only the exception shape (type and partial_content
    attribute). Tightens after T7/T8 when usage capture on partial streams
    lands. Zero retry budgets keep the test to a single attempt.
    """
    model = openai_model(retry_policy=zero_retry_policy())
    llm = get_llm(model, auto_fallback=False)
    messages = [Message.user("Write a detailed 500 word essay about the history of the sea.")]

    with pytest.raises(StreamingTimeoutError) as excinfo:
        llm.stream_invoke(messages, timeout=0.5)

    assert hasattr(excinfo.value, "partial_content")
    assert isinstance(excinfo.value.partial_content, str)


# ---------------------------------------------------------------------------
# B4: structured output (gates T1, T2)
# ---------------------------------------------------------------------------


# supported_on_vllm: measured on the baseline. vLLM's /v1/messages accepts
# but silently ignores Anthropic's output_format, so beta.messages.parse
# never returns parsed_output and claude_tool structured output cannot work
# against this endpoint regardless of karenina-side changes. Asserted as a
# ParseError so the limitation stays visible. Revisit if vLLM gains
# Anthropic structured output support.
B4_MODELS = [
    pytest.param("openai_endpoint", openai_model, True, id="openai_endpoint"),
    pytest.param("claude_tool", claude_tool_model, False, id="claude_tool"),
    # deep_agents warns-and-ignores max_retries on with_structured_output,
    # documented adapter behavior. We pass max_retries=None to avoid the warning.
    pytest.param("langchain_deep_agents", deep_agents_model, True, id="deep_agents"),
]


@pytest.mark.parametrize(("interface", "make_model", "supported_on_vllm"), B4_MODELS)
def test_b4_structured_output(interface: str, make_model, supported_on_vllm: bool) -> None:
    """B4: with_structured_output returns schema-conformant JSON content.

    The supported adapters serialize the parsed model into response.content
    as JSON, so the assertion is uniform. The claude_tool param documents
    the vLLM endpoint limitation (see B4_MODELS comment). Gates T1 and T2
    (retry routing must not break the structured paths).
    """
    require_adapter(interface)
    llm = get_llm(make_model(), auto_fallback=False)

    if not supported_on_vllm:
        # max_retries=0 keeps the documented failure to a single attempt.
        structured = llm.with_structured_output(CapitalAnswer, max_retries=0)
        with pytest.raises(ParseError):
            asyncio.run(structured.ainvoke([Message.user(SIMPLE_QUESTION)]))
        return

    structured = llm.with_structured_output(CapitalAnswer)
    response = asyncio.run(structured.ainvoke([Message.user(SIMPLE_QUESTION)]))

    payload = json.loads(response.content)
    assert "paris" in str(payload.get("city", "")).lower()


# ---------------------------------------------------------------------------
# B5: parser ports (gates T1, T2)
# ---------------------------------------------------------------------------


# supported_on_vllm: the claude_tool parser delegates to beta.messages.parse,
# which vLLM's /v1/messages cannot satisfy (output_format ignored, see
# B4_MODELS comment). Asserted as a ParseError on the baseline.
B5_PARSERS = [
    pytest.param("openai_endpoint", openai_model, True, id="openai_endpoint"),
    pytest.param("claude_tool", claude_tool_model, False, id="claude_tool"),
    pytest.param("langchain_deep_agents", deep_agents_model, True, id="deep_agents"),
]


@pytest.mark.parametrize(("interface", "make_model", "supported_on_vllm"), B5_PARSERS)
def test_b5_parse_to_pydantic(interface: str, make_model, supported_on_vllm: bool) -> None:
    """B5: parse_to_pydantic and aparse_to_pydantic extract structured data.

    Gates T1 (deep_agents parser retry routing) and T2 (claude paths
    through RetryExecutor). The claude_tool param documents the vLLM
    endpoint limitation and only asserts the sync path to bound runtime
    (each doomed attempt costs a full thinking-enabled generation).
    """
    require_adapter(interface)
    parser = get_parser(make_model(), auto_fallback=False)
    messages = _parse_messages()

    if not supported_on_vllm:
        with pytest.raises(ParseError):
            parser.parse_to_pydantic(messages, CapitalAnswer)
        return

    sync_result = parser.parse_to_pydantic(messages, CapitalAnswer)
    assert "paris" in sync_result.parsed.city.lower()

    async_result = asyncio.run(parser.aparse_to_pydantic(messages, CapitalAnswer))
    assert "paris" in async_result.parsed.city.lower()


# ---------------------------------------------------------------------------
# B6: concurrency cap observed at the adapter leaf (gates T13)
# ---------------------------------------------------------------------------


def _install_counter(monkeypatch: pytest.MonkeyPatch) -> InFlightCounter:
    """Patch the langchain adapter async leaves with an in-flight counter.

    Generation goes through stream_invoke -> _astream_with_timeout and
    parsing goes through structured ainvoke, so both are wrapped. The
    wrapper sleeps briefly before delegating to force overlap.
    """
    counter = InFlightCounter()
    monkeypatch.setattr(
        LangChainLLMAdapter,
        "ainvoke",
        counted_async(LangChainLLMAdapter.ainvoke, counter),
    )
    monkeypatch.setattr(
        LangChainLLMAdapter,
        "_astream_with_timeout",
        counted_async(LangChainLLMAdapter._astream_with_timeout, counter),
    )
    return counter


def test_b6_concurrency_cap_scenario_batch(scenario_benchmark: Benchmark, monkeypatch: pytest.MonkeyPatch) -> None:
    """B6a: scenario batch with max_concurrent_requests=2 stays within the cap.

    On the baseline the cap is enforced by the threading.Semaphore around
    the sync invoke wrappers plus the worker-thread bound. Gates T13 (the
    GlobalLLMLimiter must keep max_observed <= cap).
    """
    counter = _install_counter(monkeypatch)
    config = _scenario_config(max_workers=2, max_concurrent_requests=2)

    result_set = scenario_benchmark.run_verification(config=config, run_name="live-b6-scenario")

    assert len(result_set) > 0
    assert counter.total_calls >= 2
    assert counter.max_observed <= 2


def test_b6_concurrency_cap_qa_batch(qa_subset: Benchmark, monkeypatch: pytest.MonkeyPatch) -> None:
    """B6b: QA batch with async_max_workers=2 stays within the worker bound.

    On the baseline max_concurrent_requests is dead on the QA path, so the
    enforced bound is the worker count. After T13 this test gains a
    parametrization with async_max_workers > max_concurrent_requests to
    prove the limiter (not the thread pool) is what caps in-flight calls.
    Gates T13.
    """
    counter = _install_counter(monkeypatch)
    config = _qa_config(max_workers=2)

    result_set = qa_subset.run_verification(config=config, run_name="live-b6-qa")

    assert len(result_set) == 3
    assert counter.total_calls >= 3
    assert counter.max_observed <= 2


# ---------------------------------------------------------------------------
# B7: full QA run with clean teardown (gates T4, T6, T10, T12, T14)
# ---------------------------------------------------------------------------


def test_b7_qa_run_verification_clean_teardown(qa_subset: Benchmark, caplog: pytest.LogCaptureFixture) -> None:
    """B7: 3 questions through run_verification with parallel workers.

    Asserts results complete AND that no async teardown debris appears in
    captured warnings or logs ("Event loop is closed", "coroutine ... was
    never awaited", destroyed pending tasks). Gates T4 (MCP teardown), T6
    (sync-wrapper timeouts), T10/T12 (lifecycle extraction), T14 (portal
    pool consolidation).
    """
    config = _qa_config(max_workers=3)

    with caplog.at_level(logging.WARNING), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result_set = qa_subset.run_verification(config=config, run_name="live-b7-qa")
        gc.collect()

    assert len(result_set) == 3

    observed = [str(w.message) for w in caught] + [record.getMessage() for record in caplog.records]
    problems = find_teardown_problems(observed)
    assert not problems, f"async teardown debris detected: {problems}"


# ---------------------------------------------------------------------------
# B8: scenario batch end-to-end with clean teardown (gates T11, T14)
# ---------------------------------------------------------------------------


def test_b8_scenario_run_clean_teardown(scenario_benchmark: Benchmark, caplog: pytest.LogCaptureFixture) -> None:
    """B8: 2 scenarios end-to-end, same teardown assertions as B7.

    Gates T11 (pre-teardown aclose in sequential paths) and T14 (the
    scenario side of the portal-pool consolidation).
    """
    config = _scenario_config(max_workers=2)

    with caplog.at_level(logging.WARNING), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result_set = scenario_benchmark.run_verification(config=config, run_name="live-b8-scenario")
        gc.collect()

    assert len(result_set) > 0

    observed = [str(w.message) for w in caught] + [record.getMessage() for record in caplog.records]
    problems = find_teardown_problems(observed)
    assert not problems, f"async teardown debris detected: {problems}"


# ---------------------------------------------------------------------------
# B9: retry + telemetry against a dead port, no GPU needed (gates T1, T2, T3)
# ---------------------------------------------------------------------------

# Each param carries expect_recorded: whether track_retries observes at
# least one recorded retry on the UNMODIFIED baseline. The fix tasks flip
# the False entries to True as deliberate test edits:
#   - langchain-sync-in-portal: measured on the baseline, anyio's
#     BlockingPortal.call DOES propagate the caller's contextvars, so
#     telemetry is recorded. This contradicts the audit assumption and is
#     kept as a regression guard: T3 must keep it True.
#   - langchain-sync-thread-fallback: invoke() called while an event loop
#     runs in the calling thread dispatches to a fresh ThreadPoolExecutor
#     thread. On the baseline that thread did not inherit the context and
#     telemetry was lost. Flipped to True at T3: the fallback re-enters a
#     copy of the caller's context (run_coro_in_thread).
#   - claude-tool-async: retries happened inside the Anthropic SDK on the
#     baseline, RetryExecutor never saw them. Flipped to True at T2: the
#     SDK clients run with max_retries=0 and RetryExecutor owns retries.
#   - deep-agents-async: ainvoke had no retry layer on the baseline (the
#     confirmed zero-retry bug). Flipped to True at T1: ainvoke now routes
#     through RetryExecutor.
B9_LLM_CASES = [
    pytest.param("openai_endpoint", "async", True, id="langchain-async"),
    pytest.param("openai_endpoint", "sync", True, id="langchain-sync-no-portal"),
    pytest.param("openai_endpoint", "sync_portal", True, id="langchain-sync-in-portal"),  # T3 keeps this True
    pytest.param("openai_endpoint", "sync_thread_fallback", True, id="langchain-sync-thread-fallback"),  # flipped at T3
    pytest.param("claude_tool", "async", True, id="claude-tool-async"),  # flipped at T2
    pytest.param("langchain_deep_agents", "async", True, id="deep-agents-async"),  # flipped at T1
]


@pytest.mark.parametrize(("interface", "mode", "expect_recorded"), B9_LLM_CASES)
def test_b9_retry_telemetry_dead_port(interface: str, mode: str, expect_recorded: bool) -> None:
    """B9: dead-port invocation raises, and telemetry matches the baseline.

    Uses a tight RetryPolicy (2 connection attempts, zero backoff) against
    http://127.0.0.1:1 so every attempt fails in milliseconds. Gates T1
    (deep_agents retry routing), T2 (claude_tool through RetryExecutor),
    and T3 (telemetry propagation across portal dispatch).
    """
    import anthropic
    import openai

    require_adapter(interface)
    model = dead_port_model(interface)
    llm = get_llm(model, auto_fallback=False)
    messages = [Message.user("ping")]
    policy = tight_retry_policy()
    # claude_tool surfaces the raw Anthropic SDK error, the langchain-routed
    # interfaces surface the OpenAI SDK error.
    expected_exc = anthropic.APIConnectionError if interface == "claude_tool" else openai.APIConnectionError

    with track_retries(policy) as tracker, pytest.raises(expected_exc):
        if mode == "async":
            asyncio.run(llm.ainvoke(messages))
        elif mode == "sync":
            llm.invoke(messages)
        elif mode == "sync_thread_fallback":
            # Calling sync invoke while a loop runs in this thread makes
            # the adapter dispatch to a fresh ThreadPoolExecutor thread,
            # the path where retry telemetry is lost on the baseline.
            async def call_sync_from_coroutine() -> None:
                llm.invoke(messages)

            asyncio.run(call_sync_from_coroutine())
        else:
            from anyio.from_thread import start_blocking_portal

            from karenina.benchmark.verification.executor import set_async_portal

            with start_blocking_portal() as portal:
                set_async_portal(portal)
                try:
                    llm.invoke(messages)
                finally:
                    set_async_portal(None)

    recorded = sum(entry["used"] for entry in tracker.values())
    if expect_recorded:
        assert recorded >= 1, f"expected recorded retries, tracker={tracker}"
    else:
        assert recorded == 0, f"baseline expects zero recorded retries, tracker={tracker}"


def test_b9_retry_telemetry_claude_sdk_parser_dead_port() -> None:
    """B9: claude_agent_sdk parser against a dead port records retries.

    The parser talks to the endpoint directly (AsyncOpenAI for custom base
    URLs, no CLI subprocess). Since T2 it routes API calls through
    RetryExecutor (SDK clients at max_retries=0), so track_retries observes
    the connection retries while the public contract still raises
    ParseError.
    """
    require_adapter("claude_agent_sdk")
    parser = get_parser(dead_port_model("claude_agent_sdk"), auto_fallback=False)
    messages = _parse_messages()

    with track_retries(tight_retry_policy()) as tracker, pytest.raises(ParseError):
        asyncio.run(parser.aparse_to_pydantic(messages, CapitalAnswer))

    recorded = sum(entry["used"] for entry in tracker.values())
    assert recorded >= 1, f"expected recorded retries after T2, tracker={tracker}"


# ---------------------------------------------------------------------------
# B10: claude_agent_sdk agent via the docker container wrapper (gates T5, T0a)
# ---------------------------------------------------------------------------


@pytest.fixture
def colima_shared_workspace() -> Iterator[Path]:
    """Workspace under the user home so the docker VM can bind-mount it.

    Colima only shares ``$HOME`` (and a few VM-local paths) with the VM.
    pytest's ``tmp_path`` lives under ``/private/var/folders``, which is not
    shared, so a ``--volume`` bind of it appears empty inside the container.
    """
    root = Path.home() / ".cache" / "karenina-live-docker"
    root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="b10_ws_", dir=root))
    yield workspace
    shutil.rmtree(workspace, ignore_errors=True)


def test_b10_claude_sdk_container_agent(colima_shared_workspace: Path, caplog: pytest.LogCaptureFixture) -> None:
    """B10: containerized claude_agent_sdk agent against vLLM /v1/messages.

    Exercises _build_options env forwarding, the docker CLI wrapper path,
    and the pinned CLI (2.1.146) end-to-end. Docker-gated: requires
    KARENINA_LIVE_DOCKER_TESTS=1 plus a passing docker preflight. Gates T5
    (wall-clock guard must not break container runs) and validates the T0a
    image.
    """
    reason = docker_gate_reason()
    if reason is not None:
        pytest.skip(reason)
    require_adapter("claude_agent_sdk")

    workspace = colima_shared_workspace
    (workspace / "notes.txt").write_text("The code word for this exercise is 'tangerine'.\n")

    agent = get_agent(claude_sdk_container_model(), auto_fallback=False)
    agent_config = AgentConfig(
        max_turns=8,
        timeout=300.0,
        workspace_path=workspace,
    )

    with caplog.at_level(logging.WARNING), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = agent.run(
            messages=[Message.user("Read notes.txt in your working directory and report the code word it contains.")],
            config=agent_config,
        )
        gc.collect()

    assert result.final_response.strip()
    assert "tangerine" in result.final_response.lower()
    assert result.trace_messages, "expected a non-empty trace"
    assert result.trace_messages[-1].role.value == "assistant"
    assert result.usage.total_tokens > 0

    observed = [str(w.message) for w in caught] + [record.getMessage() for record in caplog.records]
    problems = find_teardown_problems(observed)
    assert not problems, f"async teardown debris detected: {problems}"


# ---------------------------------------------------------------------------
# B11: langchain_deep_agents agent run, no container (gates T1, T14)
# ---------------------------------------------------------------------------


def test_b11_deep_agents_filesystem_agent(tmp_path: Path) -> None:
    """B11: deep_agents agent with the default FilesystemBackend.

    Simple file-read task in a tmp workspace, direct to vLLM. Gates T1
    (deep_agents retry routing must not change agent-loop behavior) and
    T14 (executor consolidation must keep agent runs working).
    """
    require_adapter("langchain_deep_agents")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("The code word for this exercise is 'tangerine'.\n")

    agent = get_agent(deep_agents_model(), auto_fallback=False)
    agent_config = AgentConfig(
        max_turns=8,
        timeout=300.0,
        workspace_path=workspace,
    )

    result = agent.run(
        messages=[Message.user("Read the file notes.txt and report the code word it contains.")],
        config=agent_config,
    )

    assert result.final_response.strip()
    assert "tangerine" in result.final_response.lower()
    assert result.trace_messages, "expected a non-empty trace"
    assert result.usage.total_tokens > 0
