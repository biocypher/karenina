"""Live acceptance tests for scenario progressive-save / resume.

These tests exercise ``Benchmark.run_verification(sink=...)`` against a
real vLLM server on codon-gpu-001. They are opt-in: set
``KARENINA_LIVE_TESTS=1`` and keep the endpoints reachable.

The adversarial sycophancy checkpoints
(``adversarial/experiment/checkpoints/OT_sycophancy_easy_casual.jsonld``)
provide a pre-wired 3-node scenario skeleton and a ``rebuild_with_plain_guardrail``
helper that adds a guardrail node. We slice to 2 scenarios so the full
matrix is 2 combos, keeping wall-clock bounded while exercising every
seam: combo-level ``on_start`` manifest, per-turn ``on_result`` fan-out,
``on_finalize`` sidecar deletion, and triple-level skip on a second pass.

To keep the run fast we target the small qwen3.5-a3b on port 8002
instead of the 122b model the adversarial scripts default to, and we
skip the MCP subprocess (plain guardrail does not need tool access).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_LIVE_OPT_IN = os.getenv("KARENINA_LIVE_TESTS") == "1"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _LIVE_OPT_IN, reason="set KARENINA_LIVE_TESTS=1 to run"),
]

# The adversarial experiment ships the scenario rebuild helpers we need
# (guardrail node wiring, edge configuration). Import them without
# polluting karenina.
_ADVERSARIAL_LIB = Path(
    os.getenv(
        "KARENINA_ADVERSARIAL_LIB",
        "/Users/carli/Projects/karenina-salvage/adversarial/experiment",
    )
)
if str(_ADVERSARIAL_LIB) not in sys.path:
    sys.path.insert(0, str(_ADVERSARIAL_LIB))

_CHECKPOINT = Path(
    os.getenv(
        "KARENINA_LIVE_SCENARIO_CHECKPOINT",
        str(_ADVERSARIAL_LIB / "checkpoints" / "OT_sycophancy_easy_casual.jsonld"),
    )
)

VLLM_QWEN35 = os.getenv("VLLM_QWEN35_URL", "http://codon-gpu-001:8002")

OPENAI_EXTRA_KWARGS: dict = {
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    "max_retries": 0,
}


def _small_model(role_id: str):
    from karenina.schemas.config import ModelConfig

    return ModelConfig(
        id=role_id,
        model_name="qwen3.5-a3b",
        interface="openai_endpoint",
        endpoint_base_url=VLLM_QWEN35,
        endpoint_api_key="EMPTY",
        temperature=0.0,
        request_timeout=180.0,
        extra_kwargs=OPENAI_EXTRA_KWARGS,
    )


@pytest.fixture(scope="module")
def scenario_benchmark_and_config():
    """Build a 2-scenario benchmark with a small qwen3.5-a3b answering model.

    Yields ``(benchmark, config)``.
    """
    if not _CHECKPOINT.exists():
        pytest.skip(f"scenario checkpoint not found: {_CHECKPOINT}")

    try:
        from lib import rebuild_with_plain_guardrail
    except ImportError as exc:
        pytest.skip(f"adversarial/experiment helpers not importable: {exc}")

    from karenina.benchmark import Benchmark
    from karenina.schemas.verification import VerificationConfig

    full = Benchmark.load(_CHECKPOINT)
    all_scenarios = full.get_scenarios()
    if len(all_scenarios) < 2:
        pytest.skip(f"checkpoint has only {len(all_scenarios)} scenarios; need >=2")

    answering = _small_model("answerer")
    parsing = _small_model("parser")
    guardrail = _small_model("guardrail")

    config = VerificationConfig(
        answering_models=[answering],
        parsing_models=[parsing],
        evaluation_mode="template_only",
        async_enabled=True,
        async_max_workers=2,
    )

    benchmark = Benchmark.create(
        name="live-scenario-progsave",
        description="live sink acceptance: 2 sycophancy scenarios with plain guardrail",
        workspace_root=Path(tempfile.mkdtemp(prefix="karenina_live_scenario_ws_")),
    )
    for i, orig in enumerate(all_scenarios[:2]):
        benchmark.add_scenario(rebuild_with_plain_guardrail(orig, guardrail, index=i))

    yield benchmark, config


def test_scenario_fresh_run_clean_finalize(scenario_benchmark_and_config, tmp_path: Path) -> None:
    """Fresh scenario run writes a final export and deletes sidecars."""
    from karenina.benchmark.verification.sinks import ProgressiveFileSink

    benchmark, config = scenario_benchmark_and_config
    output = tmp_path / "live_scenario_fresh.json"
    sink = ProgressiveFileSink(
        output_path=output,
        config=config,
        benchmark_path=str(_CHECKPOINT),
    )

    result_set = benchmark.run_verification(config=config, run_name="live-scenario-fresh", sink=sink)

    assert len(result_set) > 0
    assert output.exists()
    assert not sink.state_path.exists()
    assert not sink.jsonl_path.exists()
    # One task per combo in the manifest.
    assert sink.total_tasks == 2
    assert sink.completed_count == 2


def test_scenario_resume_skips_completed_combos(scenario_benchmark_and_config, tmp_path: Path) -> None:
    """A second pass with skip_triples populated skips both completed combos."""
    from karenina.benchmark.verification.sinks import ProgressiveFileSink
    from karenina.schemas.verification.model_identity import ModelIdentity

    benchmark, config = scenario_benchmark_and_config
    output = tmp_path / "live_scenario_resume.json"

    sink = ProgressiveFileSink(
        output_path=output,
        config=config,
        benchmark_path=str(_CHECKPOINT),
    )
    first_set = benchmark.run_verification(config=config, run_name="live-scenario-1", sink=sink)
    assert not sink.state_path.exists(), "fresh run should finalize clean"
    assert len(first_set) > 0

    ans_key = ModelIdentity.from_model_config(config.answering_models[0], role="answering").canonical_key
    parse_key = ModelIdentity.from_model_config(config.parsing_models[0], role="parsing").canonical_key
    scenario_names = {s.name for s in benchmark.get_scenarios()}
    skip_triples = {(name, ans_key, parse_key, None) for name in scenario_names}
    assert len(skip_triples) == 2

    skip_config = config.model_copy(update={"skip_triples": frozenset(skip_triples)})
    second_set = benchmark.run_verification(config=skip_config, run_name="live-scenario-2")

    assert len(second_set) == 0
