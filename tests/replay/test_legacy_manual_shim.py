"""Tests the legacy interface='manual' auto-translation path."""

from __future__ import annotations

import pytest

from karenina.adapters.manual import ManualTraces
from karenina.benchmark import Benchmark
from karenina.replay import ReplayStore


def _build_benchmark_with_one_question() -> Benchmark:
    bm = Benchmark.create(name="shim-test", version="1.0.0")
    bm.add_question(
        question="What is X?",
        raw_answer="X",
    )
    return bm


@pytest.mark.unit
def test_from_manual_traces_builds_store_keyed_by_question_id():
    bm = _build_benchmark_with_one_question()
    mt = ManualTraces(bm)
    mt.register_traces({"What is X?": "X is a variable"}, map_to_id=True)

    store = ReplayStore.from_manual_traces(mt, benchmark=bm, miss_policy="strict")
    assert store.miss_policy == "strict"

    from karenina.utils.checkpoint import generate_question_id

    question_id = generate_question_id("What is X?")
    entry = store.lookup(question_id=question_id)
    assert entry is not None
    assert "X is a variable" in entry.raw_trace


@pytest.mark.unit
def test_benchmark_run_verification_auto_builds_replay_store_for_manual_interface(monkeypatch):
    bm = _build_benchmark_with_one_question()
    mt = ManualTraces(bm)
    mt.register_traces({"What is X?": "X is a variable"}, map_to_id=True)

    captured: dict = {}

    # Capture the config that the underlying VerificationManager receives.
    # This is the dispatch boundary right after the auto-build runs.
    original_vm_run = bm._verification_manager.run_verification

    def _capture_config(config, *args, **kwargs):
        captured["config"] = config
        return original_vm_run(config, *args, **kwargs)

    monkeypatch.setattr(bm._verification_manager, "run_verification", _capture_config)

    # Also stub out run_single_model_verification so the captured run
    # does not actually try to call any LLMs.
    def _fake_run_single(**_kwargs):
        from karenina.schemas.verification.result import VerificationResult

        return VerificationResult.model_construct()

    monkeypatch.setattr(
        "karenina.benchmark.verification.runner.run_single_model_verification",
        _fake_run_single,
    )

    from karenina.schemas.config import ModelConfig
    from karenina.schemas.verification.config import VerificationConfig

    manual_model = ModelConfig(
        id="manual-1",
        model_name="manual",
        model_provider="manual",
        interface="manual",
        manual_traces=mt,
    )
    judge_model = ModelConfig(id="j", model_name="j", model_provider="anthropic")

    config = VerificationConfig(
        answering_models=[manual_model],
        parsing_models=[judge_model],
    )

    # The downstream pipeline runs with the stubbed
    # run_single_model_verification and completes without error. The
    # contract under test is "by the time VerificationManager.run_verification
    # is called, config.replay_store has been auto-populated."
    bm.run_verification(config)

    captured_config = captured.get("config")
    assert captured_config is not None, "VerificationManager.run_verification was never called"
    assert captured_config.replay_store is not None
    assert isinstance(captured_config.replay_store, ReplayStore)
    assert captured_config.replay_store.miss_policy == "strict"
