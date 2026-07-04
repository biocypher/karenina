"""Live-model end-to-end test for the QA replay path.

Gated on KARENINA_LIVE_TESTS=1 and endpoint reachability via the
shared fixtures in tests/replay/conftest.py. Skipped in offline
environments.

Goal: prove that capturing a live VerificationResultSet, persisting
it to JSON, reloading, and re-running with the loaded ReplayStore
produces a bit-identical raw_trace and a verify_result of True,
in dramatically less wall-clock time than the live baseline.
"""

from __future__ import annotations

import time

import pytest

from karenina.benchmark import Benchmark
from karenina.replay import ReplayStore
from karenina.schemas.verification.config import VerificationConfig

TEMPLATE = """
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.verified_field import VerifiedField
from karenina.schemas.primitives.comparisons import BooleanMatch


class Answer(BaseAnswer):
    value: int = VerifiedField(
        description="The integer answer to the math question",
        verify_with=BooleanMatch(),
        ground_truth=42,
        default=0,
    )

    def verify(self) -> bool:
        return self.value == 42
"""


@pytest.mark.live
@pytest.mark.integration
def test_qa_live_capture_save_load_replay(tmp_path, live_primary_model, live_parsing_model):
    bm = Benchmark.create(name="live-qa-replay", version="1.0.0")
    bm.add_question(
        question="What is 6 times 7? Answer with just the number.",
        raw_answer="42",
        answer_template=TEMPLATE,
    )

    live_config = VerificationConfig(
        answering_models=[live_primary_model],
        parsing_models=[live_parsing_model],
        request_timeout=180.0,
    )

    # 1. Live baseline run
    live_start = time.monotonic()
    live_result = bm.run_verification(live_config)
    live_elapsed = time.monotonic() - live_start
    assert len(live_result.results) == 1
    vr = live_result.results[0]
    assert vr.template.verify_result is True, f"Live run failed; raw_trace={vr.template.raw_llm_response!r}"

    # 2. Capture
    store = live_result.to_replay_store(include_parsed=True)
    assert len(store.entries) == 1
    captured_key, captured_entry = store.entries[0]
    assert captured_entry.raw_trace == vr.template.raw_llm_response
    # parsed_answer_fields should match what the live judge produced
    assert captured_entry.parsed_answer_fields == vr.template.parsed_llm_response

    # 3. Save + reload
    path = tmp_path / "qa_live_replay.json"
    store.save(path)
    reloaded = ReplayStore.load(path)
    assert len(reloaded.entries) == 1

    # 4. Replay: wire the reloaded store into a new config with the same
    # models. We do NOT monkey-patch anything; the short-circuit inside
    # GenerateAnswerStage should bypass the live call entirely. The
    # parsed bypass inside ParseTemplateStage should also skip the
    # judge because parsed_answer_fields was captured.
    replay_config = VerificationConfig(
        answering_models=[live_primary_model],
        parsing_models=[live_parsing_model],
        request_timeout=180.0,
        replay_store=reloaded,
    )

    replay_start = time.monotonic()
    replay_result = bm.run_verification(replay_config)
    replay_elapsed = time.monotonic() - replay_start

    assert len(replay_result.results) == 1
    replay_vr = replay_result.results[0]
    assert replay_vr.template.raw_llm_response == vr.template.raw_llm_response
    assert replay_vr.template.verify_result is True

    # Sanity: replay must be dramatically faster than live (no network
    # round trips). The 10s ceiling catches accidental regressions
    # where the replay path silently still hits the LLM.
    assert replay_elapsed < 10.0, f"Replay took {replay_elapsed:.1f}s; expected well under 10s"
    assert replay_elapsed < live_elapsed, (
        f"Replay ({replay_elapsed:.1f}s) was not faster than live ({live_elapsed:.1f}s)"
    )
