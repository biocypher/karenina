"""Live-model hybrid scenario replay (setup canned, followup live).

Mirrors the offline scenario replay test idea but runs against the
real vLLM endpoint. Gated on KARENINA_LIVE_TESTS=1 via the shared
conftest. Skipped in offline environments.

Goal: prove that capturing one turn of a multi-turn scenario, then
re-running with that turn canned and the next turn forced through
the live model, reproduces the canned content bit-identically and
runs the live followup successfully, in less wall-clock time than
the full live baseline.
"""

from __future__ import annotations

import time

import pytest

from karenina.benchmark import Benchmark
from karenina.replay import ReplayStore
from karenina.schemas.entities.question import Question
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode
from karenina.schemas.verification.config import VerificationConfig

TEMPLATE = """
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.verified_field import VerifiedField
from karenina.schemas.primitives.comparisons import BooleanMatch


class Answer(BaseAnswer):
    answer: str = VerifiedField(
        description="The free-form answer text",
        verify_with=BooleanMatch(),
        ground_truth="non-empty",
        default="",
    )

    def verify(self) -> bool:
        return len(self.answer.strip()) > 0
"""


def _build_two_node_scenario_benchmark() -> Benchmark:
    setup_q = Question(
        question="Name one gene associated with cystic fibrosis. Just the gene symbol.",
        raw_answer="CFTR",
        keywords=[],
        answer_template=TEMPLATE,
    )
    followup_q = Question(
        question="What chromosome is that gene on? Just the chromosome number.",
        raw_answer="7",
        keywords=[],
        answer_template=TEMPLATE,
    )
    defn = ScenarioDefinition(
        name="live-setup-followup",
        entry_node="setup",
        nodes={
            "setup": ScenarioNode(node_id="setup", question=setup_q),
            "followup": ScenarioNode(node_id="followup", question=followup_q),
        },
        edges=[
            ScenarioEdge(source="setup", target="followup"),
            ScenarioEdge(source="followup", target=END),
        ],
        outcome_criteria=[],
    )
    bm = Benchmark.create(name="live-scenario-replay", version="1.0.0")
    bm.add_scenario(defn)
    return bm


@pytest.mark.live
@pytest.mark.integration
def test_scenario_live_hybrid_replay(tmp_path, live_primary_model, live_parsing_model):
    bm = _build_two_node_scenario_benchmark()

    base_config = VerificationConfig(
        answering_models=[live_primary_model],
        parsing_models=[live_parsing_model],
        request_timeout=180.0,
    )

    # 1. Live baseline for both nodes
    start_live = time.monotonic()
    live_result = bm.run_verification(base_config)
    live_elapsed = time.monotonic() - start_live
    assert live_result.scenario_results is not None
    assert len(live_result.scenario_results) == 1
    scen = live_result.scenario_results[0]
    assert len(scen.history) == 2
    setup_live_raw = scen.history[0].raw_response

    # 2. Capture the full result, then drop the followup entries to
    # force the followup turn to run live during replay.
    store = live_result.to_replay_store(include_parsed=True)
    store.entries = [(key, entry) for (key, entry) in store.entries if key.scenario_node == "setup"]
    store._rebuild_indexes()  # private but documented; required after direct entries mutation
    assert len(store.entries) == 1

    # Save + reload round trip for good measure
    path = tmp_path / "scenario_live_replay.json"
    store.save(path)
    reloaded = ReplayStore.load(path)

    # 3. Replay: setup canned, followup live
    replay_config = VerificationConfig(
        answering_models=[live_primary_model],
        parsing_models=[live_parsing_model],
        request_timeout=180.0,
        replay_store=reloaded,
    )

    start_replay = time.monotonic()
    replay_result = bm.run_verification(replay_config)
    replay_elapsed = time.monotonic() - start_replay

    assert replay_result.scenario_results is not None
    replay_scen = replay_result.scenario_results[0]
    assert len(replay_scen.history) == 2

    # Setup turn should match the captured content bit-for-bit
    assert replay_scen.history[0].raw_response == setup_live_raw

    # Followup turn should have run live against the model. We cannot
    # assert on its exact text (the LLM is non-deterministic even at
    # temperature=0) but we can assert it produced content.
    assert replay_scen.history[1].raw_response.strip() != ""

    # Sanity: hybrid replay should be meaningfully faster than full
    # live because one of two turns was skipped. Allow a generous
    # margin; the point is to catch regressions where both turns hit
    # the model.
    assert replay_elapsed < live_elapsed, (
        f"Replay ({replay_elapsed:.1f}s) was not faster than live ({live_elapsed:.1f}s)"
    )
