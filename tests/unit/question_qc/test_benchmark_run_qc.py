"""Unit tests for Benchmark.run_qc facade with fake agents (no live models)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from karenina import Benchmark
from karenina.question_qc import QcConfig, RoleModelConfig
from karenina.question_qc.models import QcResult, QcTurn
from karenina.schemas.config import ModelConfig


class ScriptedAgent:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self._last: QcTurn | None = None

    def reset_session(self) -> None:
        self._last = None

    def snapshot_turn(self) -> QcTurn | None:
        return self._last

    async def cancel_turn(self) -> None:
        return None

    async def run_turn(self, prompt: str, **kwargs) -> QcTurn:  # type: ignore[no-untyped-def]
        if not self.responses:
            turn = QcTurn(text="", error="exhausted", prompt=prompt, stop_reason="error")
            self._last = turn
            return turn
        text = self.responses.pop(0)
        turn = QcTurn(
            text=text,
            prompt=prompt,
            stop_reason="completed",
            raw_trace=text,
            trace_messages=[{"role": "assistant", "content": text}],
            stage=kwargs.get("stage"),
        )
        self._last = turn
        return turn


def _dummy_config() -> QcConfig:
    dummy = ModelConfig(
        id="dummy",
        model_name="dummy",
        model_provider="openai",
        interface="langchain",
    )
    return QcConfig(
        proposer=RoleModelConfig(model=dummy),
        validator=RoleModelConfig(model=dummy),
        reviewer=RoleModelConfig(model=dummy),
        runtime={
            "investigation_seconds": 30,
            "wrap_up_seconds": 5,
            "conclusion_seconds": 5,
        },
    )


@pytest.fixture
def benchmark_with_qa() -> Benchmark:
    b = Benchmark.create(name="QC Test", version="0.1.0")
    b.add_question(
        question="What is the target of venetoclax?",
        raw_answer="BCL2",
    )
    b.add_question(
        question="Empty answer should be skipped by default",
        raw_answer="",
    )
    return b


def test_run_qc_default_skips_empty_raw_answer(benchmark_with_qa: Benchmark) -> None:
    config = _dummy_config()
    scripted = [
        ScriptedAgent(
            [json.dumps({"decision": "propose", "witness": "W", "explanation": "e"})]
        ),
        ScriptedAgent(
            [json.dumps({"supports_claim": True, "reasoning": "r", "quality_issues": ""})]
        ),
        ScriptedAgent(
            [json.dumps({"classification": "supported", "reasoning": "r", "quality_issues": ""})]
        ),
    ]
    agent_iter = iter(scripted)

    with patch(
        "karenina.question_qc.runner.build_qc_agent",
        side_effect=lambda *a, **k: next(agent_iter),
    ):
        results = benchmark_with_qa.run_qc(config, async_enabled=False)

    assert len(results) == 1
    assert results.results[0].terminal_status == "supported"
    assert results.results[0].witness == "W"
    assert results.results[0].attempts[0].proposer_turn is not None
    assert results.results[0].attempts[0].proposer_turn.trace_messages


def test_run_qc_question_ids_filter(benchmark_with_qa: Benchmark) -> None:
    ids = benchmark_with_qa.get_question_ids()
    qid = ids[0]
    config = _dummy_config()

    def fake_eval(question, cfg, *, evidence_context, run_name):  # type: ignore[no-untyped-def]
        return QcResult(
            question_id=question.question_id,
            terminal_status="inconclusive",
            run_name=run_name,
        )

    with patch("karenina.question_qc.runner._evaluate_one_sync", side_effect=fake_eval):
        results = benchmark_with_qa.run_qc(config, question_ids=[qid], async_enabled=False)

    assert len(results) == 1
    assert results.results[0].question_id == qid


def test_run_qc_missing_question_raises(benchmark_with_qa: Benchmark) -> None:
    config = _dummy_config()
    with pytest.raises(ValueError, match="not found"):
        benchmark_with_qa.run_qc(config, question_ids=["does-not-exist"], async_enabled=False)
