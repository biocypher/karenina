"""Build exact first-turn scenario replay with Karenina."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import tiktoken

from karenina.benchmark import Benchmark, ResultsIOManager, VerificationConfig
from karenina.replay import ReplayKey, ReplayStore, ScenarioReplayBuilder, capture_from_result_set
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import ModelIdentity, VerificationResult
from karenina.utils.checkpoint import generate_question_id
from paper.otp_sycophancy_scenarios.config import answerer_model, reference_parser_model

logger = logging.getLogger(__name__)

MAX_REPLAY_TOKENS = 125_000
MAX_REPLAY_CHARS = 300_000
TOKENIZER = "cl100k_base"
_ENCODER = tiktoken.get_encoding(TOKENIZER)

_SOURCE_NAMES = {
    "qwen3.5-122b-a10b": {"qwen3.5-122b-a10b"},
    "claude-haiku-4-5": {"claude-haiku-4-5", "claude-haiku-4-5-20251001"},
}


@dataclass(frozen=True, slots=True)
class ReplayBuild:
    """Projected replay store, filtered benchmark, and exclusion records."""

    benchmark: Benchmark
    store: ReplayStore
    exclusions: list[dict[str, object]]
    source_rows: int


@lru_cache(maxsize=8)
def _selected_rows_cached(path: Path, answerer: str) -> tuple[VerificationResult, ...]:
    """Load the exact first replicate parsed by Claude Opus."""
    result_set = ResultsIOManager.load_result_set_from_json(path)
    rows = [
        result
        for result in result_set.results
        if result.metadata.replicate == 1
        and result.metadata.answering.model_name in _SOURCE_NAMES[answerer]
        and result.metadata.parsing.model_name == "claude-opus-4-6"
    ]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for result in rows:
        if result.metadata.question_id in seen:
            duplicates.add(result.metadata.question_id)
        seen.add(result.metadata.question_id)
    if duplicates:
        raise ValueError(f"Duplicate first-replicate replay rows: {sorted(duplicates)[:5]}")
    return tuple(rows)


def _selected_rows(path: Path, answerer: str) -> list[VerificationResult]:
    """Return a fresh list backed by the process-local validated input cache."""
    return list(_selected_rows_cached(path, answerer))


def _normalize_model_identity(store: ReplayStore, answerer: str, regime: str) -> ReplayStore:
    """Map archived model aliases to the configured answerer identity."""
    current_model = answerer_model(
        answerer,
        mcp_url="http://replay.invalid/mcp" if regime == "mcp" else None,
    )
    current = ModelIdentity.from_model_config(current_model, role="answering").display_string
    normalized = ReplayStore(miss_policy="fall_through")
    for key, entry in store.entries:
        normalized.register(
            ReplayKey(
                question_id=key.question_id,
                scenario_id=key.scenario_id,
                scenario_node=key.scenario_node,
                answering_model_id=current,
                visit_index=key.visit_index,
                replicate=key.replicate,
            ),
            entry.model_copy(update={"captured_model_id": current}),
        )
    return normalized


@lru_cache(maxsize=512)
def _trace_size(trace: str) -> tuple[int, int]:
    """Cache token and character counts reused across four graph variants."""
    return len(_ENCODER.encode(trace)), len(trace)


def _eligibility(result: VerificationResult) -> tuple[list[str], int, int]:
    """Apply replay exclusions while retaining explicit abstentions."""
    template = result.template
    trace = template.raw_llm_response if template else ""
    tokens, chars = _trace_size(trace)
    failure = result.metadata.failure
    category = failure.category.value if failure else None
    explicit_abstention = category == "abstention" and bool(trace.strip())
    reasons: list[str] = []
    if (template is None or template.parsed_llm_response is None) and not explicit_abstention:
        reasons.append("missing_parsed_answer_fields")
    if failure is not None and category not in {"content", "abstention"}:
        reasons.append("non_content_qa_failure")
    if category == "abstention" and not explicit_abstention:
        reasons.append("empty_abstention_trace")
    if tokens > MAX_REPLAY_TOKENS:
        reasons.append("ask_replay_token_limit")
    if chars > MAX_REPLAY_CHARS:
        reasons.append("ask_replay_char_limit")
    return reasons, tokens, chars


def build_ask_replay(
    benchmark: Benchmark,
    qa_results_path: Path,
    *,
    answerer: str,
    regime: str = "parametric",
) -> ReplayBuild:
    """Capture, project, filter, and strictly validate ask replay."""
    rows = _selected_rows(qa_results_path, answerer)
    source_set = VerificationResultSet(results=rows)
    qa_store = capture_from_result_set(
        source_set,
        include_parsed=True,
        include_agent_traces=True,
        only_successful=False,
        replicate_selector="first",
    )
    qa_store = _normalize_model_identity(qa_store, answerer, regime)
    projection_answerer = answerer_model(
        answerer,
        mcp_url="http://replay.invalid/mcp" if regime == "mcp" else None,
    )
    config = VerificationConfig(
        answering_models=[projection_answerer],
        parsing_models=[reference_parser_model()],
        evaluation_mode="template_only",
    )
    builder = ScenarioReplayBuilder(benchmark, config=config, miss_policy="strict")
    builder.add_qa(qa_store, target_nodes=["ask"])
    projection = builder.validate()
    if projection.unmatched_targets:
        preview = [target.model_dump() for target in projection.unmatched_targets[:5]]
        raise ValueError(f"Missing ask replay coverage: {preview}")
    projected = builder.build(strict=True)

    rows_by_question = {result.metadata.question_id: result for result in rows}
    exclusions: list[dict[str, object]] = []
    excluded_scenarios: set[str] = set()
    for key, _entry in projected.entries:
        result = rows_by_question[key.question_id]
        reasons, tokens, chars = _eligibility(result)
        if reasons:
            scenario_id = key.scenario_id or ""
            excluded_scenarios.add(scenario_id)
            failure = result.metadata.failure
            exclusions.append(
                {
                    "scenario_id": scenario_id,
                    "question_id": key.question_id,
                    "reasons": reasons,
                    "raw_trace_tokens": tokens,
                    "raw_trace_chars": chars,
                    "failure_category": failure.category.value if failure else None,
                    "failure_stage": failure.stage if failure else None,
                }
            )

    # Coverage is validated explicitly for every selected ``ask`` node below.
    # Other nodes are live continuations and therefore must fall through.
    filtered = ReplayStore(miss_policy="fall_through")
    for key, entry in projected.entries:
        if key.scenario_id not in excluded_scenarios:
            filtered.register(key, entry)
    for scenario_id in excluded_scenarios:
        benchmark.remove_scenario(scenario_id)
    if not benchmark.get_scenarios():
        raise ValueError("Replay exclusions removed every scenario")
    _validate_coverage(benchmark, filtered)
    logger.info(
        "Built strict ask replay for %s: %d source rows, %d scenarios, %d exclusions",
        answerer,
        len(rows),
        len(benchmark.get_scenarios()),
        len(exclusions),
    )
    return ReplayBuild(benchmark=benchmark, store=filtered, exclusions=exclusions, source_rows=len(rows))


def _validate_coverage(benchmark: Benchmark, store: ReplayStore) -> None:
    """Refuse to run if any selected ask node could fall through to a model."""
    missing: list[str] = []
    for scenario in benchmark.get_scenarios():
        ask = scenario.nodes["ask"]
        question_id = generate_question_id(ask.question.question)
        if not store.has_any_for(question_id=question_id, scenario_id=scenario.name, scenario_node="ask"):
            missing.append(scenario.name)
    if missing:
        raise ValueError(f"Missing ask replay entries after filtering: {missing[:5]}")


__all__ = ["MAX_REPLAY_CHARS", "MAX_REPLAY_TOKENS", "ReplayBuild", "build_ask_replay"]
