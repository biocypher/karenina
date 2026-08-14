"""Rerun and analyze evidence-grounding judgments over MCP QA results."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from karenina.benchmark import (
    RowContext,
    count_deep_judgment_reasoning_tokens,
    evaluate_rubric_on_results,
    mask_graphql_schema_messages,
)
from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric
from karenina.schemas.verification import VerificationResult
from paper.common.bootstrap import input_path
from paper.common.qa_results import REFERENCE_JUDGE, iter_results, load_benchmark_text
from paper.config import (
    OTP_BENCHMARK_JSONLD,
    OTP_MCP_RESULTS,
    RESPONSE_GROUNDING_PROMPT,
    RESPONSE_GROUNDING_SCORES,
    RESPONSE_GROUNDING_TRACES,
)
from paper.otp_response_characterization.analyses.failure_tree import score_response_shapes
from paper.otp_response_characterization.analyses.maraviroc import (
    EvidenceKey,
    collect_trace_support,
)
from paper.otp_response_characterization.analyses.no_tool_call import score_no_tool_rows
from paper.otp_response_characterization.config import (
    MAX_STAGE1_INPUT_TOKENS,
    POST_HOC_WORKERS,
    evidence_grounding_judge,
    gpt_oss_judge,
)

logger = logging.getLogger(__name__)

TRAIT_NAME = "EvidenceGroundedAnswer"
KEY_COLUMNS = ["question_id", "answerer", "judge", "replicate"]
EMPTY_CONTENT_COLUMNS = ["EmptyTrace", "EmptyTrailingAI", "NoAIFinalMessage"]
PANEL_STATUSES = (
    "outside_reference_judge",
    "not_correct",
    "regex_characterized",
    "tool_less",
    "unscored",
    "grounded",
    "ungrounded",
)
MARAVIROC_APPROVAL_QUESTION_ID = "urn:uuid:question-when-was-maraviroc-first-approved-fe6f3f89"
VERBALIZATION_MISMATCH_QUESTION_IDS = frozenset(
    {
        "urn:uuid:question-what's-the-genetic-constraint-of-ensg00000143631-5fbd2683",
        "urn:uuid:question-does-kras-have-a-favourable-small-molecule-tractab-7c70d704",
    }
)
PEER_FAMILY_THRESHOLD = 4
class _Identity(BaseModel):
    """Minimal model identity stored in judgment keys."""

    model_config = ConfigDict(extra="allow")
    model_name: str


class _EvidenceKeyModel(BaseModel):
    """Validated key stored in one evidence score record."""

    model_config = ConfigDict(extra="allow")
    question_id: str
    answering: _Identity
    parsing: _Identity
    replicate: int
    result_id: str


class _EvidenceScoreRecord(BaseModel):
    """Validated stored evidence score record."""

    model_config = ConfigDict(extra="allow")
    key: _EvidenceKeyModel
    rubric_addon: dict[str, bool | None] | None


def _key_from_record(record: _EvidenceScoreRecord) -> EvidenceKey:
    """Convert a validated score record to the analysis identity."""
    return (
        record.key.question_id,
        record.key.answering.model_name,
        record.key.parsing.model_name,
        record.key.replicate,
    )


def load_evidence_scores(path: Path) -> dict[EvidenceKey, bool | None]:
    """Load stored evidence verdicts without coercing null to false."""
    scores: dict[EvidenceKey, bool | None] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = _EvidenceScoreRecord.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid evidence score at {path}:{line_number}: {error}") from error
            key = _key_from_record(record)
            if key in scores:
                raise ValueError(f"Duplicate evidence score identity: {key}")
            verdict = record.rubric_addon.get(TRAIT_NAME) if record.rubric_addon else None
            if verdict is not None and not isinstance(verdict, bool):
                raise ValueError(f"Invalid evidence verdict for {key}: {verdict!r}")
            scores[key] = verdict
    return scores


def _frame_key(row: Any) -> EvidenceKey:
    """Return the evidence identity from a named dataframe row."""
    return (str(row.question_id), str(row.answerer), str(row.judge), int(row.replicate))


def build_panel_rows(
    response_scores: pd.DataFrame,
    no_tool_scores: pd.DataFrame,
    evidence_scores: dict[EvidenceKey, bool | None],
) -> pd.DataFrame:
    """Join inputs and assign one mutually exclusive panel status per row."""
    if response_scores.duplicated(KEY_COLUMNS).any() or no_tool_scores.duplicated(KEY_COLUMNS).any():
        raise ValueError("Evidence analysis inputs contain duplicate row identities")
    merged = response_scores.merge(
        no_tool_scores[KEY_COLUMNS + ["no_tool_call", "empty_trace"]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if merged[["no_tool_call", "empty_trace"]].isna().any().any():
        raise ValueError("No-tool scores do not cover every response-shape row")
    source_keys = {_frame_key(row) for row in merged.itertuples(index=False)}
    missing = source_keys.difference(evidence_scores)
    extra = set(evidence_scores).difference(source_keys)
    if missing or extra:
        raise ValueError(
            f"Evidence score join is incomplete: missing={len(missing)}, extra={len(extra)}"
        )

    merged["evidence_grounded"] = [
        evidence_scores[_frame_key(row)] for row in merged.itertuples(index=False)
    ]
    merged["regex_characterized"] = merged[EMPTY_CONTENT_COLUMNS].astype(bool).any(axis=1)
    merged["tool_less"] = merged["no_tool_call"].astype(bool) & ~merged["regex_characterized"]

    def status(row: Any) -> str:
        if row.judge != REFERENCE_JUDGE:
            return "outside_reference_judge"
        if row.outcome_class != "pass":
            return "not_correct"
        if row.regex_characterized:
            return "regex_characterized"
        if row.tool_less:
            return "tool_less"
        if row.evidence_grounded is None:
            return "unscored"
        return "grounded" if row.evidence_grounded else "ungrounded"

    merged["panel_status"] = [status(row) for row in merged.itertuples(index=False)]
    unknown = set(merged["panel_status"]).difference(PANEL_STATUSES)
    if unknown or len(merged) != merged["panel_status"].notna().sum():
        raise ValueError(f"Evidence panel status partition is invalid: {sorted(unknown)}")
    return merged


def _peer_counts(panel: pd.DataFrame) -> dict[str, tuple[set[str], int]]:
    """Group ungrounded rows by question using distinct answerer families."""
    peers: dict[str, tuple[set[str], int]] = {}
    flagged = panel[panel["panel_status"] == "ungrounded"]
    for question_id, cohort in flagged.groupby("question_id"):
        peers[str(question_id)] = (set(cohort["answerer"].astype(str)), len(cohort))
    return peers


def build_summary(
    panel: pd.DataFrame,
    trace_support: dict[EvidenceKey, tuple[bool, bool]] | None = None,
) -> pd.DataFrame:
    """Build grounding, exclusion, answerer, peer, and Maraviroc metrics."""
    trace_support = trace_support or {}
    reference = panel[panel["judge"] == REFERENCE_JUDGE]
    correct = reference[reference["outcome_class"] == "pass"]
    status_counts = Counter(correct["panel_status"])
    evaluated = correct[correct["panel_status"].isin(["grounded", "ungrounded"])]
    ungrounded = evaluated[evaluated["panel_status"] == "ungrounded"]
    metrics: list[tuple[str, int | float]] = [
        ("reference_judge_rows", len(reference)),
        ("correct_reference_judge_rows", len(correct)),
        ("regex_characterized_ref_judge_rows", int(reference["regex_characterized"].sum())),
        ("regex_tool_less_ref_judge_rows", int(reference["tool_less"].sum())),
        ("llm_successfully_evaluated_ref_judge_residual_rows", len(evaluated)),
        ("evidence_ungrounded_ref_judge_residual_rows", len(ungrounded)),
        ("figure_5b_grounded_rows", status_counts["grounded"]),
        ("figure_5b_ungrounded_rows", status_counts["ungrounded"]),
        ("figure_5b_tool_less_omitted_rows", status_counts["tool_less"]),
        ("figure_5b_unscored_omitted_rows", status_counts["unscored"]),
        (
            "evidence_ungrounded_ref_judge_residual_rate",
            100 * len(ungrounded) / len(evaluated) if len(evaluated) else 0.0,
        ),
    ]
    for status_name in PANEL_STATUSES:
        metrics.append((f"panel_status_{status_name}_rows", int((panel["panel_status"] == status_name).sum())))

    for answerer, cohort in evaluated.groupby("answerer", sort=True):
        flagged = int((cohort["panel_status"] == "ungrounded").sum())
        slug = str(answerer).replace(".", "").replace("-", "_")
        metrics.extend(
            [
                (f"{slug}_evidence_ungrounded_rows", flagged),
                (f"{slug}_evidence_ungrounded_denominator_rows", len(cohort)),
                (f"{slug}_evidence_ungrounded_rate", 100 * flagged / len(cohort)),
            ]
        )

    peers = _peer_counts(panel)
    for threshold in (4, 6, 7):
        metrics.append(
            (
                f"evidence_grounding_peer_count_{threshold}plus_families",
                sum(len(families) >= threshold for families, _count in peers.values()),
            )
        )
    metrics.append(
        (
            "maraviroc_style_peer_count",
            sum(
                len(families) >= PEER_FAMILY_THRESHOLD
                and question_id != MARAVIROC_APPROVAL_QUESTION_ID
                and question_id not in VERBALIZATION_MISMATCH_QUESTION_IDS
                for question_id, (families, _count) in peers.items()
            ),
        )
    )

    maraviroc = ungrounded[ungrounded["question_id"] == MARAVIROC_APPROVAL_QUESTION_ID]
    maraviroc_keys = {_frame_key(row) for row in maraviroc.itertuples(index=False)}
    metrics.extend(
        [
            ("maraviroc_approval_ref_judge_flagged_answerers", maraviroc["answerer"].nunique()),
            ("maraviroc_approval_ref_judge_flagged_rows", len(maraviroc)),
            (
                "maraviroc_approval_ref_judge_approval_status_rows",
                sum(trace_support.get(key, (False, False))[0] for key in maraviroc_keys),
            ),
            (
                "maraviroc_approval_ref_judge_first_approval_date_rows",
                sum(trace_support.get(key, (False, False))[1] for key in maraviroc_keys),
            ),
        ]
    )
    return pd.DataFrame(metrics, columns=["metric", "value"])


def build_peer_table(panel: pd.DataFrame, benchmark: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """List questions with at least four flagged answerer families."""
    records: list[dict[str, Any]] = []
    for question_id, (families, replicate_count) in _peer_counts(panel).items():
        if len(families) < PEER_FAMILY_THRESHOLD:
            continue
        question, answer = benchmark.get(question_id, ("", ""))
        records.append(
            {
                "question_id": question_id,
                "question_text": question,
                "ground_truth": answer,
                "families_flagged": len(families),
                "replicates_flagged": replicate_count,
                "cohort": (
                    "verbalization_mismatch"
                    if question_id in VERBALIZATION_MISMATCH_QUESTION_IDS
                    else "ungrounded_correct"
                ),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["families_flagged", "replicates_flagged", "question_id"],
        ascending=[False, False, True],
        ignore_index=True,
    ) if records else pd.DataFrame(
        columns=[
            "question_id",
            "question_text",
            "ground_truth",
            "families_flagged",
            "replicates_flagged",
            "cohort",
        ]
    )


def _sha256(path: Path) -> str:
    """Hash a stored judgment input for manifest tracking."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(
    traces_path: Path,
    scores_path: Path,
    panel: pd.DataFrame,
) -> dict[str, Any]:
    """Describe the stored stochastic judgment run and verify token-gate skips."""
    record_count = 0
    boolean_verdicts = 0
    skipped: Counter[str] = Counter()
    modes: set[str] = set()
    oversize: set[tuple[str, str, int]] = set()
    with traces_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record_count += 1
            mode = record.get("mode")
            if mode:
                modes.add(str(mode))
            verdict = (record.get("stage2") or {}).get("verdict_value")
            if isinstance(verdict, bool):
                boolean_verdicts += 1
            reason = record.get("skipped")
            if reason:
                skipped[str(reason)] += 1
                if reason == "oversize_tokens":
                    key = record.get("key") or {}
                    replicate = key.get("replicate")
                    if not isinstance(replicate, int):
                        raise ValueError("Stored token-gate record has no integer replicate")
                    oversize.add(
                        (
                            str(key.get("question_id")),
                            str((key.get("answering") or {}).get("model_name")),
                            replicate,
                        )
                    )
    unscored = panel[panel["panel_status"] == "unscored"]
    unscored_identities = {
        (str(row.question_id), str(row.answerer), int(row.replicate))
        for row in unscored.itertuples(index=False)
    }
    if oversize != unscored_identities:
        raise ValueError("Unscored panel rows do not match stored token-gate skips")
    return {
        "schema_version": "1",
        "source_files": {
            "scores": {"path": str(scores_path), "sha256": _sha256(scores_path)},
            "traces": {"path": str(traces_path), "sha256": _sha256(traces_path)},
        },
        "stored_judgment_run": {
            "trace_records": record_count,
            "boolean_verdicts": boolean_verdicts,
            "skipped_by_reason": dict(sorted(skipped.items())),
            "modes": sorted(modes),
            "stochastic_outputs_vary": True,
        },
    }


@lru_cache(maxsize=1)
def _rubric_description() -> str:
    """Load the evidence-grounding trait text from the data deposit."""
    path = input_path(RESPONSE_GROUNDING_PROMPT)
    text = path.read_text(encoding="utf-8")
    marker = "\n## Notes\n"
    return text.split(marker, 1)[0].strip()


def _grounding_rubric(*, deep_judgment: bool) -> Rubric:
    """Build the Boolean rubric for basic or two-stage judgment."""
    return Rubric(
        llm_traits=[
            LLMRubricTrait(
                name=TRAIT_NAME,
                summary="answer grounded in retrieved evidence",
                description=_rubric_description(),
                kind="boolean",
                min_score=None,
                max_score=None,
                classes=None,
                higher_is_better=True,
                deep_judgment_enabled=deep_judgment,
                deep_judgment_excerpt_enabled=False,
                deep_judgment_max_excerpts=None,
                deep_judgment_fuzzy_match_threshold=None,
                deep_judgment_excerpt_retry_attempts=None,
                deep_judgment_search_enabled=False,
            )
        ]
    )


def _rubric_for_result(
    result: VerificationResult,
    _base: Rubric,
    benchmark: dict[str, tuple[str, str]],
    *,
    deep_judgment: bool,
) -> Rubric:
    """Append the benchmark question and reference context for each row."""
    question, reference = benchmark[result.metadata.question_id]
    description = (
        f"{_rubric_description()}\n\n---\n\n## Per-row context\n\n"
        f"**QUESTION:**\n{question}\n\n"
        f"**REFERENCE ANSWER:**\n{reference}\n---\n"
    )
    return Rubric(
        llm_traits=[
            LLMRubricTrait(
                name=TRAIT_NAME,
                summary="answer grounded in retrieved evidence",
                description=description,
                kind="boolean",
                min_score=None,
                max_score=None,
                classes=None,
                higher_is_better=True,
                deep_judgment_enabled=deep_judgment,
                deep_judgment_excerpt_enabled=False,
                deep_judgment_max_excerpts=None,
                deep_judgment_fuzzy_match_threshold=None,
                deep_judgment_excerpt_retry_attempts=None,
                deep_judgment_search_enabled=False,
            )
        ]
    )


class _GroundingInputs:
    """Cache masking and token-gate work across parser siblings."""

    def __init__(
        self,
        benchmark: dict[str, tuple[str, str]],
        judge: Any,
    ) -> None:
        self._benchmark = benchmark
        self._judge = judge
        self._masked: dict[tuple[str, str, int | None], str] = {}
        self._eligibility: dict[tuple[str, str, int | None], bool] = {}
        self.skipped: set[tuple[str, str, int | None]] = set()

    @staticmethod
    def identity(result: VerificationResult) -> tuple[str, str, int | None]:
        """Return the generated-answer identity used for deduplication."""
        return (
            result.metadata.question_id,
            result.metadata.answering.canonical_key,
            result.metadata.replicate,
        )

    def text(self, result: VerificationResult) -> str:
        """Return the schema-masked trace, cached across parser siblings."""
        identity = self.identity(result)
        if identity not in self._masked:
            raw = result.template.raw_llm_response if result.template else ""
            self._masked[identity] = mask_graphql_schema_messages(raw).text
        return self._masked[identity]

    def eligible(self, result: VerificationResult) -> bool:
        """Select eligible rows and enforce the 120k stage-one input limit."""
        template = result.template
        if template is None or template.verify_result is not True:
            return False
        if not result.metadata.answering.tools:
            return False
        identity = self.identity(result)
        if identity in self._eligibility:
            return self._eligibility[identity]
        rubric = _rubric_for_result(
            result,
            _grounding_rubric(deep_judgment=True),
            self._benchmark,
            deep_judgment=True,
        )
        prompt_tokens = count_deep_judgment_reasoning_tokens(
            self.text(result),
            rubric.llm_traits[0],
            parsing_model=self._judge,
        )
        if prompt_tokens > MAX_STAGE1_INPUT_TOKENS:
            self.skipped.add(identity)
            self._eligibility[identity] = False
            return False
        self._eligibility[identity] = True
        return True


def _result_evidence_key(result: VerificationResult) -> EvidenceKey:
    """Return the panel identity for one validated result row."""
    replicate = result.metadata.replicate
    if replicate is None:
        raise ValueError("Evidence-grounding source row has no replicate")
    return (
        result.metadata.question_id,
        result.metadata.answering.model_name,
        result.metadata.parsing.model_name,
        replicate,
    )


def _write_fresh_scores(path: Path, scores: dict[EvidenceKey, bool | None]) -> None:
    """Write fresh evidence scores as a compact audit JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for key, verdict in scores.items():
            question_id, answerer, judge, replicate = key
            record = {
                "question_id": question_id,
                "answering_model": answerer,
                "parsing_model": judge,
                "replicate": replicate,
                "evidence_grounded": verdict,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _fresh_evidence_scores(
    out_dir: Path,
    benchmark: dict[str, tuple[str, str]],
    *,
    limit: int | None = None,
    deep_judgment: bool = True,
) -> tuple[dict[EvidenceKey, bool | None], dict[str, Any]]:
    """Rerun the grounding rubric from validated MCP results."""
    source_by_id: dict[str, EvidenceKey] = {}
    scores: dict[EvidenceKey, bool | None] = {}
    for result in iter_results(input_path(OTP_MCP_RESULTS)):
        key = _result_evidence_key(result)
        if key in scores:
            raise ValueError(f"Duplicate evidence source identity: {key}")
        scores[key] = None
        source_by_id[result.metadata.result_id] = key

    judge = evidence_grounding_judge()
    inputs = _GroundingInputs(benchmark, judge)
    selected = iter_results(input_path(OTP_MCP_RESULTS))
    if limit is not None:
        eligible_results: list[VerificationResult] = []
        seen: set[tuple[str, str, int | None]] = set()
        for result in selected:
            identity = inputs.identity(result)
            if identity in seen or not inputs.eligible(result):
                continue
            seen.add(identity)
            eligible_results.append(result)
            if len(eligible_results) == limit:
                break
        selected = iter(eligible_results)
        row_filter = None
    else:
        row_filter = inputs.eligible

    logger.info(
        "Calling GPT-OSS for the evidence-grounding %s run",
        "smoke" if limit is not None else "full",
    )
    base_rubric = _grounding_rubric(deep_judgment=deep_judgment)
    judgments = evaluate_rubric_on_results(
        selected,
        base_rubric,
        judge,
        text_selector=inputs.text,
        row_context=lambda result: RowContext(
            question=benchmark[result.metadata.question_id][0],
            ground_truth=benchmark[result.metadata.question_id][1],
        ),
        row_filter=row_filter,
        rubric_factory=lambda result, rubric: _rubric_for_result(
            result,
            rubric,
            benchmark,
            deep_judgment=deep_judgment,
        ),
        max_workers=1 if limit is not None else POST_HOC_WORKERS,
    )
    judged = 0
    for judgment in judgments:
        if judgment.error is not None:
            raise RuntimeError(f"Live evidence-grounding judgment failed: {judgment.error}")
        verdict = judgment.scores.get(TRAIT_NAME)
        if not isinstance(verdict, bool):
            raise RuntimeError(f"Live evidence judge returned an invalid verdict: {verdict!r}")
        for result_id in judgment.sibling_result_ids:
            scores[source_by_id[result_id]] = verdict
        judged += 1

    path = out_dir / "evidence_grounding_judgments.jsonl"
    _write_fresh_scores(path, scores)
    manifest = {
        "schema_version": "1",
        "judgment_source": "fresh",
        "rubric_judge": {
            "model_name": "gpt-oss-120b",
            "interface": "openai_endpoint",
            "temperature": 0.0,
        },
        "evaluation": {
            "flow": (
                "two sequential calls: free-form evidence review, then Boolean extraction"
                if deep_judgment
                else "single structured Boolean call"
            ),
            "schema_masking": True,
            "max_stage1_input_tokens": MAX_STAGE1_INPUT_TOKENS,
            "input_tokenizer": "judge endpoint chat template",
            "max_output_tokens_per_call": 16_384,
            "unique_traces_judged": judged,
            "unique_traces_skipped_by_input_gate": len(inputs.skipped),
            "stochastic_outputs_vary": True,
        },
    }
    return scores, manifest


def run(out_dir: Path, *, reuse_stored_judgments: bool = False) -> None:
    """Rerun the rubric by default and write evidence-grounding tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Rebuilding deterministic exclusions through TaskEval")
    response_scores = score_response_shapes(
        iter_results(input_path(OTP_MCP_RESULTS)),
        "mcp",
        gpt_oss_judge(),
    )
    no_tool_scores = score_no_tool_rows(
        iter_results(input_path(OTP_MCP_RESULTS)),
        gpt_oss_judge(),
    )
    benchmark = load_benchmark_text(input_path(OTP_BENCHMARK_JSONLD))
    if reuse_stored_judgments:
        logger.info("Reusing archived evidence-grounding judgments")
        evidence_scores = load_evidence_scores(
            input_path(RESPONSE_GROUNDING_SCORES)
        )
    else:
        evidence_scores, manifest = _fresh_evidence_scores(out_dir, benchmark)
    panel = build_panel_rows(response_scores, no_tool_scores, evidence_scores)

    maraviroc_rows = panel[
        (panel["panel_status"] == "ungrounded")
        & (
            panel["question_id"] == MARAVIROC_APPROVAL_QUESTION_ID
        )
    ]
    maraviroc_keys = {_frame_key(row) for row in maraviroc_rows.itertuples(index=False)}
    support = collect_trace_support(
        iter_results(input_path(OTP_MCP_RESULTS)),
        maraviroc_keys,
    )
    summary = build_summary(panel, support)
    peers = build_peer_table(panel, benchmark)
    if reuse_stored_judgments:
        manifest = build_run_manifest(
            input_path(RESPONSE_GROUNDING_TRACES),
            input_path(RESPONSE_GROUNDING_SCORES),
            panel,
        )
        manifest["judgment_source"] = "archived"

    panel.to_csv(out_dir / "evidence_grounding_panel_rows.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "evidence_grounding_summary.tsv", sep="\t", index=False)
    peers.to_csv(out_dir / "evidence_grounding_peers.tsv", sep="\t", index=False)
    (out_dir / "evidence_grounding_run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_smoke(limit: int) -> None:
    """Judge a small live slice through TaskEval."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    benchmark = load_benchmark_text(input_path(OTP_BENCHMARK_JSONLD))
    smoke_dir = Path(__file__).resolve().parents[1] / "out" / "smoke" / "evidence_grounding"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    scores, _manifest = _fresh_evidence_scores(
        smoke_dir,
        benchmark,
        limit=limit,
        deep_judgment=True,
    )
    verdicts = [value for value in scores.values() if isinstance(value, bool)]
    logger.info("Fresh grounding smoke produced %d parser-row verdicts", len(verdicts))
