"""Deterministic response-shape and failure-tree analysis."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import pandas as pd

from karenina.benchmark import evaluate_rubric_on_results
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities.rubric import Rubric
from karenina.schemas.results import ResultRowKey
from karenina.schemas.verification import VerificationResult
from paper.common.bootstrap import input_path
from paper.common.qa_results import REFERENCE_JUDGE, QAResultRow, iter_results
from paper.config import OTP_MCP_RESULTS, OTP_PARAMETRIC_RESULTS
from paper.otp_response_characterization.config import (
    FAILURE_SHAPE_TRAITS,
    POST_HOC_WORKERS,
    gpt_oss_judge,
)

logger = logging.getLogger(__name__)

KEY_COLUMNS = ["question_id", "answerer", "regime", "replicate"]
TRAIT_COLUMNS = [trait.name for trait in FAILURE_SHAPE_TRAITS]
PRIMITIVE_COLUMNS = [
    "no_usable_output",
    "blank_final_assistant",
    "tool_loop_cutoff",
]
FLAG_COLUMNS = [*PRIMITIVE_COLUMNS, "technical_failure"]


def _track_results(
    results: Iterable[VerificationResult],
    regime: str,
    rows_by_key: dict[ResultRowKey, QAResultRow],
) -> Iterator[VerificationResult]:
    """Record reduced row metadata while yielding validated results."""
    for result in results:
        key = ResultRowKey.from_result(result)
        if key in rows_by_key:
            raise ValueError(f"Duplicate QA result key: {key}")
        rows_by_key[key] = QAResultRow.from_result(result, regime)
        yield result


def score_response_shapes(
    results: Iterable[VerificationResult],
    regime: str,
    parsing_model: ModelConfig,
) -> pd.DataFrame:
    """Evaluate response-shape traits through the TaskEval post-hoc facade.

    Args:
        results: Stream of validated Karenina result rows.
        regime: User-facing experimental regime label.
        parsing_model: Standard parsing model configuration. Regex traits do
            not call it, but TaskEval requires the normal evaluation config.

    Returns:
        One scored record per parser sibling.

    Raises:
        ValueError: If stored result identities are duplicated or incomplete.
        RuntimeError: If TaskEval reports a row evaluation failure.
    """
    rows_by_key: dict[ResultRowKey, QAResultRow] = {}
    records: list[dict[str, object]] = []
    judgments = evaluate_rubric_on_results(
        _track_results(results, regime, rows_by_key),
        Rubric(regex_traits=FAILURE_SHAPE_TRAITS),
        parsing_model,
        collapse_parser_siblings=True,
        max_workers=POST_HOC_WORKERS,
    )
    for judgment in judgments:
        if judgment.error is not None:
            raise RuntimeError(f"TaskEval failed for response-shape row {judgment.key}: {judgment.error}")
        flags = {name: judgment.scores.get(name) for name in TRAIT_COLUMNS}
        if not all(isinstance(value, bool) for value in flags.values()):
            raise ValueError(f"TaskEval returned incomplete response-shape scores for {judgment.key}")
        for sibling_key in judgment.sibling_keys:
            row = rows_by_key[sibling_key]
            records.append(
                {
                    "question_id": row.question_id,
                    "answerer": row.answerer,
                    "regime": row.regime,
                    "replicate": row.replicate,
                    "judge": row.parser,
                    "outcome_class": row.outcome,
                    **flags,
                }
            )
    return pd.DataFrame(
        records,
        columns=[*KEY_COLUMNS, "judge", "outcome_class", *TRAIT_COLUMNS],
    )


def build_instance_flags(scores: pd.DataFrame) -> pd.DataFrame:
    """Collapse parser siblings into one exclusive response classification."""
    required = set(KEY_COLUMNS + TRAIT_COLUMNS)
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Response scores are missing columns: {', '.join(sorted(missing))}")

    siblings = scores.groupby(KEY_COLUMNS, dropna=False)[TRAIT_COLUMNS].nunique(dropna=False)
    if not siblings.le(1).all().all():
        raise ValueError("Parser siblings have inconsistent response-shape flags")

    unique = scores.drop_duplicates(KEY_COLUMNS).copy()
    flags = unique[KEY_COLUMNS].copy()
    flags["no_usable_output"] = unique["EmptyTrace"].astype(bool)
    flags["blank_final_assistant"] = unique["EmptyTrailingAI"].astype(bool)
    flags["tool_loop_cutoff"] = unique["NoAIFinalMessage"].astype(bool) | unique["BracketedTraceNote"].astype(bool)
    if flags[PRIMITIVE_COLUMNS].sum(axis=1).gt(1).any():
        raise ValueError("Response-shape technical-failure classes overlap")
    flags["technical_failure"] = flags[PRIMITIVE_COLUMNS].any(axis=1)
    return flags.reset_index(drop=True)


def summarize_response_shapes(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build judge-level and generated-answer response-shape summaries."""
    summary_rows: list[dict[str, object]] = []
    unique_rows: list[dict[str, object]] = []
    answerer_rows: list[dict[str, object]] = []

    for regime, cohort in scores.groupby("regime", sort=True):
        total_judges = len(cohort)
        total_unique = len(cohort.drop_duplicates(KEY_COLUMNS))
        any_malformed = cohort[TRAIT_COLUMNS].any(axis=1)
        trait_masks = {
            "AnyMalformedOutput": any_malformed,
            **{trait: cohort[trait].astype(bool) for trait in TRAIT_COLUMNS},
        }
        for trait, mask in trait_masks.items():
            hits = cohort[mask]
            unique_hits = hits.drop_duplicates(KEY_COLUMNS)
            if trait != "AnyMalformedOutput":
                summary_rows.append(
                    {
                        "regime": regime,
                        "trait": trait,
                        "hits": len(hits),
                        "total": total_judges,
                        "rate": len(hits) / total_judges if total_judges else 0.0,
                    }
                )
            unique_rows.append(
                {
                    "regime": regime,
                    "trait": trait,
                    "judge_instances": len(hits),
                    "unique_instances": len(unique_hits),
                    "judge_instance_rate": len(hits) / total_judges if total_judges else 0.0,
                    "unique_instance_rate": len(unique_hits) / total_unique if total_unique else 0.0,
                }
            )
            for answerer, answerer_hits in hits.groupby("answerer", sort=True):
                answerer_rows.append(
                    {
                        "regime": regime,
                        "trait": trait,
                        "answerer": answerer,
                        "judge_instances": len(answerer_hits),
                        "unique_instances": len(answerer_hits.drop_duplicates(KEY_COLUMNS)),
                    }
                )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(unique_rows),
        pd.DataFrame(answerer_rows),
    )


def load_blank_final_counts(path: Path) -> Counter[str]:
    """Count successful stored classifications for blank-final traces."""
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            result_id = str(record.get("representative_result_id") or "")
            classification = record.get("classification")
            if not result_id or not classification:
                raise ValueError(f"Incomplete blank-final judgment at {path}:{line_number}")
            if result_id in seen:
                raise ValueError(f"Duplicate blank-final judgment for {result_id}")
            seen.add(result_id)
            if record.get("error") is None:
                counts[str(classification)] += 1
    return counts


def build_failure_tree_edges(
    scores: pd.DataFrame,
    flags: pd.DataFrame,
    blank_final_counts: Mapping[str, int],
    reference_judge: str = REFERENCE_JUDGE,
) -> pd.DataFrame:
    """Build an exclusive failure-tree edge table from current inputs."""
    reference = scores[scores["judge"] == reference_judge][[*KEY_COLUMNS, "outcome_class"]].copy()
    if reference.duplicated(KEY_COLUMNS).any():
        raise ValueError("Reference-judge rows contain duplicate generated-answer keys")
    reference = reference.merge(flags, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if reference[FLAG_COLUMNS].isna().any().any():
        raise ValueError("Response flags do not cover every reference-judge row")
    if ((reference["outcome_class"] == "pass") & reference["technical_failure"]).any():
        raise ValueError("A reference-judge pass was flagged as a technical failure")

    failures = reference[reference["outcome_class"] != "pass"].copy()
    failures["failure_tree_class"] = "content"
    failures.loc[failures["technical_failure"], "failure_tree_class"] = "technical"
    failures.loc[failures["outcome_class"] == "abstain", "failure_tree_class"] = "abstention"

    def count(regime: str, tree_class: str) -> int:
        return int(((failures["regime"] == regime) & (failures["failure_tree_class"] == tree_class)).sum())

    regime_totals = failures.groupby("regime").size().to_dict()
    parametric_total = int(regime_totals.get("parametric", 0))
    mcp_total = int(regime_totals.get("mcp", 0))
    parametric = {name: count("parametric", name) for name in ("content", "abstention", "technical")}
    mcp = {name: count("mcp", name) for name in ("content", "abstention", "technical")}

    mcp_technical = failures[(failures["regime"] == "mcp") & (failures["failure_tree_class"] == "technical")]
    primitives = {column: int(mcp_technical[column].sum()) for column in PRIMITIVE_COLUMNS}
    blank_classes = {
        "no_answer": int(blank_final_counts.get("no_answer_gave_up", 0)),
        "wrong_result": int(blank_final_counts.get("wrong_result_no_final_message", 0)),
        "answer_present": int(blank_final_counts.get("answer_present_no_final_message", 0)),
    }
    checks = {
        "root regime split": parametric_total + mcp_total == len(failures),
        "parametric failure split": sum(parametric.values()) == parametric_total,
        "MCP failure split": sum(mcp.values()) == mcp_total,
        "MCP technical split": sum(primitives.values()) == mcp["technical"],
        "blank-final split": sum(blank_classes.values()) == primitives["blank_final_assistant"],
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"Failure tree is not internally consistent: {', '.join(failed)}")

    rows = [
        ("all_failed", "parametric_failed", "regime = parametric", parametric_total),
        ("all_failed", "mcp_failed", "regime = MCP", mcp_total),
        ("parametric_failed", "parametric_content", "biological content failure", parametric["content"]),
        ("parametric_failed", "parametric_abstention", "abstention guard", parametric["abstention"]),
        ("parametric_failed", "parametric_malformed", "technical failure", parametric["technical"]),
        ("mcp_failed", "mcp_content", "biological content failure", mcp["content"]),
        ("mcp_failed", "mcp_abstention", "abstention guard", mcp["abstention"]),
        ("mcp_failed", "mcp_malformed", "technical failure", mcp["technical"]),
        ("mcp_malformed", "mcp_empty_trace", "fully empty trace", primitives["no_usable_output"]),
        ("mcp_malformed", "mcp_blank_final", "blank final assistant message", primitives["blank_final_assistant"]),
        ("mcp_malformed", "mcp_tool_cutoff", "tool-loop cutoff", primitives["tool_loop_cutoff"]),
        ("mcp_blank_final", "mcp_no_answer", "LLM: no answer reached", blank_classes["no_answer"]),
        ("mcp_blank_final", "mcp_wrong_result", "LLM: wrong result reached", blank_classes["wrong_result"]),
        ("mcp_blank_final", "mcp_answer_present", "LLM: correct answer present", blank_classes["answer_present"]),
    ]
    return pd.DataFrame(
        [
            {
                "edge_order": index,
                "parent": parent,
                "child": child,
                "condition": condition,
                "count": edge_count,
            }
            for index, (parent, child, condition, edge_count) in enumerate(rows, start=1)
        ]
    )


def run(out_dir: Path, *, missing_final_judgments: Path) -> None:
    """Write response-shape summaries and the failure tree into ``out_dir``."""
    logger.info("Evaluating deterministic response-shape traits")
    parsing_model = gpt_oss_judge()
    scores = pd.concat(
        [
            score_response_shapes(
                iter_results(input_path(OTP_PARAMETRIC_RESULTS)),
                "parametric",
                parsing_model,
            ),
            score_response_shapes(
                iter_results(input_path(OTP_MCP_RESULTS)),
                "mcp",
                parsing_model,
            ),
        ],
        ignore_index=True,
    )
    flags = build_instance_flags(scores)
    summary, unique_summary, by_answerer = summarize_response_shapes(scores)
    blank_counts = load_blank_final_counts(missing_final_judgments)
    edges = build_failure_tree_edges(scores, flags, blank_counts)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "error_characterization_summary.tsv", sep="\t", index=False)
    unique_summary.to_csv(out_dir / "error_characterization_unique_summary.tsv", sep="\t", index=False)
    by_answerer.to_csv(out_dir / "error_characterization_unique_by_answerer.tsv", sep="\t", index=False)
    flags.astype(dict.fromkeys(FLAG_COLUMNS, int)).to_csv(
        out_dir / "error_characterization_instance_flags.tsv", sep="\t", index=False
    )
    edges.to_csv(out_dir / "failure_mode_tree_edges.tsv", sep="\t", index=False)
