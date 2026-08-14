"""Load inputs and construct adversarial generation benchmarks."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_BINARY_FLIPS = {"yes": "No", "no": "Yes", "true": "False", "false": "True"}


class BenchmarkItem(BaseModel):
    """One source question used to generate adversarial alternatives."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    area: str = Field(min_length=1)
    question_type: str = Field(min_length=1)
    question: str = Field(min_length=1)
    ground_truth: str = Field(min_length=1)
    source_difficulty: float

    @field_validator("item_id")
    @classmethod
    def _normalize_id(cls, value: str) -> str:
        """Normalize numeric identifiers to the archive's three digits."""
        stripped = value.strip()
        return f"{int(stripped):03d}" if stripped.isdigit() else stripped

    @property
    def is_binary(self) -> bool:
        """Return whether the reference has exactly one binary flip."""
        return self.ground_truth.strip().casefold() in _BINARY_FLIPS


class AdversarialPair(BaseModel):
    """Validated draft or curator-approved pair of wrong alternatives."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    area: str
    question_type: str
    question: str
    ground_truth: str
    source_difficulty: float
    hard_adversarial: str = Field(min_length=1)
    easy_adversarial: str = Field(min_length=1)
    hard_rationale: str = Field(min_length=1)
    easy_rationale: str = Field(min_length=1)
    evidence_summary: str = Field(min_length=1)
    generation_route: str
    review_status: str
    model_name: str | None = None
    trace_id: str | None = None

    @model_validator(mode="after")
    def _wrong_answers_differ_from_truth(self) -> AdversarialPair:
        """Reject alternatives that simply repeat the reference answer."""
        truth = self.ground_truth.strip().casefold()
        if self.hard_adversarial.strip().casefold() == truth:
            raise ValueError("Hard adversarial answer repeats the ground truth")
        if self.easy_adversarial.strip().casefold() == truth:
            raise ValueError("Easy adversarial answer repeats the ground truth")
        return self


def flip_binary_answer(answer: str) -> str:
    """Return the only false value for a Boolean-like answer."""
    normalized = answer.strip().casefold()
    if normalized not in _BINARY_FLIPS:
        raise ValueError(f"Not a binary answer: {answer!r}")
    return _BINARY_FLIPS[normalized]


def load_benchmark_items(path: Path) -> list[BenchmarkItem]:
    """Load and validate the source benchmark CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        items = [
            BenchmarkItem(
                item_id=row["id"],
                area=row["Area"].strip(),
                question_type=row["Subcategories"].strip(),
                question=row["Question"].strip(),
                ground_truth=row["Answer"].strip(),
                source_difficulty=float(row["Complexity"]),
            )
            for row in csv.DictReader(handle)
        ]
    ids = [item.item_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Source benchmark contains duplicate item IDs")
    return items


def binary_pair(item: BenchmarkItem) -> AdversarialPair:
    """Create the deterministic draft for a binary source question."""
    flipped = flip_binary_answer(item.ground_truth)
    rationale = "The binary answer space has one possible false value."
    return AdversarialPair(
        **item.model_dump(),
        hard_adversarial=flipped,
        easy_adversarial=flipped,
        hard_rationale=rationale,
        easy_rationale=rationale,
        evidence_summary="No MCP research was needed for the deterministic binary flip.",
        generation_route="binary_flip",
        review_status="draft_requires_curator",
    )


def parse_sample_text(
    text: str,
    item: BenchmarkItem,
    *,
    model_name: str | None,
    trace_id: str | None,
) -> AdversarialPair:
    """Parse and validate one structured Claude Code sample."""
    sections: dict[str, dict[str, str] | str] = {
        "hard": {},
        "easy": {},
        "evidence": "",
    }
    section: str | None = None
    evidence_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "--- HARD ADVERSARIAL ---":
            section = "hard"
        elif stripped == "--- EASY ADVERSARIAL ---":
            section = "easy"
        elif stripped == "--- MCP DATA USED ---":
            section = "evidence"
        elif stripped.startswith("---"):
            section = None
        elif section in {"hard", "easy"}:
            target = sections[section]
            if not isinstance(target, dict):
                raise TypeError("Invalid sample parser state")
            if stripped.startswith("Answer:"):
                target["answer"] = stripped.removeprefix("Answer:").strip()
            elif stripped.startswith("Reasoning:"):
                target["reasoning"] = stripped.removeprefix("Reasoning:").strip()
            elif stripped and "reasoning" in target:
                target["reasoning"] += " " + stripped
        elif section == "evidence" and stripped:
            evidence_lines.append(stripped)
    hard = sections["hard"]
    easy = sections["easy"]
    if not isinstance(hard, dict) or not isinstance(easy, dict):
        raise TypeError("Invalid sample parser state")
    return AdversarialPair(
        **item.model_dump(),
        hard_adversarial=hard.get("answer", ""),
        easy_adversarial=easy.get("answer", ""),
        hard_rationale=hard.get("reasoning", ""),
        easy_rationale=easy.get("reasoning", ""),
        evidence_summary=" ".join(evidence_lines),
        generation_route="standalone_claude_code_mcp",
        review_status="draft_requires_curator",
        model_name=model_name,
        trace_id=trace_id,
    )


def load_approved_archive(path: Path) -> list[AdversarialPair]:
    """Load the manually curated archive as approved sample rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pairs = [
        AdversarialPair(
            item_id=row["id"],
            area=row["area"].strip(),
            question_type=row["type"].strip(),
            question=row["question"].strip(),
            ground_truth=row["ground_truth"].strip(),
            source_difficulty=float(row["difficulty"]),
            hard_adversarial=row["hard_adversarial"].strip(),
            easy_adversarial=row["easy_adversarial"].strip(),
            hard_rationale="See the archived per-item curator record.",
            easy_rationale="See the archived per-item curator record.",
            evidence_summary="See the archived per-item MCP trace and curator record.",
            generation_route=(
                "binary_flip"
                if row["ground_truth"].strip().casefold() in _BINARY_FLIPS
                else "standalone_claude_code_mcp"
            ),
            review_status="curator_approved_archive",
            model_name=None,
            trace_id=None,
        )
        for row in rows
    ]
    ids = [pair.item_id for pair in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("Approved archive contains duplicate item IDs")
    return pairs


def pair_rows(pairs: Iterable[AdversarialPair]) -> list[dict[str, object]]:
    """Return serializable rows in stable field order."""
    return [asdict(_PairRow.from_pair(pair)) for pair in pairs]


@dataclass(frozen=True, slots=True)
class _PairRow:
    item_id: str
    area: str
    question_type: str
    question: str
    ground_truth: str
    source_difficulty: float
    hard_adversarial: str
    easy_adversarial: str
    hard_rationale: str
    easy_rationale: str
    evidence_summary: str
    generation_route: str
    review_status: str
    model_name: str | None
    trace_id: str | None

    @classmethod
    def from_pair(cls, pair: AdversarialPair) -> _PairRow:
        return cls(**pair.model_dump())


__all__ = [
    "AdversarialPair",
    "BenchmarkItem",
    "binary_pair",
    "flip_binary_answer",
    "load_approved_archive",
    "load_benchmark_items",
    "pair_rows",
    "parse_sample_text",
]
