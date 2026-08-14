"""Direct summaries for adversarial alternative generation."""

from __future__ import annotations

import csv
import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from paper.otp_adversarial_generation.generation import AdversarialPair, pair_rows

logger = logging.getLogger(__name__)


def summarize_pairs(pairs: Iterable[AdversarialPair]) -> list[dict[str, object]]:
    """Count rows by source strata and generation route."""
    counts: Counter[tuple[str, str, float, str, str]] = Counter(
        (
            pair.area,
            pair.question_type,
            pair.source_difficulty,
            pair.generation_route,
            pair.review_status,
        )
        for pair in pairs
    )
    return [
        {
            "area": area,
            "question_type": question_type,
            "source_difficulty": difficulty,
            "generation_route": route,
            "review_status": status,
            "count": count,
        }
        for (area, question_type, difficulty, route, status), count in sorted(counts.items())
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write records as a CSV with a stable header."""
    if not rows:
        raise ValueError(f"Cannot write an empty analysis table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(pairs: list[AdversarialPair], output_dir: Path) -> None:
    """Write validated pairs and direct composition summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "adversarial_pairs.csv", pair_rows(pairs))
    _write_csv(output_dir / "pair_composition.csv", summarize_pairs(pairs))
    logger.info("Wrote adversarial pair analysis to %s", output_dir)


__all__ = ["summarize_pairs", "write_analysis"]
