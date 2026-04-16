"""Build INDEX.md from already-rendered case metadata.

The indexer does not open case files. Callers (materializer.py) supply
small IndexEntry records capturing what the index needs to know.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

REASON_TRUNCATE_CHARS = 100
PASS_TABLE_LIMIT = 50


@dataclass(frozen=True)
class IndexEntry:
    """Minimal projection of a materialized case for INDEX.md assembly."""

    case_id: str
    bucket_path: str
    outcome: str
    category: str
    group: str
    stage: str
    reason: str


def _directory_map() -> list[str]:
    return [
        "- `benchmark/`: the checkpoint's questions, templates, and rubric.",
        "- `passes/`: one file per passing case.",
        "- `failures/<category>/`: one file per failure, bucketed by category.",
        "- `case_assets/`: offloaded trace content referenced from case files.",
        "- `PROMPT.md`: the prompt the analyst agent is asked to follow.",
        "- `REPORT.md`: the agent's output; absent until a launcher runs.",
    ]


def build_index_markdown(
    *,
    benchmark_name: str,
    answering_model: str,
    run_timestamp: str,
    entries: list[IndexEntry],
    run_id: str | None = None,
) -> str:
    passes = [e for e in entries if e.outcome == "pass"]
    failures = [e for e in entries if e.outcome == "failure"]
    total = len(entries)
    pass_count = len(passes)
    fail_count = len(failures)
    pass_pct = (pass_count / total * 100.0) if total else 0.0

    lines: list[str] = [
        f"# Error Analysis: {benchmark_name}",
        "",
        f"- Model: {answering_model}",
        f"- Run timestamp: {run_timestamp}",
    ]
    if run_id is not None:
        lines.append(f"- Run id: {run_id}")
    lines.extend(
        [
            f"- Total: {total}",
            f"- Passed: {pass_count} ({pass_pct:.0f}%)",
            f"- Failed: {fail_count}",
            "",
            "## Failure breakdown",
            "",
            "| Group | Category | Count |",
            "| --- | --- | --- |",
        ]
    )
    by_cat: Counter[tuple[str, str]] = Counter()
    for e in failures:
        by_cat[(e.group, e.category)] += 1
    for (group, category), count in sorted(by_cat.items()):
        lines.append(f"| {group} | {category} | {count} |")

    lines += ["", "## Directory", ""]
    lines += _directory_map()

    lines += ["", "## Failures by category", ""]
    grouped: dict[str, list[IndexEntry]] = defaultdict(list)
    for e in failures:
        grouped[e.category].append(e)
    for category in sorted(grouped):
        bucket_entries = grouped[category]
        lines.append(f"### {category} ({len(bucket_entries)})")
        lines.append("")
        lines.append("| ID | Stage | One-line reason |")
        lines.append("| --- | --- | --- |")
        for e in bucket_entries:
            reason = (e.reason or "").replace("\n", " ").strip()
            if len(reason) > REASON_TRUNCATE_CHARS:
                reason = reason[:REASON_TRUNCATE_CHARS]
            lines.append(f"| [{e.case_id}]({e.bucket_path}) | {e.stage} | {reason} |")
        lines.append("")

    lines += ["## Passes", ""]
    if passes:
        lines += ["| ID | Stage |", "| --- | --- |"]
        for e in passes[:PASS_TABLE_LIMIT]:
            lines.append(f"| {e.case_id} | {e.stage or '-'} |")
        if len(passes) > PASS_TABLE_LIMIT:
            lines.append(f"_(... and {len(passes) - PASS_TABLE_LIMIT} more)_")
    else:
        lines.append("_No passing cases._")

    return "\n".join(lines) + "\n"
