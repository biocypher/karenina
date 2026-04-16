"""Materialize a VerificationResultSet + Benchmark into an analysis directory.

This module owns: the partition over results vs scenario_results, filename
sanitization and collision handling, benchmark artifact writing, the
force/REPORT.previous.md rules, and the INDEX.md assembly (via indexer).
Per-case rendering is delegated to case_renderer.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from karenina.benchmark.error_analysis.exceptions import MaterializationError
from karenina.schemas.results.verification_result_set import VerificationResultSet
from karenina.schemas.scenario.state import ScenarioExecutionResult
from karenina.schemas.verification.result import VerificationResult
from karenina.schemas.verification.result_components import VerificationResultMetadata

logger = logging.getLogger(__name__)

_ID_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_id(raw: str) -> str:
    """Replace any character that is not alphanumeric, underscore, or dash with an underscore.

    Args:
        raw: Original identifier (question_id, scenario_id, etc.).

    Returns:
        Sanitized identifier safe for use as a filename component.
    """
    return _ID_SANITIZE_RE.sub("_", raw)


def _hash_suffix(result_id: str) -> str:
    """Compute an 8-char SHA-1 suffix for collision disambiguation."""
    return hashlib.sha1(result_id.encode("utf-8")).hexdigest()[:8]


def case_filename(
    *,
    metadata: VerificationResultMetadata | None = None,
    scenario: ScenarioExecutionResult | None = None,
    monotonic_n: int = 1,
    existing: set[str] | None = None,
) -> str:
    """Generate the on-disk filename for a QA or scenario case.

    QA cases use ``q_{question_id}[__rep_{N}].md``. Scenario cases use
    ``scenario_{scenario_id}__run_{N}.md`` where N is the scenario's
    ``replicate`` if present or a monotonic counter otherwise. On
    collision with ``existing``, append an 8-char SHA-1 suffix derived
    from the result_id (QA) or ``scenario_id:monotonic_n`` (scenario).

    Args:
        metadata: VerificationResultMetadata for a QA case. Mutually
            exclusive with ``scenario``.
        scenario: ScenarioExecutionResult for a scenario case.
        monotonic_n: Fallback scenario run number when
            ``scenario.replicate`` is None.
        existing: Set of filenames already assigned in the current
            analysis directory; triggers the SHA-1 suffix on match.

    Returns:
        The filename (basename only, no directory component).
    """
    if scenario is None and metadata is None:
        raise ValueError("case_filename requires either metadata or scenario")
    existing = existing or set()
    if scenario is not None:
        raw_id = sanitize_id(scenario.scenario_id)
        n = scenario.replicate if scenario.replicate is not None else monotonic_n
        base = f"scenario_{raw_id}__run_{n}.md"
    else:
        # metadata is guaranteed non-None here by the guard at the top;
        # the `assert` is for the type checker only (runtime safety is
        # provided by the ValueError above, which survives python -O).
        assert metadata is not None
        raw_id = sanitize_id(metadata.question_id)
        base = f"q_{raw_id}"
        if metadata.replicate is not None:
            base = f"{base}__rep_{metadata.replicate}"
        base = f"{base}.md"

    if base not in existing:
        return base
    # Collision: append an 8-char SHA-1 suffix of the result_id.
    if metadata is not None:
        suffix = _hash_suffix(metadata.result_id)
    else:
        # Scenario collision: use scenario_id + run number.
        assert scenario is not None
        suffix = _hash_suffix(f"{scenario.scenario_id}:{monotonic_n}")
    stem, _, _ = base.rpartition(".md")
    hashed = f"{stem}__h{suffix}.md"
    if hashed in existing:
        raise MaterializationError(
            "Filename collision after SHA-1 disambiguation.",
            details={"base": base, "hashed": hashed},
        )
    return hashed


def partition_results(
    result_set: VerificationResultSet,
) -> tuple[list[VerificationResult], list[ScenarioExecutionResult]]:
    """Split VerificationResultSet into classical QA results and scenario runs.

    Classical QA results are those whose ``metadata.scenario_id`` is None.
    Scenario runs come directly from ``result_set.scenario_results``.

    Args:
        result_set: The aggregated result set to partition.

    Returns:
        A 2-tuple ``(classical, scenarios)``.

    Raises:
        MaterializationError: If any result carries ``scenario_id`` but
            ``scenario_results`` is None (legacy result set; the
            aggregated per-scenario view is required for rendering).
    """
    scenario_aware = [r for r in result_set.results if r.metadata.scenario_id is not None]
    if scenario_aware and not result_set.scenario_results:
        raise MaterializationError(
            "Result set carries scenario turns but no aggregated scenario_results; "
            "this usually means the run was produced by an older pipeline version. "
            "Re-run the benchmark to produce a result set with scenario_results populated.",
            details={"scenario_turn_count": len(scenario_aware)},
        )
    classical = [r for r in result_set.results if r.metadata.scenario_id is None]
    scenarios: list[Any] = list(result_set.scenario_results or [])
    return classical, scenarios
