"""Optional integrity checks for the downloadable paper data deposit."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from karenina.benchmark import Benchmark, ResultsIOManager
from karenina.schemas.verification import VerificationResult
from paper.bixbench_harness_comparison.config import selected_conditions
from paper.bixbench_harness_comparison.run import discover_archived_sources
from paper.common.bootstrap import data_root, input_path
from paper.config import (
    BIXBENCH_ARCHIVED_BURDENS,
    BIXBENCH_ARCHIVED_RUNS,
    BIXBENCH_BENCHMARK_JSONLD,
    CITATION_ARCHIVED_JUDGMENTS,
    CITATION_ARCHIVED_SELECTED,
    OTP_ADVERSARIAL_BENCHMARK_CSV,
    OTP_ADVERSARIAL_CSV,
    OTP_BENCHMARK_JSONLD,
    OTP_MCP_RESULTS,
    OTP_PARAMETRIC_RESULTS,
    RESPONSE_EMPTY_TRAILING_JUDGMENTS,
    RESPONSE_GROUNDING_PROMPT,
    RESPONSE_GROUNDING_SCORES,
    RESPONSE_GROUNDING_TRACES,
    SYCOPHANCY_ABSTENTION_JUDGMENTS,
    SYCOPHANCY_CAVE_GROUNDING_JUDGMENTS,
    SYCOPHANCY_CAVE_REGEX_JUDGMENTS,
    SYCOPHANCY_DEFINITIVE_RESULTS,
)
from paper.otp_model_comparison.config import ANSWERER_NAMES
from paper.otp_sycophancy_scenarios.config import (
    ANSWERERS,
    DIFFICULTIES,
    FRAMINGS,
    REGIMES,
)
from paper.otp_sycophancy_scenarios.run import _load_archived_cells

pytestmark = pytest.mark.paper_archive

REQUIRED_ARCHIVE_MEMBERS = (
    OTP_BENCHMARK_JSONLD,
    OTP_PARAMETRIC_RESULTS,
    OTP_MCP_RESULTS,
    OTP_ADVERSARIAL_BENCHMARK_CSV,
    OTP_ADVERSARIAL_CSV,
    RESPONSE_EMPTY_TRAILING_JUDGMENTS,
    RESPONSE_GROUNDING_SCORES,
    RESPONSE_GROUNDING_TRACES,
    RESPONSE_GROUNDING_PROMPT,
    CITATION_ARCHIVED_SELECTED,
    CITATION_ARCHIVED_JUDGMENTS,
    SYCOPHANCY_DEFINITIVE_RESULTS,
    SYCOPHANCY_ABSTENTION_JUDGMENTS,
    SYCOPHANCY_CAVE_REGEX_JUDGMENTS,
    SYCOPHANCY_CAVE_GROUNDING_JUDGMENTS,
    BIXBENCH_BENCHMARK_JSONLD,
    BIXBENCH_ARCHIVED_RUNS,
    BIXBENCH_ARCHIVED_BURDENS,
)


@pytest.fixture(scope="module", autouse=True)
def configured_archive() -> None:
    """Skip cleanly when the optional data deposit cannot be discovered."""
    try:
        data_root()
    except RuntimeError as error:
        pytest.skip(str(error))


def _iter_results(path: Path) -> Iterator[VerificationResult]:
    return cast(Iterator[VerificationResult], ResultsIOManager.iter_from_json(path))


def test_every_configured_archive_member_exists() -> None:
    """Every path referenced by a regeneration entry point must resolve."""
    for relative_path in REQUIRED_ARCHIVE_MEMBERS:
        assert input_path(relative_path).exists()


@pytest.mark.parametrize("relative_path", [OTP_PARAMETRIC_RESULTS, OTP_MCP_RESULTS])
def test_model_comparison_export_covers_the_configured_matrix(relative_path: str) -> None:
    """Each retained arm must cover every configured QA cell exactly once."""
    benchmark = Benchmark.load(input_path(OTP_BENCHMARK_JSONLD))
    question_ids = set(benchmark.get_question_ids())
    required = {
        (question_id, answerer, parser, replicate)
        for question_id in question_ids
        for answerer in ANSWERER_NAMES
        for parser in ANSWERER_NAMES
        for replicate in range(1, 4)
    }
    observed: set[tuple[str, str, str, int]] = set()
    row_count = 0
    for result in _iter_results(input_path(relative_path)):
        row_count += 1
        observed.add(
            (
                result.metadata.question_id,
                result.metadata.answering.model_name,
                result.metadata.parsing.model_name,
                result.metadata.replicate or 1,
            )
        )

    assert row_count == len(observed)
    assert observed == required


def test_scenario_and_bixbench_archives_cover_the_configured_cells() -> None:
    """The two crossed experiments must have every selected archived cell."""
    scenario_cells = _load_archived_cells(
        answerers=ANSWERERS,
        regimes=REGIMES,
        difficulties=DIFFICULTIES,
        framings=FRAMINGS,
        limit=None,
    )
    assert len(scenario_cells) == len(ANSWERERS) * len(REGIMES) * len(DIFFICULTIES) * len(FRAMINGS)

    conditions = selected_conditions()
    bixbench_sources = discover_archived_sources(input_path(BIXBENCH_ARCHIVED_RUNS), conditions)
    assert len(bixbench_sources) == len(conditions) * 3
