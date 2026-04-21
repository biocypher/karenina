"""Regenerate the byte-exact golden fixtures under this directory.

Usage (from karenina/):
    uv run python -m tests.fixtures.export_format.regen_fixtures

Run manually whenever the on-disk format intentionally changes. Review
the git diff of the fixture files carefully before committing.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import karenina.benchmark.verification.stages.helpers.results_exporter as exporter_mod
from karenina.benchmark.verification.stages.helpers import (
    export_verification_results_json_stream,
)
from tests.fixtures.export_format.fixture_builders import (
    FIXED_EXPORT_TIMESTAMP,
    FIXED_KARENINA_VERSION,
    build_empty_job,
    build_empty_results,
    build_full_job,
    build_full_results,
    build_full_rubric,
)

FIXTURE_DIR = Path(__file__).resolve().parent


def _write_with_determinism(fn: Callable[[], None]) -> None:
    """Wrap the function in the same monkeypatches the tests use."""
    original_strftime = time.strftime
    original_version = exporter_mod.get_karenina_version

    def fake_strftime(fmt: str, t: object = None) -> str:
        return FIXED_EXPORT_TIMESTAMP

    try:
        time.strftime = fake_strftime
        exporter_mod.get_karenina_version = lambda: FIXED_KARENINA_VERSION
        fn()
    finally:
        time.strftime = original_strftime
        exporter_mod.get_karenina_version = original_version


def regen_empty() -> None:
    out_path = FIXTURE_DIR / "results_export_empty.json"
    export_verification_results_json_stream(
        build_empty_job(),
        iter(build_empty_results().results),
        out_path=out_path,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


def regen_full() -> None:
    out_path = FIXTURE_DIR / "results_export_full.json"
    export_verification_results_json_stream(
        build_full_job(),
        iter(build_full_results().results),
        build_full_rubric(),
        is_complete=True,
        out_path=out_path,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


def main() -> int:
    _write_with_determinism(regen_empty)
    _write_with_determinism(regen_full)
    return 0


if __name__ == "__main__":
    sys.exit(main())
