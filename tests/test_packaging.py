"""Packaging regression tests.

These guard the wheel build configuration in ``pyproject.toml``. The failure
mode they exist for: a ``force-include`` entry pointing at a subtree that
``packages`` already covers makes hatchling abort the build with

    ValueError: A second file is being added to the wheel archive at the
    same path: ...

which breaks ``pip install`` straight from the repository.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
import zipfile
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Package data that must reach the installed wheel. Both are read at runtime:
# the ADeLe rubric texts via importlib.resources, the error-analysis prompts
# via the packaged prompt loader.
REQUIRED_WHEEL_PATHS = [
    "karenina/integrations/adele/data/AS.txt",
    "karenina/benchmark/error_analysis/prompts/default_prompt.md",
    "karenina/benchmark/error_analysis/prompts/rubric_guide.md",
]


def _wheel_config() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    return pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]


@pytest.mark.unit
def test_force_include_does_not_duplicate_packages() -> None:
    """No force-include source may sit inside a configured ``packages`` root.

    hatchling treats a file arriving at the same archive path twice as an
    error, not a no-op, so such an entry fails every wheel build.
    """
    config = _wheel_config()
    package_roots = [(REPO_ROOT / package).resolve() for package in config.get("packages", [])]

    offenders = []
    for source in config.get("force-include", {}):
        source_path = (REPO_ROOT / source).resolve()
        for root in package_roots:
            if source_path == root or root in source_path.parents:
                offenders.append(f"{source} is already covered by packages entry {root.relative_to(REPO_ROOT)}")

    assert not offenders, "force-include duplicates files already selected by `packages`:\n" + "\n".join(offenders)


@pytest.mark.integration
@pytest.mark.slow
def test_wheel_builds_and_ships_package_data(tmp_path: Path) -> None:
    """The wheel builds, has no duplicate entries, and carries its data files."""
    if shutil.which("uv"):
        command = ["uv", "build", "--wheel", "--out-dir", str(tmp_path), str(REPO_ROOT)]
    else:
        command = [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path), str(REPO_ROOT)]

    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"wheel build failed:\n{result.stdout}\n{result.stderr}"

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    names = zipfile.ZipFile(wheels[0]).namelist()

    duplicates = [name for name, count in Counter(names).items() if count > 1]
    assert not duplicates, f"wheel contains duplicate archive entries: {duplicates}"

    missing = [path for path in REQUIRED_WHEEL_PATHS if path not in names]
    assert not missing, f"wheel is missing package data: {missing}"

    adele_data = [name for name in names if name.startswith("karenina/integrations/adele/data/")]
    expected_adele = len(list((REPO_ROOT / "src/karenina/integrations/adele/data").glob("*.txt")))
    assert len(adele_data) == expected_adele, (
        f"expected {expected_adele} ADeLe data files in the wheel, found {len(adele_data)}"
    )
