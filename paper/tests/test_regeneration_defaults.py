"""Guard the user-facing contract that full entry points regenerate by default."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from paper.bixbench_harness_comparison import run as bixbench_run
from paper.otp_adversarial_generation import run as adversarial_run
from paper.otp_benchmark_curation import simplified as curation_run
from paper.otp_citation_audit import run as citation_run
from paper.otp_model_comparison import run as comparison_run
from paper.otp_sycophancy_scenarios import run as sycophancy_run


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module", "required_defaults"),
    [
        (comparison_run, {"reuse_stored_results": False}),
        (citation_run, {"reuse_stored_judgments": False, "screen_only": False}),
        (adversarial_run, {"reuse_stored_samples": False, "limit": None}),
        (
            sycophancy_run,
            {"reuse_stored_results": False, "reuse_stored_judgments": False},
        ),
        (
            bixbench_run,
            {"reuse_stored_results": False, "reuse_stored_judgments": False},
        ),
    ],
)
def test_full_entry_points_do_not_reuse_archives_by_default(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    required_defaults: dict[str, object],
) -> None:
    """No-argument run commands must select model-calling paths."""
    captured: dict[str, object] = {}

    def capture(_output: Path, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(module, "bootstrap", lambda _verbose=False: None)
    monkeypatch.setattr(module, "run", capture)
    monkeypatch.setattr(sys, "argv", [f"paper.{module.__name__}"])

    module.main()  # type: ignore[attr-defined]

    for key, value in required_defaults.items():
        assert captured[key] == value


@pytest.mark.unit
def test_curation_processes_the_full_source_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The benchmark-curation entry point must not select a small slice implicitly."""
    source = tmp_path / "questions.xlsx"
    source.touch()
    captured: dict[str, object] = {}

    def capture(source_path: Path, output_dir: Path, **kwargs: object) -> None:
        captured.update({"source": source_path, "output_dir": output_dir, **kwargs})

    monkeypatch.setattr(curation_run, "bootstrap", lambda _verbose=False: None)
    monkeypatch.setattr(curation_run, "build_draft_benchmark", capture)
    monkeypatch.setattr(sys, "argv", ["curation", "--source", str(source)])

    curation_run.main()

    assert captured["source"] == source
    assert captured["limit"] is None


@pytest.mark.unit
def test_paper_controller_installs_every_fresh_run_extra() -> None:
    """The controller image must include the BixBench DeepAgents adapter."""
    dockerfile = (Path(__file__).parents[1] / "docker" / "Dockerfile").read_text()
    sync_line = next(line for line in dockerfile.splitlines() if line.startswith("RUN uv sync"))
    assert "--extra paper-analysis" in sync_line
    assert "--extra deep-agents" in sync_line


@pytest.mark.unit
def test_paper_controller_excludes_credentials_and_run_outputs() -> None:
    """Local credentials and generated traces must not enter image builds."""
    repository_root = Path(__file__).parents[2]
    dockerignore = {
        line.strip()
        for line in (repository_root / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    dockerfile = (repository_root / "paper" / "docker" / "Dockerfile").read_text()
    assert ".env" in dockerignore
    assert "**/out" in dockerignore
    assert "COPY . " not in dockerfile
