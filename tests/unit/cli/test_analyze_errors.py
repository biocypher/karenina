"""Tests for the `karenina analyze-errors` CLI command."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from karenina.cli import app
from karenina.schemas.entities.question import Question
from karenina.schemas.results.failure import FailureCategory
from karenina.schemas.results.verification_result_set import VerificationResultSet
from tests.unit.benchmark.error_analysis.fixtures import make_failure, make_pass
from tests.unit.benchmark.error_analysis.test_materializer_build import _StubBenchmark


@pytest.fixture
def sample_inputs(tmp_path):
    q = Question(question="2+2?", raw_answer="4")
    bench = _StubBenchmark(name="sample", questions=[q])
    passed = make_pass(question_id=q.id)
    failed = make_failure(
        question_id=q.id,
        category=FailureCategory.CONTENT,
        stage="verify_template",
        reason="oops",
    )
    rs = VerificationResultSet(results=[passed, failed], scenario_results=None)

    results_path = tmp_path / "results.json"
    results_path.write_text(rs.model_dump_json())

    # The CLI expects a real Benchmark JSON-LD file; intercept Benchmark.load in
    # the test by monkeypatching instead. Return a placeholder path.
    return rs, bench, results_path


@pytest.fixture
def runner():
    return CliRunner(mix_stderr=False)


@pytest.mark.unit
class TestAnalyzeErrorsCli:
    def test_unknown_launcher_exits_with_2(self, runner, tmp_path, sample_inputs, monkeypatch):
        _rs, bench, results_path = sample_inputs
        monkeypatch.setattr(
            "karenina.benchmark.benchmark.Benchmark.load",
            classmethod(lambda _cls, _path, **_kwargs: bench),
        )
        result = runner.invoke(
            app,
            [
                "analyze-errors",
                "--results",
                str(results_path),
                "--checkpoint",
                str(tmp_path / "does-not-matter.json"),
                "--out-dir",
                str(tmp_path / "out"),
                "--launcher",
                "no-such-launcher",
            ],
        )
        assert result.exit_code == 2
        assert "prepare-only" in result.stdout or "prepare-only" in result.stderr

    def test_missing_report_exits_with_3(self, runner, tmp_path, sample_inputs, monkeypatch):
        _rs, bench, results_path = sample_inputs
        monkeypatch.setattr(
            "karenina.benchmark.benchmark.Benchmark.load",
            classmethod(lambda _cls, _path, **_kwargs: bench),
        )
        result = runner.invoke(
            app,
            [
                "analyze-errors",
                "--results",
                str(results_path),
                "--checkpoint",
                str(tmp_path / "any.json"),
                "--out-dir",
                str(tmp_path / "out"),
            ],
        )
        # prepare-only launcher raises LauncherNoOutputError.
        assert result.exit_code == 3

    def test_force_on_nonempty_preserves_prior_report(
        self,
        runner,
        tmp_path,
        sample_inputs,
        monkeypatch,
    ):
        _rs, bench, results_path = sample_inputs
        monkeypatch.setattr(
            "karenina.benchmark.benchmark.Benchmark.load",
            classmethod(lambda _cls, _path, **_kwargs: bench),
        )
        out = tmp_path / "out"
        out.mkdir()
        (out / "REPORT.md").write_text("old analysis")

        # Pre-existing prior analysis: facade renames it to REPORT.previous.md
        # but the default prepare-only launcher still fails with exit 3.
        result = runner.invoke(
            app,
            [
                "analyze-errors",
                "--results",
                str(results_path),
                "--checkpoint",
                str(tmp_path / "any.json"),
                "--out-dir",
                str(out),
                "--force",
            ],
        )
        assert result.exit_code == 3
        assert (out / "REPORT.previous.md").read_text() == "old analysis"
