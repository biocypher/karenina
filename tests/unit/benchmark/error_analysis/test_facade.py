"""Tests for the analyze_errors facade."""

from __future__ import annotations

from pathlib import Path

import pytest

from karenina.benchmark.error_analysis import analyze_errors
from karenina.benchmark.error_analysis.exceptions import (
    LauncherNoOutputError,
    LauncherNotFoundError,
)
from karenina.schemas.entities.question import Question
from karenina.schemas.results.failure import FailureCategory
from karenina.schemas.results.verification_result_set import VerificationResultSet

from .fixtures import make_failure, make_pass
from .test_materializer_build import _StubBenchmark


@pytest.fixture
def tiny_benchmark():
    q1 = Question(question="2+2?", raw_answer="4")
    return _StubBenchmark(name="sample", questions=[q1])


@pytest.fixture
def tiny_result_set(tiny_benchmark):
    (q1,) = tiny_benchmark.questions
    passed = make_pass(question_id=q1.id)
    failed = make_failure(
        question_id=q1.id,
        category=FailureCategory.CONTENT,
        stage="verify_template",
        reason="oops",
    )
    return VerificationResultSet(results=[passed, failed], scenario_results=None)


class _RecordingLauncher:
    def __init__(self):
        self.seen: list[Path] = []

    def run(self, analysis_dir: Path, **_):
        self.seen.append(analysis_dir)
        (analysis_dir / "REPORT.md").write_text("# Findings\nnothing to see here.")
        return analysis_dir / "REPORT.md"


@pytest.mark.unit
class TestFacade:
    def test_default_launcher_requires_user_to_produce_report(
        self,
        tmp_path,
        tiny_benchmark,
        tiny_result_set,
    ):
        with pytest.raises(LauncherNoOutputError):
            analyze_errors(
                results=tiny_result_set,
                checkpoint=tiny_benchmark,
                out_dir=tmp_path / "analysis",
            )

    def test_instance_launcher_is_used_verbatim(
        self,
        tmp_path,
        tiny_benchmark,
        tiny_result_set,
    ):
        launcher = _RecordingLauncher()
        out = tmp_path / "analysis"
        report_path = analyze_errors(
            results=tiny_result_set,
            checkpoint=tiny_benchmark,
            out_dir=out,
            launcher=launcher,
        )
        assert report_path == out / "REPORT.md"
        assert launcher.seen == [out]
        assert report_path.read_text().startswith("# Findings")

    def test_unknown_named_launcher_raises(
        self,
        tmp_path,
        tiny_benchmark,
        tiny_result_set,
    ):
        with pytest.raises(LauncherNotFoundError):
            analyze_errors(
                results=tiny_result_set,
                checkpoint=tiny_benchmark,
                out_dir=tmp_path / "analysis",
                launcher="no-such-launcher",
            )

    def test_prompt_md_written_with_substitutions(
        self,
        tmp_path,
        tiny_benchmark,
        tiny_result_set,
    ):
        launcher = _RecordingLauncher()
        out = tmp_path / "analysis"
        analyze_errors(
            results=tiny_result_set,
            checkpoint=tiny_benchmark,
            out_dir=out,
            launcher=launcher,
        )
        prompt = (out / "PROMPT.md").read_text()
        assert "sample" in prompt  # benchmark name
        assert "anthropic:claude-opus-4-6" in prompt
