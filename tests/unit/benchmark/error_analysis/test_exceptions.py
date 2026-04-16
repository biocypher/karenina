"""Tests for the error-analysis exception hierarchy."""

from __future__ import annotations

import subprocess

import pytest

from karenina.benchmark.error_analysis.exceptions import (
    ErrorAnalysisError,
    LauncherExecutionError,
    LauncherNoOutputError,
    LauncherNotFoundError,
    LauncherUnavailableError,
    MaterializationError,
)
from karenina.exceptions import KareninaError


@pytest.mark.unit
class TestErrorAnalysisExceptions:
    def test_base_inherits_from_karenina_error(self):
        assert issubclass(ErrorAnalysisError, KareninaError)

    def test_all_leaves_inherit_from_base(self):
        for cls in (
            MaterializationError,
            LauncherNotFoundError,
            LauncherUnavailableError,
            LauncherExecutionError,
            LauncherNoOutputError,
        ):
            assert issubclass(cls, ErrorAnalysisError)

    def test_launcher_not_found_includes_registered_names(self):
        exc = LauncherNotFoundError("codex", registered=["prepare-only", "claude-code"])
        assert "codex" in str(exc)
        assert "prepare-only" in str(exc)
        assert "claude-code" in str(exc)

    def test_launcher_unavailable_includes_hint(self):
        exc = LauncherUnavailableError(
            launcher="claude-code",
            hint="Install via https://docs.anthropic.com/claude/code",
        )
        assert "claude-code" in str(exc)
        assert "https://docs.anthropic.com/claude/code" in str(exc)

    def test_launcher_execution_error_preserves_returncode_and_stderr_tail(self):
        called = subprocess.CalledProcessError(
            returncode=42,
            cmd=["claude", "-p", "@PROMPT.md"],
            stderr=b"x" * 3000,
        )
        exc = LauncherExecutionError("claude-code", called)
        assert exc.returncode == 42
        # last 2048 bytes preserved
        assert exc.stderr_tail == "x" * 2048
        assert exc.__cause__ is None  # explicit chaining is caller's job; stash, don't chain by default

    def test_launcher_no_output_error_mentions_report_path(self, tmp_path):
        exc = LauncherNoOutputError(analysis_dir=tmp_path)
        assert str(tmp_path / "REPORT.md") in str(exc)

    def test_materialization_error_preserves_detail_map(self):
        exc = MaterializationError(
            "missing template for question q_foo",
            details={"question_id": "q_foo"},
        )
        assert exc.details == {"question_id": "q_foo"}
