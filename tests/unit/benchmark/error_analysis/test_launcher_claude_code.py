"""Tests for the opt-in claude-code launcher (all subprocess calls mocked)."""

from __future__ import annotations

import subprocess

import pytest


@pytest.mark.unit
class TestClaudeCodeLauncher:
    def test_registers_when_imported(self):
        import importlib

        importlib.import_module("karenina.benchmark.error_analysis.launchers.claude_code")
        from karenina.benchmark.error_analysis.launcher import list_launchers

        assert "claude-code" in list_launchers()

    def test_missing_binary_raises_unavailable(self, tmp_path, monkeypatch):
        from karenina.benchmark.error_analysis.exceptions import LauncherUnavailableError
        from karenina.benchmark.error_analysis.launchers.claude_code import ClaudeCodeLauncher

        monkeypatch.setattr(
            "karenina.benchmark.error_analysis.launchers.claude_code.shutil.which",
            lambda _name: None,
        )
        with pytest.raises(LauncherUnavailableError) as exc:
            ClaudeCodeLauncher().run(tmp_path)
        assert "claude-code" in str(exc.value)

    def test_invokes_subprocess_with_expected_args(self, tmp_path, monkeypatch):
        from karenina.benchmark.error_analysis.launchers import claude_code

        (tmp_path / "PROMPT.md").write_text("prompt")
        calls: list[dict] = []

        def _which(_name: str):
            return "/usr/local/bin/claude"

        def _run(cmd, **kwargs):
            calls.append({"cmd": cmd, "kwargs": kwargs})
            (tmp_path / "REPORT.md").write_text("generated")
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(claude_code.shutil, "which", _which)
        monkeypatch.setattr(claude_code.subprocess, "run", _run)

        report = claude_code.ClaudeCodeLauncher().run(tmp_path, timeout=60)
        assert report == tmp_path / "REPORT.md"
        assert calls[0]["kwargs"]["cwd"] == tmp_path
        assert calls[0]["kwargs"]["timeout"] == 60
        assert "-p" in calls[0]["cmd"]
        assert any("@" in part and "PROMPT.md" in part for part in calls[0]["cmd"])

    def test_subprocess_failure_wraps_in_launcher_execution_error(self, tmp_path, monkeypatch):
        from karenina.benchmark.error_analysis.exceptions import LauncherExecutionError
        from karenina.benchmark.error_analysis.launchers import claude_code

        monkeypatch.setattr(claude_code.shutil, "which", lambda _name: "/usr/local/bin/claude")

        def _run(cmd, **kwargs):
            raise subprocess.CalledProcessError(
                returncode=2,
                cmd=cmd,
                stderr=b"permission denied",
            )

        monkeypatch.setattr(claude_code.subprocess, "run", _run)
        with pytest.raises(LauncherExecutionError) as exc:
            claude_code.ClaudeCodeLauncher().run(tmp_path)
        assert exc.value.returncode == 2
        assert "permission denied" in exc.value.stderr_tail
