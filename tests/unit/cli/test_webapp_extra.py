"""Tests for the optional webapp packaging entry point."""

from __future__ import annotations

import tomllib
from pathlib import Path

from typer.testing import CliRunner

from karenina.cli import app

runner = CliRunner()


def test_webapp_extra_depends_on_karenina_server() -> None:
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    webapp_deps = data["project"]["optional-dependencies"]["webapp"]

    assert any(dep.startswith("karenina-server") for dep in webapp_deps)


def test_serve_command_is_registered_in_core_cli() -> None:
    result = runner.invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "Start the Karenina webapp server" in result.stdout
    assert "Skip first-time setup" in result.stdout
