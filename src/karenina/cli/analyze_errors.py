"""`karenina analyze-errors` CLI command."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from karenina.benchmark.error_analysis import analyze_errors as _analyze_errors
from karenina.benchmark.error_analysis.exceptions import (
    ErrorAnalysisError,
    LauncherExecutionError,
    LauncherNoOutputError,
    LauncherNotFoundError,
    LauncherUnavailableError,
    MaterializationError,
)

logger = logging.getLogger(__name__)


def analyze_errors(
    results: Path = typer.Option(..., "--results", help="Path to VerificationResultSet JSON export"),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to benchmark JSON-LD file"),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory to create or overwrite"),
    prompt: Path | None = typer.Option(None, "--prompt", help="Custom PROMPT.md source"),
    launcher: str = typer.Option("prepare-only", "--launcher", help="Registered launcher name"),
    max_trace_chars: int | None = typer.Option(
        None, "--max-trace-chars", help="Override KARENINA_TRACE_TRUNCATION_THRESHOLD"
    ),
    timeout: int = typer.Option(1800, "--timeout", help="Launcher timeout in seconds"),
    force: bool = typer.Option(False, "--force", help="Allow writing into a non-empty out-dir"),
) -> None:
    """Materialize a verification run and optionally run an error-analyst agent."""
    try:
        report_path = _analyze_errors(
            results=results,
            checkpoint=checkpoint,
            out_dir=out_dir,
            prompt_path=prompt,
            launcher=launcher,
            launcher_kwargs={"timeout": timeout},
            max_trace_chars=max_trace_chars,
            force=force,
        )
    except MaterializationError as exc:
        typer.echo(f"Materialization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except LauncherNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except LauncherUnavailableError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except LauncherExecutionError as exc:
        typer.echo(f"Launcher subprocess failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except LauncherNoOutputError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3) from exc
    except ErrorAnalysisError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(str(report_path.resolve()))
