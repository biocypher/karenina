"""
Karenina CLI - Command-line interface for benchmark verification.

This module provides a Typer-based CLI for running benchmark verifications from the command line.
"""

import logging

import typer

logger = logging.getLogger(__name__)

# Create the main Typer app
app = typer.Typer(
    name="karenina",
    help="Karenina - LLM Benchmark Verification CLI",
    add_completion=False,
    invoke_without_command=True,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        from karenina import __version__

        print(f"karenina {__version__}")
        raise typer.Exit()


@app.callback()
def callback(
    ctx: typer.Context,
    _version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """
    Karenina CLI for running benchmark verifications.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand was invoked, show help
        print(ctx.get_help())


try:
    from .verify import verify

    app.command(name="verify")(verify)
except ImportError as e:
    logger.warning("verify command unavailable: %s", e)

try:
    from .status import verify_status

    app.command(name="verify-status")(verify_status)
except ImportError as e:
    logger.warning("verify-status command unavailable: %s", e)

try:
    from .preset import preset_app

    app.add_typer(preset_app, name="preset", help="Manage verification presets")
except ImportError as e:
    logger.warning("preset command unavailable: %s", e)

try:
    from .serve import init, serve

    app.command(name="serve")(serve)
    app.command(name="init")(init)
except ImportError as e:
    logger.warning("serve/init commands unavailable: %s", e)

try:
    from .optimize import optimize

    app.command(name="optimize")(optimize)
except ImportError as e:
    # GEPA not installed
    logger.debug("optimize command unavailable: %s", e)

try:
    from .optimize import optimize_history

    app.command(name="optimize-history")(optimize_history)
except ImportError as e:
    # GEPA not installed
    logger.debug("optimize-history command unavailable: %s", e)

try:
    from .optimize import optimize_compare

    app.command(name="optimize-compare")(optimize_compare)
except ImportError as e:
    # GEPA not installed
    logger.debug("optimize-compare command unavailable: %s", e)

try:
    from .analyze_errors import analyze_errors as _analyze_errors_cmd

    app.command(name="analyze-errors")(_analyze_errors_cmd)
except ImportError as e:  # pragma: no cover: defensive guard only
    logger.debug("analyze-errors command unavailable: %s", e)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
