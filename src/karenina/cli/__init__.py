"""
Karenina CLI - Command-line interface for benchmark verification.

This module provides a Typer-based CLI for running benchmark verifications from the command line.
"""

import typer

# Create the main Typer app
app = typer.Typer(
    name="karenina",
    help="Karenina - LLM Benchmark Verification CLI",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context) -> None:
    """
    Karenina CLI for running benchmark verifications.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand was invoked, show help
        print(ctx.get_help())


# Import subcommands (will be implemented in subsequent phases)
# These imports will be added as we implement each command module
try:
    from .preset import preset_app
    from .serve import init, serve
    from .status import verify_status
    from .verify import verify

    # Register verify command directly on main app
    app.command(name="verify")(verify)

    # Register verify-status command for inspecting progressive save state
    app.command(name="verify-status")(verify_status)

    # Register preset as a sub-command group
    app.add_typer(preset_app, name="preset", help="Manage verification presets")

    # Register serve and init commands for webapp
    app.command(name="serve")(serve)
    app.command(name="init")(init)
except ImportError:
    # Commands not yet implemented
    pass

# Import GEPA optimization commands (optional - requires gepa package)
try:
    from .optimize import optimize, optimize_compare, optimize_history

    # Register optimize command directly on main app
    app.command(name="optimize")(optimize)

    # Register optimize-history command for viewing past runs
    app.command(name="optimize-history")(optimize_history)

    # Register optimize-compare command for comparing runs
    app.command(name="optimize-compare")(optimize_compare)
except ImportError:
    # GEPA optimization not available
    pass


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
