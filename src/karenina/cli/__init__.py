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
)


@app.callback()
def callback() -> None:
    """
    Karenina CLI for running benchmark verifications.

    Use 'karenina --help' to see available commands.
    """
    pass


# Import subcommands (will be implemented in subsequent phases)
# These imports will be added as we implement each command module
try:
    from .preset import preset_app
    from .status import verify_status
    from .verify import verify

    # Register verify command directly on main app
    app.command(name="verify")(verify)

    # Register verify-status command for inspecting progressive save state
    app.command(name="verify-status")(verify_status)

    # Register preset as a sub-command group
    app.add_typer(preset_app, name="preset", help="Manage verification presets")
except ImportError:
    # Commands not yet implemented
    pass


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
