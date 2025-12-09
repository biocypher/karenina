"""
Preset management commands.

This module implements preset management subcommands:
- list: List all available presets
- show: Show preset details
- delete: Delete preset
"""

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from karenina.schemas import VerificationConfig

from .utils import get_preset_path, list_presets

# Create preset app
preset_app = typer.Typer(name="preset", help="Manage verification presets")
console = Console()


@preset_app.command(name="list")
def list_presets_cmd() -> None:
    """
    List all available presets.
    """
    presets = list_presets()

    if not presets:
        console.print("[yellow]No presets found.[/yellow]")
        console.print("Presets are stored in the 'presets/' directory by default.")
        console.print("You can override this with the KARENINA_PRESETS_DIR environment variable.")
        return

    # Create a rich table
    table = Table(title="Available Presets", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Modified", style="green")

    for preset in presets:
        name = preset.get("name", "")
        modified = preset.get("modified", "")[:10] if preset.get("modified") else ""

        table.add_row(name, modified)

    console.print(table)
    console.print(f"\n[dim]Total: {len(presets)} preset(s)[/dim]")


@preset_app.command(name="show")
def show_preset(name_or_path: Annotated[str, typer.Argument(help="Preset name or path")]) -> None:
    """
    Show preset details.
    """
    try:
        # Resolve preset path
        preset_path = get_preset_path(name_or_path)

        # Load preset
        config = VerificationConfig.from_preset(preset_path)

        # Display preset info
        console.print(f"\n[bold cyan]Preset: {preset_path.stem}[/bold cyan]")
        console.print(f"[dim]Path: {preset_path}[/dim]")

        # Display configuration as formatted JSON
        console.print("\n[bold]Configuration:[/bold]")
        config_dict = config.model_dump(mode="json", exclude_none=True)
        config_json = json.dumps(config_dict, indent=2)
        syntax = Syntax(config_json, "json", theme="monokai", line_numbers=False)
        console.print(syntax)

        # Summary
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Answering models: {len(config.answering_models)}")
        console.print(f"  Parsing models: {len(config.parsing_models)}")
        console.print(f"  Replicates: {config.replicate_count}")
        console.print(f"  Rubric enabled: {config.rubric_enabled}")
        console.print(f"  Abstention enabled: {config.abstention_enabled}")
        console.print(f"  Embedding check enabled: {config.embedding_check_enabled}")
        console.print(f"  Deep judgment enabled: {config.deep_judgment_enabled}")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]Error loading preset: {e}[/red]")
        raise typer.Exit(code=1) from e


@preset_app.command(name="delete")
def delete_preset(name_or_path: Annotated[str, typer.Argument(help="Preset name or path")]) -> None:
    """
    Delete preset.
    """
    try:
        # Resolve preset path
        preset_path = get_preset_path(name_or_path)

        # Confirm deletion
        console.print(f"[yellow]About to delete preset:[/yellow] {preset_path}")
        confirm = typer.confirm("Are you sure?")

        if not confirm:
            console.print("[dim]Deletion cancelled.[/dim]")
            return

        # Delete file
        preset_path.unlink()
        console.print("[green]✓ Preset deleted successfully![/green]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]Error deleting preset: {e}[/red]")
        raise typer.Exit(code=1) from e
