"""
Preset management commands.

This module implements preset management subcommands:
- list: List all available presets
- show: Show preset details
- create: Create preset from config file
- delete: Delete preset
"""

import json
from pathlib import Path
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
        console.print("Presets are stored in the 'benchmark_presets/' directory by default.")
        console.print("You can override this with the KARENINA_PRESETS_DIR environment variable.")
        return

    # Create a rich table
    table = Table(title="Available Presets", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Created", style="green")

    for preset in presets:
        name = preset.get("name", "")
        description = preset.get("description") or "[dim]No description[/dim]"
        created = preset.get("created_at", "")[:10] if preset.get("created_at") else ""

        table.add_row(name, description, created)

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

        # Read the raw file for metadata
        with open(preset_path) as f:
            preset_data = json.load(f)

        # Display preset info
        console.print(f"\n[bold cyan]Preset: {preset_data.get('name', 'Unknown')}[/bold cyan]")
        console.print(f"[dim]Path: {preset_path}[/dim]")

        if preset_data.get("description"):
            console.print(f"\n[bold]Description:[/bold] {preset_data['description']}")

        console.print(f"\n[bold]Created:[/bold] {preset_data.get('created_at', 'Unknown')}")
        console.print(f"[bold]Updated:[/bold] {preset_data.get('updated_at', 'Unknown')}")

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


@preset_app.command(name="create")
def create_preset(
    name: Annotated[str, typer.Argument(help="Preset name")],
    config_json: Annotated[Path, typer.Argument(help="Path to config JSON file")],
    description: Annotated[str | None, typer.Option(help="Preset description")] = None,
) -> None:
    """
    Create preset from config file.
    """
    try:
        # Load config from JSON
        if not config_json.exists():
            console.print(f"[red]Error: Config file not found: {config_json}[/red]")
            raise typer.Exit(code=1)

        with open(config_json) as f:
            config_data = json.load(f)

        # Create VerificationConfig
        config = VerificationConfig(**config_data)

        # Save as preset
        preset_info = config.save_preset(name=name, description=description)

        console.print("[green]✓ Preset created successfully![/green]")
        console.print(f"\n[bold]Name:[/bold] {preset_info['name']}")
        console.print(f"[bold]Path:[/bold] {preset_info['filepath']}")
        if description:
            console.print(f"[bold]Description:[/bold] {description}")

    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON in config file: {e}[/red]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]Error creating preset: {e}[/red]")
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
