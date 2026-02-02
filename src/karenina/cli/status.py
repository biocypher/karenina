"""
Status command for inspecting progressive save state files.

This module provides the 'karenina verify-status' command to check
the progress of interrupted verification jobs.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from karenina.utils.progressive_save import ProgressiveJobStatus, inspect_state_file

console = Console()


def _format_elapsed_time(seconds: float | None) -> str:
    """Format elapsed time in human-readable format."""
    if seconds is None:
        return "unknown"

    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def _format_file_size(size_bytes: int | None) -> str:
    """Format file size in human-readable format."""
    if size_bytes is None:
        return "N/A"

    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def verify_status(
    state_file: Annotated[Path, typer.Argument(help="Path to the .state file to inspect")],
    show_tasks: Annotated[
        bool,
        typer.Option(
            "--show-tasks",
            "-t",
            help="Show detailed list of completed and pending task IDs",
        ),
    ] = False,
    show_questions: Annotated[
        bool,
        typer.Option(
            "--show-questions",
            "-q",
            help="Show list of completed and pending question IDs",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output status as JSON",
        ),
    ] = False,
) -> None:
    """
    Inspect a progressive save state file and show job status.

    This command shows the progress of an interrupted verification job,
    including completed and pending tasks, elapsed time, and configuration.

    Examples:
        # Basic status
        karenina verify-status results.json.state

        # Show question IDs
        karenina verify-status results.json.state --show-questions

        # Show all task IDs (verbose)
        karenina verify-status results.json.state --show-tasks

        # Output as JSON for scripting
        karenina verify-status results.json.state --json
    """
    try:
        status = inspect_state_file(state_file)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e

    if json_output:
        _output_json(status)
        return

    _output_rich(status, show_tasks, show_questions)


def _output_json(status: ProgressiveJobStatus) -> None:
    """Output status as JSON."""
    import json

    output = {
        "state_file": str(status.state_file_path),
        "output_path": str(status.output_path),
        "benchmark_path": status.benchmark_path,
        "progress": {
            "total_tasks": status.total_tasks,
            "completed_count": status.completed_count,
            "pending_count": status.pending_count,
            "progress_percent": status.progress_percent,
        },
        "timing": {
            "created_at": status.created_at,
            "last_updated_at": status.last_updated_at,
            "elapsed_seconds": status.elapsed_time,
        },
        "config": {
            "answering_models": status.answering_models,
            "parsing_models": status.parsing_models,
            "replicate_count": status.replicate_count,
        },
        "files": {
            "tmp_file_exists": status.tmp_file_exists,
            "tmp_file_size": status.tmp_file_size,
        },
        "completed_question_ids": status.completed_question_ids,
        "pending_question_ids": status.pending_question_ids,
        "completed_task_ids": status.completed_task_ids,
        "pending_task_ids": status.pending_task_ids,
    }

    console.print(json.dumps(output, indent=2))


def _output_rich(status: ProgressiveJobStatus, show_tasks: bool, show_questions: bool) -> None:
    """Output status with rich formatting."""
    # Header panel
    if status.pending_count == 0:
        header_style = "green"
        header_text = "Job Complete"
    else:
        header_style = "yellow"
        header_text = "Job In Progress"

    console.print(Panel(f"[bold]{header_text}[/bold]", style=header_style))

    # Progress bar
    console.print("\n[bold]Progress[/bold]")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Tasks", total=status.total_tasks)
        progress.update(task, completed=status.completed_count)

    # Summary table
    console.print("\n[bold]Summary[/bold]")
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Key", style="cyan")
    summary_table.add_column("Value")

    summary_table.add_row("State file", str(status.state_file_path))
    summary_table.add_row("Output path", str(status.output_path))
    summary_table.add_row("Benchmark", status.benchmark_path)
    summary_table.add_row("", "")
    summary_table.add_row("Completed", f"[green]{status.completed_count}[/green]")
    summary_table.add_row("Pending", f"[yellow]{status.pending_count}[/yellow]" if status.pending_count > 0 else "0")
    summary_table.add_row("Total", str(status.total_tasks))
    summary_table.add_row("", "")
    summary_table.add_row("Started", status.created_at)
    summary_table.add_row("Last update", status.last_updated_at)
    summary_table.add_row("Elapsed", _format_elapsed_time(status.elapsed_time))

    console.print(summary_table)

    # Config table
    console.print("\n[bold]Configuration[/bold]")
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="cyan")
    config_table.add_column("Value")

    config_table.add_row("Answering models", ", ".join(status.answering_models) or "none")
    config_table.add_row("Parsing models", ", ".join(status.parsing_models) or "none")
    config_table.add_row("Replicates", str(status.replicate_count))

    console.print(config_table)

    # File status
    console.print("\n[bold]Files[/bold]")
    file_table = Table(show_header=False, box=None, padding=(0, 2))
    file_table.add_column("Key", style="cyan")
    file_table.add_column("Value")

    tmp_status = "[green]exists[/green]" if status.tmp_file_exists else "[red]missing[/red]"
    file_table.add_row("Results file (.tmp)", f"{tmp_status} ({_format_file_size(status.tmp_file_size)})")
    file_table.add_row("State file (.state)", "[green]exists[/green]")

    console.print(file_table)

    # Question IDs
    if show_questions:
        console.print("\n[bold]Question IDs[/bold]")

        if status.completed_question_ids:
            console.print(f"\n[green]Completed ({len(status.completed_question_ids)}):[/green]")
            for qid in status.completed_question_ids:
                console.print(f"  {qid}")

        if status.pending_question_ids:
            console.print(f"\n[yellow]Pending ({len(status.pending_question_ids)}):[/yellow]")
            for qid in status.pending_question_ids:
                console.print(f"  {qid}")

    # Task IDs (verbose)
    if show_tasks:
        console.print("\n[bold]Task IDs[/bold]")

        if status.completed_task_ids:
            console.print(f"\n[green]Completed ({len(status.completed_task_ids)}):[/green]")
            for tid in status.completed_task_ids:
                console.print(f"  [dim]{tid}[/dim]")

        if status.pending_task_ids:
            console.print(f"\n[yellow]Pending ({len(status.pending_task_ids)}):[/yellow]")
            for tid in status.pending_task_ids:
                console.print(f"  [dim]{tid}[/dim]")

    # Resume hint
    if status.pending_count > 0:
        console.print(f"\n[dim]To resume: karenina verify --resume {status.state_file_path}[/dim]")
