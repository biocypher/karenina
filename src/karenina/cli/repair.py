"""CLI command for repairing selected rows of a completed verification run."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.console import Console

from karenina.benchmark import Benchmark, RepairSelection, repair_results_export
from karenina.schemas import VerificationConfig

console = Console()


def repair(
    source: Annotated[Path, typer.Argument(help="Completed JSON result export to repair")],
    benchmark_path: Annotated[Path, typer.Option("--benchmark", help="Benchmark JSON-LD source")],
    preset: Annotated[Path, typer.Option(help="VerificationConfig preset for the original run")],
    mode: Annotated[str, typer.Option(help="Repair execution mode: replay or live")] = "replay",
    output: Annotated[Path | None, typer.Option(help="Destination JSON. Defaults to safe in-place repair")] = None,
    question_id: Annotated[list[str] | None, typer.Option("--question-id", help="Question ID filter")] = None,
    answerer_key: Annotated[
        list[str] | None,
        typer.Option("--answerer-key", help="Canonical answering model key filter"),
    ] = None,
    parser_key: Annotated[
        list[str] | None,
        typer.Option("--parser-key", help="Canonical parsing model key filter"),
    ] = None,
    replicate: Annotated[list[int] | None, typer.Option(help="Replicate filter")] = None,
    failure_group: Annotated[list[str] | None, typer.Option(help="Failure group filter")] = None,
    failure_category: Annotated[list[str] | None, typer.Option(help="Failure category filter")] = None,
    failure_stage: Annotated[list[str] | None, typer.Option(help="Failure stage filter")] = None,
    select_all: Annotated[bool, typer.Option("--all", help="Repair every row explicitly")] = False,
    dry_run: Annotated[bool, typer.Option(help="Show selected identities without model calls or writes")] = False,
) -> None:
    """Repair selected rows through replay or fresh answer generation."""
    if mode not in {"replay", "live"}:
        console.print("[red]Mode must be 'replay' or 'live'.[/red]")
        raise typer.Exit(code=2)
    selection = RepairSelection(
        question_ids=set(question_id or []),
        answerer_keys=set(answerer_key or []),
        parser_keys=set(parser_key or []),
        replicates=set(replicate or []),
        failure_groups=set(failure_group or []),
        failure_categories=set(failure_category or []),
        failure_stages=set(failure_stage or []),
        select_all=select_all,
    )
    if not selection.has_filters():
        console.print("[red]Provide at least one selection filter or pass --all.[/red]")
        raise typer.Exit(code=2)
    try:
        config = VerificationConfig.from_preset(preset)
        benchmark = Benchmark.load(benchmark_path)
        outcome = repair_results_export(
            benchmark,
            source,
            config,
            selection,
            mode=cast(Literal["replay", "live"], mode),
            output_path=output,
            dry_run=dry_run,
        )
    except Exception as exc:
        console.print(f"[red]Repair failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    if dry_run:
        console.print(f"[cyan]Selected {outcome.selected_count} row(s):[/cyan]")
        for key in outcome.selected_keys:
            console.print(key)
        return
    console.print(f"[green]Repaired {outcome.replaced_count} row(s) in {outcome.output_path}.[/green]")
    if outcome.backup_path is not None:
        console.print(f"Backup: {outcome.backup_path}")
    if outcome.provenance_path is not None:
        console.print(f"Provenance: {outcome.provenance_path}")
