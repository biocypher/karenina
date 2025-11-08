"""
Verify command for running benchmark verifications.

This module implements the main 'karenina verify' command.
"""

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn

from karenina.benchmark import Benchmark
from karenina.benchmark.exporter import export_verification_results_csv, export_verification_results_json
from karenina.benchmark.verification.batch_runner import run_verification_batch
from karenina.schemas import VerificationConfig, VerificationResult

from .utils import (
    create_export_job,
    filter_templates_by_ids,
    filter_templates_by_indices,
    get_preset_path,
    parse_question_indices,
    validate_output_path,
)

console = Console()


def verify(
    benchmark_path: Annotated[str, typer.Argument(help="Path to benchmark JSON-LD file")],
    preset: Annotated[Path | None, typer.Option(help="Path to preset configuration")] = None,
    output: Annotated[Path | None, typer.Option(help="Output file (.json or .csv)")] = None,
    questions: Annotated[str | None, typer.Option(help="Question indices (e.g., '0,1,2' or '5-10')")] = None,
    question_ids: Annotated[str | None, typer.Option(help="Comma-separated question IDs")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show progress bar")] = False,
    interactive: Annotated[bool, typer.Option("--interactive", help="Interactive configuration mode")] = False,
    _mode: Annotated[str, typer.Option("--mode", help="Interactive mode (basic or advanced)")] = "basic",
) -> None:
    """
    Run verification on a benchmark.

    Examples:
        # Verify with preset, filter 2 questions
        karenina verify checkpoint.jsonld --preset default.json --questions 0,1

        # Verify range with output
        karenina verify checkpoint.jsonld --preset default.json --questions 0-5 --output results.json

        # Interactive mode
        karenina verify checkpoint.jsonld --interactive --mode basic
    """
    try:
        # Step 1: Validate output path upfront (fail fast)
        output_format = None
        if output:
            try:
                output_format = validate_output_path(output)
            except ValueError as e:
                console.print(f"[red]Error: {e}[/red]")
                raise typer.Exit(code=1) from e

        # Step 2: Load benchmark
        console.print("[cyan]Loading benchmark...[/cyan]")
        try:
            benchmark = Benchmark.load(Path(benchmark_path))
        except Exception as e:
            console.print(f"[red]Error loading benchmark: {e}[/red]")
            raise typer.Exit(code=1) from e

        console.print(f"[green]✓ Loaded benchmark: {benchmark.name or 'Unnamed'}[/green]")
        console.print(f"  Total questions: {len(benchmark.get_all_questions())}")

        # Step 3: Load or build config
        selected_question_indices = None
        if interactive:
            from .interactive import build_config_interactively

            config, selected_question_indices = build_config_interactively(benchmark, mode=_mode)
        elif preset:
            console.print("[cyan]Loading preset...[/cyan]")
            try:
                preset_path = get_preset_path(str(preset))
                config = VerificationConfig.from_preset(preset_path)
            except Exception as e:
                console.print(f"[red]Error loading preset: {e}[/red]")
                raise typer.Exit(code=1) from e

            console.print(f"[green]✓ Loaded preset from: {preset_path}[/green]")
        else:
            console.print("[red]Error: Either --preset or --interactive is required[/red]")
            raise typer.Exit(code=1)

        # Step 4: Get and filter templates
        all_templates = benchmark.get_finished_templates()

        if not all_templates:
            console.print("[yellow]Warning: No finished templates found in benchmark[/yellow]")
            raise typer.Exit(code=1)

        # Filter templates
        templates = all_templates
        if selected_question_indices is not None:
            # Use indices from interactive mode
            templates = filter_templates_by_indices(all_templates, selected_question_indices)
            console.print(f"[dim]Using {len(templates)} question(s) from interactive selection[/dim]")
        elif questions:
            try:
                indices = parse_question_indices(questions, len(all_templates))
                templates = filter_templates_by_indices(all_templates, indices)
                console.print(f"[dim]Filtered to {len(templates)} question(s) by indices[/dim]")
            except ValueError as e:
                console.print(f"[red]Error parsing question indices: {e}[/red]")
                raise typer.Exit(code=1) from e
        elif question_ids:
            ids = [id.strip() for id in question_ids.split(",")]
            templates = filter_templates_by_ids(all_templates, ids)
            console.print(f"[dim]Filtered to {len(templates)} question(s) by IDs[/dim]")

        if not templates:
            console.print("[red]Error: No templates to verify after filtering[/red]")
            raise typer.Exit(code=1)

        # Step 5: Run verification
        console.print("\n[bold cyan]Starting verification...[/bold cyan]")
        console.print(f"  Questions: {len(templates)}")
        console.print(f"  Answering models: {len(config.answering_models)}")
        console.print(f"  Parsing models: {len(config.parsing_models)}")
        console.print(f"  Replicates: {config.replicate_count}")
        console.print()

        start_time = time.time()

        # Run verification with optional progress bar
        if verbose:
            # Calculate total verifications
            total_verifications = (
                len(templates) * len(config.answering_models) * len(config.parsing_models) * config.replicate_count
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Verifying questions...", total=total_verifications)

                def progress_callback(current: int, _total: int, result: VerificationResult | None = None) -> None:
                    """Update progress bar on each verification completion."""
                    progress.update(task, completed=current)
                    if result and result.verify_result is not None:
                        # Add pass/fail indicator to description
                        status = "✓" if result.verify_result else "✗"
                        progress.update(task, description=f"Verifying questions... {status}")

                results = run_verification_batch(
                    templates=templates,
                    config=config,
                    run_name="cli-verification",
                    global_rubric=benchmark.get_global_rubric(),
                    progress_callback=progress_callback,
                )
        else:
            results = run_verification_batch(
                templates=templates,
                config=config,
                run_name="cli-verification",
                global_rubric=benchmark.get_global_rubric(),
                progress_callback=None,
            )

        end_time = time.time()
        duration = end_time - start_time

        # Step 6: Export or display results
        if output:
            console.print(f"\n[cyan]Exporting results to {output}...[/cyan]")

            # Create job for export (comprehensive backend format)
            job = create_export_job(results, config, "cli-verification", start_time, end_time)

            # Export using backend exporter (includes all fields)
            if output_format == "json":
                export_data = export_verification_results_json(job, results)
            else:  # csv
                export_data = export_verification_results_csv(job, results, benchmark.get_global_rubric())

            # Write to file
            output.write_text(export_data)
            console.print(f"[green]✓ Results exported to {output}[/green]")

        # Display summary
        total = len(results)
        passed = sum(1 for r in results.values() if r.verify_result)
        failed = total - passed
        errors = sum(1 for r in results.values() if not r.completed_without_errors)

        console.print("\n[bold]Verification Summary:[/bold]")
        console.print(f"  Total: {total}")
        console.print(f"  Passed: [green]{passed}[/green]")
        console.print(f"  Failed: [red]{failed}[/red]")
        if errors > 0:
            console.print(f"  Errors: [yellow]{errors}[/yellow]")
        console.print(f"  Duration: {duration:.2f}s")

        console.print("\n[green]✓ Verification complete![/green]")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(code=1) from e
