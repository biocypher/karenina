"""
Output, display, and progress helpers for the verify command.

Handles result export, summary display, progress bar rendering,
output validation, and interactive output prompts.
"""

from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt

from karenina.benchmark import Benchmark
from karenina.benchmark.verification import (
    export_verification_results_csv,
    export_verification_results_json,
)
from karenina.benchmark.verification.batch_runner import run_verification_batch
from karenina.benchmark.verification.sinks import ProgressiveFileSink
from karenina.schemas import FinishedTemplate, VerificationConfig, VerificationResult, VerificationResultSet

from .utils import cli_error, create_export_job, validate_output_path

console = Console()


def validate_output_and_prompt(
    output: Path | None,
    progressive_save: bool,
) -> tuple[Path | None, str | None]:
    """
    Validate output path and prompt for one if needed.

    Returns:
        Tuple of (output_path, output_format) - both may be None if no output wanted
    """
    output_format = None

    # Validate output path upfront (fail fast)
    if output:
        try:
            output_format = validate_output_path(output)
        except ValueError as e:
            cli_error(str(e), e)

    # Validate progressive save requires output
    if progressive_save and not output:
        cli_error("--progressive-save requires --output to be specified")

    return output, output_format


def prompt_for_output_interactive(
    progressive_save: bool,
) -> tuple[Path | None, str | None, bool]:
    """
    Prompt for output file in interactive mode.

    Returns:
        Tuple of (output_path, output_format, progressive_save_updated)
    """
    if progressive_save:
        console.print("\n[yellow]Progressive save requires an output file.[/yellow]")
        save_to_file = True
    else:
        save_to_file = Confirm.ask("\nWould you like to save the verification results to a file?", default=True)

    if save_to_file:
        format_choice = Prompt.ask("Output format", choices=["json", "csv"], default="json")
        default_filename = f"verification_results.{format_choice}"
        output_filename = Prompt.ask("Output filename", default=default_filename)

        output = Path(output_filename)

        try:
            output_format = validate_output_path(output)
            console.print(f"[green]✓ Results will be saved to: {output}[/green]")
            return output, output_format, progressive_save
        except ValueError as e:
            console.print(f"[yellow]Warning: {e}. Results will not be saved to file.[/yellow]")
            if progressive_save:
                console.print("[yellow]Disabling progressive save due to invalid output path.[/yellow]")
            return None, None, False

    return None, None, progressive_save


def run_verification_with_progress(
    templates: list[FinishedTemplate],
    config: VerificationConfig,
    benchmark: Benchmark,
    progressive_sink: ProgressiveFileSink | None,
    show_progress: bool,
) -> VerificationResultSet:
    """Run verification with optional progress bar display.

    Incremental persistence is handled by the sink; the progress callback
    here is UI-only.
    """
    if show_progress:
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
            progress_task: TaskID = progress.add_task("Verifying questions...", total=total_verifications)

            def progress_callback(current: int, _total: int, result: VerificationResult | None = None) -> None:
                progress.update(progress_task, completed=current)
                if result and result.template and result.template.verify_result is not None:
                    status = "\u2713" if result.template.verify_result else "\u2717"
                    progress.update(progress_task, description=f"Verifying questions... {status}")

            results = run_verification_batch(
                templates=templates,
                config=config,
                run_name="cli-verification",
                global_rubric=benchmark.get_global_rubric(),
                progress_callback=progress_callback,
                sink=progressive_sink,
            )
    else:
        results = run_verification_batch(
            templates=templates,
            config=config,
            run_name="cli-verification",
            global_rubric=benchmark.get_global_rubric(),
            sink=progressive_sink,
        )

    return results


def export_results(
    output: Path,
    output_format: str,
    final_results: VerificationResultSet,
    config: VerificationConfig,
    benchmark: Benchmark,
    start_time: float,
    end_time: float,
) -> None:
    """Export verification results to file.

    When a :class:`ProgressiveFileSink` is in use its ``on_finalize`` has
    already written the JSON export; calling this function re-serializes
    with fresh job metadata so CSV output and the non-progressive path are
    handled uniformly.
    """
    console.print(f"\n[cyan]Exporting results to {output}...[/cyan]")

    job = create_export_job(final_results, config, "cli-verification", start_time, end_time)

    global_rubric = benchmark.get_global_rubric()
    if output_format == "json":
        export_data = export_verification_results_json(job, final_results, global_rubric, is_complete=True)
    else:  # csv
        export_data = export_verification_results_csv(job, final_results, global_rubric)

    output.write_text(export_data)
    console.print(f"[green]✓ Results exported to {output}[/green]")


def display_summary(
    final_results: VerificationResultSet,
    duration: float,
) -> None:
    """
    Display verification summary.
    """
    total = len(final_results)
    passed = sum(1 for r in final_results if r.template and r.template.verify_result)
    execution_errors = sum(1 for r in final_results if r.metadata.failure is not None)
    failures = total - passed
    not_passed = failures - execution_errors

    console.print("\n[bold]Verification Summary:[/bold]")
    console.print(f"  Total: {total}")
    console.print(f"  Passed: [green]{passed}[/green]")
    if failures > 0:
        console.print(f"  Failures: [red]{failures}[/red]")
        if execution_errors > 0:
            console.print(f"    Execution Errors: [yellow]{execution_errors}[/yellow]")
        if not_passed > 0:
            console.print(f"    Not Passed: [red]{not_passed}[/red]")
    else:
        console.print("  Failures: [green]0[/green]")
    console.print(f"  Duration: {duration:.2f}s")

    if failures == 0:
        console.print("\n[green]\u2713 Verification complete![/green]")
    else:
        console.print(f"\n[yellow]\u26a0 Verification complete with {failures} failure(s)[/yellow]")
