"""
Optimize command for GEPA-based prompt optimization.

This module implements the 'karenina optimize' command for automated
optimization of prompts, instructions, and tool descriptions using GEPA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .utils import cli_error

if TYPE_CHECKING:
    from karenina.benchmark import Benchmark
    from karenina.schemas import VerificationConfig

console = Console()

VALID_TARGETS = {"answering_system_prompt", "parsing_instructions", "mcp_tool_descriptions"}


def _check_gepa_available() -> bool:
    """Check if GEPA is available."""
    try:
        from karenina.integrations.gepa import GEPA_AVAILABLE

        return GEPA_AVAILABLE
    except ImportError:
        return False


def _validate_optimize_params(
    target: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float | None,
) -> None:
    """Validate optimization parameters. Exits on invalid input."""
    total_ratio = train_ratio + val_ratio + (test_ratio or 0.0)
    if abs(total_ratio - 1.0) > 0.001:
        cli_error(f"Split ratios must sum to 1.0, got {total_ratio}")

    for t in target:
        if t not in VALID_TARGETS:
            cli_error(f"Invalid target '{t}'. Valid targets: {', '.join(VALID_TARGETS)}")


def _load_optimize_resources(
    checkpoint: Path,
    preset: Path | None,
    seed_prompt: str | None,
    targets: list[str],
) -> tuple[Benchmark, VerificationConfig | None, dict[str, str]]:
    """Load benchmark, preset config, and seed prompts.

    Returns (benchmark, config, seed_prompts).
    """
    from karenina.benchmark import Benchmark
    from karenina.schemas import VerificationConfig

    console.print(f"[blue]Loading benchmark:[/blue] {checkpoint}")
    try:
        benchmark = Benchmark.load(checkpoint)
    except FileNotFoundError:
        cli_error(f"Benchmark file not found: {checkpoint}")
    except Exception as e:
        cli_error(f"loading benchmark: {e}", e)
    console.print(f"  {benchmark.question_count} questions, {benchmark.finished_count} finished")

    config = None
    if preset:
        console.print(f"[blue]Loading preset:[/blue] {preset}")
        try:
            with open(preset) as f:
                preset_data = json.load(f)
        except FileNotFoundError:
            cli_error(f"Preset file not found: {preset}")
        except json.JSONDecodeError as e:
            cli_error(f"Invalid JSON in preset: {e}", e)
        config = VerificationConfig.model_validate(preset_data)

    seed_prompts: dict[str, str] = {}
    if seed_prompt:
        seed_prompts[targets[0]] = seed_prompt

    return benchmark, config, seed_prompts


def _display_optimize_results(result: Any) -> None:
    """Display optimization results as a Rich table with optional prompt previews."""
    console.print("")
    console.print("[bold green]Optimization Complete![/bold green]")
    console.print("")

    table = Table(title="Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Train Score", f"{result.train_score:.3f}")
    table.add_row("Val Score", f"{result.val_score:.3f}")
    if result.test_score is not None:
        table.add_row("Test Score", f"{result.test_score:.3f}")
    table.add_row("Improvement", f"{result.improvement:+.1%}")

    console.print(table)

    if result.answering_system_prompt:
        console.print("")
        console.print("[bold]Optimized Answering System Prompt:[/bold]")
        prompt = result.answering_system_prompt
        console.print(f"[dim]{prompt[:200]}...[/dim]" if len(prompt) > 200 else f"[dim]{prompt}[/dim]")

    if result.parsing_instructions:
        console.print("")
        console.print("[bold]Optimized Parsing Instructions:[/bold]")
        instructions = result.parsing_instructions
        console.print(
            f"[dim]{instructions[:200]}...[/dim]" if len(instructions) > 200 else f"[dim]{instructions}[/dim]"
        )


def _export_optimize_results(
    result: Any,
    benchmark_name: str,
    targets: list[str],
    output: Path | None,
    prompts_output: Path | None,
) -> None:
    """Export optimization results to preset and/or prompts JSON files."""
    if prompts_output:
        from karenina.integrations.gepa import export_prompts_json

        optimized_prompts: dict[str, str] = {}
        if result.answering_system_prompt:
            optimized_prompts["answering_system_prompt"] = result.answering_system_prompt
        if result.parsing_instructions:
            optimized_prompts["parsing_instructions"] = result.parsing_instructions
        if result.mcp_tool_descriptions:
            for tool_name, desc in result.mcp_tool_descriptions.items():
                optimized_prompts[f"mcp_tool_{tool_name}"] = desc

        export_prompts_json(
            optimized_prompts,
            metadata={
                "benchmark": benchmark_name,
                "targets": targets,
                "train_score": result.train_score,
                "val_score": result.val_score,
                "test_score": result.test_score,
                "improvement": result.improvement,
            },
            output_path=prompts_output,
        )
        console.print(f"\n[blue]Prompts exported to:[/blue] {prompts_output}")

    if output:
        console.print(f"[blue]Preset exported to:[/blue] {output}")


def _display_history_table(runs: list[Any]) -> None:
    """Display optimization history as a Rich table."""
    table = Table(title=f"Optimization History ({len(runs)} runs)")
    table.add_column("Run ID", style="cyan")
    table.add_column("Timestamp", style="dim")
    table.add_column("Benchmark")
    table.add_column("Targets")
    table.add_column("Val Score", style="green")
    table.add_column("Improvement", style="yellow")

    for run in runs:
        table.add_row(
            run.run_id,
            run.timestamp.strftime("%Y-%m-%d %H:%M"),
            run.benchmark_name[:20] + "..." if len(run.benchmark_name) > 20 else run.benchmark_name,
            ", ".join(run.targets)[:30],
            f"{run.val_score:.3f}",
            f"{run.improvement:+.1%}",
        )

    console.print(table)


def optimize(
    checkpoint: Annotated[
        Path,
        typer.Argument(
            help="Path to the JSON-LD benchmark checkpoint file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    target: Annotated[
        list[str] | None,
        typer.Option(
            "--target",
            "-t",
            help="Component to optimize. Can be specified multiple times. "
            "Valid: answering_system_prompt, parsing_instructions, mcp_tool_descriptions",
        ),
    ] = None,
    preset: Annotated[
        Path | None,
        typer.Option(
            "--preset",
            "-p",
            help="Path to a verification preset JSON file for base config",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to save optimized preset JSON file",
        ),
    ] = None,
    prompts_output: Annotated[
        Path | None,
        typer.Option(
            "--prompts-output",
            help="Path to save just the optimized prompts JSON file",
        ),
    ] = None,
    seed_prompt: Annotated[
        str | None,
        typer.Option(
            "--seed-prompt",
            "-s",
            help="Initial seed prompt for optimization",
        ),
    ] = None,
    reflection_model: Annotated[
        str,
        typer.Option(
            "--reflection-model",
            "-r",
            help="Model for GEPA's reflection LLM",
        ),
    ] = "openai/gpt-4o",
    max_calls: Annotated[
        int,
        typer.Option(
            "--max-calls",
            "-m",
            help="Maximum number of GEPA optimization iterations",
        ),
    ] = 150,
    train_ratio: Annotated[
        float,
        typer.Option(
            "--train-ratio",
            help="Fraction of questions for training (0-1)",
        ),
    ] = 0.8,
    val_ratio: Annotated[
        float,
        typer.Option(
            "--val-ratio",
            help="Fraction of questions for validation (0-1)",
        ),
    ] = 0.2,
    test_ratio: Annotated[
        float | None,
        typer.Option(
            "--test-ratio",
            help="Optional fraction for test set (0-1). If set, will evaluate final prompts on held-out test set.",
        ),
    ] = None,
    frontier_type: Annotated[
        str,
        typer.Option(
            "--frontier-type",
            help="Pareto frontier tracking: 'instance', 'objective' (recommended), 'hybrid', 'cartesian'",
        ),
    ] = "objective",
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Random seed for reproducibility",
        ),
    ] = None,
    tracker_path: Annotated[
        Path | None,
        typer.Option(
            "--tracker",
            help="Path to SQLite file for tracking optimization history",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed progress: iteration updates, score improvements, Pareto frontier changes, and final summary",
        ),
    ] = False,
) -> None:
    """
    Optimize prompts and instructions using GEPA.

    Uses GEPA (Generative Evolutionary Prompt Advancement) to automatically
    optimize text components used in benchmark verification. Karenina's
    verification pipeline serves as the optimization metric.

    Examples:

        # Basic optimization
        karenina optimize benchmark.jsonld --target answering_system_prompt

        # With preset and output
        karenina optimize benchmark.jsonld -p preset.json -o optimized.json

        # Multiple targets
        karenina optimize benchmark.jsonld -t answering_system_prompt -t parsing_instructions

        # With test set evaluation
        karenina optimize benchmark.jsonld --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
    """
    if not _check_gepa_available():
        cli_error("GEPA is not installed. Install with: pip install karenina[gepa]")

    if target is None:
        target = ["answering_system_prompt"]

    _validate_optimize_params(target, train_ratio, val_ratio, test_ratio)

    try:
        benchmark, config, seed_prompts = _load_optimize_resources(checkpoint, preset, seed_prompt, target)

        def progress_callback(percentage: float, message: str) -> None:
            if verbose:
                console.print(f"  [{percentage:5.1f}%] {message}")

        console.print("")
        console.print("[bold green]Starting GEPA optimization...[/bold green]")
        console.print(f"  Targets: {', '.join(target)}")
        console.print(f"  Reflection model: {reflection_model}")
        console.print(f"  Max iterations: {max_calls}")
        console.print(
            f"  Split: {train_ratio:.0%} train, {val_ratio:.0%} val"
            + (f", {test_ratio:.0%} test" if test_ratio else "")
        )
        console.print("")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Optimizing...", total=None)

            result = benchmark.optimize(
                targets=target,
                config=config,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                reflection_model=reflection_model,
                max_metric_calls=max_calls,
                frontier_type=cast(Literal["instance", "objective", "hybrid", "cartesian"], frontier_type),
                seed_prompts=seed_prompts if seed_prompts else None,
                tracker_path=tracker_path,
                export_preset_path=output,
                progress_callback=progress_callback if verbose else None,
                verbose=verbose,
            )

            progress.update(task, description="Complete!")

        _display_optimize_results(result)
        _export_optimize_results(result, benchmark.name, target, output, prompts_output)

    except KeyboardInterrupt:
        console.print("\n[yellow]Optimization interrupted.[/yellow]")
        raise typer.Exit(code=130) from None
    except typer.Exit:
        raise
    except Exception as e:
        cli_error(str(e), e)


def optimize_history(
    benchmark_name: Annotated[
        str | None,
        typer.Argument(
            help="Optional benchmark name to filter history",
        ),
    ] = None,
    tracker_path: Annotated[
        Path,
        typer.Option(
            "--tracker",
            help="Path to SQLite file with optimization history",
        ),
    ] = Path("~/.karenina/optimization_history.db"),
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of runs to show",
        ),
    ] = 20,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: table, json, csv",
        ),
    ] = "table",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to export history (for json/csv formats)",
        ),
    ] = None,
) -> None:
    """
    View optimization history.

    Shows past optimization runs from the tracking database.

    Examples:

        # View all history
        karenina optimize-history

        # Filter by benchmark
        karenina optimize-history my_benchmark

        # Export as JSON
        karenina optimize-history --format json -o history.json
    """
    # Check GEPA availability
    if not _check_gepa_available():
        cli_error("GEPA integration not available. Install with: pip install karenina[gepa]")

    try:
        from karenina.integrations.gepa import OptimizationTracker

        tracker_path = tracker_path.expanduser()
        if not tracker_path.exists():
            console.print(f"[yellow]No optimization history found at {tracker_path}[/yellow]")
            return

        tracker = OptimizationTracker(tracker_path)
        runs = tracker.list_runs(benchmark_name=benchmark_name, limit=limit)

        if not runs:
            console.print("[yellow]No optimization runs found.[/yellow]")
            return

        if format == "table":
            _display_history_table(runs)

        elif format in ("json", "csv"):
            # Type narrowing - we've confirmed format is "json" or "csv"
            format_literal: Literal["json", "csv"] = "json" if format == "json" else "csv"
            history_str = tracker.export_history(format=format_literal, benchmark_name=benchmark_name)

            if output:
                with open(output, "w") as f:
                    f.write(history_str)
                console.print(f"[blue]History exported to:[/blue] {output}")
            else:
                console.print(history_str)

        else:
            cli_error(f"Invalid format '{format}'. Use: table, json, csv")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130) from None
    except typer.Exit:
        raise
    except Exception as e:
        cli_error(str(e), e)


def optimize_compare(
    run_ids: Annotated[
        list[str],
        typer.Argument(
            help="Run IDs to compare (2-5 runs)",
        ),
    ],
    tracker_path: Annotated[
        Path,
        typer.Option(
            "--tracker",
            help="Path to SQLite file with optimization history",
        ),
    ] = Path("~/.karenina/optimization_history.db"),
) -> None:
    """
    Compare multiple optimization runs.

    Shows a side-by-side comparison of optimization runs.

    Examples:

        karenina optimize-compare abc123 def456 ghi789
    """
    if len(run_ids) < 2:
        cli_error("Need at least 2 run IDs to compare")

    if len(run_ids) > 5:
        cli_error("Maximum 5 runs can be compared at once")

    # Check GEPA availability
    if not _check_gepa_available():
        cli_error("GEPA integration not available. Install with: pip install karenina[gepa]")

    try:
        from karenina.integrations.gepa import OptimizationTracker

        tracker_path = tracker_path.expanduser()
        if not tracker_path.exists():
            cli_error(f"No optimization history found at {tracker_path}")

        tracker = OptimizationTracker(tracker_path)
        comparison = tracker.compare_runs(run_ids)

        if not comparison["runs"]:
            console.print("[yellow]No matching runs found.[/yellow]")
            return

        # Build comparison table
        table = Table(title="Run Comparison")
        table.add_column("Metric", style="cyan")

        for run_id in run_ids:
            if run_id in comparison["runs"]:
                style = "bold green" if run_id == comparison["best"]["val_score"] else None
                table.add_column(run_id, style=style)

        # Add rows
        table.add_row(
            "Benchmark",
            *[
                comparison["runs"].get(rid, {}).get("benchmark_name", "N/A")
                for rid in run_ids
                if rid in comparison["runs"]
            ],
        )
        table.add_row(
            "Targets",
            *[
                ", ".join(comparison["runs"].get(rid, {}).get("targets", []))
                for rid in run_ids
                if rid in comparison["runs"]
            ],
        )
        table.add_row(
            "Val Score",
            *[f"{comparison['metrics']['val_score'].get(rid, 0):.3f}" for rid in run_ids if rid in comparison["runs"]],
        )
        table.add_row(
            "Improvement",
            *[
                f"{comparison['metrics']['improvement'].get(rid, 0):+.1%}"
                for rid in run_ids
                if rid in comparison["runs"]
            ],
        )
        table.add_row(
            "Metric Calls",
            *[str(comparison["metrics"]["metric_calls"].get(rid, 0)) for rid in run_ids if rid in comparison["runs"]],
        )

        console.print(table)

        # Highlight best
        console.print("")
        console.print(f"[bold green]Best val_score:[/bold green] {comparison['best']['val_score']}")
        console.print(f"[bold yellow]Best improvement:[/bold yellow] {comparison['best']['improvement']}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130) from None
    except typer.Exit:
        raise
    except Exception as e:
        cli_error(str(e), e)


# Create sub-app for optimize commands
optimize_app = typer.Typer(
    name="optimize",
    help="GEPA-based prompt optimization commands",
)

# Register commands
optimize_app.command(name="run")(optimize)
optimize_app.command(name="history")(optimize_history)
optimize_app.command(name="compare")(optimize_compare)
